#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../attention/data.h"
#include "transformer.h"

void print_data_samples(float* X, float* y, int seq_len, int feature_dim) {
    printf("Sample data for inspection:\n");
    printf("============================\n");
    for (int i = 0; i < 3; i++) {
        print_sample_data(X, y, i, seq_len, feature_dim);
    }
}

void train_model(Transformer* transformer, float* X, float* y, int num_samples, int batch_size, int num_epochs, float learning_rate) {
    const int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    // Allocate device memory for input and output
    int seq_size = batch_size * transformer->seq_len * transformer->d_model;
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, seq_size * sizeof(float)));
    
    printf("Starting training...\n");
    printf("Architecture: d_model=%d, seq_len=%d, mlp_hidden=%d, batch_size=%d, num_samples=%d, num_batches=%d\n\n", 
           transformer->d_model, transformer->seq_len, transformer->mlp_hidden, batch_size, num_samples, num_batches);
    
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = (start_idx + batch_size > num_samples) ? num_samples : start_idx + batch_size;
            if (end_idx - start_idx < batch_size) continue;
            
            float* X_batch = X + start_idx * transformer->seq_len * transformer->d_model;
            float* y_batch = y + start_idx * transformer->seq_len * transformer->d_model;
            
            // Copy batch data to device
            CHECK_CUDA(cudaMemcpy(d_X, X_batch, seq_size * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_y, y_batch, seq_size * sizeof(float), cudaMemcpyHostToDevice));
            
            forward_pass_transformer(transformer, d_X);
            total_loss += calculate_loss_transformer(transformer, d_y);

            if (epoch < num_epochs) {
                zero_gradients_transformer(transformer);
                backward_pass_transformer(transformer, d_X);
                update_weights_transformer(transformer, learning_rate);
            }
        }

        if (epoch % 2 == 0) {
            printf("Epoch [%d/%d], Average Loss: %.8f\n", epoch, num_epochs, total_loss / num_batches);
        }
    }
    
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_y));
}

void evaluate_model(Transformer* transformer, float* X_eval, int eval_samples, int seq_len, int feature_dim, int batch_size) {
    const int eval_batches = (eval_samples + batch_size - 1) / batch_size;
    int correct_predictions = 0, total_predictions = 0;
    
    // Allocate device memory
    int seq_size = batch_size * seq_len * feature_dim;
    float *d_X;
    CHECK_CUDA(cudaMalloc(&d_X, seq_size * sizeof(float)));
    
    // Allocate host memory for predictions
    float* predictions = (float*)malloc(seq_size * sizeof(float));
    
    for (int batch = 0; batch < eval_batches; batch++) {
        int start_idx = batch * batch_size;
        int end_idx = (start_idx + batch_size > eval_samples) ? eval_samples : start_idx + batch_size;
        if (end_idx - start_idx < batch_size) continue;
        
        float* X_batch = X_eval + start_idx * seq_len * feature_dim;
        
        // Copy batch data to device and run forward pass
        CHECK_CUDA(cudaMemcpy(d_X, X_batch, seq_size * sizeof(float), cudaMemcpyHostToDevice));
        forward_pass_transformer(transformer, d_X);
        
        // Copy predictions back to host
        CHECK_CUDA(cudaMemcpy(predictions, transformer->attention2->d_layer_output, 
                             seq_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int sample = 0; sample < batch_size; sample++) {
            int global_sample = start_idx + sample;
            
            // Find expected max row
            int expected_max_row = 0;
            float max_val = X_eval[global_sample * seq_len * feature_dim + 0];
            for (int seq = 1; seq < seq_len; seq++) {
                float val = X_eval[global_sample * seq_len * feature_dim + seq * feature_dim + 0];
                if (val > max_val) {
                    max_val = val;
                    expected_max_row = seq;
                }
            }
            
            // Check prediction accuracy
            int sample_correct = 1;
            float tolerance = 0.5f;
            
            for (int seq = 0; seq < seq_len && sample_correct; seq++) {
                for (int feat = 0; feat < feature_dim && sample_correct; feat++) {
                    float predicted = predictions[sample * seq_len * feature_dim + seq * feature_dim + feat];
                    float expected = X_eval[global_sample * seq_len * feature_dim + expected_max_row * feature_dim + feat];
                    
                    if (fabsf(predicted - expected) > tolerance) {
                        sample_correct = 0;
                    }
                }
            }
            
            if (sample_correct) correct_predictions++;
            total_predictions++;
        }
    }
    
    printf("Transformer Task Accuracy on NEW data: %d/%d (%.1f%%)\n", 
           correct_predictions, total_predictions, 
           (100.0f * correct_predictions) / total_predictions);
    
    free(predictions);
    CHECK_CUDA(cudaFree(d_X));
}

void print_evaluation_samples(Transformer* transformer, float* X_eval, float* y_eval, int seq_len, int feature_dim, int batch_size) {
    printf("\nSample Predictions from NEW evaluation data (first 5 samples):\n");
    printf("=============================================================\n");

    // Allocate device memory
    int seq_size = batch_size * seq_len * feature_dim;
    float *d_X;
    CHECK_CUDA(cudaMalloc(&d_X, seq_size * sizeof(float)));
    
    // Allocate host memory for predictions
    float* predictions = (float*)malloc(seq_size * sizeof(float));
    
    // Copy first batch to device and run forward pass
    CHECK_CUDA(cudaMemcpy(d_X, X_eval, seq_size * sizeof(float), cudaMemcpyHostToDevice));
    forward_pass_transformer(transformer, d_X);
    
    // Copy predictions back to host
    CHECK_CUDA(cudaMemcpy(predictions, transformer->attention2->d_layer_output, 
                         seq_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int sample = 0; sample < 5; sample++) {
        printf("\nSample %d:\n", sample);
        printf("Input:\n");
        
        for (int seq = 0; seq < seq_len; seq++) {
            printf("  [");
            for (int feat = 0; feat < feature_dim; feat++) {
                printf("%6.2f", X_eval[sample * seq_len * feature_dim + seq * feature_dim + feat]);
                if (feat < feature_dim - 1) printf(", ");
            }
            printf("]\n");
        }
        
        // Find expected max row
        int expected_max_row = 0;
        float max_val = X_eval[sample * seq_len * feature_dim + 0];
        for (int seq = 1; seq < seq_len; seq++) {
            float val = X_eval[sample * seq_len * feature_dim + seq * feature_dim + 0];
            if (val > max_val) {
                max_val = val;
                expected_max_row = seq;
            }
        }
        
        printf("Expected max row: %d (value: %.2f)\n", expected_max_row, max_val);
        printf("Model Output:\n");
        
        for (int seq = 0; seq < seq_len; seq++) {
            printf("  [");
            for (int feat = 0; feat < feature_dim; feat++) {
                printf("%6.2f", predictions[sample * seq_len * feature_dim + seq * feature_dim + feat]);
                if (feat < feature_dim - 1) printf(", ");
            }
            printf("]\n");
        }
        
        printf("Target Output:\n");
        
        for (int seq = 0; seq < seq_len; seq++) {
            printf("  [");
            for (int feat = 0; feat < feature_dim; feat++) {
                printf("%6.2f", y_eval[sample * seq_len * feature_dim + seq * feature_dim + feat]);
                if (feat < feature_dim - 1) printf(", ");
            }
            printf("]\n");
        }
    }
    
    // Calculate MSE per feature
    printf("\nMSE per feature (first evaluation batch):\n");
    for (int feat = 0; feat < feature_dim; feat++) {
        float mse = 0.0f;
        for (int sample = 0; sample < batch_size; sample++) {
            for (int seq = 0; seq < seq_len; seq++) {
                float pred = predictions[sample * seq_len * feature_dim + seq * feature_dim + feat];
                float actual = y_eval[sample * seq_len * feature_dim + seq * feature_dim + feat];
                float diff = pred - actual;
                mse += diff * diff;
            }
        }
        mse /= (batch_size * seq_len);
        printf("Feature %d MSE: %.6f\n", feat, mse);
    }
    
    free(predictions);
    CHECK_CUDA(cudaFree(d_X));
}

int main() {
    srand(time(NULL));

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    const int seq_len = 16, feature_dim = 8, num_samples = 65536, batch_size = 512, mlp_hidden = 128;
    
    float *X, *y;
    generate_attention_data(&X, &y, num_samples, seq_len, feature_dim);
    print_data_samples(X, y, seq_len, feature_dim);
    
    Transformer* transformer = init_transformer(feature_dim, seq_len, batch_size, mlp_hidden, cublas_handle);
    train_model(transformer, X, y, num_samples, batch_size, 50, 0.001f);

    // Get timestamp and save
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_model.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_data.csv", localtime(&now));

    save_transformer(transformer, model_fname);
    save_data(X, y, num_samples, seq_len, feature_dim, data_fname);
    
    printf("\nVerifying saved model...\n");
    Transformer* loaded_transformer = load_transformer(model_fname, batch_size, cublas_handle);
    
    // Allocate device memory for verification
    int seq_size = batch_size * seq_len * feature_dim;
    float *d_X, *d_y;
    CHECK_CUDA(cudaMalloc(&d_X, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, seq_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, seq_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y, seq_size * sizeof(float), cudaMemcpyHostToDevice));
    
    forward_pass_transformer(loaded_transformer, d_X);
    printf("Loss with loaded model (sample batch): %.8f\n", calculate_loss_transformer(loaded_transformer, d_y));

    // Generate and evaluate on new data
    printf("\nGenerating new evaluation dataset...\n");
    const int eval_samples = 2048;
    float *X_eval, *y_eval;
    generate_attention_data(&X_eval, &y_eval, eval_samples, seq_len, feature_dim);
    
    printf("\nEvaluating model performance on NEW data...\n");
    evaluate_model(loaded_transformer, X_eval, eval_samples, seq_len, feature_dim, batch_size);
    print_evaluation_samples(loaded_transformer, X_eval, y_eval, seq_len, feature_dim, batch_size);
    
    free(X); free(y); free(X_eval); free(y_eval);
    CHECK_CUDA(cudaFree(d_X)); CHECK_CUDA(cudaFree(d_y));
    free_transformer(transformer); free_transformer(loaded_transformer);
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    return 0;
}