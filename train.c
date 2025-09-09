#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "data.h"
#include "transformer.h"

int main() {
    srand(time(NULL));
    openblas_set_num_threads(4);

    // Parameters
    const int seq_len = 128;
    const int d_model = 64;
    const int hidden_dim = 256;
    const int num_samples = 1024;
    const int batch_size = 32;
    const int num_layers = 3;
    
    // Generate synthetic data
    float *X, *y;
    generate_data(&X, &y, seq_len, num_samples, d_model, -5.0f, 5.0f);
    
    // Initialize transformer
    Transformer* transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, false);
    
    // Training parameters
    const int num_epochs = 50;
    const float learning_rate = 0.0003f;
    const int num_batches = num_samples / batch_size;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs + 1; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Calculate batch offset
            int batch_offset = batch * batch_size * seq_len * d_model;

            // Forward pass
            forward_pass_transformer(transformer, &X[batch_offset]);
            
            // Calculate loss
            float loss = calculate_loss_transformer(transformer, &y[batch_offset]);
            epoch_loss += loss;

            // Don't update weights after final evaluation
            if (epoch == num_epochs) continue;

            // Backward pass
            zero_gradients_transformer(transformer);
            backward_pass_transformer(transformer, &X[batch_offset], NULL);
            
            // Update weights
            update_weights_transformer(transformer, learning_rate);
        }
        
        epoch_loss /= num_batches;

        // Print progress
        if (epoch % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.8f\n", epoch, num_epochs, epoch_loss);
        }
    }

    // Get timestamp for filenames
    char model_fname[64], data_fname[64];
    time_t now = time(NULL);
    strftime(model_fname, sizeof(model_fname), "%Y%m%d_%H%M%S_transformer.bin", localtime(&now));
    strftime(data_fname, sizeof(data_fname), "%Y%m%d_%H%M%S_transformer_data.csv", localtime(&now));

    // Save model and data with timestamped filenames
    save_transformer(transformer, model_fname);
    save_data(X, y, seq_len, num_samples, d_model, data_fname);
    
    // Load the model back and verify
    printf("\nVerifying saved model...\n");

    // Load the model back with original batch_size
    Transformer* loaded_transformer = load_transformer(model_fname, batch_size);

    // Forward pass with loaded model on first batch
    forward_pass_transformer(loaded_transformer, X);
    
    // Get predictions from the final MLP layer
    float* output = loaded_transformer->mlp_layers[loaded_transformer->num_layers - 1]->layer_output;

    // Evaluate model performance on first batch
    printf("Feature\tR²\t\tMAE\t\tSample Predictions\n");
    printf("-------\t--------\t--------\t--------------------------------\n");

    for (int d = 0; d < d_model; d++) {
        // Calculate mean for R² across all positions and batches for this feature
        float y_mean = 0.0f;
        int total_elements = batch_size * seq_len;
        
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                y_mean += y[idx];
            }
        }
        y_mean /= total_elements;
        
        // Calculate R² and MAE for this feature
        float ss_res = 0.0f, ss_tot = 0.0f, mae = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                float pred = output[idx];
                float actual = y[idx];
                float diff = pred - actual;
                
                ss_res += diff * diff;
                ss_tot += (actual - y_mean) * (actual - y_mean);
                mae += fabs(diff);
            }
        }
        
        float r2 = 1.0f - (ss_res / ss_tot);
        mae /= total_elements;
        
        // Print summary with sample predictions from first batch, first few positions
        printf("d%d\t%.6f\t%.3f\t\t", d, r2, mae);
        for (int sample = 0; sample < 3; sample++) {
            // Show predictions from batch 0, positions 0, 1, 2
            int idx = 0 * seq_len * d_model + sample * d_model + d;
            float pred = output[idx];
            float actual = y[idx];
            printf("%.2f/%.2f ", pred, actual);
        }
        printf("\n");
    }
    
    // Cleanup
    free(X);
    free(y);
    free_transformer(transformer);
    free_transformer(loaded_transformer);
    
    return 0;
}