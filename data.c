#include "data.h"

void generate_data(float** X, float** y, int seq_len, int num_samples, int d_model,
                   float range_min, float range_max) {
    // Row-major layout: [num_samples x seq_len x d_model]
    const int total = num_samples * seq_len * d_model;
    
    *X = (float*)malloc(total * sizeof(float));
    *y = (float*)malloc(total * sizeof(float));
    
    // Fill X with random data
    float range = range_max - range_min;
    for (int i = 0; i < total; i++) {
        (*X)[i] = range_min + ((float)rand() / (float)RAND_MAX) * range;
    }
    
    // Create attention matrix A: [seq_len × seq_len] - this is shared across all samples
    float* A = (float*)malloc(seq_len * seq_len * sizeof(float));
    float a_scale = 1.0f / sqrtf(seq_len);
    
    for (int i = 0; i < seq_len * seq_len; i++) {
        A[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * a_scale;
    }
    
    // Row-wise softmax on A to create proper attention weights
    for (int i = 0; i < seq_len; i++) {
        float max_val = -1e30f;
        for (int j = 0; j < seq_len; j++) {
            float v = A[i * seq_len + j];
            if (v > max_val) max_val = v;
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            float exp_val = expf(A[i * seq_len + j] - max_val);
            A[i * seq_len + j] = exp_val;
            sum += exp_val;
        }
        
        for (int j = 0; j < seq_len; j++) {
            A[i * seq_len + j] /= sum;
        }
    }
    
    // Create MLP-like transformation matrix W: [d_model × d_model] - this is also shared
    float* W = (float*)malloc(d_model * d_model * sizeof(float));
    float w_scale = 1.0f / sqrtf(d_model);
    
    for (int i = 0; i < d_model * d_model; i++) {
        W[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * w_scale;
    }
    
    // Allocate temporary buffer for attention output
    float* attn_output = (float*)malloc(total * sizeof(float));
    
    // Step 1: Apply attention transformation for each batch: attn_output = A * X
    for (int b = 0; b < num_samples; b++) {
        float* X_b = &(*X)[b * seq_len * d_model];
        float* attn_b = &attn_output[b * seq_len * d_model];
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, d_model, seq_len,
                    1.0f, A, seq_len,
                    X_b, d_model,
                    0.0f, attn_b, d_model);
    }
    
    // Step 2: First residual connection: attn_output += X
    cblas_saxpy(total, 1.0f, *X, 1, attn_output, 1);
    
    // Step 3: Apply MLP transformation: y = attn_output * W
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                num_samples * seq_len, d_model, d_model,
                1.0f, attn_output, d_model,
                W, d_model,
                0.0f, *y, d_model);
    
    // Step 4: Second residual connection: y += attn_output
    cblas_saxpy(total, 1.0f, attn_output, 1, *y, 1);
    
    // Add small noise to make the problem more realistic
    float noise_scale = range * 0.001f;
    for (int i = 0; i < total; i++) {
        float noise = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * noise_scale;
        (*y)[i] += noise;
    }
    
    free(A);
    free(W);
    free(attn_output);
    
    printf("Generated transformer data: %d samples, seq_len %d, d_model %d\n", 
           num_samples, seq_len, d_model);
}

void save_data(float* X, float* y, int seq_len, int num_samples, int d_model,
               const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: cannot write %s\n", filename);
        return;
    }
    
    // Header: batch_id, seq_pos, then features, then targets
    fprintf(f, "batch_id,seq_pos,");
    for (int d = 0; d < d_model; d++) {
        fprintf(f, "x_d%d,", d);
    }
    for (int d = 0; d < d_model; d++) {
        fprintf(f, "y_d%d%s", d, d == d_model-1 ? "\n" : ",");
    }
    
    // Data: one row per (batch, sequence_position)
    for (int b = 0; b < num_samples; b++) {
        for (int t = 0; t < seq_len; t++) {
            fprintf(f, "%d,%d,", b, t);
            
            // X features for this (batch, position)
            for (int d = 0; d < d_model; d++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                fprintf(f, "%.6f,", X[idx]);
            }
            
            // Y features for this (batch, position)
            for (int d = 0; d < d_model; d++) {
                int idx = b * seq_len * d_model + t * d_model + d;
                fprintf(f, "%.6f%s", y[idx], d == d_model-1 ? "\n" : ",");
            }
        }
    }
    
    fclose(f);
    printf("Data saved to %s\n", filename);
}