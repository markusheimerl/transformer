#include "layernorm.h"

// Initialize LayerNorm
LayerNorm* init_layernorm(int d_model, int seq_len, int batch_size) {
    LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
    
    // Store dimensions
    ln->d_model = d_model;
    ln->seq_len = seq_len;
    ln->batch_size = batch_size;
    ln->epsilon = 1e-6f;
    
    // Initialize Adam parameters
    ln->beta1 = 0.9f;
    ln->beta2 = 0.999f;
    ln->epsilon_adam = 1e-8f;
    ln->t = 0;
    ln->weight_decay = 0.01f;
    
    int total_seq = batch_size * seq_len;
    
    // Allocate device memory for parameters
    CHECK_CUDA(cudaMalloc(&ln->d_gamma, d_model * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ln->d_beta, d_model * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ln->d_gamma_grad, d_model * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ln->d_beta_grad, d_model * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&ln->d_gamma_m, d_model * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ln->d_gamma_v, d_model * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ln->d_beta_m, d_model * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ln->d_beta_v, d_model * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&ln->d_normalized, total_seq * d_model * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ln->d_mean, total_seq * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ln->d_var, total_seq * sizeof(float)));
    
    // Allocate device memory for backward pass buffers
    CHECK_CUDA(cudaMalloc(&ln->d_grad_input, total_seq * d_model * sizeof(float)));
    
    // Initialize parameters on host then copy to device
    float* h_gamma = (float*)malloc(d_model * sizeof(float));
    float* h_beta = (float*)malloc(d_model * sizeof(float));
    
    // Initialize gamma to 1.0 and beta to 0.0 (standard LayerNorm initialization)
    for (int i = 0; i < d_model; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }
    
    CHECK_CUDA(cudaMemcpy(ln->d_gamma, h_gamma, d_model * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ln->d_beta, h_beta, d_model * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(ln->d_gamma_m, 0, d_model * sizeof(float)));
    CHECK_CUDA(cudaMemset(ln->d_gamma_v, 0, d_model * sizeof(float)));
    CHECK_CUDA(cudaMemset(ln->d_beta_m, 0, d_model * sizeof(float)));
    CHECK_CUDA(cudaMemset(ln->d_beta_v, 0, d_model * sizeof(float)));
    
    free(h_gamma);
    free(h_beta);
    
    return ln;
}

// Free LayerNorm memory
void free_layernorm(LayerNorm* ln) {
    cudaFree(ln->d_gamma);
    cudaFree(ln->d_beta);
    cudaFree(ln->d_gamma_grad);
    cudaFree(ln->d_beta_grad);
    cudaFree(ln->d_gamma_m);
    cudaFree(ln->d_gamma_v);
    cudaFree(ln->d_beta_m);
    cudaFree(ln->d_beta_v);
    cudaFree(ln->d_normalized);
    cudaFree(ln->d_mean);
    cudaFree(ln->d_var);
    cudaFree(ln->d_grad_input);
    free(ln);
}

// CUDA kernel for LayerNorm forward pass
__global__ void layernorm_forward_kernel(float* output, float* input, float* gamma, float* beta,
                                        float* mean, float* var, int batch_size, int seq_len,
                                        int d_model, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_seq = batch_size * seq_len;
    
    if (idx < total_seq) {
        // Calculate mean for this sequence position
        float sum = 0.0f;
        for (int i = 0; i < d_model; i++) {
            sum += input[idx * d_model + i];
        }
        float mu = sum / d_model;
        mean[idx] = mu;
        
        // Calculate variance for this sequence position
        float var_sum = 0.0f;
        for (int i = 0; i < d_model; i++) {
            float diff = input[idx * d_model + i] - mu;
            var_sum += diff * diff;
        }
        float variance = var_sum / d_model;
        var[idx] = variance;
        
        // Normalize and apply scale/shift
        float inv_std = rsqrtf(variance + epsilon);
        for (int i = 0; i < d_model; i++) {
            float normalized = (input[idx * d_model + i] - mu) * inv_std;
            output[idx * d_model + i] = gamma[i] * normalized + beta[i];
        }
    }
}

// CUDA kernel for LayerNorm backward pass
__global__ void layernorm_backward_kernel(float* grad_input, float* grad_output, float* grad_gamma,
                                         float* grad_beta, float* input, float* gamma,
                                         float* mean, float* var, int batch_size, int seq_len,
                                         int d_model, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_seq = batch_size * seq_len;
    
    if (idx < total_seq) {
        float mu = mean[idx];
        float variance = var[idx];
        float inv_std = rsqrtf(variance + epsilon);
        
        // Calculate gradient sums
        float grad_var_sum = 0.0f;
        float grad_mean_sum = 0.0f;
        
        for (int i = 0; i < d_model; i++) {
            float x_centered = input[idx * d_model + i] - mu;
            float grad_out = grad_output[idx * d_model + i];
            
            // Accumulate gradients for gamma and beta (across all samples)
            atomicAdd(&grad_gamma[i], grad_out * x_centered * inv_std);
            atomicAdd(&grad_beta[i], grad_out);
            
            // Accumulate for variance and mean gradients
            grad_var_sum += grad_out * gamma[i] * x_centered;
            grad_mean_sum += grad_out * gamma[i];
        }
        
        grad_var_sum *= -0.5f * inv_std * inv_std * inv_std;
        grad_mean_sum *= -inv_std;
        
        // Calculate input gradients
        for (int i = 0; i < d_model; i++) {
            float x_centered = input[idx * d_model + i] - mu;
            float grad_out = grad_output[idx * d_model + i];
            
            grad_input[idx * d_model + i] = grad_out * gamma[i] * inv_std +
                                           grad_var_sum * 2.0f * x_centered / d_model +
                                           grad_mean_sum / d_model;
        }
    }
}

// Forward pass
void forward_pass_layernorm(LayerNorm* ln, float* d_input) {
    int total_seq = ln->batch_size * ln->seq_len;
    int block_size = 256;
    int num_blocks = (total_seq + block_size - 1) / block_size;
    
    layernorm_forward_kernel<<<num_blocks, block_size>>>(
        ln->d_normalized, d_input, ln->d_gamma, ln->d_beta,
        ln->d_mean, ln->d_var, ln->batch_size, ln->seq_len,
        ln->d_model, ln->epsilon
    );
}

// Zero gradients
void zero_gradients_layernorm(LayerNorm* ln) {
    CHECK_CUDA(cudaMemset(ln->d_gamma_grad, 0, ln->d_model * sizeof(float)));
    CHECK_CUDA(cudaMemset(ln->d_beta_grad, 0, ln->d_model * sizeof(float)));
}

// Backward pass
void backward_pass_layernorm(LayerNorm* ln, float* d_grad_output, float* d_input) {
    int total_seq = ln->batch_size * ln->seq_len;
    int block_size = 256;
    int num_blocks = (total_seq + block_size - 1) / block_size;
    
    // Initialize gradients to zero
    CHECK_CUDA(cudaMemset(ln->d_gamma_grad, 0, ln->d_model * sizeof(float)));
    CHECK_CUDA(cudaMemset(ln->d_beta_grad, 0, ln->d_model * sizeof(float)));
    
    layernorm_backward_kernel<<<num_blocks, block_size>>>(
        ln->d_grad_input, d_grad_output, ln->d_gamma_grad, ln->d_beta_grad,
        d_input, ln->d_gamma, ln->d_mean, ln->d_var,
        ln->batch_size, ln->seq_len, ln->d_model, ln->epsilon
    );
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_layernorm(float* weight, float* grad, float* m, float* v,
                                              float beta1, float beta2, float epsilon, float learning_rate,
                                              float weight_decay, float alpha_t, int size, int total_seq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / total_seq;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights using AdamW
void update_weights_layernorm(LayerNorm* ln, float learning_rate) {
    ln->t++;
    
    float beta1_t = powf(ln->beta1, ln->t);
    float beta2_t = powf(ln->beta2, ln->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    int total_seq = ln->batch_size * ln->seq_len;
    int num_blocks = (ln->d_model + block_size - 1) / block_size;
    
    // Update gamma weights
    adamw_update_kernel_layernorm<<<num_blocks, block_size>>>(
        ln->d_gamma, ln->d_gamma_grad, ln->d_gamma_m, ln->d_gamma_v,
        ln->beta1, ln->beta2, ln->epsilon_adam, learning_rate, ln->weight_decay,
        alpha_t, ln->d_model, total_seq
    );
    
    // Update beta weights
    adamw_update_kernel_layernorm<<<num_blocks, block_size>>>(
        ln->d_beta, ln->d_beta_grad, ln->d_beta_m, ln->d_beta_v,
        ln->beta1, ln->beta2, ln->epsilon_adam, learning_rate, ln->weight_decay,
        alpha_t, ln->d_model, total_seq
    );
}

// Save LayerNorm parameters
void save_layernorm(LayerNorm* ln, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&ln->d_model, sizeof(int), 1, file);
    fwrite(&ln->seq_len, sizeof(int), 1, file);
    fwrite(&ln->batch_size, sizeof(int), 1, file);
    
    // Allocate temporary host memory
    float* h_gamma = (float*)malloc(ln->d_model * sizeof(float));
    float* h_beta = (float*)malloc(ln->d_model * sizeof(float));
    
    // Copy parameters from device to host
    CHECK_CUDA(cudaMemcpy(h_gamma, ln->d_gamma, ln->d_model * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_beta, ln->d_beta, ln->d_model * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_gamma, sizeof(float), ln->d_model, file);
    fwrite(h_beta, sizeof(float), ln->d_model, file);
    
    // Save Adam state
    fwrite(&ln->t, sizeof(int), 1, file);
    
    float* h_gamma_m = (float*)malloc(ln->d_model * sizeof(float));
    float* h_gamma_v = (float*)malloc(ln->d_model * sizeof(float));
    float* h_beta_m = (float*)malloc(ln->d_model * sizeof(float));
    float* h_beta_v = (float*)malloc(ln->d_model * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_gamma_m, ln->d_gamma_m, ln->d_model * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_gamma_v, ln->d_gamma_v, ln->d_model * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_beta_m, ln->d_beta_m, ln->d_model * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_beta_v, ln->d_beta_v, ln->d_model * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_gamma_m, sizeof(float), ln->d_model, file);
    fwrite(h_gamma_v, sizeof(float), ln->d_model, file);
    fwrite(h_beta_m, sizeof(float), ln->d_model, file);
    fwrite(h_beta_v, sizeof(float), ln->d_model, file);
    
    // Free temporary host memory
    free(h_gamma); free(h_beta);
    free(h_gamma_m); free(h_gamma_v);
    free(h_beta_m); free(h_beta_v);
    
    fclose(file);
    printf("LayerNorm saved to %s\n", filename);
}

// Load LayerNorm parameters
LayerNorm* load_layernorm(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int d_model, seq_len, stored_batch_size;
    fread(&d_model, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    LayerNorm* ln = init_layernorm(d_model, seq_len, batch_size);
    
    // Load parameters
    float* h_gamma = (float*)malloc(d_model * sizeof(float));
    float* h_beta = (float*)malloc(d_model * sizeof(float));
    
    fread(h_gamma, sizeof(float), d_model, file);
    fread(h_beta, sizeof(float), d_model, file);
    
    CHECK_CUDA(cudaMemcpy(ln->d_gamma, h_gamma, d_model * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ln->d_beta, h_beta, d_model * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Load Adam state
    fread(&ln->t, sizeof(int), 1, file);
    
    float* h_gamma_m = (float*)malloc(d_model * sizeof(float));
    float* h_gamma_v = (float*)malloc(d_model * sizeof(float));
    float* h_beta_m = (float*)malloc(d_model * sizeof(float));
    float* h_beta_v = (float*)malloc(d_model * sizeof(float));
    
    fread(h_gamma_m, sizeof(float), d_model, file);
    fread(h_gamma_v, sizeof(float), d_model, file);
    fread(h_beta_m, sizeof(float), d_model, file);
    fread(h_beta_v, sizeof(float), d_model, file);
    
    CHECK_CUDA(cudaMemcpy(ln->d_gamma_m, h_gamma_m, d_model * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ln->d_gamma_v, h_gamma_v, d_model * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ln->d_beta_m, h_beta_m, d_model * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ln->d_beta_v, h_beta_v, d_model * sizeof(float), cudaMemcpyDeviceToHost));
    
    free(h_gamma); free(h_beta);
    free(h_gamma_m); free(h_gamma_v);
    free(h_beta_m); free(h_beta_v);
    
    fclose(file);
    printf("LayerNorm loaded from %s\n", filename);
    return ln;
}