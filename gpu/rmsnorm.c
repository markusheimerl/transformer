#include "rmsnorm.h"

// Initialize the RMSNorm layer
RMSNorm* init_rms_norm(int d_model, int batch_size, int seq_len, cublasHandle_t cublas_handle) {
    RMSNorm* norm = (RMSNorm*)malloc(sizeof(RMSNorm));
    
    // Store dimensions
    norm->d_model = d_model;
    norm->seq_len = seq_len;
    norm->batch_size = batch_size;
    norm->cublas_handle = cublas_handle;
    
    // Initialize Adam parameters
    norm->beta1 = 0.9f;
    norm->beta2 = 0.999f;
    norm->epsilon = 1e-8f;
    norm->t = 0;
    norm->weight_decay = 0.01f;
    
    int total_seq = batch_size * seq_len;
    int weight_size = d_model;
    
    // Allocate host memory for weight initialization
    float* h_weight = (float*)malloc(weight_size * sizeof(float));
    
    // Initialize weights on host to 1.0 (identity)
    for (int i = 0; i < weight_size; i++) {
        h_weight[i] = 1.0f;
    }
    
    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&norm->d_weight, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&norm->d_weight_grad, weight_size * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&norm->d_weight_m, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&norm->d_weight_v, weight_size * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&norm->d_output, total_seq * d_model * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&norm->d_mean_sq, total_seq * sizeof(float)));
    
    // Allocate device memory for backward pass buffers
    CHECK_CUDA(cudaMalloc(&norm->d_error_output, total_seq * d_model * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&norm->d_grad_input, total_seq * d_model * sizeof(float)));
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(norm->d_weight, h_weight, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(norm->d_weight_m, 0, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(norm->d_weight_v, 0, weight_size * sizeof(float)));
    
    // Free host memory
    free(h_weight);
    
    return norm;
}

// Free network memory
void free_rms_norm(RMSNorm* norm) {
    // Free device memory
    cudaFree(norm->d_weight);
    cudaFree(norm->d_weight_grad);
    cudaFree(norm->d_weight_m);
    cudaFree(norm->d_weight_v);
    cudaFree(norm->d_output);
    cudaFree(norm->d_mean_sq);
    cudaFree(norm->d_error_output);
    cudaFree(norm->d_grad_input);
    free(norm);
}

// CUDA kernel for computing mean of squared values per sequence element
__global__ void compute_mean_sq_kernel(float* mean_sq, float* input, int batch_size, int seq_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_seq = batch_size * seq_len;
    
    if (idx < total_seq) {
        int start_idx = idx * d_model;
        float sum_sq = 0.0f;
        
        for (int i = 0; i < d_model; i++) {
            float val = input[start_idx + i];
            sum_sq += val * val;
        }
        
        mean_sq[idx] = sum_sq / d_model;
    }
}

// CUDA kernel for RMSNorm forward pass
__global__ void rms_norm_forward_kernel(float* output, float* mean_sq, float* input, float* weight, int batch_size, int seq_len, int d_model, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len * d_model) {
        int sample = idx / (seq_len * d_model);
        int seq = (idx / d_model) % seq_len;
        int feat = idx % d_model;
        
        int input_idx = sample * seq_len * d_model + seq * d_model + feat;
        int mean_idx = sample * seq_len + seq;
        
        float mean = mean_sq[mean_idx];
        float val = input[input_idx];
        float inv_rms = 1.0f / sqrtf(mean + epsilon);
        output[input_idx] = val * inv_rms * weight[feat];
    }
}

// Forward pass
void forward_pass_rms_norm(RMSNorm* norm, float* d_X) {
    int total_seq = norm->batch_size * norm->seq_len;
    int total_size = total_seq * norm->d_model;
    
    // Compute mean of squared inputs per sequence element
    int block_size = 256;
    int num_blocks = (total_seq + block_size - 1) / block_size;
    compute_mean_sq_kernel<<<num_blocks, block_size>>>(
        norm->d_mean_sq,
        d_X,
        norm->batch_size,
        norm->seq_len,
        norm->d_model
    );
    
    // Apply RMSNorm
    num_blocks = (total_size + block_size - 1) / block_size;
    rms_norm_forward_kernel<<<num_blocks, block_size>>>(
        norm->d_output,
        norm->d_mean_sq,
        d_X,
        norm->d_weight,
        norm->batch_size,
        norm->seq_len,
        norm->d_model,
        norm->epsilon
    );
}

// Zero gradients
void zero_gradients_rms_norm(RMSNorm* norm) {
    int weight_size = norm->d_model;
    CHECK_CUDA(cudaMemset(norm->d_weight_grad, 0, weight_size * sizeof(float)));
}

// CUDA kernel for RMSNorm backward pass
__global__ void rms_norm_backward_kernel(float* grad_input, float* grad_weight, float* input, float* weight, float* mean_sq, float* error_output, int batch_size, int seq_len, int d_model, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len * d_model) {
        int sample = idx / (seq_len * d_model);
        int seq = (idx / d_model) % seq_len;
        int feat = idx % d_model;
        
        int input_idx = sample * seq_len * d_model + seq * d_model + feat;
        int mean_idx = sample * seq_len + seq;
        
        float val = input[input_idx];
        float mean = mean_sq[mean_idx];
        float inv_rms = 1.0f / sqrtf(mean + epsilon);
        float w = weight[feat];
        
        float grad = error_output[input_idx];
        grad_input[input_idx] = grad * w * inv_rms;
        atomicAdd(&grad_weight[feat], grad * val * inv_rms);
    }
}

// Backward pass
void backward_pass_rms_norm(RMSNorm* norm, float* d_X) {
    int total_seq = norm->batch_size * norm->seq_len;
    int total_size = total_seq * norm->d_model;
    
    // Calculate gradient w.r.t. input and weight
    int block_size = 256;
    int num_blocks = (total_size + block_size - 1) / block_size;
    rms_norm_backward_kernel<<<num_blocks, block_size>>>(
        norm->d_grad_input,
        norm->d_weight_grad,
        d_X,
        norm->d_weight,
        norm->d_mean_sq,
        norm->d_error_output,
        norm->batch_size,
        norm->seq_len,
        norm->d_model,
        norm->epsilon
    );
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_rms_norm(float* weight, float* grad, float* m, float* v, float beta1, float beta2, float epsilon, float learning_rate, float weight_decay, float alpha_t, int size, int total_seq) {
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
void update_weights_rms_norm(RMSNorm* norm, float learning_rate) {
    norm->t++;
    
    float beta1_t = powf(norm->beta1, norm->t);
    float beta2_t = powf(norm->beta2, norm->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int weight_size = norm->d_model;
    int total_seq = norm->batch_size * norm->seq_len;
    int block_size = 256;
    int num_blocks = (weight_size + block_size - 1) / block_size;
    
    // Update weight
    adamw_update_kernel_rms_norm<<<num_blocks, block_size>>>(
        norm->d_weight, 
        norm->d_weight_grad, 
        norm->d_weight_m, 
        norm->d_weight_v,
        norm->beta1, 
        norm->beta2, 
        norm->epsilon, 
        learning_rate, 
        norm->weight_decay,
        alpha_t, 
        weight_size, 
        total_seq
    );
}

// Save model weights to binary file
void save_rms_norm(RMSNorm* norm, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&norm->d_model, sizeof(int), 1, file);
    fwrite(&norm->seq_len, sizeof(int), 1, file);
    fwrite(&norm->batch_size, sizeof(int), 1, file);
    
    // Save weights
    int weight_size = norm->d_model;
    
    // Allocate temporary host memory
    float* h_weight = (float*)malloc(weight_size * sizeof(float));
    
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(h_weight, norm->d_weight, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_weight, sizeof(float), weight_size, file);
    
    // Save Adam state
    fwrite(&norm->t, sizeof(int), 1, file);
    
    float* h_weight_m = (float*)malloc(weight_size * sizeof(float));
    float* h_weight_v = (float*)malloc(weight_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_weight_m, norm->d_weight_m, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_weight_v, norm->d_weight_v, weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_weight_m, sizeof(float), weight_size, file);
    fwrite(h_weight_v, sizeof(float), weight_size, file);
    
    // Free temporary host memory
    free(h_weight);
    free(h_weight_m);
    free(h_weight_v);
    
    fclose(file);
    printf("RMSNorm saved to %s\n", filename);
}

// Load model weights from binary file
RMSNorm* load_rms_norm(const char* filename, int custom_batch_size, int seq_len, cublasHandle_t cublas_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int d_model, stored_seq_len, stored_batch_size;
    fread(&d_model, sizeof(int), 1, file);
    fread(&stored_seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    RMSNorm* norm = init_rms_norm(d_model, batch_size, seq_len, cublas_handle);
    
    // Load weights
    int weight_size = d_model;
    
    float* h_weight = (float*)malloc(weight_size * sizeof(float));
    
    fread(h_weight, sizeof(float), weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(norm->d_weight, h_weight, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    fread(&norm->t, sizeof(int), 1, file);
    
    float* h_weight_m = (float*)malloc(weight_size * sizeof(float));
    float* h_weight_v = (float*)malloc(weight_size * sizeof(float));
    
    fread(h_weight_m, sizeof(float), weight_size, file);
    fread(h_weight_v, sizeof(float), weight_size, file);
    
    CHECK_CUDA(cudaMemcpy(norm->d_weight_m, h_weight_m, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(norm->d_weight_v, h_weight_v, weight_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free temporary host memory
    free(h_weight);
    free(h_weight_m);
    free(h_weight_v);
    
    fclose(file);
    printf("RMSNorm loaded from %s\n", filename);
    return norm;
}