#ifndef RMSNORM_H
#define RMSNORM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLAS Error checking macro
#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    // Device pointers for weights and gradients
    float* d_weight;      // [d_model] - Learnable scaling parameters
    float* d_weight_grad; // [d_model] - Gradient accumulator
    
    // Device pointers for Adam parameters
    float* d_weight_m;    // First moment for weight
    float* d_weight_v;    // Second moment for weight
    float beta1;          // Exponential decay rate for first moment
    float beta2;          // Exponential decay rate for second moment
    float epsilon;        // Small constant for numerical stability
    int t;                // Time step
    float weight_decay;   // Weight decay parameter for AdamW
    
    // Device pointers for forward pass buffers
    float* d_output;      // [batch_size * seq_len x d_model] - Normalized output
    float* d_mean_sq;     // [batch_size * seq_len] - Mean of squared inputs
    
    // Device pointers for backward pass buffers
    float* d_error_output; // [batch_size * seq_len x d_model] - Error gradients
    float* d_grad_input;   // [batch_size * seq_len x d_model] - Input gradients
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int d_model;      // Model dimension
    int seq_len;      // Sequence length
    int batch_size;   // Batch size
} RMSNorm;

// CUDA kernel prototypes
__global__ void rms_norm_forward_kernel(float* output, float* mean_sq, float* input, float* weight, int batch_size, int seq_len, int d_model, float epsilon);
__global__ void rms_norm_backward_kernel(float* grad_input, float* grad_weight, float* input, float* weight, float* mean_sq, float* error_output, int batch_size, int seq_len, int d_model, float epsilon);
__global__ void adamw_update_kernel_rms_norm(float* weight, float* grad, float* m, float* v, float beta1, float beta2, float epsilon, float learning_rate, float weight_decay, float alpha_t, int size, int total_seq);

// Function prototypes
RMSNorm* init_rms_norm(int d_model, int batch_size, int seq_len, cublasHandle_t cublas_handle);
void free_rms_norm(RMSNorm* norm);
void forward_pass_rms_norm(RMSNorm* norm, float* d_X);
void zero_gradients_rms_norm(RMSNorm* norm);
void backward_pass_rms_norm(RMSNorm* norm, float* d_X);
void update_weights_rms_norm(RMSNorm* norm, float learning_rate);
void save_rms_norm(RMSNorm* norm, const char* filename);
RMSNorm* load_rms_norm(const char* filename, int custom_batch_size, int seq_len, cublasHandle_t cublas_handle);

#endif