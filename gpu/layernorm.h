#ifndef LAYERNORM_H
#define LAYERNORM_H

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
    // Device pointers for parameters
    float* d_gamma;      // [d_model] - Scale parameters
    float* d_beta;       // [d_model] - Shift parameters
    float* d_gamma_grad; // [d_model] - Gradient accumulators
    float* d_beta_grad;  // [d_model] - Gradient accumulators
    
    // Device pointers for Adam parameters
    float* d_gamma_m;    // First moment for gamma
    float* d_gamma_v;    // Second moment for gamma
    float* d_beta_m;     // First moment for beta
    float* d_beta_v;     // Second moment for beta
    float beta1;         // Exponential decay rate for first moment
    float beta2;         // Exponential decay rate for second moment
    float epsilon_adam;  // Small constant for numerical stability in Adam
    int t;               // Time step
    float weight_decay;  // Weight decay parameter for AdamW
    
    // Device pointers for forward pass buffers
    float* d_normalized; // [batch_size * seq_len x d_model] - Normalized activations
    float* d_mean;       // [batch_size * seq_len] - Mean for each sequence position
    float* d_var;        // [batch_size * seq_len] - Variance for each sequence position
    
    // Device pointers for backward pass buffers
    float* d_grad_input; // [batch_size * seq_len x d_model] - Gradient w.r.t. input
    
    // Dimensions
    int d_model;         // Model dimension
    int batch_size;      // Batch size
    int seq_len;         // Sequence length
    float epsilon;       // Small constant for numerical stability in LayerNorm
} LayerNorm;

// CUDA kernel prototypes
__global__ void layernorm_forward_kernel(float* output, float* input, float* gamma, float* beta, 
                                        float* mean, float* var, int batch_size, int seq_len, 
                                        int d_model, float epsilon);
__global__ void layernorm_backward_kernel(float* grad_input, float* grad_output, float* grad_gamma, 
                                         float* grad_beta, float* input, float* gamma, 
                                         float* mean, float* var, int batch_size, int seq_len, 
                                         int d_model, float epsilon);
__global__ void adamw_update_kernel_layernorm(float* weight, float* grad, float* m, float* v, 
                                              float beta1, float beta2, float epsilon, float learning_rate, 
                                              float weight_decay, float alpha_t, int size, int total_seq);

// Function prototypes
LayerNorm* init_layernorm(int d_model, int seq_len, int batch_size);
void free_layernorm(LayerNorm* ln);
void forward_pass_layernorm(LayerNorm* ln, float* d_input);
void zero_gradients_layernorm(LayerNorm* ln);
void backward_pass_layernorm(LayerNorm* ln, float* d_grad_output, float* d_input);
void update_weights_layernorm(LayerNorm* ln, float learning_rate);
void save_layernorm(LayerNorm* ln, const char* filename);
LayerNorm* load_layernorm(const char* filename, int custom_batch_size);

#endif