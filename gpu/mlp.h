#ifndef MLP_H
#define MLP_H

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
    float* d_W1;      // [hidden_dim x input_dim]
    float* d_W2;      // [output_dim x hidden_dim]
    float* d_W1_grad; // [hidden_dim x input_dim]
    float* d_W2_grad; // [output_dim x hidden_dim]
    
    // Device pointers for Adam parameters
    float* d_W1_m;    // First moment for W1
    float* d_W1_v;    // Second moment for W1
    float* d_W2_m;    // First moment for W2
    float* d_W2_v;    // Second moment for W2
    float beta1;      // Exponential decay rate for first moment
    float beta2;      // Exponential decay rate for second moment
    float epsilon;    // Small constant for numerical stability
    int t;            // Time step
    float weight_decay; // Weight decay parameter for AdamW
    
    // Device pointers for forward pass buffers
    float* d_layer_preact;  // [batch_size * seq_len x hidden_dim]
    float* d_layer_postact; // [batch_size * seq_len x hidden_dim]
    float* d_layer_output;  // [batch_size * seq_len x output_dim]
    
    // Device pointers for backward pass buffers
    float* d_error_hidden;  // [batch_size * seq_len x hidden_dim]
    float* d_error_output;  // [batch_size * seq_len x output_dim]

    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int batch_size;
    int seq_len;
} MLP;

// CUDA kernel prototypes
__global__ void swish_forward_kernel_mlp(float* output, float* pre_activation, int size);
__global__ void swish_backward_kernel_mlp(float* error_hidden, float* pre_activation, int size);
__global__ void adamw_update_kernel_mlp(float* weight, float* grad, float* m, float* v, float beta1, float beta2, float epsilon, float learning_rate, float weight_decay, float alpha_t, int size, int batch_size);

// Function prototypes
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int batch_size, int seq_len, cublasHandle_t cublas_handle);
void free_mlp(MLP* mlp);
void forward_pass_mlp(MLP* mlp, float* d_X);
float calculate_loss_mlp(MLP* mlp, float* d_y);
void zero_gradients_mlp(MLP* mlp);
void backward_pass_mlp(MLP* mlp, float* d_X);
void update_weights_mlp(MLP* mlp, float learning_rate);
void save_mlp(MLP* mlp, const char* filename);
MLP* load_mlp(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle);

#endif