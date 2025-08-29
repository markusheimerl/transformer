#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../attention/gpu/attention.h"
#include "mlp.h"

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
    // Component modules
    Attention** attention_layers;
    MLP** mlp_layers;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
    
    // Dimensions
    int d_model;      // Model dimension (feature_dim)
    int seq_len;      // Sequence length
    int batch_size;   // Batch size
    int mlp_hidden;   // MLP hidden dimension
    int num_layers;   // Number of transformer layers
} Transformer;

// Function prototypes
Transformer* init_transformer(int d_model, int seq_len, int batch_size, int mlp_hidden, int num_layers, cublasHandle_t cublas_handle);
void free_transformer(Transformer* transformer);
void forward_pass_transformer(Transformer* transformer, float* d_X);
float calculate_loss_transformer(Transformer* transformer, float* d_y);
void zero_gradients_transformer(Transformer* transformer);
void backward_pass_transformer(Transformer* transformer, float* d_X);
void update_weights_transformer(Transformer* transformer, float learning_rate);
void save_transformer(Transformer* transformer, const char* filename);
Transformer* load_transformer(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle);

#endif