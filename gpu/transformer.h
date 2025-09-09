#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include "../attention/gpu/attention.h"
#include "../mlp/gpu/mlp.h"

typedef struct {
    // Layer 1 components
    Attention* attention1;
    MLP* mlp1;
    
    // Layer 2 components  
    Attention* attention2;
    MLP* mlp2;
    
    // cuBLAS handles
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    int hidden_dim;
} Transformer;

// Function prototypes
Transformer* init_transformer(int seq_len, int d_model, int hidden_dim, int batch_size, bool is_causal, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle);
void free_transformer(Transformer* transformer);
void forward_pass_transformer(Transformer* transformer, float* d_X);
float calculate_loss_transformer(Transformer* transformer, float* d_y);
void zero_gradients_transformer(Transformer* transformer);
void backward_pass_transformer(Transformer* transformer, float* d_X, float* d_grad_X);
void update_weights_transformer(Transformer* transformer, float learning_rate);
void save_transformer(Transformer* transformer, const char* filename);
Transformer* load_transformer(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle);

#endif