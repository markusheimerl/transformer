#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <cblas.h>
#include "attention/attention.h"
#include "mlp/mlp.h"

typedef struct {
    Attention** attention_layers;
    MLP** mlp_layers;
    
    // Dimensions
    int seq_len;
    int d_model;
    int batch_size;
    int hidden_dim;
    int num_layers;
} Transformer;

// Function prototypes
Transformer* init_transformer(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, bool is_causal);
void free_transformer(Transformer* transformer);
void forward_pass_transformer(Transformer* transformer, float* X);
float calculate_loss_transformer(Transformer* transformer, float* y);
void zero_gradients_transformer(Transformer* transformer);
void backward_pass_transformer(Transformer* transformer, float* X, float* grad_X);
void update_weights_transformer(Transformer* transformer, float learning_rate, int effective_batch_size);
void reset_optimizer_transformer(Transformer* transformer);
void save_transformer(Transformer* transformer, const char* filename);
Transformer* load_transformer(const char* filename, int custom_batch_size);

#endif