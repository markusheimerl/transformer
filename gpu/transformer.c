#include "transformer.h"

// Initialize the transformer
Transformer* init_transformer(int seq_len, int d_model, int hidden_dim, int batch_size, bool is_causal, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle) {
    Transformer* transformer = (Transformer*)malloc(sizeof(Transformer));
    
    // Store dimensions and handles
    transformer->seq_len = seq_len;
    transformer->d_model = d_model;
    transformer->batch_size = batch_size;
    transformer->hidden_dim = hidden_dim;
    transformer->cublas_handle = cublas_handle;
    transformer->cublaslt_handle = cublaslt_handle;
    
    // Initialize attention layer
    transformer->attention = init_attention(seq_len, d_model, batch_size, is_causal, cublas_handle, cublaslt_handle);
    
    // Initialize MLP layer (input_dim = d_model, output_dim = d_model)
    transformer->mlp = init_mlp(d_model, hidden_dim, d_model, batch_size * seq_len, cublas_handle, cublaslt_handle);
    
    return transformer;
}

// Free transformer memory
void free_transformer(Transformer* transformer) {
    free_attention(transformer->attention);
    free_mlp(transformer->mlp);
    free(transformer);
}

// Forward pass: X -> Attention -> MLP -> Output
void forward_pass_transformer(Transformer* transformer, float* d_X) {
    // Step 1: Attention layer
    forward_pass_attention(transformer->attention, d_X);
    
    // Step 2: MLP layer (use attention output directly)
    forward_pass_mlp(transformer->mlp, transformer->attention->d_output);
}

// Calculate loss using MLP's output
float calculate_loss_transformer(Transformer* transformer, float* d_y) {
    return calculate_loss_mlp(transformer->mlp, d_y);
}

// Zero gradients for all components
void zero_gradients_transformer(Transformer* transformer) {
    zero_gradients_attention(transformer->attention);
    zero_gradients_mlp(transformer->mlp);
}

// Backward pass: MLP -> Attention
void backward_pass_transformer(Transformer* transformer, float* d_X, float* d_grad_X) {
    // Step 1: Backward through MLP (this computes gradients for MLP weights and writes to mlp->d_grad_output)
    backward_pass_mlp(transformer->mlp, transformer->attention->d_output, transformer->attention->d_grad_output);
    
    // Step 2: Backward through attention (uses attention->d_grad_output directly)
    backward_pass_attention(transformer->attention, d_X, d_grad_X);
}

// Update weights for all components
void update_weights_transformer(Transformer* transformer, float learning_rate) {
    update_weights_attention(transformer->attention, learning_rate);
    update_weights_mlp(transformer->mlp, learning_rate);
}

// Save transformer to binary file
void save_transformer(Transformer* transformer, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&transformer->seq_len, sizeof(int), 1, file);
    fwrite(&transformer->d_model, sizeof(int), 1, file);
    fwrite(&transformer->batch_size, sizeof(int), 1, file);
    fwrite(&transformer->hidden_dim, sizeof(int), 1, file);
    
    // Save attention is_causal flag
    fwrite(&transformer->attention->is_causal, sizeof(bool), 1, file);
    
    fclose(file);
    
    // Save attention and MLP components separately
    char attn_filename[256], mlp_filename[256];
    snprintf(attn_filename, sizeof(attn_filename), "%s.attn", filename);
    snprintf(mlp_filename, sizeof(mlp_filename), "%s.mlp", filename);
    
    save_attention(transformer->attention, attn_filename);
    save_mlp(transformer->mlp, mlp_filename);
    
    printf("Transformer saved to %s (with .attn and .mlp components)\n", filename);
}

// Load transformer from binary file
Transformer* load_transformer(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int seq_len, d_model, stored_batch_size, hidden_dim;
    bool is_causal;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    
    fclose(file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Load attention and MLP components
    char attn_filename[256], mlp_filename[256];
    snprintf(attn_filename, sizeof(attn_filename), "%s.attn", filename);
    snprintf(mlp_filename, sizeof(mlp_filename), "%s.mlp", filename);
    
    Attention* attention = load_attention(attn_filename, batch_size, cublas_handle, cublaslt_handle);
    MLP* mlp = load_mlp(mlp_filename, batch_size * seq_len, cublas_handle, cublaslt_handle);
    
    if (!attention || !mlp) {
        if (attention) free_attention(attention);
        if (mlp) free_mlp(mlp);
        return NULL;
    }
    
    // Create transformer and assign loaded components
    Transformer* transformer = (Transformer*)malloc(sizeof(Transformer));
    transformer->seq_len = seq_len;
    transformer->d_model = d_model;
    transformer->batch_size = batch_size;
    transformer->hidden_dim = hidden_dim;
    transformer->cublas_handle = cublas_handle;
    transformer->cublaslt_handle = cublaslt_handle;
    transformer->attention = attention;
    transformer->mlp = mlp;
    
    printf("Transformer loaded from %s\n", filename);
    return transformer;
}