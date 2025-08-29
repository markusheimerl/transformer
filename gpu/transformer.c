#include "transformer.h"

// Initialize the transformer
Transformer* init_transformer(int d_model, int seq_len, int batch_size, int mlp_hidden, cublasHandle_t cublas_handle) {
    Transformer* transformer = (Transformer*)malloc(sizeof(Transformer));
    
    // Store dimensions
    transformer->d_model = d_model;
    transformer->seq_len = seq_len;
    transformer->batch_size = batch_size;
    transformer->mlp_hidden = mlp_hidden;
    transformer->cublas_handle = cublas_handle;
    
    // Initialize attention module
    transformer->attention = init_attention(d_model, seq_len, batch_size, cublas_handle);
    
    // Initialize MLP module
    transformer->mlp = init_mlp(d_model, mlp_hidden, d_model, batch_size, seq_len, cublas_handle);
    
    return transformer;
}

// Free memory
void free_transformer(Transformer* transformer) {
    free_attention(transformer->attention);
    free_mlp(transformer->mlp);
    free(transformer);
}

// Forward pass
void forward_pass_transformer(Transformer* transformer, float* d_X) {
    // Step 1: Attention layer
    forward_pass_attention(transformer->attention, d_X);
    
    // Step 2: MLP layer
    forward_pass_mlp(transformer->mlp, transformer->attention->d_layer_output);
}

// Calculate loss
float calculate_loss_transformer(Transformer* transformer, float* d_y) {
    // ∂L/∂Y = Y - Y_true
    MLP* mlp = transformer->mlp;
    float loss = 0.0f;
    int total_size = mlp->batch_size * mlp->seq_len * mlp->output_dim;

    const float alpha = 1.0f;
    const float beta = -1.0f;
    CHECK_CUBLAS(cublasSgeam(mlp->cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            mlp->output_dim, mlp->batch_size * mlp->seq_len,
                            &alpha, mlp->d_layer_output, mlp->output_dim,
                            &beta, d_y, mlp->output_dim,
                            mlp->d_error_output, mlp->output_dim));
    CHECK_CUBLAS(cublasSdot(mlp->cublas_handle, total_size, 
                           mlp->d_error_output, 1, mlp->d_error_output, 1, &loss));
    
    return loss / total_size;
}

// Zero gradients
void zero_gradients_transformer(Transformer* transformer) {
    zero_gradients_attention(transformer->attention);
    zero_gradients_mlp(transformer->mlp);
}

// Backward pass
void backward_pass_transformer(Transformer* transformer, float* d_X) {
    // Step 1: Backward pass through MLP
    backward_pass_mlp(transformer->mlp, transformer->attention->d_layer_output);
    
    // Step 2: Compute gradient w.r.t. MLP input
    // ∂L/∂attention_output = (∂L/∂H)(W₁)^T
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int total_seq = transformer->batch_size * transformer->seq_len;
    
    CHECK_CUBLAS(cublasSgemm(transformer->mlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            transformer->d_model, total_seq, transformer->mlp_hidden,
                            &alpha, transformer->mlp->d_W1, transformer->d_model,
                            transformer->mlp->d_error_hidden, transformer->mlp_hidden,
                            &beta, transformer->attention->d_error_output, transformer->d_model));
    
    // Step 3: Backward pass through attention
    backward_pass_attention(transformer->attention, d_X);
}

// Update weights
void update_weights_transformer(Transformer* transformer, float learning_rate) {
    update_weights_attention(transformer->attention, learning_rate);
    update_weights_mlp(transformer->mlp, learning_rate);
}

// Save transformer to binary files
void save_transformer(Transformer* transformer, const char* filename) {
    // Extract base filename without extension
    char base[256];
    strncpy(base, filename, sizeof(base) - 1);
    base[sizeof(base) - 1] = '\0';
    
    // Remove .bin extension if present
    char* dot = strrchr(base, '.');
    if (dot && strcmp(dot, ".bin") == 0) {
        *dot = '\0';
    }
    
    // Save attention module
    char attn_filename[300];
    snprintf(attn_filename, sizeof(attn_filename), "%s_attn.bin", base);
    save_attention(transformer->attention, attn_filename);
    
    // Save MLP module
    char mlp_filename[300];
    snprintf(mlp_filename, sizeof(mlp_filename), "%s_mlp.bin", base);
    save_mlp(transformer->mlp, mlp_filename);
    
    // Save transformer metadata
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    fwrite(&transformer->d_model, sizeof(int), 1, file);
    fwrite(&transformer->seq_len, sizeof(int), 1, file);
    fwrite(&transformer->batch_size, sizeof(int), 1, file);
    fwrite(&transformer->mlp_hidden, sizeof(int), 1, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load transformer from binary files
Transformer* load_transformer(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle) {
    // Load transformer metadata
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    int d_model, seq_len, stored_batch_size, mlp_hidden;
    fread(&d_model, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&mlp_hidden, sizeof(int), 1, file);
    
    fclose(file);
    
    // Use custom_batch_size if provided
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Extract base filename without extension
    char base[256];
    strncpy(base, filename, sizeof(base) - 1);
    base[sizeof(base) - 1] = '\0';
    
    // Remove .bin extension if present
    char* dot = strrchr(base, '.');
    if (dot && strcmp(dot, ".bin") == 0) {
        *dot = '\0';
    }
    
    // Load attention module
    char attn_filename[300];
    snprintf(attn_filename, sizeof(attn_filename), "%s_attn.bin", base);
    Attention* attention = load_attention(attn_filename, batch_size, cublas_handle);
    if (!attention) return NULL;
    
    // Load MLP module
    char mlp_filename[300];
    snprintf(mlp_filename, sizeof(mlp_filename), "%s_mlp.bin", base);
    MLP* mlp = load_mlp(mlp_filename, batch_size, cublas_handle);
    if (!mlp) {
        free_attention(attention);
        return NULL;
    }
    
    // Create transformer and assign loaded modules
    Transformer* transformer = (Transformer*)malloc(sizeof(Transformer));
    transformer->d_model = d_model;
    transformer->seq_len = seq_len;
    transformer->batch_size = batch_size;
    transformer->mlp_hidden = mlp_hidden;
    transformer->cublas_handle = cublas_handle;
    transformer->attention = attention;
    transformer->mlp = mlp;
    
    printf("Model loaded from %s\n", filename);
    return transformer;
}