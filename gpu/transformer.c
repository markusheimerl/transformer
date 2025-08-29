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
    transformer->mlp = init_mlp(d_model, mlp_hidden, d_model, 1, batch_size * seq_len, cublas_handle);
    
    return transformer;
}

// Free transformer memory
void free_transformer(Transformer* transformer) {
    free_attention(transformer->attention);
    free_mlp(transformer->mlp);
    free(transformer);
}

// Forward pass through transformer - clean and simple!
void forward_pass_transformer(Transformer* transformer, float* d_X) {
    // Step 1: Attention layer
    forward_pass_attention(transformer->attention, d_X);
    
    // Step 2: MLP layer
    forward_pass_mlp(transformer->mlp, transformer->attention->d_layer_output);
}

// Calculate loss
float calculate_loss_transformer(Transformer* transformer, float* d_y) {
    // Use MLP's loss calculation with the final MLP output
    return calculate_loss_mlp(transformer->mlp, d_y);
}

// Zero gradients
void zero_gradients_transformer(Transformer* transformer) {
    zero_gradients_attention(transformer->attention);
    zero_gradients_mlp(transformer->mlp);
}

// Backward pass through transformer - clean gradient flow
void backward_pass_transformer(Transformer* transformer, float* d_X) {
    // Step 1: Backward pass through MLP (using attention output as input)
    backward_pass_mlp(transformer->mlp, transformer->attention->d_layer_output);
    
    // Step 2: Copy MLP input gradients to attention output gradients for gradient flow
    int seq_size = transformer->batch_size * transformer->seq_len * transformer->d_model;
    CHECK_CUDA(cudaMemcpy(transformer->attention->d_error_output, transformer->mlp->d_error_output[0],
                         seq_size * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Step 3: Backward pass through attention
    backward_pass_attention(transformer->attention, d_X);
}

// Update weights
void update_weights_transformer(Transformer* transformer, float learning_rate) {
    update_weights_attention(transformer->attention, learning_rate);
    update_weights_mlp(transformer->mlp, learning_rate);
}

// Extract base filename without extension
static void get_base_filename(const char* filename, char* base, size_t base_size) {
    strncpy(base, filename, base_size - 1);
    base[base_size - 1] = '\0';
    
    // Remove .bin extension if present
    char* dot = strrchr(base, '.');
    if (dot && strcmp(dot, ".bin") == 0) {
        *dot = '\0';
    }
}

// Save transformer to binary files
void save_transformer(Transformer* transformer, const char* filename) {
    char base[256];
    get_base_filename(filename, base, sizeof(base));
    
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
    
    char base[256];
    get_base_filename(filename, base, sizeof(base));
    
    // Load attention module
    char attn_filename[300];
    snprintf(attn_filename, sizeof(attn_filename), "%s_attn.bin", base);
    Attention* attention = load_attention(attn_filename, batch_size, cublas_handle);
    if (!attention) return NULL;
    
    // Load MLP module
    char mlp_filename[300];
    snprintf(mlp_filename, sizeof(mlp_filename), "%s_mlp.bin", base);
    MLP* mlp = load_mlp(mlp_filename, batch_size * seq_len, cublas_handle);
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