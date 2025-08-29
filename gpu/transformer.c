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
    
    // Initialize first attention module
    transformer->attention1 = init_attention(d_model, seq_len, batch_size, cublas_handle);
    
    // Initialize first MLP module
    transformer->mlp1 = init_mlp(d_model, mlp_hidden, d_model, batch_size, seq_len, cublas_handle);
    
    // Initialize second attention module
    transformer->attention2 = init_attention(d_model, seq_len, batch_size, cublas_handle);
    
    // Initialize second MLP module
    transformer->mlp2 = init_mlp(d_model, mlp_hidden, d_model, batch_size, seq_len, cublas_handle);
    
    return transformer;
}

// Free memory
void free_transformer(Transformer* transformer) {
    free_attention(transformer->attention1);
    free_mlp(transformer->mlp1);
    free_attention(transformer->attention2);
    free_mlp(transformer->mlp2);
    free(transformer);
}

// Forward pass
void forward_pass_transformer(Transformer* transformer, float* d_X) {
    const float alpha = 1.0f;
    int total_seq = transformer->batch_size * transformer->seq_len;
    
    // Step 1: First attention layer
    forward_pass_attention(transformer->attention1, d_X);
    
    // Step 2: First residual connection - attention1_output += X
    CHECK_CUBLAS(cublasSgeam(transformer->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            transformer->d_model, total_seq,
                            &alpha, transformer->attention1->d_layer_output, transformer->d_model,
                            &alpha, d_X, transformer->d_model,
                            transformer->attention1->d_layer_output, transformer->d_model));
    
    // Step 3: First MLP layer
    forward_pass_mlp(transformer->mlp1, transformer->attention1->d_layer_output);
    
    // Step 4: Second residual connection - mlp1_output += attention1_output
    CHECK_CUBLAS(cublasSgeam(transformer->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            transformer->d_model, total_seq,
                            &alpha, transformer->mlp1->d_layer_output, transformer->d_model,
                            &alpha, transformer->attention1->d_layer_output, transformer->d_model,
                            transformer->mlp1->d_layer_output, transformer->d_model));
    
    // Step 5: Second attention layer
    forward_pass_attention(transformer->attention2, transformer->mlp1->d_layer_output);
    
    // Step 6: Third residual connection - attention2_output += mlp1_output
    CHECK_CUBLAS(cublasSgeam(transformer->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            transformer->d_model, total_seq,
                            &alpha, transformer->attention2->d_layer_output, transformer->d_model,
                            &alpha, transformer->mlp1->d_layer_output, transformer->d_model,
                            transformer->attention2->d_layer_output, transformer->d_model));
    
    // Step 7: Second MLP layer
    forward_pass_mlp(transformer->mlp2, transformer->attention2->d_layer_output);
    
    // Step 8: Fourth residual connection - mlp2_output += attention2_output
    CHECK_CUBLAS(cublasSgeam(transformer->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            transformer->d_model, total_seq,
                            &alpha, transformer->mlp2->d_layer_output, transformer->d_model,
                            &alpha, transformer->attention2->d_layer_output, transformer->d_model,
                            transformer->mlp2->d_layer_output, transformer->d_model));
}

// Calculate loss
float calculate_loss_transformer(Transformer* transformer, float* d_y) {
    // Loss is calculated against the final output
    MLP* final_mlp = transformer->mlp2;
    float loss = 0.0f;
    int total_size = final_mlp->batch_size * final_mlp->seq_len * final_mlp->output_dim;

    const float alpha = 1.0f;
    const float beta = -1.0f;
    CHECK_CUBLAS(cublasSgeam(final_mlp->cublas_handle, 
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            final_mlp->output_dim, final_mlp->batch_size * final_mlp->seq_len,
                            &alpha, final_mlp->d_layer_output, final_mlp->output_dim,
                            &beta, d_y, final_mlp->output_dim,
                            final_mlp->d_error_output, final_mlp->output_dim));
    CHECK_CUBLAS(cublasSdot(final_mlp->cublas_handle, total_size, 
                           final_mlp->d_error_output, 1, final_mlp->d_error_output, 1, &loss));
    
    return loss / total_size;
}

// Zero gradients
void zero_gradients_transformer(Transformer* transformer) {
    zero_gradients_attention(transformer->attention1);
    zero_gradients_mlp(transformer->mlp1);
    zero_gradients_attention(transformer->attention2);
    zero_gradients_mlp(transformer->mlp2);
}

// Backward pass
void backward_pass_transformer(Transformer* transformer, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int total_seq = transformer->batch_size * transformer->seq_len;
    
    // Step 1: Backward pass through second MLP
    backward_pass_mlp(transformer->mlp2, transformer->attention2->d_layer_output);
    
    // Step 2: Compute gradient w.r.t. MLP2 input and store in attention2 error buffer
    CHECK_CUBLAS(cublasSgemm(transformer->mlp2->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            transformer->d_model, total_seq, transformer->mlp_hidden,
                            &alpha, transformer->mlp2->d_W1, transformer->d_model,
                            transformer->mlp2->d_error_hidden, transformer->mlp_hidden,
                            &beta, transformer->attention2->d_error_output, transformer->d_model));
    
    // Step 3: Add gradient from fourth residual connection
    CHECK_CUBLAS(cublasSgeam(transformer->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            transformer->d_model, total_seq,
                            &alpha, transformer->attention2->d_error_output, transformer->d_model,
                            &alpha, transformer->mlp2->d_error_output, transformer->d_model,
                            transformer->attention2->d_error_output, transformer->d_model));
    
    // Step 4: Backward pass through second attention
    backward_pass_attention(transformer->attention2, transformer->mlp1->d_layer_output);
    
    // Step 5: Compute gradient w.r.t. attention2 input and store in MLP1 error buffer
    // ∂L/∂X = (∂L/∂Q)W_q^T + (∂L/∂K)W_k^T + (∂L/∂V)W_v^T
    
    // ∂L/∂X = (∂L/∂Q)W_q^T
    CHECK_CUBLAS(cublasSgemm(transformer->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            transformer->d_model, total_seq, transformer->d_model,
                            &alpha, transformer->attention2->d_W_q, transformer->d_model,
                            transformer->attention2->d_grad_Q, transformer->d_model,
                            &beta, transformer->mlp1->d_error_output, transformer->d_model));
    
    // ∂L/∂X += (∂L/∂K)W_k^T
    CHECK_CUBLAS(cublasSgemm(transformer->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            transformer->d_model, total_seq, transformer->d_model,
                            &alpha, transformer->attention2->d_W_k, transformer->d_model,
                            transformer->attention2->d_grad_K, transformer->d_model,
                            &alpha, transformer->mlp1->d_error_output, transformer->d_model));
    
    // ∂L/∂X += (∂L/∂V)W_v^T
    CHECK_CUBLAS(cublasSgemm(transformer->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            transformer->d_model, total_seq, transformer->d_model,
                            &alpha, transformer->attention2->d_W_v, transformer->d_model,
                            transformer->attention2->d_grad_V, transformer->d_model,
                            &alpha, transformer->mlp1->d_error_output, transformer->d_model));
    
    // Step 6: Add gradient from third residual connection
    CHECK_CUBLAS(cublasSgeam(transformer->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            transformer->d_model, total_seq,
                            &alpha, transformer->mlp1->d_error_output, transformer->d_model,
                            &alpha, transformer->attention2->d_error_output, transformer->d_model,
                            transformer->mlp1->d_error_output, transformer->d_model));
    
    // Step 7: Backward pass through first MLP
    backward_pass_mlp(transformer->mlp1, transformer->attention1->d_layer_output);
    
    // Step 8: Compute gradient w.r.t. MLP1 input and store in attention1 error buffer
    CHECK_CUBLAS(cublasSgemm(transformer->mlp1->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            transformer->d_model, total_seq, transformer->mlp_hidden,
                            &alpha, transformer->mlp1->d_W1, transformer->d_model,
                            transformer->mlp1->d_error_hidden, transformer->mlp_hidden,
                            &beta, transformer->attention1->d_error_output, transformer->d_model));
    
    // Step 9: Add gradient from second residual connection
    CHECK_CUBLAS(cublasSgeam(transformer->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            transformer->d_model, total_seq,
                            &alpha, transformer->attention1->d_error_output, transformer->d_model,
                            &alpha, transformer->mlp1->d_error_output, transformer->d_model,
                            transformer->attention1->d_error_output, transformer->d_model));
    
    // Step 10: Backward pass through first attention
    backward_pass_attention(transformer->attention1, d_X);
}

// Update weights
void update_weights_transformer(Transformer* transformer, float learning_rate) {
    update_weights_attention(transformer->attention1, learning_rate);
    update_weights_mlp(transformer->mlp1, learning_rate);
    update_weights_attention(transformer->attention2, learning_rate);
    update_weights_mlp(transformer->mlp2, learning_rate);
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
    
    // Save first attention module
    char attn1_filename[300];
    snprintf(attn1_filename, sizeof(attn1_filename), "%s_attn1.bin", base);
    save_attention(transformer->attention1, attn1_filename);
    
    // Save first MLP module
    char mlp1_filename[300];
    snprintf(mlp1_filename, sizeof(mlp1_filename), "%s_mlp1.bin", base);
    save_mlp(transformer->mlp1, mlp1_filename);
    
    // Save second attention module
    char attn2_filename[300];
    snprintf(attn2_filename, sizeof(attn2_filename), "%s_attn2.bin", base);
    save_attention(transformer->attention2, attn2_filename);
    
    // Save second MLP module
    char mlp2_filename[300];
    snprintf(mlp2_filename, sizeof(mlp2_filename), "%s_mlp2.bin", base);
    save_mlp(transformer->mlp2, mlp2_filename);
    
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
    
    // Load first attention module
    char attn1_filename[300];
    snprintf(attn1_filename, sizeof(attn1_filename), "%s_attn1.bin", base);
    Attention* attention1 = load_attention(attn1_filename, batch_size, cublas_handle);
    if (!attention1) return NULL;
    
    // Load first MLP module
    char mlp1_filename[300];
    snprintf(mlp1_filename, sizeof(mlp1_filename), "%s_mlp1.bin", base);
    MLP* mlp1 = load_mlp(mlp1_filename, batch_size, cublas_handle);
    if (!mlp1) {
        free_attention(attention1);
        return NULL;
    }
    
    // Load second attention module
    char attn2_filename[300];
    snprintf(attn2_filename, sizeof(attn2_filename), "%s_attn2.bin", base);
    Attention* attention2 = load_attention(attn2_filename, batch_size, cublas_handle);
    if (!attention2) {
        free_attention(attention1);
        free_mlp(mlp1);
        return NULL;
    }
    
    // Load second MLP module
    char mlp2_filename[300];
    snprintf(mlp2_filename, sizeof(mlp2_filename), "%s_mlp2.bin", base);
    MLP* mlp2 = load_mlp(mlp2_filename, batch_size, cublas_handle);
    if (!mlp2) {
        free_attention(attention1);
        free_mlp(mlp1);
        free_attention(attention2);
        return NULL;
    }
    
    // Create transformer and assign loaded modules
    Transformer* transformer = (Transformer*)malloc(sizeof(Transformer));
    transformer->d_model = d_model;
    transformer->seq_len = seq_len;
    transformer->batch_size = batch_size;
    transformer->mlp_hidden = mlp_hidden;
    transformer->cublas_handle = cublas_handle;
    transformer->attention1 = attention1;
    transformer->mlp1 = mlp1;
    transformer->attention2 = attention2;
    transformer->mlp2 = mlp2;
    
    printf("Model loaded from %s\n", filename);
    return transformer;
}