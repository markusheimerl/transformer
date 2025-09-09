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
    
    // Initialize Layer 1 components
    transformer->attention1 = init_attention(seq_len, d_model, batch_size, is_causal, cublas_handle, cublaslt_handle);
    transformer->mlp1 = init_mlp(d_model, hidden_dim, d_model, batch_size * seq_len, cublas_handle, cublaslt_handle);
    
    // Initialize Layer 2 components
    transformer->attention2 = init_attention(seq_len, d_model, batch_size, is_causal, cublas_handle, cublaslt_handle);
    transformer->mlp2 = init_mlp(d_model, hidden_dim, d_model, batch_size * seq_len, cublas_handle, cublaslt_handle);
    
    // Initialize Layer 3 components
    transformer->attention3 = init_attention(seq_len, d_model, batch_size, is_causal, cublas_handle, cublaslt_handle);
    transformer->mlp3 = init_mlp(d_model, hidden_dim, d_model, batch_size * seq_len, cublas_handle, cublaslt_handle);
    
    return transformer;
}

// Free transformer memory
void free_transformer(Transformer* transformer) {
    free_attention(transformer->attention1);
    free_mlp(transformer->mlp1);
    free_attention(transformer->attention2);
    free_mlp(transformer->mlp2);
    free_attention(transformer->attention3);
    free_mlp(transformer->mlp3);
    free(transformer);
}

// CUDA kernel for in-place residual connection: a += b
__global__ static void residual_add_kernel(float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}

// Forward pass
void forward_pass_transformer(Transformer* transformer, float* d_X) {
    int seq_batch_size = transformer->batch_size * transformer->seq_len * transformer->d_model;
    int block_size = 256;
    int num_blocks = (seq_batch_size + block_size - 1) / block_size;
    
    // ========== LAYER 1 ==========
    
    // Step 1: First attention layer
    forward_pass_attention(transformer->attention1, d_X);
    
    // Step 2: First residual connection - attention1_output += input
    residual_add_kernel<<<num_blocks, block_size>>>(
        transformer->attention1->d_output, 
        d_X, 
        seq_batch_size
    );
    
    // Step 3: First MLP layer (input: attention1->d_output)
    forward_pass_mlp(transformer->mlp1, transformer->attention1->d_output);
    
    // Step 4: Second residual connection - mlp1_output += attention1_output
    residual_add_kernel<<<num_blocks, block_size>>>(
        transformer->mlp1->d_layer_output, 
        transformer->attention1->d_output, 
        seq_batch_size
    );
    
    // ========== LAYER 2 ==========
    
    // Step 5: Second attention layer (input: mlp1->d_layer_output)
    forward_pass_attention(transformer->attention2, transformer->mlp1->d_layer_output);
    
    // Step 6: Third residual connection - attention2_output += mlp1_output
    residual_add_kernel<<<num_blocks, block_size>>>(
        transformer->attention2->d_output, 
        transformer->mlp1->d_layer_output, 
        seq_batch_size
    );
    
    // Step 7: Second MLP layer (input: attention2->d_output)
    forward_pass_mlp(transformer->mlp2, transformer->attention2->d_output);
    
    // Step 8: Fourth residual connection - mlp2_output += attention2_output
    residual_add_kernel<<<num_blocks, block_size>>>(
        transformer->mlp2->d_layer_output, 
        transformer->attention2->d_output, 
        seq_batch_size
    );
    
    // ========== LAYER 3 ==========
    
    // Step 9: Third attention layer (input: mlp2->d_layer_output)
    forward_pass_attention(transformer->attention3, transformer->mlp2->d_layer_output);
    
    // Step 10: Fifth residual connection - attention3_output += mlp2_output
    residual_add_kernel<<<num_blocks, block_size>>>(
        transformer->attention3->d_output, 
        transformer->mlp2->d_layer_output, 
        seq_batch_size
    );
    
    // Step 11: Third MLP layer (input: attention3->d_output)
    forward_pass_mlp(transformer->mlp3, transformer->attention3->d_output);
    
    // Step 12: Sixth residual connection - mlp3_output += attention3_output
    residual_add_kernel<<<num_blocks, block_size>>>(
        transformer->mlp3->d_layer_output, 
        transformer->attention3->d_output, 
        seq_batch_size
    );
    
    // Final output is in transformer->mlp3->d_layer_output
}

// Calculate loss using final MLP's output
float calculate_loss_transformer(Transformer* transformer, float* d_y) {
    return calculate_loss_mlp(transformer->mlp3, d_y);
}

// Zero gradients for all components
void zero_gradients_transformer(Transformer* transformer) {
    zero_gradients_attention(transformer->attention1);
    zero_gradients_mlp(transformer->mlp1);
    zero_gradients_attention(transformer->attention2);
    zero_gradients_mlp(transformer->mlp2);
    zero_gradients_attention(transformer->attention3);
    zero_gradients_mlp(transformer->mlp3);
}

// Backward pass
void backward_pass_transformer(Transformer* transformer, float* d_X, float* d_grad_X) {
    int seq_batch_size = transformer->batch_size * transformer->seq_len * transformer->d_model;
    int block_size = 256;
    int num_blocks = (seq_batch_size + block_size - 1) / block_size;
    
    // ========== LAYER 3 BACKWARD ==========
    
    // Step 1: Backward through third MLP
    backward_pass_mlp(transformer->mlp3, transformer->attention3->d_output, transformer->attention3->d_grad_output);
    
    // Step 2: Add gradient from sixth residual connection (mlp3_output += attention3_output)
    residual_add_kernel<<<num_blocks, block_size>>>(
        transformer->attention3->d_grad_output, 
        transformer->mlp3->d_grad_output, 
        seq_batch_size
    );
    
    // Step 3: Backward through third attention (gradient flows to mlp2->d_grad_output)
    backward_pass_attention(transformer->attention3, transformer->mlp2->d_layer_output, transformer->mlp2->d_grad_output);
    
    // Step 4: Add gradient from fifth residual connection (attention3_output += mlp2_output)
    residual_add_kernel<<<num_blocks, block_size>>>(
        transformer->mlp2->d_grad_output, 
        transformer->attention3->d_grad_output, 
        seq_batch_size
    );
    
    // ========== LAYER 2 BACKWARD ==========
    
    // Step 5: Backward through second MLP (gradient already in mlp2->d_grad_output from step 4)
    backward_pass_mlp(transformer->mlp2, transformer->attention2->d_output, transformer->attention2->d_grad_output);
    
    // Step 6: Add gradient from fourth residual connection (mlp2_output += attention2_output)
    residual_add_kernel<<<num_blocks, block_size>>>(
        transformer->attention2->d_grad_output, 
        transformer->mlp2->d_grad_output, 
        seq_batch_size
    );
    
    // Step 7: Backward through second attention (gradient flows to mlp1->d_grad_output)
    backward_pass_attention(transformer->attention2, transformer->mlp1->d_layer_output, transformer->mlp1->d_grad_output);
    
    // Step 8: Add gradient from third residual connection (attention2_output += mlp1_output)
    residual_add_kernel<<<num_blocks, block_size>>>(
        transformer->mlp1->d_grad_output, 
        transformer->attention2->d_grad_output, 
        seq_batch_size
    );
    
    // ========== LAYER 1 BACKWARD ==========
    
    // Step 9: Backward through first MLP (gradient already in mlp1->d_grad_output from step 8)
    backward_pass_mlp(transformer->mlp1, transformer->attention1->d_output, transformer->attention1->d_grad_output);
    
    // Step 10: Add gradient from second residual connection (mlp1_output += attention1_output)
    residual_add_kernel<<<num_blocks, block_size>>>(
        transformer->attention1->d_grad_output, 
        transformer->mlp1->d_grad_output, 
        seq_batch_size
    );
    
    // Step 11: Backward through first attention
    backward_pass_attention(transformer->attention1, d_X, d_grad_X);
    
    // Step 12: Add gradient from first residual connection (attention1_output += input)
    if (d_grad_X != NULL) {
        residual_add_kernel<<<num_blocks, block_size>>>(
            d_grad_X, 
            transformer->attention1->d_grad_output, 
            seq_batch_size
        );
    }
}

// Update weights for all components
void update_weights_transformer(Transformer* transformer, float learning_rate) {
    update_weights_attention(transformer->attention1, learning_rate);
    update_weights_mlp(transformer->mlp1, learning_rate);
    update_weights_attention(transformer->attention2, learning_rate);
    update_weights_mlp(transformer->mlp2, learning_rate);
    update_weights_attention(transformer->attention3, learning_rate);
    update_weights_mlp(transformer->mlp3, learning_rate);
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
    
    // Save attention is_causal flag (assuming all layers have same causal setting)
    fwrite(&transformer->attention1->is_causal, sizeof(bool), 1, file);
    
    fclose(file);
    
    // Save all components
    char attn1_filename[256], mlp1_filename[256];
    char attn2_filename[256], mlp2_filename[256];
    char attn3_filename[256], mlp3_filename[256];
    
    snprintf(attn1_filename, sizeof(attn1_filename), "%s_attn1.bin", filename);
    snprintf(mlp1_filename, sizeof(mlp1_filename), "%s_mlp1.bin", filename);
    snprintf(attn2_filename, sizeof(attn2_filename), "%s_attn2.bin", filename);
    snprintf(mlp2_filename, sizeof(mlp2_filename), "%s_mlp2.bin", filename);
    snprintf(attn3_filename, sizeof(attn3_filename), "%s_attn3.bin", filename);
    snprintf(mlp3_filename, sizeof(mlp3_filename), "%s_mlp3.bin", filename);
    
    save_attention(transformer->attention1, attn1_filename);
    save_mlp(transformer->mlp1, mlp1_filename);
    save_attention(transformer->attention2, attn2_filename);
    save_mlp(transformer->mlp2, mlp2_filename);
    save_attention(transformer->attention3, attn3_filename);
    save_mlp(transformer->mlp3, mlp3_filename);
    
    printf("Model saved to %s\n", filename);
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
    
    // Initialize transformer first
    Transformer* transformer = init_transformer(seq_len, d_model, hidden_dim, batch_size, is_causal, cublas_handle, cublaslt_handle);
    
    // Load all components
    char attn1_filename[256], mlp1_filename[256];
    char attn2_filename[256], mlp2_filename[256];
    char attn3_filename[256], mlp3_filename[256];
    
    snprintf(attn1_filename, sizeof(attn1_filename), "%s_attn1.bin", filename);
    snprintf(mlp1_filename, sizeof(mlp1_filename), "%s_mlp1.bin", filename);
    snprintf(attn2_filename, sizeof(attn2_filename), "%s_attn2.bin", filename);
    snprintf(mlp2_filename, sizeof(mlp2_filename), "%s_mlp2.bin", filename);
    snprintf(attn3_filename, sizeof(attn3_filename), "%s_attn3.bin", filename);
    snprintf(mlp3_filename, sizeof(mlp3_filename), "%s_mlp3.bin", filename);
    
    // Free the initialized components
    free_attention(transformer->attention1);
    free_mlp(transformer->mlp1);
    free_attention(transformer->attention2);
    free_mlp(transformer->mlp2);
    free_attention(transformer->attention3);
    free_mlp(transformer->mlp3);
    
    // Load the saved components
    transformer->attention1 = load_attention(attn1_filename, batch_size, cublas_handle, cublaslt_handle);
    transformer->mlp1 = load_mlp(mlp1_filename, batch_size * seq_len, cublas_handle, cublaslt_handle);
    transformer->attention2 = load_attention(attn2_filename, batch_size, cublas_handle, cublaslt_handle);
    transformer->mlp2 = load_mlp(mlp2_filename, batch_size * seq_len, cublas_handle, cublaslt_handle);
    transformer->attention3 = load_attention(attn3_filename, batch_size, cublas_handle, cublaslt_handle);
    transformer->mlp3 = load_mlp(mlp3_filename, batch_size * seq_len, cublas_handle, cublaslt_handle);
    
    printf("Model loaded from %s\n", filename);
    return transformer;
}