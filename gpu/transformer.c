#include "transformer.h"

// CUDA kernel for RMSNorm forward (parameter-free)
__global__ static void rmsnorm_forward_kernel(float* output, float* input, int batch_size, int seq_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len;
    float eps = 1e-6f;
    
    if (idx < total) {
        float* in_vec = &input[idx * d_model];
        float* out_vec = &output[idx * d_model];
        
        // Compute mean of squares
        float sum_sq = 0.0f;
        for (int i = 0; i < d_model; i++) {
            sum_sq += in_vec[i] * in_vec[i];
        }
        float rms = sqrtf(sum_sq / d_model + eps);
        
        // Normalize
        for (int i = 0; i < d_model; i++) {
            out_vec[i] = in_vec[i] / rms;
        }
    }
}

// CUDA kernel for RMSNorm backward
__global__ static void rmsnorm_backward_kernel(float* grad_input, float* grad_output, float* input, int batch_size, int seq_len, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len;
    float eps = 1e-6f;
    
    if (idx < total) {
        float* in_vec = &input[idx * d_model];
        float* grad_out_vec = &grad_output[idx * d_model];
        float* grad_in_vec = &grad_input[idx * d_model];
        
        // Compute mean of squares (same as forward)
        float sum_sq = 0.0f;
        for (int i = 0; i < d_model; i++) {
            sum_sq += in_vec[i] * in_vec[i];
        }
        float mean_sq = sum_sq / d_model;
        float rms = sqrtf(mean_sq + eps);
        float rms3 = rms * rms * rms;
        
        // Compute sum of (x_j * grad_out_j)
        float sum_grad_x = 0.0f;
        for (int i = 0; i < d_model; i++) {
            sum_grad_x += in_vec[i] * grad_out_vec[i];
        }
        
        // Compute gradient (accumulate)
        for (int i = 0; i < d_model; i++) {
            grad_in_vec[i] += (grad_out_vec[i] / rms) - (in_vec[i] * sum_grad_x) / (d_model * rms3);
        }
    }
}

// Initialize the transformer
Transformer* init_transformer(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, bool is_causal, cublasLtHandle_t cublaslt_handle) {
    Transformer* transformer = (Transformer*)malloc(sizeof(Transformer));
    
    // Store dimensions and handle
    transformer->seq_len = seq_len;
    transformer->d_model = d_model;
    transformer->batch_size = batch_size;
    transformer->hidden_dim = hidden_dim;
    transformer->num_layers = num_layers;
    transformer->cublaslt_handle = cublaslt_handle;
    
    // Allocate RMSNorm buffers
    int norm_buffer_size = batch_size * seq_len * d_model;
    CHECK_CUDA(cudaMalloc(&transformer->d_norm_buffer1, norm_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&transformer->d_norm_buffer2, norm_buffer_size * sizeof(float)));
    
    // Allocate arrays for layer components
    transformer->attention_layers = (Attention**)malloc(num_layers * sizeof(Attention*));
    transformer->mlp_layers = (MLP**)malloc(num_layers * sizeof(MLP*));
    
    // Initialize all layers
    for (int i = 0; i < num_layers; i++) {
        transformer->attention_layers[i] = init_attention(seq_len, d_model, batch_size, is_causal, cublaslt_handle);
        transformer->mlp_layers[i] = init_mlp(d_model, hidden_dim, d_model, batch_size * seq_len, cublaslt_handle);
    }
    
    return transformer;
}

// Free transformer memory
void free_transformer(Transformer* transformer) {
    // Free RMSNorm buffers
    cudaFree(transformer->d_norm_buffer1);
    cudaFree(transformer->d_norm_buffer2);
    
    // Free all layers
    for (int i = 0; i < transformer->num_layers; i++) {
        free_attention(transformer->attention_layers[i]);
        free_mlp(transformer->mlp_layers[i]);
    }
    
    // Free layer arrays
    free(transformer->attention_layers);
    free(transformer->mlp_layers);
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
    int size = transformer->batch_size * transformer->seq_len * transformer->d_model;
    int block_size = 256;
    int num_blocks = (transformer->batch_size * transformer->seq_len + block_size - 1) / block_size;
    
    // Process each layer sequentially
    for (int layer = 0; layer < transformer->num_layers; layer++) {
        float* layer_input = (layer == 0) ? d_X : transformer->mlp_layers[layer-1]->d_output;
        
        // Step 1: Apply RMSNorm before attention
        rmsnorm_forward_kernel<<<num_blocks, block_size>>>(
            transformer->d_norm_buffer1, layer_input,
            transformer->batch_size, transformer->seq_len, transformer->d_model
        );
        
        // Step 2: Attention layer
        forward_pass_attention(transformer->attention_layers[layer], transformer->d_norm_buffer1);
        
        // Step 3: First residual connection - attention_output += input
        residual_add_kernel<<<(size + 255) / 256, 256>>>(
            transformer->attention_layers[layer]->d_output, 
            layer_input, 
            size
        );
        
        // Step 4: Apply RMSNorm before MLP
        rmsnorm_forward_kernel<<<num_blocks, block_size>>>(
            transformer->d_norm_buffer2, transformer->attention_layers[layer]->d_output,
            transformer->batch_size, transformer->seq_len, transformer->d_model
        );
        
        // Step 5: MLP layer (input: d_norm_buffer2)
        forward_pass_mlp(transformer->mlp_layers[layer], transformer->d_norm_buffer2);
        
        // Step 6: Second residual connection - mlp_output += attention_output
        residual_add_kernel<<<(size + 255) / 256, 256>>>(
            transformer->mlp_layers[layer]->d_output, 
            transformer->attention_layers[layer]->d_output, 
            size
        );
    }
}

// Calculate loss
float calculate_loss_transformer(Transformer* transformer, float* d_y) {
    return calculate_loss_mlp(transformer->mlp_layers[transformer->num_layers - 1], d_y);
}

// Zero gradients
void zero_gradients_transformer(Transformer* transformer) {
    for (int i = 0; i < transformer->num_layers; i++) {
        zero_gradients_attention(transformer->attention_layers[i]);
        zero_gradients_mlp(transformer->mlp_layers[i]);
    }
}

// Backward pass
void backward_pass_transformer(Transformer* transformer, float* d_X, float* d_grad_X) {
    int size = transformer->batch_size * transformer->seq_len * transformer->d_model;
    int block_size = 256;
    int num_blocks = (transformer->batch_size * transformer->seq_len + block_size - 1) / block_size;
    
    // Process layers in reverse order
    for (int layer = transformer->num_layers - 1; layer >= 0; layer--) {
        float* layer_input = (layer == 0) ? d_X : transformer->mlp_layers[layer-1]->d_output;
        float* layer_grad_input = (layer == 0) ? d_grad_X : transformer->mlp_layers[layer-1]->d_grad_output;
        float* result1 = transformer->attention_layers[layer]->d_output;  // attention_output + layer_input from forward
        
        // Step 1: Add gradient from second residual connection (mlp_output += attention_output)
        residual_add_kernel<<<(size + 255) / 256, 256>>>(
            transformer->attention_layers[layer]->d_grad_output, 
            transformer->mlp_layers[layer]->d_grad_output, 
            size
        );
        
        // Step 2: Recompute norm2 for MLP backward
        rmsnorm_forward_kernel<<<num_blocks, block_size>>>(
            transformer->d_norm_buffer2, result1,
            transformer->batch_size, transformer->seq_len, transformer->d_model
        );
        
        // Step 3: Backward through MLP
        backward_pass_mlp(transformer->mlp_layers[layer], 
                         transformer->d_norm_buffer2,  // norm2 (input to MLP in forward)
                         transformer->d_norm_buffer1);  // write grad w.r.t. norm2 here
        
        // Step 4: Backward through RMSNorm2
        rmsnorm_backward_kernel<<<num_blocks, block_size>>>(
            transformer->attention_layers[layer]->d_grad_output,  // accumulate gradient here
            transformer->d_norm_buffer1,  // gradient w.r.t. norm2
            result1,  // input to RMSNorm2
            transformer->batch_size, transformer->seq_len, transformer->d_model
        );
        
        // Step 5: Add gradient from first residual connection (attention_output += input)
        if (layer_grad_input != NULL) {
            residual_add_kernel<<<(size + 255) / 256, 256>>>(
                layer_grad_input, 
                transformer->attention_layers[layer]->d_grad_output, 
                size
            );
        }
        
        // Step 6: Recompute norm1 for attention backward
        rmsnorm_forward_kernel<<<num_blocks, block_size>>>(
            transformer->d_norm_buffer1, layer_input,
            transformer->batch_size, transformer->seq_len, transformer->d_model
        );
        
        // Step 7: Backward through attention
        backward_pass_attention(transformer->attention_layers[layer], 
                               transformer->d_norm_buffer1,  // norm1 (input to attention in forward)
                               transformer->d_norm_buffer2);  // write grad w.r.t. norm1 here
        
        // Step 8: Backward through RMSNorm1
        if (layer_grad_input != NULL) {
            rmsnorm_backward_kernel<<<num_blocks, block_size>>>(
                layer_grad_input,  // accumulate gradient here
                transformer->d_norm_buffer2,  // gradient w.r.t. norm1
                layer_input,  // input to RMSNorm1
                transformer->batch_size, transformer->seq_len, transformer->d_model
            );
        }
    }
}

// Update weights for all components
void update_weights_transformer(Transformer* transformer, float learning_rate, int effective_batch_size) {
    for (int i = 0; i < transformer->num_layers; i++) {
        update_weights_attention(transformer->attention_layers[i], learning_rate, effective_batch_size);
        update_weights_mlp(transformer->mlp_layers[i], learning_rate, effective_batch_size);
    }
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
    fwrite(&transformer->num_layers, sizeof(int), 1, file);
    
    // Save attention is_causal flag
    fwrite(&transformer->attention_layers[0]->is_causal, sizeof(bool), 1, file);
    
    fclose(file);
    
    // Create base filename by removing .bin extension
    char base_filename[256];
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    
    // Find and remove .bin extension if it exists
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    // Save all layer components
    for (int i = 0; i < transformer->num_layers; i++) {
        char attn_filename[256], mlp_filename[256];
        
        snprintf(attn_filename, sizeof(attn_filename), "%s_attn%d.bin", base_filename, i);
        snprintf(mlp_filename, sizeof(mlp_filename), "%s_mlp%d.bin", base_filename, i);
        
        save_attention(transformer->attention_layers[i], attn_filename);
        save_mlp(transformer->mlp_layers[i], mlp_filename);
    }

    printf("Model saved to %s\n", filename);
}

// Load transformer from binary file
Transformer* load_transformer(const char* filename, int custom_batch_size, cublasLtHandle_t cublaslt_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int seq_len, d_model, stored_batch_size, hidden_dim, num_layers;
    bool is_causal;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    
    fclose(file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize transformer first
    Transformer* transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, is_causal, cublaslt_handle);
    
    // Create base filename by removing .bin extension
    char base_filename[256];
    strncpy(base_filename, filename, sizeof(base_filename) - 1);
    base_filename[sizeof(base_filename) - 1] = '\0';
    
    // Find and remove .bin extension if it exists
    char* dot_pos = strrchr(base_filename, '.');
    if (dot_pos && strcmp(dot_pos, ".bin") == 0) {
        *dot_pos = '\0';
    }
    
    // Load all layer components
    for (int i = 0; i < num_layers; i++) {
        char attn_filename[256], mlp_filename[256];
        
        snprintf(attn_filename, sizeof(attn_filename), "%s_attn%d.bin", base_filename, i);
        snprintf(mlp_filename, sizeof(mlp_filename), "%s_mlp%d.bin", base_filename, i);
        
        // Free the initialized components
        free_attention(transformer->attention_layers[i]);
        free_mlp(transformer->mlp_layers[i]);
        
        // Load the saved components
        transformer->attention_layers[i] = load_attention(attn_filename, batch_size, cublaslt_handle);
        transformer->mlp_layers[i] = load_mlp(mlp_filename, batch_size * seq_len, cublaslt_handle);
    }
    
    printf("Model loaded from %s\n", filename);
    return transformer;
}