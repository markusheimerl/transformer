#include "transformer.h"

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
    
    // Allocate arrays for layer components
    transformer->attention_layers = (Attention**)malloc(num_layers * sizeof(Attention*));
    transformer->mlp_layers = (MLP**)malloc(num_layers * sizeof(MLP*));
    
    // Allocate intermediate buffers for double-pass
    int intermediate_size = batch_size * seq_len * d_model;
    CHECK_CUDA(cudaMalloc(&transformer->d_intermediate_output, intermediate_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&transformer->d_grad_intermediate_output, intermediate_size * sizeof(float)));
    
    // Initialize all layers
    for (int i = 0; i < num_layers; i++) {
        transformer->attention_layers[i] = init_attention(seq_len, d_model, batch_size, is_causal, cublaslt_handle);
        transformer->mlp_layers[i] = init_mlp(d_model, hidden_dim, d_model, batch_size * seq_len, cublaslt_handle);
    }
    
    return transformer;
}

// Free transformer memory
void free_transformer(Transformer* transformer) {
    // Free intermediate buffers
    cudaFree(transformer->d_intermediate_output);
    cudaFree(transformer->d_grad_intermediate_output);
    
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

// Forward pass - processes through all layers twice with weight sharing
void forward_pass_transformer(Transformer* transformer, float* d_X) {
    int total_elements = transformer->batch_size * transformer->seq_len * transformer->d_model;
    
    // FIRST PASS: Process through all layers
    for (int layer = 0; layer < transformer->num_layers; layer++) {
        float* layer_input = (layer == 0) ? d_X : transformer->mlp_layers[layer-1]->d_output;
        
        // Attention layer
        forward_pass_attention(transformer->attention_layers[layer], layer_input);
        
        // First residual connection
        residual_add_kernel<<<(total_elements + 255) / 256, 256>>>(
            transformer->attention_layers[layer]->d_output, 
            layer_input, 
            total_elements
        );
        
        // MLP layer
        forward_pass_mlp(transformer->mlp_layers[layer], transformer->attention_layers[layer]->d_output);
        
        // Second residual connection
        residual_add_kernel<<<(total_elements + 255) / 256, 256>>>(
            transformer->mlp_layers[layer]->d_output, 
            transformer->attention_layers[layer]->d_output, 
            total_elements
        );
    }
    
    // Store intermediate output after first pass
    CHECK_CUDA(cudaMemcpy(transformer->d_intermediate_output, 
                         transformer->mlp_layers[transformer->num_layers-1]->d_output,
                         total_elements * sizeof(float), 
                         cudaMemcpyDeviceToDevice));
    
    // SECOND PASS: Process through all layers again with same weights
    for (int layer = 0; layer < transformer->num_layers; layer++) {
        float* layer_input = (layer == 0) ? transformer->d_intermediate_output : transformer->mlp_layers[layer-1]->d_output;
        
        // Attention layer
        forward_pass_attention(transformer->attention_layers[layer], layer_input);
        
        // First residual connection
        residual_add_kernel<<<(total_elements + 255) / 256, 256>>>(
            transformer->attention_layers[layer]->d_output, 
            layer_input, 
            total_elements
        );
        
        // MLP layer
        forward_pass_mlp(transformer->mlp_layers[layer], transformer->attention_layers[layer]->d_output);
        
        // Second residual connection
        residual_add_kernel<<<(total_elements + 255) / 256, 256>>>(
            transformer->mlp_layers[layer]->d_output, 
            transformer->attention_layers[layer]->d_output, 
            total_elements
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

// Backward pass - backprops through both passes, accumulating gradients
void backward_pass_transformer(Transformer* transformer, float* d_X, float* d_grad_X) {
    int total_elements = transformer->batch_size * transformer->seq_len * transformer->d_model;
    
    // SECOND PASS BACKWARD: Backprop through the second pass
    for (int layer = transformer->num_layers - 1; layer >= 0; layer--) {
        float* layer_input = (layer == 0) ? transformer->d_intermediate_output : transformer->mlp_layers[layer-1]->d_output;
        float* layer_grad_input = (layer == 0) ? transformer->d_grad_intermediate_output : transformer->mlp_layers[layer-1]->d_grad_output;
        
        // Backward through MLP
        backward_pass_mlp(transformer->mlp_layers[layer], 
                         transformer->attention_layers[layer]->d_output, 
                         transformer->attention_layers[layer]->d_grad_output);
        
        // Add gradient from second residual connection
        residual_add_kernel<<<(total_elements + 255) / 256, 256>>>(
            transformer->attention_layers[layer]->d_grad_output, 
            transformer->mlp_layers[layer]->d_grad_output, 
            total_elements
        );
        
        // Backward through attention
        backward_pass_attention(transformer->attention_layers[layer], layer_input, layer_grad_input);
        
        // Add gradient from first residual connection
        if (layer_grad_input != NULL) {
            residual_add_kernel<<<(total_elements + 255) / 256, 256>>>(
                layer_grad_input, 
                transformer->attention_layers[layer]->d_grad_output, 
                total_elements
            );
        }
    }
    
    // FIRST PASS BACKWARD: Backprop through the first pass
    // Gradients from second pass are now in d_grad_intermediate_output
    for (int layer = transformer->num_layers - 1; layer >= 0; layer--) {
        float* layer_input = (layer == 0) ? d_X : transformer->mlp_layers[layer-1]->d_output;
        float* layer_grad_input = (layer == 0) ? d_grad_X : transformer->mlp_layers[layer-1]->d_grad_output;
        
        // For the last layer of first pass, we need to add the gradient from second pass input
        if (layer == transformer->num_layers - 1) {
            // Add gradient flowing from the second pass
            residual_add_kernel<<<(total_elements + 255) / 256, 256>>>(
                transformer->mlp_layers[layer]->d_grad_output,
                transformer->d_grad_intermediate_output,
                total_elements
            );
        }
        
        // Backward through MLP (gradients accumulate)
        backward_pass_mlp(transformer->mlp_layers[layer], 
                         transformer->attention_layers[layer]->d_output, 
                         transformer->attention_layers[layer]->d_grad_output);
        
        // Add gradient from second residual connection
        residual_add_kernel<<<(total_elements + 255) / 256, 256>>>(
            transformer->attention_layers[layer]->d_grad_output, 
            transformer->mlp_layers[layer]->d_grad_output, 
            total_elements
        );
        
        // Backward through attention (gradients accumulate)
        backward_pass_attention(transformer->attention_layers[layer], layer_input, layer_grad_input);
        
        // Add gradient from first residual connection
        if (layer_grad_input != NULL) {
            residual_add_kernel<<<(total_elements + 255) / 256, 256>>>(
                layer_grad_input, 
                transformer->attention_layers[layer]->d_grad_output, 
                total_elements
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

// Reset optimizer state
void reset_optimizer_transformer(Transformer* transformer) {
    for (int i = 0; i < transformer->num_layers; i++) {
        reset_optimizer_attention(transformer->attention_layers[i]);
        reset_optimizer_mlp(transformer->mlp_layers[i]);
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