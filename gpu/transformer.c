#include "transformer.h"

// Initialize the transformer
Transformer* init_transformer(int seq_len, int d_model, int hidden_dim, int num_layers, int batch_size, bool is_causal, bool use_rope, cublasLtHandle_t cublaslt_handle) {
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
    
    // Initialize all layers
    for (int i = 0; i < num_layers; i++) {
        transformer->attention_layers[i] = init_attention(seq_len, d_model, batch_size, is_causal, use_rope, cublaslt_handle);
        transformer->mlp_layers[i] = init_mlp(d_model, hidden_dim, d_model, batch_size * seq_len, cublaslt_handle);
    }
    
    return transformer;
}

// Free transformer memory
void free_transformer(Transformer* transformer) {
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
    // Process each layer sequentially
    for (int layer = 0; layer < transformer->num_layers; layer++) {
        float* layer_input = (layer == 0) ? d_X : transformer->mlp_layers[layer-1]->d_output;
        
        // Step 1: Attention layer
        forward_pass_attention(transformer->attention_layers[layer], layer_input);
        
        // Step 2: First residual connection - attention_output += input
        residual_add_kernel<<<(transformer->batch_size * transformer->seq_len * transformer->d_model + 255) / 256, 256>>>(
            transformer->attention_layers[layer]->d_output, 
            layer_input, 
            transformer->batch_size * transformer->seq_len * transformer->d_model
        );
        
        // Step 3: MLP layer (input: attention->d_output)
        forward_pass_mlp(transformer->mlp_layers[layer], transformer->attention_layers[layer]->d_output);
        
        // Step 4: Second residual connection - mlp_output += attention_output
        residual_add_kernel<<<(transformer->batch_size * transformer->seq_len * transformer->d_model + 255) / 256, 256>>>(
            transformer->mlp_layers[layer]->d_output, 
            transformer->attention_layers[layer]->d_output, 
            transformer->batch_size * transformer->seq_len * transformer->d_model
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
    // Process layers in reverse order
    for (int layer = transformer->num_layers - 1; layer >= 0; layer--) {
        float* layer_input = (layer == 0) ? d_X : transformer->mlp_layers[layer-1]->d_output;
        float* layer_grad_input = (layer == 0) ? d_grad_X : transformer->mlp_layers[layer-1]->d_grad_output;
        
        // Step 1: Backward through MLP
        backward_pass_mlp(transformer->mlp_layers[layer], 
                         transformer->attention_layers[layer]->d_output, 
                         transformer->attention_layers[layer]->d_grad_output);
        
        // Step 2: Add gradient from second residual connection (mlp_output += attention_output)
        residual_add_kernel<<<(transformer->batch_size * transformer->seq_len * transformer->d_model + 255) / 256, 256>>>(
            transformer->attention_layers[layer]->d_grad_output, 
            transformer->mlp_layers[layer]->d_grad_output, 
            transformer->batch_size * transformer->seq_len * transformer->d_model
        );
        
        // Step 3: Backward through attention
        backward_pass_attention(transformer->attention_layers[layer], layer_input, layer_grad_input);
        
        // Step 4: Add gradient from first residual connection (attention_output += input)
        if (layer_grad_input != NULL) {
            residual_add_kernel<<<(transformer->batch_size * transformer->seq_len * transformer->d_model + 255) / 256, 256>>>(
                layer_grad_input, 
                transformer->attention_layers[layer]->d_grad_output, 
                transformer->batch_size * transformer->seq_len * transformer->d_model
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

// Serialize transformer to a file
void serialize_transformer(Transformer* transformer, FILE* file) {
    // Write dimensions
    fwrite(&transformer->seq_len, sizeof(int), 1, file);
    fwrite(&transformer->d_model, sizeof(int), 1, file);
    fwrite(&transformer->batch_size, sizeof(int), 1, file);
    fwrite(&transformer->hidden_dim, sizeof(int), 1, file);
    fwrite(&transformer->num_layers, sizeof(int), 1, file);
    
    // Write attention flags
    fwrite(&transformer->attention_layers[0]->is_causal, sizeof(bool), 1, file);
    fwrite(&transformer->attention_layers[0]->use_rope, sizeof(bool), 1, file);
    
    // Serialize all layers
    for (int i = 0; i < transformer->num_layers; i++) {
        serialize_attention(transformer->attention_layers[i], file);
        serialize_mlp(transformer->mlp_layers[i], file);
    }
}

// Deserialize transformer from a file
Transformer* deserialize_transformer(FILE* file, int custom_batch_size, cublasLtHandle_t cublaslt_handle) {
    // Read dimensions
    int seq_len, d_model, stored_batch_size, hidden_dim, num_layers;
    bool is_causal, use_rope;
    fread(&seq_len, sizeof(int), 1, file);
    fread(&d_model, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    fread(&is_causal, sizeof(bool), 1, file);
    fread(&use_rope, sizeof(bool), 1, file);
    
    // Use custom batch size if provided
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize transformer
    Transformer* transformer = init_transformer(seq_len, d_model, hidden_dim, num_layers, batch_size, is_causal, use_rope, cublaslt_handle);
    
    // Deserialize all layers
    for (int i = 0; i < num_layers; i++) {
        // Free the initialized components
        free_attention(transformer->attention_layers[i]);
        free_mlp(transformer->mlp_layers[i]);
        
        // Deserialize the saved components
        transformer->attention_layers[i] = deserialize_attention(file, batch_size, cublaslt_handle);
        transformer->mlp_layers[i] = deserialize_mlp(file, batch_size * seq_len, cublaslt_handle);
    }
    
    return transformer;
}