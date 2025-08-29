#include "transformer.h"

// Initialize the transformer
Transformer* init_transformer(int d_model, int seq_len, int batch_size, int mlp_hidden, int num_layers, cublasHandle_t cublas_handle) {
    Transformer* transformer = (Transformer*)malloc(sizeof(Transformer));
    
    // Store dimensions
    transformer->d_model = d_model;
    transformer->seq_len = seq_len;
    transformer->batch_size = batch_size;
    transformer->mlp_hidden = mlp_hidden;
    transformer->num_layers = num_layers;
    transformer->cublas_handle = cublas_handle;
    transformer->t_scalars = 0;
    
    // Allocate arrays for layers
    transformer->attention_layers = (Attention**)malloc(num_layers * sizeof(Attention*));
    transformer->mlp_layers = (MLP**)malloc(num_layers * sizeof(MLP*));
    
    // Allocate and initialize residual scalars
    CHECK_CUDA(cudaMalloc(&transformer->d_gamma_attn, num_layers * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&transformer->d_gamma_mlp, num_layers * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&transformer->d_gamma_attn_grad, num_layers * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&transformer->d_gamma_mlp_grad, num_layers * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&transformer->d_gamma_attn_m, num_layers * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&transformer->d_gamma_attn_v, num_layers * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&transformer->d_gamma_mlp_m, num_layers * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&transformer->d_gamma_mlp_v, num_layers * sizeof(float)));
    
    // Initialize scalars to small values (0.1) on host then copy
    float* h_gamma = (float*)malloc(num_layers * sizeof(float));
    for (int i = 0; i < num_layers; i++) {
        h_gamma[i] = 0.1f;  // Small initial value
    }
    CHECK_CUDA(cudaMemcpy(transformer->d_gamma_attn, h_gamma, num_layers * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(transformer->d_gamma_mlp, h_gamma, num_layers * sizeof(float), cudaMemcpyHostToDevice));
    free(h_gamma);
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(transformer->d_gamma_attn_m, 0, num_layers * sizeof(float)));
    CHECK_CUDA(cudaMemset(transformer->d_gamma_attn_v, 0, num_layers * sizeof(float)));
    CHECK_CUDA(cudaMemset(transformer->d_gamma_mlp_m, 0, num_layers * sizeof(float)));
    CHECK_CUDA(cudaMemset(transformer->d_gamma_mlp_v, 0, num_layers * sizeof(float)));
    
    // Initialize all layers
    for (int i = 0; i < num_layers; i++) {
        transformer->attention_layers[i] = init_attention(d_model, seq_len, batch_size, cublas_handle);
        transformer->mlp_layers[i] = init_mlp(d_model, mlp_hidden, d_model, batch_size, seq_len, cublas_handle);
    }
    
    return transformer;
}

// Free memory
void free_transformer(Transformer* transformer) {
    for (int i = 0; i < transformer->num_layers; i++) {
        free_attention(transformer->attention_layers[i]);
        free_mlp(transformer->mlp_layers[i]);
    }
    cudaFree(transformer->d_gamma_attn);
    cudaFree(transformer->d_gamma_mlp);
    cudaFree(transformer->d_gamma_attn_grad);
    cudaFree(transformer->d_gamma_mlp_grad);
    cudaFree(transformer->d_gamma_attn_m);
    cudaFree(transformer->d_gamma_attn_v);
    cudaFree(transformer->d_gamma_mlp_m);
    cudaFree(transformer->d_gamma_mlp_v);
    free(transformer->attention_layers);
    free(transformer->mlp_layers);
    free(transformer);
}

// Forward pass
void forward_pass_transformer(Transformer* transformer, float* d_X) {
    int total_seq = transformer->batch_size * transformer->seq_len;
    
    // Set cuBLAS to use device pointers for scalars
    cublasPointerMode_t original_mode;
    CHECK_CUBLAS(cublasGetPointerMode(transformer->cublas_handle, &original_mode));
    CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

    // Forward pass through layers
    for (int layer = 0; layer < transformer->num_layers; layer++) {
        Attention* current_attn = transformer->attention_layers[layer];
        MLP* current_mlp = transformer->mlp_layers[layer];
        
        // Determine input for this layer
        float* layer_input = (layer == 0) ? d_X : transformer->mlp_layers[layer-1]->d_layer_output;
        
        // Step 1: Attention layer (uses host pointer mode internally)
        CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_HOST));
        forward_pass_attention(current_attn, layer_input);
        CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
        
        // Step 2: First residual connection - attention_output += gamma_attn * input
        CHECK_CUBLAS(cublasSaxpy(transformer->cublas_handle,
                                transformer->d_model * total_seq,
                                transformer->d_gamma_attn + layer,
                                layer_input, 1,
                                current_attn->d_layer_output, 1));
        
        // Step 3: MLP layer (uses host pointer mode internally)
        CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_HOST));
        forward_pass_mlp(current_mlp, current_attn->d_layer_output);
        CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
        
        // Step 4: Second residual connection - mlp_output += gamma_mlp * attention_output
        CHECK_CUBLAS(cublasSaxpy(transformer->cublas_handle,
                                transformer->d_model * total_seq,
                                transformer->d_gamma_mlp + layer,
                                current_attn->d_layer_output, 1,
                                current_mlp->d_layer_output, 1));
    }
    
    // Restore original pointer mode
    CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, original_mode));
}

// Calculate loss
float calculate_loss_transformer(Transformer* transformer, float* d_y) {
    // Loss is calculated against the final output
    MLP* final_mlp = transformer->mlp_layers[transformer->num_layers-1];
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
    for (int i = 0; i < transformer->num_layers; i++) {
        zero_gradients_attention(transformer->attention_layers[i]);
        zero_gradients_mlp(transformer->mlp_layers[i]);
    }
    CHECK_CUDA(cudaMemset(transformer->d_gamma_attn_grad, 0, transformer->num_layers * sizeof(float)));
    CHECK_CUDA(cudaMemset(transformer->d_gamma_mlp_grad, 0, transformer->num_layers * sizeof(float)));
}

// Backward pass
void backward_pass_transformer(Transformer* transformer, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int total_seq = transformer->batch_size * transformer->seq_len;
    int total_size = total_seq * transformer->d_model;
    
    // Set cuBLAS to use device pointers for scalars
    cublasPointerMode_t original_mode;
    CHECK_CUBLAS(cublasGetPointerMode(transformer->cublas_handle, &original_mode));
    CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
    
    // Backward pass through layers in reverse order
    for (int layer = transformer->num_layers - 1; layer >= 0; layer--) {
        Attention* current_attn = transformer->attention_layers[layer];
        MLP* current_mlp = transformer->mlp_layers[layer];
        
        // Determine the layer input that was used in forward pass
        float* layer_input = (layer == 0) ? d_X : transformer->mlp_layers[layer-1]->d_layer_output;
        
        // Step 1: Backward pass through current MLP (uses host pointer mode internally)
        CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_HOST));
        backward_pass_mlp(current_mlp, current_attn->d_layer_output);
        CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
        
        // Step 2: Compute gradient w.r.t. MLP input and store in attention error buffer
        CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_HOST));
        CHECK_CUBLAS(cublasSgemm(current_mlp->cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                transformer->d_model, total_seq, transformer->mlp_hidden,
                                &alpha, current_mlp->d_W1, transformer->d_model,
                                current_mlp->d_error_hidden, transformer->mlp_hidden,
                                &beta, current_attn->d_error_output, transformer->d_model));
        CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
        
        // Step 3: Gradient w.r.t gamma_mlp: d_gamma_mlp += sum(d_output * attention_output)
        CHECK_CUBLAS(cublasSdot(transformer->cublas_handle, total_size,
                               current_mlp->d_error_output, 1,
                               current_attn->d_layer_output, 1,
                               transformer->d_gamma_mlp_grad + layer));
        
        // Step 4: Add gradient from residual connection - attention_error += gamma_mlp * mlp_error
        CHECK_CUBLAS(cublasSaxpy(transformer->cublas_handle,
                                total_size,
                                transformer->d_gamma_mlp + layer,
                                current_mlp->d_error_output, 1,
                                current_attn->d_error_output, 1));
        
        // Step 5: Backward pass through current attention (uses host pointer mode internally)
        CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_HOST));
        backward_pass_attention(current_attn, layer_input);
        CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
        
        // Step 6: Gradient w.r.t gamma_attn: d_gamma_attn += sum(d_output * input)
        CHECK_CUBLAS(cublasSdot(transformer->cublas_handle, total_size,
                               current_attn->d_error_output, 1,
                               layer_input, 1,
                               transformer->d_gamma_attn_grad + layer));
        
        if (layer > 0) {
            // Step 7: Compute gradient w.r.t. attention input
            MLP* prev_mlp = transformer->mlp_layers[layer-1];
            
            CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_HOST));
            
            // ∂L/∂X = (∂L/∂Q)W_q^T
            CHECK_CUBLAS(cublasSgemm(transformer->cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    transformer->d_model, total_seq, transformer->d_model,
                                    &alpha, current_attn->d_W_q, transformer->d_model,
                                    current_attn->d_grad_Q, transformer->d_model,
                                    &beta, prev_mlp->d_error_output, transformer->d_model));
            
            // ∂L/∂X += (∂L/∂K)W_k^T
            CHECK_CUBLAS(cublasSgemm(transformer->cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    transformer->d_model, total_seq, transformer->d_model,
                                    &alpha, current_attn->d_W_k, transformer->d_model,
                                    current_attn->d_grad_K, transformer->d_model,
                                    &alpha, prev_mlp->d_error_output, transformer->d_model));
            
            // ∂L/∂X += (∂L/∂V)W_v^T
            CHECK_CUBLAS(cublasSgemm(transformer->cublas_handle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    transformer->d_model, total_seq, transformer->d_model,
                                    &alpha, current_attn->d_W_v, transformer->d_model,
                                    current_attn->d_grad_V, transformer->d_model,
                                    &alpha, prev_mlp->d_error_output, transformer->d_model));
            
            CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
            
            // Step 8: Add gradient from residual connection - prev_mlp_error += gamma_attn * attention_error
            CHECK_CUBLAS(cublasSaxpy(transformer->cublas_handle,
                                    total_size,
                                    transformer->d_gamma_attn + layer,
                                    current_attn->d_error_output, 1,
                                    prev_mlp->d_error_output, 1));
        }
    }
    
    // Restore original pointer mode
    CHECK_CUBLAS(cublasSetPointerMode(transformer->cublas_handle, original_mode));
}

// Update weights
void update_weights_transformer(Transformer* transformer, float learning_rate) {
    // Update layer weights
    for (int i = 0; i < transformer->num_layers; i++) {
        update_weights_attention(transformer->attention_layers[i], learning_rate);
        update_weights_mlp(transformer->mlp_layers[i], learning_rate);
    }
    
    // Update residual scalars using AdamW
    transformer->t_scalars++;
    
    float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f, weight_decay = 0.01f;
    float beta1_t = powf(beta1, transformer->t_scalars);
    float beta2_t = powf(beta2, transformer->t_scalars);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int total_seq = transformer->batch_size * transformer->seq_len;
    
    // Use the existing adamw_update_kernel from attention module
    int block_size = 256;
    int num_blocks = (transformer->num_layers + block_size - 1) / block_size;
    
    // Update gamma_attn
    adamw_update_kernel_attention<<<num_blocks, block_size>>>(
        transformer->d_gamma_attn, transformer->d_gamma_attn_grad,
        transformer->d_gamma_attn_m, transformer->d_gamma_attn_v,
        beta1, beta2, epsilon, learning_rate, weight_decay,
        alpha_t, transformer->num_layers, total_seq
    );
    
    // Update gamma_mlp
    adamw_update_kernel_attention<<<num_blocks, block_size>>>(
        transformer->d_gamma_mlp, transformer->d_gamma_mlp_grad,
        transformer->d_gamma_mlp_m, transformer->d_gamma_mlp_v,
        beta1, beta2, epsilon, learning_rate, weight_decay,
        alpha_t, transformer->num_layers, total_seq
    );
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
    
    // Save all layers
    for (int i = 0; i < transformer->num_layers; i++) {
        // Save attention module
        char attn_filename[300];
        snprintf(attn_filename, sizeof(attn_filename), "%s_attn%d.bin", base, i);
        save_attention(transformer->attention_layers[i], attn_filename);
        
        // Save MLP module
        char mlp_filename[300];
        snprintf(mlp_filename, sizeof(mlp_filename), "%s_mlp%d.bin", base, i);
        save_mlp(transformer->mlp_layers[i], mlp_filename);
    }
    
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
    fwrite(&transformer->num_layers, sizeof(int), 1, file);
    
    // Save residual scalars
    float* h_gamma_attn = (float*)malloc(transformer->num_layers * sizeof(float));
    float* h_gamma_mlp = (float*)malloc(transformer->num_layers * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_gamma_attn, transformer->d_gamma_attn, transformer->num_layers * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_gamma_mlp, transformer->d_gamma_mlp, transformer->num_layers * sizeof(float), cudaMemcpyDeviceToHost));
    fwrite(h_gamma_attn, sizeof(float), transformer->num_layers, file);
    fwrite(h_gamma_mlp, sizeof(float), transformer->num_layers, file);
    fwrite(&transformer->t_scalars, sizeof(int), 1, file);
    
    // Save Adam state for scalars
    float* h_gamma_attn_m = (float*)malloc(transformer->num_layers * sizeof(float));
    float* h_gamma_attn_v = (float*)malloc(transformer->num_layers * sizeof(float));
    float* h_gamma_mlp_m = (float*)malloc(transformer->num_layers * sizeof(float));
    float* h_gamma_mlp_v = (float*)malloc(transformer->num_layers * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_gamma_attn_m, transformer->d_gamma_attn_m, transformer->num_layers * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_gamma_attn_v, transformer->d_gamma_attn_v, transformer->num_layers * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_gamma_mlp_m, transformer->d_gamma_mlp_m, transformer->num_layers * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_gamma_mlp_v, transformer->d_gamma_mlp_v, transformer->num_layers * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_gamma_attn_m, sizeof(float), transformer->num_layers, file);
    fwrite(h_gamma_attn_v, sizeof(float), transformer->num_layers, file);
    fwrite(h_gamma_mlp_m, sizeof(float), transformer->num_layers, file);
    fwrite(h_gamma_mlp_v, sizeof(float), transformer->num_layers, file);
    
    free(h_gamma_attn); free(h_gamma_mlp);
    free(h_gamma_attn_m); free(h_gamma_attn_v);
    free(h_gamma_mlp_m); free(h_gamma_mlp_v);
    
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
    
    int d_model, seq_len, stored_batch_size, mlp_hidden, num_layers;
    fread(&d_model, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&mlp_hidden, sizeof(int), 1, file);
    fread(&num_layers, sizeof(int), 1, file);
    
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
    
    // Create transformer structure
    Transformer* transformer = init_transformer(d_model, seq_len, batch_size, mlp_hidden, num_layers, cublas_handle);
    
    // Load all layers
    for (int i = 0; i < num_layers; i++) {
        // Free the initialized layers first
        free_attention(transformer->attention_layers[i]);
        free_mlp(transformer->mlp_layers[i]);
        
        // Load attention module
        char attn_filename[300];
        snprintf(attn_filename, sizeof(attn_filename), "%s_attn%d.bin", base, i);
        transformer->attention_layers[i] = load_attention(attn_filename, batch_size, cublas_handle);

        // Load MLP module
        char mlp_filename[300];
        snprintf(mlp_filename, sizeof(mlp_filename), "%s_mlp%d.bin", base, i);
        transformer->mlp_layers[i] = load_mlp(mlp_filename, batch_size, cublas_handle);
    }
    
    // Load residual scalars
    float* h_gamma_attn = (float*)malloc(num_layers * sizeof(float));
    float* h_gamma_mlp = (float*)malloc(num_layers * sizeof(float));
    fread(h_gamma_attn, sizeof(float), num_layers, file);
    fread(h_gamma_mlp, sizeof(float), num_layers, file);
    fread(&transformer->t_scalars, sizeof(int), 1, file);
    
    CHECK_CUDA(cudaMemcpy(transformer->d_gamma_attn, h_gamma_attn, num_layers * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(transformer->d_gamma_mlp, h_gamma_mlp, num_layers * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state for scalars
    float* h_gamma_attn_m = (float*)malloc(num_layers * sizeof(float));
    float* h_gamma_attn_v = (float*)malloc(num_layers * sizeof(float));
    float* h_gamma_mlp_m = (float*)malloc(num_layers * sizeof(float));
    float* h_gamma_mlp_v = (float*)malloc(num_layers * sizeof(float));
    
    fread(h_gamma_attn_m, sizeof(float), num_layers, file);
    fread(h_gamma_attn_v, sizeof(float), num_layers, file);
    fread(h_gamma_mlp_m, sizeof(float), num_layers, file);
    fread(h_gamma_mlp_v, sizeof(float), num_layers, file);
    
    CHECK_CUDA(cudaMemcpy(transformer->d_gamma_attn_m, h_gamma_attn_m, num_layers * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(transformer->d_gamma_attn_v, h_gamma_attn_v, num_layers * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(transformer->d_gamma_mlp_m, h_gamma_mlp_m, num_layers * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(transformer->d_gamma_mlp_v, h_gamma_mlp_v, num_layers * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_gamma_attn); free(h_gamma_mlp);
    free(h_gamma_attn_m); free(h_gamma_attn_v);
    free(h_gamma_mlp_m); free(h_gamma_mlp_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return transformer;
}