#include "mlp.h"

// Initialize the MLP
MLP* init_mlp(int input_dim, int hidden_dim, int output_dim, int batch_size, int seq_len, cublasHandle_t cublas_handle) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    
    // Store dimensions
    mlp->input_dim = input_dim;
    mlp->hidden_dim = hidden_dim;
    mlp->output_dim = output_dim;
    mlp->batch_size = batch_size;
    mlp->seq_len = seq_len;
    
    // Initialize Adam parameters
    mlp->beta1 = 0.9f;
    mlp->beta2 = 0.999f;
    mlp->epsilon = 1e-8f;
    mlp->t = 0;
    mlp->weight_decay = 0.01f;
    
    // Initialize cuBLAS
    mlp->cublas_handle = cublas_handle;
    
    int w1_size = hidden_dim * input_dim;
    int w2_size = output_dim * hidden_dim;
    int total_seq = batch_size * seq_len;
    int hidden_buffer_size = total_seq * hidden_dim;
    int output_buffer_size = total_seq * output_dim;
    
    // Allocate host memory for weight initialization
    float* h_W1 = (float*)malloc(w1_size * sizeof(float));
    float* h_W2 = (float*)malloc(w2_size * sizeof(float));
    
    // Initialize weights on host
    float scale_W1 = 1.0f / sqrtf(input_dim);
    float scale_W2 = 1.0f / sqrtf(hidden_dim);
    
    for (int i = 0; i < w1_size; i++) {
        h_W1[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W1;
    }
    
    for (int i = 0; i < w2_size; i++) {
        h_W2[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_W2;
    }
    
    // Allocate device memory for weights and gradients
    CHECK_CUDA(cudaMalloc(&mlp->d_W1, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_grad, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_grad, w2_size * sizeof(float)));
    
    // Allocate device memory for Adam parameters
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_m, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W1_v, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_m, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_W2_v, w2_size * sizeof(float)));
    
    // Allocate device memory for forward pass buffers
    CHECK_CUDA(cudaMalloc(&mlp->d_layer_preact, hidden_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_layer_postact, hidden_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_layer_output, output_buffer_size * sizeof(float)));
    
    // Allocate device memory for backward pass buffers
    CHECK_CUDA(cudaMalloc(&mlp->d_error_hidden, hidden_buffer_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&mlp->d_error_output, output_buffer_size * sizeof(float)));
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(mlp->d_W1, h_W1, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2, h_W2, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam parameters to zero
    CHECK_CUDA(cudaMemset(mlp->d_W1_m, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W1_v, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_m, 0, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_v, 0, w2_size * sizeof(float)));
    
    // Free host memory
    free(h_W1); free(h_W2);
    
    return mlp;
}

// Free MLP memory
void free_mlp(MLP* mlp) {
    // Free device memory
    cudaFree(mlp->d_W1); cudaFree(mlp->d_W2);
    cudaFree(mlp->d_W1_grad); cudaFree(mlp->d_W2_grad);
    cudaFree(mlp->d_W1_m); cudaFree(mlp->d_W1_v);
    cudaFree(mlp->d_W2_m); cudaFree(mlp->d_W2_v);
    cudaFree(mlp->d_layer_preact); cudaFree(mlp->d_layer_postact);
    cudaFree(mlp->d_layer_output);
    cudaFree(mlp->d_error_hidden); cudaFree(mlp->d_error_output);
    free(mlp);
}

// CUDA kernel for Swish activation
__global__ void swish_forward_kernel_mlp(float* output, float* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float h = pre_activation[idx];
        output[idx] = h / (1.0f + expf(-h));
    }
}

// CUDA kernel for Swish derivative
__global__ void swish_backward_kernel_mlp(float* error_hidden, float* pre_activation, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float h = pre_activation[idx];
        float sigmoid = 1.0f / (1.0f + expf(-h));
        error_hidden[idx] *= sigmoid + h * sigmoid * (1.0f - sigmoid);
    }
}

// Forward pass
void forward_pass_mlp(MLP* mlp, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int total_seq = mlp->batch_size * mlp->seq_len;
    
    // H = XW₁
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            mlp->hidden_dim, total_seq, mlp->input_dim,
                            &alpha, mlp->d_W1, mlp->input_dim,
                            d_X, mlp->input_dim,
                            &beta, mlp->d_layer_preact, mlp->hidden_dim));

    // S = Hσ(H)
    int block_size = 256;
    int num_blocks = (total_seq * mlp->hidden_dim + block_size - 1) / block_size;
    swish_forward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_layer_postact,
        mlp->d_layer_preact,
        total_seq * mlp->hidden_dim
    );

    // Y = SW₂
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            mlp->output_dim, total_seq, mlp->hidden_dim,
                            &alpha, mlp->d_W2, mlp->hidden_dim,
                            mlp->d_layer_postact, mlp->hidden_dim,
                            &beta, mlp->d_layer_output, mlp->output_dim));
}

// Zero gradients
void zero_gradients_mlp(MLP* mlp) {
    int w1_size = mlp->hidden_dim * mlp->input_dim;
    int w2_size = mlp->output_dim * mlp->hidden_dim;
    
    CHECK_CUDA(cudaMemset(mlp->d_W1_grad, 0, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(mlp->d_W2_grad, 0, w2_size * sizeof(float)));
}

// Backward pass
void backward_pass_mlp(MLP* mlp, float* d_X) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int total_seq = mlp->batch_size * mlp->seq_len;

    // ∂L/∂W₂ = S^T(∂L/∂Y)
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            mlp->hidden_dim, mlp->output_dim, total_seq,
                            &alpha, mlp->d_layer_postact, mlp->hidden_dim,
                            mlp->d_error_output, mlp->output_dim,
                            &alpha, mlp->d_W2_grad, mlp->hidden_dim));

    // ∂L/∂S = (∂L/∂Y)(W₂)^T
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            mlp->hidden_dim, total_seq, mlp->output_dim,
                            &alpha, mlp->d_W2, mlp->hidden_dim,
                            mlp->d_error_output, mlp->output_dim,
                            &beta, mlp->d_error_hidden, mlp->hidden_dim));

    // ∂L/∂H = ∂L/∂S ⊙ [σ(H) + Hσ(H)(1-σ(H))]
    int block_size = 256;
    int num_blocks = (total_seq * mlp->hidden_dim + block_size - 1) / block_size;
    swish_backward_kernel_mlp<<<num_blocks, block_size>>>(
        mlp->d_error_hidden,
        mlp->d_layer_preact,
        total_seq * mlp->hidden_dim
    );

    // ∂L/∂W₁ = X^T(∂L/∂H)
    CHECK_CUBLAS(cublasSgemm(mlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            mlp->input_dim, mlp->hidden_dim, total_seq,
                            &alpha, d_X, mlp->input_dim,
                            mlp->d_error_hidden, mlp->hidden_dim,
                            &alpha, mlp->d_W1_grad, mlp->input_dim));
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel_mlp(float* weight, float* grad, float* m, float* v,
                                        float beta1, float beta2, float epsilon, float learning_rate,
                                        float weight_decay, float alpha_t, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        // m = β₁m + (1-β₁)(∂L/∂W)
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // v = β₂v + (1-β₂)(∂L/∂W)²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        // W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Update weights using AdamW
void update_weights_mlp(MLP* mlp, float learning_rate) {
    mlp->t++;
    
    float beta1_t = powf(mlp->beta1, mlp->t);
    float beta2_t = powf(mlp->beta2, mlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    int block_size = 256;
    int total_seq = mlp->batch_size * mlp->seq_len;
    
    int w1_size = mlp->hidden_dim * mlp->input_dim;
    int w2_size = mlp->output_dim * mlp->hidden_dim;
    
    // Update W1 weights
    int W1_blocks = (w1_size + block_size - 1) / block_size;
    adamw_update_kernel_mlp<<<W1_blocks, block_size>>>(
        mlp->d_W1, mlp->d_W1_grad, mlp->d_W1_m, mlp->d_W1_v,
        mlp->beta1, mlp->beta2, mlp->epsilon, learning_rate, mlp->weight_decay,
        alpha_t, w1_size, total_seq
    );
    
    // Update W2 weights
    int W2_blocks = (w2_size + block_size - 1) / block_size;
    adamw_update_kernel_mlp<<<W2_blocks, block_size>>>(
        mlp->d_W2, mlp->d_W2_grad, mlp->d_W2_m, mlp->d_W2_v,
        mlp->beta1, mlp->beta2, mlp->epsilon, learning_rate, mlp->weight_decay,
        alpha_t, w2_size, total_seq
    );
}

// Save MLP weights to binary file
void save_mlp(MLP* mlp, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&mlp->input_dim, sizeof(int), 1, file);
    fwrite(&mlp->hidden_dim, sizeof(int), 1, file);
    fwrite(&mlp->output_dim, sizeof(int), 1, file);
    fwrite(&mlp->batch_size, sizeof(int), 1, file);
    fwrite(&mlp->seq_len, sizeof(int), 1, file);
    
    int w1_size = mlp->hidden_dim * mlp->input_dim;
    int w2_size = mlp->output_dim * mlp->hidden_dim;
    
    // Allocate temporary host memory for weights
    float* h_W1 = (float*)malloc(w1_size * sizeof(float));
    float* h_W2 = (float*)malloc(w2_size * sizeof(float));
    
    // Copy weights from device to host
    CHECK_CUDA(cudaMemcpy(h_W1, mlp->d_W1, w1_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2, mlp->d_W2, w2_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_W1, sizeof(float), w1_size, file);
    fwrite(h_W2, sizeof(float), w2_size, file);
    
    // Save Adam state
    fwrite(&mlp->t, sizeof(int), 1, file);
    
    float* h_W1_m = (float*)malloc(w1_size * sizeof(float));
    float* h_W1_v = (float*)malloc(w1_size * sizeof(float));
    float* h_W2_m = (float*)malloc(w2_size * sizeof(float));
    float* h_W2_v = (float*)malloc(w2_size * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_W1_m, mlp->d_W1_m, w1_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W1_v, mlp->d_W1_v, w1_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2_m, mlp->d_W2_m, w2_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_W2_v, mlp->d_W2_v, w2_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(h_W1_m, sizeof(float), w1_size, file);
    fwrite(h_W1_v, sizeof(float), w1_size, file);
    fwrite(h_W2_m, sizeof(float), w2_size, file);
    fwrite(h_W2_v, sizeof(float), w2_size, file);
    
    // Free temporary host memory
    free(h_W1); free(h_W2);
    free(h_W1_m); free(h_W1_v);
    free(h_W2_m); free(h_W2_v);

    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load MLP weights from binary file
MLP* load_mlp(const char* filename, int custom_batch_size, cublasHandle_t cublas_handle) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, hidden_dim, output_dim, stored_batch_size, seq_len;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    fread(&seq_len, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    MLP* mlp = init_mlp(input_dim, hidden_dim, output_dim, batch_size, seq_len, cublas_handle);
    
    int w1_size = hidden_dim * input_dim;
    int w2_size = output_dim * hidden_dim;
    
    // Load weights
    float* h_W1 = (float*)malloc(w1_size * sizeof(float));
    float* h_W2 = (float*)malloc(w2_size * sizeof(float));
    
    fread(h_W1, sizeof(float), w1_size, file);
    fread(h_W2, sizeof(float), w2_size, file);
    
    CHECK_CUDA(cudaMemcpy(mlp->d_W1, h_W1, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2, h_W2, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Load Adam state
    fread(&mlp->t, sizeof(int), 1, file);
    
    float* h_W1_m = (float*)malloc(w1_size * sizeof(float));
    float* h_W1_v = (float*)malloc(w1_size * sizeof(float));
    float* h_W2_m = (float*)malloc(w2_size * sizeof(float));
    float* h_W2_v = (float*)malloc(w2_size * sizeof(float));
    
    fread(h_W1_m, sizeof(float), w1_size, file);
    fread(h_W1_v, sizeof(float), w1_size, file);
    fread(h_W2_m, sizeof(float), w2_size, file);
    fread(h_W2_v, sizeof(float), w2_size, file);
    
    CHECK_CUDA(cudaMemcpy(mlp->d_W1_m, h_W1_m, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W1_v, h_W1_v, w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2_m, h_W2_m, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(mlp->d_W2_v, h_W2_v, w2_size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_W1); free(h_W2);
    free(h_W1_m); free(h_W1_v);
    free(h_W2_m); free(h_W2_v);
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return mlp;
}