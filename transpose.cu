#include<cstdio>
#include<cuda_runtime.h>

__global__ void transpose2D(const float *in, float *out, int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // x → columns
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // y → rows
    
    if (row < rows && col < cols){
        int in_index = row * cols + col;      // Read from (row, col)
        int out_index = col * rows + row;     // Write to (col, row) - transposed!
        out[out_index] = in[in_index];
    }
}

int main(){
    int batch_size = 100;
    int rows = 28;
    int cols = 28;
    int matrix_size = rows * cols;

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,    // x → columns: (28+15)/16 = 2
              (rows + block.y - 1) / block.y);   // y → rows: (28+15)/16 = 2
    
    size_t bytes_per_matrix = rows * cols * sizeof(float);
    size_t total_bytes = batch_size * bytes_per_matrix;
    
    // Allocate host memory
    float *h_input = (float*)malloc(total_bytes);
    float *h_output = (float*)malloc(total_bytes);
    
    // Initialize input
    for (int batch = 0; batch < batch_size; batch++){
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                int idx = batch * matrix_size + i * cols + j;
                h_input[idx] = i * cols + j;  // Value = position
            }
        }
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, total_bytes);
    cudaMalloc((void**)&d_output, total_bytes);
    
    // Copy all batches to device
    cudaMemcpy(d_input, h_input, total_bytes, cudaMemcpyHostToDevice);
    
    // Process each batch with 2D transpose
    for (int batch = 0; batch < batch_size; batch++){
        float *d_in_batch = d_input + batch * matrix_size;
        float *d_out_batch = d_output + batch * matrix_size;
        
        transpose2D<<<grid, block>>>(d_in_batch, d_out_batch, rows, cols);
    }
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Copy results back
    cudaMemcpy(h_output, d_output, total_bytes, cudaMemcpyDeviceToHost);
    
    // Verify first batch
    printf("Input batch 0 (first 5x5):\n");
    for (int i = 0; i < 5; i++){
        for (int j = 0; j < 5; j++){
            printf("%3.0f ", h_input[i * cols + j]);
        }
        printf("\n");
    }
    
    printf("\nTransposed batch 0 (first 5x5):\n");
    for (int i = 0; i < 5; i++){
        for (int j = 0; j < 5; j++){
            printf("%3.0f ", h_output[i * rows + j]);
        }
        printf("\n");
    }
    
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}