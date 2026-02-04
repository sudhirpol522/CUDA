#include <stdio.h>
#include <cuda_runtime.h>

__global__ void gemm_kernel(const float* A, const float* B, float* C,
                            int M_rows, int N_cols, int K_shared_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M_rows && column < N_cols) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < K_shared_dim; k_idx++) {
            sum += A[row * K_shared_dim + k_idx] * B[k_idx * N_cols + column];
        }
        C[row * N_cols + column] = sum;
    }
}

int main() {
    // Matrix dimensions: C(M×N) = A(M×K) × B(K×N)
    int M = 4;   // Rows of A and C
    int K = 3;   // Cols of A, Rows of B
    int N = 5;   // Cols of B and C
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    // Initialize A and B
    printf("Matrix A (%d,%d):\n", M, K);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = i + j;  // Simple pattern
            printf("%.0f ", h_A[i * K + j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix B (%d,%d):\n", K, N);
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = i * 10 + j;  // Simple pattern
            printf("%.0f ", h_B[i * N + j]);
        }
        printf("\n");
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // Launch configuration
    dim3 block(16, 16);  // 16×16 = 256 threads per block
    dim3 grid((N + block.x - 1) / block.x,    // Covers N columns
              (M + block.y - 1) / block.y);   // Covers M rows
    // For M=4, N=5: grid = (1, 1) - only 1 block needed!
    
    printf("\nLaunching kernel with grid(%d,%d) and block(%d,%d)\n", 
           grid.x, grid.y, block.x, block.y);
    
    // Launch kernel
    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Print result
    printf("\nMatrix C (%d,%d) = A , B:\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.0f ", h_C[i * N + j]);
        }
        printf("\n");
    }
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}