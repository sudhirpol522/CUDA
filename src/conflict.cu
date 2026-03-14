#include <stdio.h>
#include <cuda_runtime.h>

__global__ void no_conflict_kernel(int *d_out) {
    __shared__ int smem[32];

    int tid = threadIdx.x;

    // NO CONFLICT: Stride of 1
    // Thread 0 -> Bank 0, Thread 1 -> Bank 1, ... Thread 31 -> Bank 31
    smem[tid] = tid;

    __syncthreads();

    d_out[tid] = smem[tid];
}

__global__ void conflict_kernel(int *d_out) {
    __shared__ int smem[128];

    int tid = threadIdx.x;

    // 2-WAY BANK CONFLICT: Stride of 2
    // Thread 0 -> Bank 0, Thread 1 -> Bank 2, ... but
    // Thread 0 and Thread 16 both access Bank 0 -> 2-way conflict
    smem[tid * 4] = tid;

    __syncthreads();

    d_out[tid] = smem[tid * 4];
}

int main() {
    int *d_out;
    int size = 64 * sizeof(int);
    int h_out[64];

    cudaMalloc((void **)&d_out, size);

    printf("Running no_conflict_kernel...\n");
    no_conflict_kernel<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("no_conflict output[0..3]: %d %d %d %d\n", h_out[0], h_out[1], h_out[2], h_out[3]);

    printf("Running conflict_kernel...\n");
    conflict_kernel<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 64 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("conflict output[0..3]: %d %d %d %d\n", h_out[0], h_out[2], h_out[4], h_out[6]);

    cudaFree(d_out);
    return 0;
}