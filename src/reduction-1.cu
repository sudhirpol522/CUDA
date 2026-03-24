#include <iostream>
#include <cuda_runtime.h>

//major leanring is that
//1. shared memory is used to store the partial sums of the block



// ---------------------------------------------------------
// 1. The Kernel (Intra-Block Reduction using Shared Memory)
// ---------------------------------------------------------

// shared memory is used to store the partial sums of the block
__global__ void reduction_kernel(float* d_out, float* d_in, unsigned int size) {
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Dynamically allocated shared memory
    extern __shared__ float s_data[];

    // Load elements into shared memory. 
    // If the thread index is out of bounds, pad with 0.0f so it doesn't affect the sum.
    s_data[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.0f;
    
    // Ensure all threads in the block have finished writing to shared memory
    __syncthreads(); 

    // Perform the tree-based reduction
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        // Thread synchronous reduction
        if ((threadIdx.x % (stride * 2)) == 0) { // he must ahe, because of the race condition avoid krnyasathi, ithe global index nhi use kel ithe threadidx use kel ahe so.....
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        }
        // Ensure all additions for this stride are done before moving to the next
        __syncthreads();
    }

    // Thread 0 of each block writes its final partial sum to global memory
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = s_data[0];
    }
}

// ---------------------------------------------------------
// 2. The Host Wrapper (Inter-Block Reduction)
// ---------------------------------------------------------
void reduction(float *d_out, float *d_in, int n_threads, int size) {
    // Copy input to output array so we can perform the reduction "in-place"
    // Note: d_out must be allocated to hold at least 'size' elements!
    cudaMemcpy(d_out, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);

    int current_size = size;
    
    // Keep launching the kernel until we are down to a single block/element
    while(current_size > 1) {
        int n_blocks = (current_size + n_threads - 1) / n_threads;
        int shared_mem_bytes = n_threads * sizeof(float);

        // Launch kernel. We use d_out as both input and output.
        reduction_kernel<<<n_blocks, n_threads, shared_mem_bytes>>>(d_out, d_out, current_size);

        // Wait for the GPU to finish this launch before starting the next one
        cudaDeviceSynchronize(); 

        // The number of elements to process next time is exactly the number of blocks 
        // that just finished generating partial sums.
        current_size = n_blocks;
    }
}

// ---------------------------------------------------------
// 3. Main Execution
// ---------------------------------------------------------
int main() {
    // Setup parameters
    const int ARRAY_SIZE = 4096;
    const int THREADS_PER_BLOCK = 1024;
    size_t bytes = ARRAY_SIZE * sizeof(float);

    // Allocate host memory
    float* h_in = new float[ARRAY_SIZE];
    float h_out = 0.0f; // We only need one float on the host to hold the final answer

    // Initialize the array with 1.0f (so the final sum should exactly equal ARRAY_SIZE)
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = 1.0f;
    }

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes); // d_out needs to be the same size for the in-place copy

    // Copy data from Host to Device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Call our reduction wrapper
    reduction(d_out, d_in, THREADS_PER_BLOCK, ARRAY_SIZE);

    // Copy the final result (which sits at index 0 of d_out) back to the host
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result
    std::cout << "Expected Sum: " << ARRAY_SIZE << std::endl;
    std::cout << "Actual Sum:   " << h_out << std::endl;

    if (h_out == ARRAY_SIZE) {
        std::cout << "SUCCESS! Reduction worked correctly." << std::endl;
    } else {
        std::cout << "FAILED! Result does not match." << std::endl;
    }

    // Cleanup memory
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;

    return 0;
}