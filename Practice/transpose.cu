#include<cstdio>
#include<cuda_runtime.h>

__global__ void demo(float *input, float *output, int n)
{
    if (threadIdx.x < n)
    {
        int i = threadIdx.x;
        int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
        int block_id=blockIdx.x;
        printf("Thread %d: block %d: input[%d] = %f\n", thread_id, block_id, i, input[i]);
        output[i] = input[i] * 2;
        printf("Thread %d: block %d: output[%d] = %f\n", thread_id, block_id, i, output[i]);
    }
}

__global__ void block_max_kernel(const float* input, float* block_max, int N)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    __shared__ float shared_max[8];

    float local_max = -1e20f;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x)
    {
        local_max = max(local_max, input[i]);
        // printf("thread %d: index %d: %f\n", tid, i, input[i]);
        // printf("Thread %d: block %d: local_max = %f\n", tid, blockIdx.x, local_max);
    }
    
    // printf("Thread %d: block %d: local_max = %f\n", tid, blockIdx.x, local_max);
    shared_max[tid] = local_max;
    printf("Thread %d: block %d: shared_max[%d] = %f\n", tid, blockIdx.x, tid, shared_max[tid]);
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            shared_max[tid] = max(shared_max[tid] , shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        block_max[blockIdx.x] = shared_max[0];
    }
}


int main()
{
    int n=1000;
    float *h_input=(float*)malloc(n*sizeof(float));
    float *h_output=(float*)malloc(n*sizeof(float));
    float *d_input,*d_output;
    for (int i=0;i<n;i++)
    {
        h_input[i]=(i+1)*(i+1);
    }
    cudaMalloc((void**)&d_input,n*sizeof(float));
    cudaMalloc((void**)&d_output,n*sizeof(float));
    cudaMemcpy(d_input,h_input,n*sizeof(float),cudaMemcpyHostToDevice);
    block_max_kernel<<<2,4>>>(d_input,d_output,n);
    cudaDeviceSynchronize(); // Ensures printf output is flushed

    cudaMemcpy(h_output,d_output,n*sizeof(float),cudaMemcpyDeviceToHost);
    for (int i=0;i<n;i++)
    {
        printf("%f, %f \n",h_input[i],h_output[i]);
    }
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}