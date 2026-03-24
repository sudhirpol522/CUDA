#include<stdio.h>
#include<cuda_runtime.h>

__global__ void naive_reduction(float *input, int stride,int n){
    int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
    if(thread_id<n){
        input[thread_id]+=input[thread_id+stride];
    }
}

int main(){
    int n=50;
    float *input=(float*)malloc(n*sizeof(float));
    for(int i=0;i<n;i++){
        input[i]=i;
    }
    float *d_input;
    cudaMalloc((void**)&d_input,n*sizeof(float));
    cudaMemcpy(d_input,input,n*sizeof(float),cudaMemcpyHostToDevice);
    for(int stride=1;stride<n;stride*=2){
        naive_reduction<<<(n+1023)/1024,1024>>>(d_input,stride,n);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(input,d_input,n*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<n;i++){
        printf("%f\n", input[i]);
    }
    free(input);
    cudaFree(d_input);
    return 0;
}