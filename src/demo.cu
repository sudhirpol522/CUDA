#include<stdio.h>
#include<iostream>
#include<cuda_runtime.h>

__global__ void demo(float *input, int stride, int n){
    int global_index=blockIdx.x*blockDim.x+threadIdx.x;
    if(global_index<n){
        input[global_index]+=input[global_index+stride];
    }
}

int main(){
    int n=100;
    float *input=(float*)malloc(n*sizeof(float));
    for(int i=0;i<n;i++){
        input[i]=i+1;
    }
    float *d_input;
    cudaMalloc((void**)&d_input,n*sizeof(float));
    cudaMemcpy(d_input,input,n*sizeof(float),cudaMemcpyHostToDevice);
    for(int stride=1;stride<n;stride*=2){
        demo<<<(n+1023)/1024,1024>>>(d_input,stride,n);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    cudaMemcpy(input,d_input,n*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<n;i++){
        std::cout<<input[i]<<" ";
    }
    std::cout<<"\n";
    free(input);
    cudaFree(d_input);
    return 0;
}