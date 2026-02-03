#include<cstdio>
#include<cuda_runtime.h>

__global__ void transpose(float *a,float *b,int rows,int cols){
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    if (row<rows && col<cols){
        b[col*rows+row]=a[row*cols+col];
    }
}

int main(){
    int rows=8;
    int cols=8;
    dim3 block(16,16);
    dim3 grid((cols+block.x-1)/block.x,(rows+block.y-1)/block.y);
    size_t bytes=rows*cols*sizeof(float);
    float *h_a=(float*)malloc(bytes);
    float *h_b=(float*)malloc(bytes);
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            h_a[i*cols+j]=i;
        }
    }
    float *d_a,*d_b;
    cudaMalloc((void**)&d_a,bytes);
    cudaMalloc((void**)&d_b,bytes);
    cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice);
    transpose<<<grid,block>>>(d_a,d_b,rows,cols);
    cudaDeviceSynchronize();
    cudaGetLastError();
    cudaMemcpy(h_b,d_b,bytes,cudaMemcpyDeviceToHost);
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            printf("%f ",h_b[i*cols+j]);
        }
        printf("\n");
    }
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}