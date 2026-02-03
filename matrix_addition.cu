#include<cstdio>
#include<cuda_runtime.h>

__global__ void add2DMatrix(float *a,float *b,float *c, int rows, int cols){
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int col=blockIdx.y*blockDim.y+threadIdx.y;
    if (row<rows && col<cols){
        c[row*cols+col]=a[row*cols+col]+b[row*cols+col];
    }
}

int main(){
    int rows=8;
    int cols=8;
    
    dim3 block(2,2);
    dim3 grid((cols+block.x-1)/block.x,(rows+block.y-1)/block.y);

    size_t bytes=rows*cols*sizeof(float);
    float *h_a=(float*)malloc(bytes);
    float *h_b=(float*)malloc(bytes);
    float *h_c=(float*)malloc(bytes);
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            h_a[i*cols+j]=i;
            h_b[i*cols+j]=j;
        }
    }
    float *d_a,*d_b,*d_c;
    cudaMalloc((void**)&d_a,bytes);
    cudaMalloc((void**)&d_b,bytes);
    cudaMalloc((void**)&d_c,bytes);
    cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice);
    add2DMatrix<<<grid,block>>>(d_a,d_b,d_c,rows,cols);
    cudaDeviceSynchronize();
    cudaGetLastError();
    cudaMemcpy(h_c,d_c,bytes,cudaMemcpyDeviceToHost);
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            printf("%f ",h_c[i*cols+j]);
        }
        printf("\n");
    }
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}