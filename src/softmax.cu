#include<cstdio>
#include<cuda_runtime.h>

__global__ void softmax(float *x,float *y, int no_of_rows, int number_of_columns){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if (row<no_of_rows && col<number_of_columns){
        float max_val=-1e20f;;
        for (int i=0;i<number_of_columns;i++){
            if (x[row*number_of_columns+i]>max_val){
                max_val=x[row*number_of_columns+i];
            }
        }
        float sum=0.0f;
        for (int i=0;i<number_of_columns;i++){
            sum+=exp(x[row*number_of_columns+i]-max_val);
        }
    
            y[row*number_of_columns+col]=exp(x[row*number_of_columns+col]-max_val)/sum;
        
    }
}

int main(){
    int no_of_rows=20;
    int number_of_columns=10;
    size_t bytes=no_of_rows*number_of_columns*sizeof(float);
    dim3 block(16,16);
    dim3 grid((number_of_columns+block.x-1)/block.x,(no_of_rows+block.y-1)/block.y);
    float *h_x=(float*)malloc(bytes);
    float *h_y=(float*)malloc(bytes);
    float *d_x,*d_y;
    for (int i=0;i<no_of_rows;i++){
        for (int j=0;j<number_of_columns;j++){
            h_x[i*number_of_columns+j]=i+j;
        }
    }
    cudaMalloc((void**)&d_x,bytes);
    cudaMalloc((void**)&d_y,bytes);
    cudaMemcpy(d_x,h_x,bytes,cudaMemcpyHostToDevice);
    softmax<<<grid,block>>>(d_x,d_y,no_of_rows,number_of_columns);
    cudaDeviceSynchronize();
    cudaGetLastError();
    cudaMemcpy(h_y,d_y,bytes,cudaMemcpyDeviceToHost);
    for (int i=0;i<no_of_rows;i++){
        for (int j=0;j<number_of_columns;j++){
            printf("%f ",h_y[i*number_of_columns+j]);
        }
        printf("\n");
    }
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
