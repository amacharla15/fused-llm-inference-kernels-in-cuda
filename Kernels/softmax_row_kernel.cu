#include <cfloat>

__global__ void softmax_row_kernel(const float* x, float* out, int rows, int cols) {
    float temp=-FLT_MAX;
    int tid=threadIdx.x;
    __shared__ float s[256];
    for(int i=tid;i<cols;i=i+blockDim.x){
        temp=max(temp,x[blockIdx.x*cols+i]);
    }
    s[tid]=temp;
    __syncthreads();
    temp=-FLT_MAX;
    if(tid<32){
        for(int i=tid;i<blockDim.x;i=i+32){
            temp=max(temp,s[i]);
        }
        float val=temp;
        val=max(val,__shfl_down_sync(0xffffffff,val,16));
        val=max(val,__shfl_down_sync(0xffffffff,val,8));
        val=max(val,__shfl_down_sync(0xffffffff,val,4));
        val=max(val,__shfl_down_sync(0xffffffff,val,2));
        val=max(val,__shfl_down_sync(0xffffffff,val,1));
        if(tid==0){
            s[0]=val;
        }
    }
    __syncthreads();
    for(int i=tid;i<cols;i=i+blockDim.x){
        out[cols*blockIdx.x+i]=expf(x[cols*blockIdx.x+i]-s[0]);
    }
    temp=0.0f;
    for(int i=tid;i<cols;i=i+blockDim.x){
        temp=temp+out[cols*blockIdx.x+i];
    }
    s[tid]=temp;
    __syncthreads();
    temp=0.0f;
    if(tid<32){
        for(int i=tid;i<blockDim.x;i=i+32){
            temp=temp+s[i];
        }
        float val=temp;
        val+=__shfl_down_sync(0xffffffff,val,16);
        val+=__shfl_down_sync(0xffffffff,val,8);
        val+=__shfl_down_sync(0xffffffff,val,4);
        val+=__shfl_down_sync(0xffffffff,val,2);
        val+=__shfl_down_sync(0xffffffff,val,1);
        if(tid==0){
            s[0]=val;
        }
    }
    __syncthreads();
    for(int i=tid;i<cols;i=i+blockDim.x){
        out[cols*blockIdx.x+i]/=s[0];
    }
}

int threads=256;
size_t smem=(threads/32)*sizeof(float);
softmax_row_kernel<<<rows,threads,smem>>>(d_x,d_out,rows,cols);