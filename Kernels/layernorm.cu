__global__ void layernorm_kernel(const float* x,const float* weight,const float* bias,float* out,int n,float eps){
    int tid=threadIdx.x;
    int row=blockIdx.x;
    float temp=0.0f;
    extern __shared__ float s[];
    for(int i=tid;i<n;i+=blockDim.x){
        temp+=x[row*n+i];
    }
    s[tid]=temp;
    __syncthreads();
    temp=0.0f;
    if(tid<32){
        for(int i=tid;i<blockDim.x;i+=32){
            temp+=s[i];
        }
        float val=temp;
        val+=__shfl_down_sync(0xffffffff,val,16);
        val+=__shfl_down_sync(0xffffffff,val,8);
        val+=__shfl_down_sync(0xffffffff,val,4);
        val+=__shfl_down_sync(0xffffffff,val,2);
        val+=__shfl_down_sync(0xffffffff,val,1);
        if(tid==0){
            s[0]=val/n;
        }
    }
    __syncthreads();
    float mean=s[0];
    temp=0.0f;
    for(int i=tid;i<n;i+=blockDim.x){
        float diff=x[row*n+i]-mean;
        temp+=diff*diff;
    }
    s[tid]=temp;
    __syncthreads();
    temp=0.0f;
    if(tid<32){
        for(int i=tid;i<blockDim.x;i+=32){
            temp+=s[i];
        }
        float val=temp;
        val+=__shfl_down_sync(0xffffffff,val,16);
        val+=__shfl_down_sync(0xffffffff,val,8);
        val+=__shfl_down_sync(0xffffffff,val,4);
        val+=__shfl_down_sync(0xffffffff,val,2);
        val+=__shfl_down_sync(0xffffffff,val,1);
        if(tid==0){
            s[0]=val/n;
        }
    }
    __syncthreads();
    float variance=s[0];
    float inv_std=rsqrtf(variance+eps);
    for(int i=tid;i<n;i+=blockDim.x){
        float normalized=(x[row*n+i]-mean)*inv_std;
        out[row*n+i]=normalized*weight[i]+bias[i];
    }
}
// Launch
int threads=256;
size_t shared_bytes=threads*sizeof(float);
layernorm_kernel<<<rows,threads,shared_bytes>>>(d_x,d_weight,d_bias,d_out,n,1e-5f);
cudaDeviceSynchronize();