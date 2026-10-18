__global__ void fused_add_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int n,
    float eps
) {
    int tid = threadIdx.x;
    __shared__ float s[256];
    for (int i=tid;i<n;i+=blockDim.x){
        residual_out[blockIdx.x*n+i]=x[blockIdx.x*n+i]+residual[blockIdx.x*n+i];
    }
    __syncthreads();
    float temp=0.0f;
    for (int i=tid;i<n;i+=blockDim.x){
        temp+=residual_out[blockIdx.x*n+i]*residual_out[blockIdx.x*n+i];
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
    float inv_rms=1.0f/sqrtf(s[0]+eps);
    for(int i=tid;i<n;i+=blockDim.x){
        out[blockIdx.x*n+i]=inv_rms*residual_out[blockIdx.x*n+i]*weight[i];
    }
}

int threads=256;
fused_add_rmsnorm_kernel<<<rows,threads>>>(
    d_x,d_residual,d_weight,d_out,d_residual_out,n,1e-5f);