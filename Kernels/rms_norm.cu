__global__ void rmsnorm_kernel(const float* x,const float* weight,float* out,int n,float eps){
    int row=blockIdx.x;
    int tid=threadIdx.x;
    int row_start=row*n;
    int lane=tid%32;
    int warp_id=tid/32;
    int num_warps=blockDim.x/32;
    __shared__ float s[32];
    float local_sum=0.0f;
    for(int i=tid;i<n;i+=blockDim.x){
        float value=x[row_start+i];
        local_sum+=value*value;
    }
    float val=local_sum;
    val+=__shfl_down_sync(0xffffffff,val,16);
    val+=__shfl_down_sync(0xffffffff,val,8);
    val+=__shfl_down_sync(0xffffffff,val,4);
    val+=__shfl_down_sync(0xffffffff,val,2);
    val+=__shfl_down_sync(0xffffffff,val,1);
    if(lane==0){
        s[warp_id]=val;
    }
    __syncthreads();
    if(warp_id==0){
        float block_sum=0.0f;
        if(lane<num_warps){
            block_sum=s[lane];
        }
        block_sum+=__shfl_down_sync(0xffffffff,block_sum,16);
        block_sum+=__shfl_down_sync(0xffffffff,block_sum,8);
        block_sum+=__shfl_down_sync(0xffffffff,block_sum,4);
        block_sum+=__shfl_down_sync(0xffffffff,block_sum,2);
        block_sum+=__shfl_down_sync(0xffffffff,block_sum,1);
        if(lane==0){
            s[0]=1.0f/sqrtf(block_sum/n+eps);
        }
    }
    __syncthreads();
    for(int i=tid;i<n;i+=blockDim.x){
        out[row_start+i]=x[row_start+i]*s[0]*weight[i];
    }
}
// Launch
int threads=256;
rmsnorm_kernel<<<rows,threads>>>(d_x,d_weight,d_out,n,1e-5f);
cudaDeviceSynchronize();