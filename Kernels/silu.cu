__global__ void silu_kernel(const float* x,float* out,int n){
    int threadsperblock=blockDim.x;
    int gid=threadsperblock*blockIdx.x+threadIdx.x;
    if(gid<n){
        out[gid]=x[gid]/(1.0f+expf(-x[gid]));
    }
}
// Launch
int threads=256;
int blocks=(n+threads-1)/threads;
silu_kernel<<<blocks,threads>>>(d_x,d_out,n);
cudaDeviceSynchronize();