__global__ void add_residual_kernel(const float* x,const float* residual,float* out,int n){
    int threadsperblock=blockDim.x;
    int gid=blockIdx.x*threadsperblock+threadIdx.x;
    if(gid<n){
        out[gid]=x[gid]+residual[gid];
    }
}
// Launch
int threads=256;
int blocks=(n+threads-1)/threads;
add_residual_kernel<<<blocks,threads>>>(d_x,d_residual,d_out,n);
cudaDeviceSynchronize();