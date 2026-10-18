__global__ void gelu_kernel(const float* x,float* out,int n){
    int threadsperblock=blockDim.x;
    int gid=threadsperblock*blockIdx.x+threadIdx.x;
    if(gid<n){
        out[gid]=0.5f*x[gid]*(1.0f+tanhf(0.7978845608f*(x[gid]+0.044715f*x[gid]*x[gid]*x[gid])));
    }
}
// Launch
int threads=256;
int blocks=(n+threads-1)/threads;
gelu_kernel<<<blocks,threads>>>(d_x,d_out,n);
cudaDeviceSynchronize();