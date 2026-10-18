__global__ void swiglu_kernel(const float* gate,const float* up,float* out,int n){
    int threadsperblock=blockDim.x;
    int gid=threadsperblock*blockIdx.x+threadIdx.x;
    if(gid<n){
        out[gid]=(gate[gid]/(1.0f+expf(-gate[gid])))*up[gid];
    }
}
// Launch
int threads=256;
int blocks=(n+threads-1)/threads;
swiglu_kernel<<<blocks,threads>>>(d_gate,d_up,d_out,n);
cudaDeviceSynchronize();