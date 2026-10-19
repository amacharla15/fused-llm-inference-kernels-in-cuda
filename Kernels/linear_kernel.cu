__global__ void linear_kernel(const float* x, const float* weight,
                              const float* bias, float* out,
                              int M, int N, int K) {
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<M*N){
        int row=gid/N;
        int col=gid%N;
        float temp=0.0f;
        for(int i=0;i<K;i++){
            temp+=x[row*K+i]*weight[col*K+i];
        }
        if(bias!=nullptr){
            temp+=bias[col];
        }
        out[gid]=temp;
    }
}

int threads=256;
int blocks=(M*N+threads-1)/threads;
linear_kernel<<<blocks,threads>>>(d_x,d_w,d_b,d_out,M,N,K);