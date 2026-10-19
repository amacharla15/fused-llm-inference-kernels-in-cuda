__global__ void fused_linear_bias_gelu_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, int M, int N, int K) {
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
        temp=0.5f*temp*(1.0f+tanhf(sqrtf(2.0f/3.14159265f)*(temp+0.044715f*temp*temp*temp)));
        out[gid]=temp;
    }
}

int threads=256;
int blocks=(M*N+threads-1)/threads;
fused_linear_bias_gelu_kernel<<<blocks,threads>>>(d_x,d_weight,d_bias,d_out,M,N,K);