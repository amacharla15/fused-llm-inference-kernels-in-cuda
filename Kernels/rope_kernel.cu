__global__ void rope_kernel(float* q, float* k,
                            const float* cos_table, const float* sin_table,
                            int seq_len, int n_heads, int head_dim) {
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int half=head_dim/2;
    int length=seq_len*n_heads*half;
    if(gid<length){
        int pair=gid%half;
        int head=(gid/half)%n_heads;
        int position=gid/(n_heads*half);
        int head_start=(position*n_heads+head)*head_dim;
        int even=head_start+2*pair;
        int odd=even+1;
        int table_idx=position*half+pair;
        float c=cos_table[table_idx];
        float s=sin_table[table_idx];
        float q_even=q[even];
        float q_odd=q[odd];
        float k_even=k[even];
        float k_odd=k[odd];
        q[even]=q_even*c-q_odd*s;
        q[odd]=q_even*s+q_odd*c;
        k[even]=k_even*c-k_odd*s;
        k[odd]=k_even*s+k_odd*c;
    }
}

int half=head_dim/2;
int total=seq_len*n_heads*half;
int threads=256;
int blocks=(total+threads-1)/threads;
rope_kernel<<<blocks,threads>>>(dq,dk,dcos,dsin,seq_len,n_heads,head_dim);