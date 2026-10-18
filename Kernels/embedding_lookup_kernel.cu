__global__ void embedding_lookup_kernel(const int* token_ids, const float* weight, float* out, int seq_len, int vocab_size, int embed_dim) {
    int gid=blockDim.x*blockIdx.x+threadIdx.x;
    int temp=seq_len*embed_dim;
    if(gid<temp){
        int token_position=gid/embed_dim;
        int embedding_position=gid%embed_dim;
        int token_id=token_ids[token_position];
        float var=weight[token_id*embed_dim+embedding_position];
        out[gid]=var;
    }
}

int total=seq_len*embed_dim;
int threads=256;
int blocks=(total+threads-1)/threads;
embedding_lookup_kernel<<<blocks,threads>>>(d_ids,d_weight,d_out,seq_len,vocab_size,embed_dim);