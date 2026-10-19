void run_transformer_ffn(const float* x, const float* residual,
                         const float* norm_weight, const float* w_gate,
                         const float* w_up, const float* w_down, float* out,
                         int M, int hidden_dim, int intermediate_dim,
                         float eps) {
    int total=M*hidden_dim;
    int threads=256;
    int blocks=(total+threads-1)/threads;
    float *residual_out,*normalized,*mlp_out;
    
    cudaMalloc((void**)&residual_out,total*sizeof(float));
    cudaMalloc((void**)&normalized,total*sizeof(float));
    cudaMalloc((void**)&mlp_out,total*sizeof(float));

    rmsnorm_residual_block(x,residual,norm_weight,normalized,residual_out,M,hidden_dim,eps);
    mlp_swiglu_forward(normalized,w_gate,w_up,w_down,mlp_out,M,hidden_dim,intermediate_dim);
    add_residual_kernel<<<blocks,threads>>>(residual_out,mlp_out,out,total);


    cudaFree(residual_out);
    cudaFree(normalized);
    cudaFree(mlp_out);
}