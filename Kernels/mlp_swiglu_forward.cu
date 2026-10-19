void mlp_swiglu_forward(
    const float* x,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    float* out,
    int M,
    int hidden_dim,
    int intermediate_dim
) {
    float *gate, *up, *activated;
    int total=M*intermediate_dim;
    int threads=256;
    int blocks=(total+threads-1)/threads;

    cudaMalloc((void**)&gate,total*sizeof(float));
    cudaMalloc((void**)&up,total*sizeof(float));
    cudaMalloc((void**)&activated,total*sizeof(float));

    linear_kernel<<<blocks,threads>>>(x,w_gate,nullptr,gate,M,intermediate_dim,hidden_dim);
    linear_kernel<<<blocks,threads>>>(x,w_up,nullptr,up,M,intermediate_dim,hidden_dim);
    swiglu_kernel<<<blocks,threads>>>(gate,up,activated,total);

    int down_total=M*hidden_dim;
    int down_blocks=(down_total+threads-1)/threads;
    linear_kernel<<<down_blocks,threads>>>(activated,w_down,nullptr,out,M,hidden_dim,intermediate_dim);
    
    cudaFree(gate);
    cudaFree(up);
    cudaFree(activated);
}