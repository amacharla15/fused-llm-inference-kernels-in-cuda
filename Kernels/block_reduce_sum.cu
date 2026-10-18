__device__ float block_reduce_sum(float val,float* shared){
    int warp_id=threadIdx.x/32;
    int lane=threadIdx.x%32;
    val=warp_reduce_sum(val);
    if(lane==0){
        shared[warp_id]=val;
    }
    __syncthreads();
    int num_warps=(blockDim.x+31)/32;
    if(warp_id==0){
        if(lane<num_warps){
            val=shared[lane];
        }else{
            val=0.0f;
        }
        val=warp_reduce_sum(val);
    }
    return val;
}
// No launch: called inside another kernel