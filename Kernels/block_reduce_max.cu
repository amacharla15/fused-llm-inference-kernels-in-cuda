__device__ float block_reduce_max(float val,float* shared){
    int warp_id=threadIdx.x/32;
    int lane=threadIdx.x%32;
    val=warp_reduce_max(val);
    if(lane==0){
        shared[warp_id]=val;
    }
    __syncthreads();
    int num_warps=(blockDim.x+31)/32;
    if(warp_id==0){
        if(lane<num_warps){
            val=shared[lane];
        }else{
            val=-INFINITY;
        }
        val=warp_reduce_max(val);
    }
    return val;
}
// No launch: called inside another kernel