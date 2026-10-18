__device__ float warp_reduce_sum(float val){
    val+=__shfl_xor_sync(0xffffffff,val,16);
    val+=__shfl_xor_sync(0xffffffff,val,8);
    val+=__shfl_xor_sync(0xffffffff,val,4);
    val+=__shfl_xor_sync(0xffffffff,val,2);
    val+=__shfl_xor_sync(0xffffffff,val,1);
    return val;
}
