__device__ float warp_reduce_max(float val){
    val=max(val,__shfl_xor_sync(0xffffffff,val,16));
    val=max(val,__shfl_xor_sync(0xffffffff,val,8));
    val=max(val,__shfl_xor_sync(0xffffffff,val,4));
    val=max(val,__shfl_xor_sync(0xffffffff,val,2));
    val=max(val,__shfl_xor_sync(0xffffffff,val,1));
    return val;
}
// No launch: called inside another kernel