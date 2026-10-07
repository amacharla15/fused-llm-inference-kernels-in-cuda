"""
Fused LLM Inference Kernels in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - warp_reduce_sum
__device__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

# Step 2 - warp_reduce_max
__device__ float warp_reduce_max(float val) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = max(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = max(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = max(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = max(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

# Step 3 - block_reduce_sum
__device__ float block_reduce_sum(float val, float* shared) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        if (lane < num_warps) {
            val = shared[lane];
        } else {
            val = 0.0f;
        }
        val = warp_reduce_sum(val);
    }
    return val;
}

# Step 4 - block_reduce_max
__device__ float block_reduce_max(float val, float* shared) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    val = warp_reduce_max(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        if (lane < num_warps) {
            val = shared[lane];
        } else {
            val = -INFINITY;
        }
        val = warp_reduce_max(val);
    }
    return val;
}

# Step 5 - add_residual_kernel
__global__ void add_residual_kernel(const float* x, const float* residual,
                                    float* out, int n) {
  // TODO: implement elementwise residual addition out[i] = x[i] + residual[i]
  long threadsperblock=blockDim.x;
  long blockspergrid=gridDim.x;
  int gid=blockIdx.x*threadsperblock+threadIdx.x;
  if(gid<n){
    out[gid]=x[gid]+residual[gid];
  }

  (void)x; (void)residual; (void)out; (void)n;
}

# Step 6 - gelu_kernel
__global__ void gelu_kernel(const float* x, float* out, int n) {
    // TODO: Apply GELU (tanh approximation) elementwise to x, write into out
    long threadsperblock=blockDim.x;
    long blockspergrid=gridDim.x;
    long gid = threadsperblock*blockIdx.x+threadIdx.x;
    if(gid<n){
        out[gid]=0.5*x[gid]*(1+tanhf((0.7978845608*(x[gid]+0.044715*x[gid]*x[gid]*x[gid]))));

    }
}

# Step 7 - silu_kernel
__global__ void silu_kernel(const float* x, float* out, int n) {
    // TODO: apply SiLU elementwise: out[i] = x[i] / (1 + exp(-x[i]))
    int threadsperblock=blockDim.x;
    int blockspergrid=gridDim.x;
    int gid= threadsperblock*blockIdx.x+threadIdx.x;
    if(gid<n){
        out[gid]=x[gid]/(1+expf(-1*x[gid]));
    }
    (void)x; (void)out; (void)n;
}

# Step 8 - swiglu_kernel
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    // TODO: out[i] = silu(gate[i]) * up[i] for all i in [0, n)
    int threadsperblock=blockDim.x;
    int blockspergrid=gridDim.x;
    int gid = threadsperblock*blockIdx.x+threadIdx.x;
    if(gid<n){
        out[gid]=(gate[gid]/(1+expf(-1*gate[gid])))*up[gid];
    }

    (void)gate; (void)up; (void)out; (void)n;
}

# Step 9 - rmsnorm_kernel
#include <cuda_runtime.h>

__global__ void rmsnorm_kernel(const float* x, const float* weight,
                               float* out, int n, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int row_start = row * n;

    int lane = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;

    __shared__ float s[32];

    float local_sum = 0.0f;

    for (int i = tid; i < n; i += blockDim.x) {
        float value = x[row_start + i];
        local_sum += value * value;
    }

    float val = local_sum;

    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (lane == 0) {
        s[warp_id] = val;
    }

    __syncthreads();

    if (warp_id == 0) {
        float block_sum = 0.0f;

        if (lane < num_warps) {
            block_sum = s[lane];
        }

        block_sum += __shfl_down_sync(0xffffffff, block_sum, 16);
        block_sum += __shfl_down_sync(0xffffffff, block_sum, 8);
        block_sum += __shfl_down_sync(0xffffffff, block_sum, 4);
        block_sum += __shfl_down_sync(0xffffffff, block_sum, 2);
        block_sum += __shfl_down_sync(0xffffffff, block_sum, 1);

        if (lane == 0) {
            s[0] = 1.0f / sqrtf(block_sum / n + eps);
        }
    }

    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        out[row_start + i] =
            x[row_start + i] * s[0] * weight[i];
    }
}

# Step 10 - layernorm_kernel (not yet solved)
# TODO: implement

# Step 11 - fused_add_rmsnorm_kernel (not yet solved)
# TODO: implement

# Step 12 - softmax_row_kernel (not yet solved)
# TODO: implement

# Step 13 - causal_softmax_kernel (not yet solved)
# TODO: implement

# Step 14 - embedding_lookup_kernel (not yet solved)
# TODO: implement

# Step 15 - rope_kernel (not yet solved)
# TODO: implement

# Step 16 - linear_kernel (not yet solved)
# TODO: implement

# Step 17 - fused_linear_bias_gelu_kernel (not yet solved)
# TODO: implement

# Step 18 - mlp_swiglu_forward (not yet solved)
# TODO: implement

# Step 19 - rmsnorm_residual_block (not yet solved)
# TODO: implement

# Step 20 - run_transformer_ffn (not yet solved)
# TODO: implement

