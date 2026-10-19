# LLM Inference Kernels in CUDA
 
CUDA C++ implementations of the operators in a Llama-style transformer feed-forward path, written from scratch: warp and block reductions, elementwise activations, RMSNorm and LayerNorm, row and causal softmax, embedding lookup, rotary position embeddings, linear layers, a SwiGLU MLP, and a complete pre-norm FFN block with residual connections.
 
## Kernels
 
| # | Kernel | Computes | Parallelization |
|---|---|---|---|
| 1 | `warp_reduce_sum` | Sum across a warp | Butterfly reduction with `__shfl_xor_sync`; every lane ends with the result |
| 2 | `warp_reduce_max` | Max across a warp | Same butterfly pattern |
| 3 | `block_reduce_sum` | Sum across a block | Warp reduction, one partial per warp in shared memory, then warp 0 reduces the partials |
| 4 | `block_reduce_max` | Max across a block | Same two-level scheme |
| 5 | `add_residual_kernel` | `out = x + residual` | One thread per element |
| 6 | `gelu_kernel` | GELU (tanh approximation) | One thread per element |
| 7 | `silu_kernel` | `x · sigmoid(x)` | One thread per element |
| 8 | `swiglu_kernel` | `silu(gate) · up` | One thread per element; activation and gating in a single pass |
| 9 | `rmsnorm_kernel` | `x / sqrt(mean(x²) + eps) · w` | One block per row; strided partial sums, warp shuffles, per-warp partials in shared memory |
| 10 | `layernorm_kernel` | `(x − mean) / sqrt(var + eps) · w + b` | One block per row; two block reductions (mean, then variance) |
| 11 | `fused_add_rmsnorm_kernel` | `h = x + residual`, `out = RMSNorm(h)` | One block per row; a single launch writes both `h` and `out` |
| 12 | `softmax_row_kernel` | Row softmax | One block per row; max reduction, `exp(x − max)`, sum reduction, normalize |
| 13 | `causal_softmax_kernel` | Masked softmax for attention scores | As above; row `r` keeps columns `0..r`, later columns are zero |
| 14 | `embedding_lookup_kernel` | `out[t] = W[token_ids[t]]` | One thread per output element |
| 15 | `rope_kernel` | Rotary position embedding on q and k, in place | One thread per (position, head, pair); interleaved pairs, precomputed cos/sin tables |
| 16 | `linear_kernel` | `x · Wᵀ + b` | One thread per output element |
| 17 | `fused_linear_bias_gelu_kernel` | `GELU(x · Wᵀ + b)` | Linear, bias and activation in one kernel; the pre-activation is never written to memory |
| 18 | `mlp_swiglu_forward` (host) | `down(silu(x · W_gateᵀ) · (x · W_upᵀ))` | Gate and up projections, SwiGLU, down projection |
| 19 | `rmsnorm_residual_block` (host) | Residual add + RMSNorm over all rows | Launches kernel 11 |
| 20 | `run_transformer_ffn` (host) | `h = x + residual`, `out = h + MLP(RMSNorm(h))` | Kernel 11, then the SwiGLU MLP, then the residual add |