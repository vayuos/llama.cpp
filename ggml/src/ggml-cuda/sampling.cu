#include "sampling.cuh"
#include "sampling.h"
#include "top-k.cuh"
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

// GPU-exclusive decode stream (nullptr when not in use)
cudaStream_t ggml_cuda_decode_stream = nullptr;

// ============================================================================
// State Management Kernels
// ============================================================================

// [STRICT] GPU-Resident State Update Kernel
// Increments position and n_past, and updates token history entirely on GPU.
__global__ void update_state_kernel(int32_t * pos, int32_t * n_past, int32_t * history, const int32_t * token, int32_t n_history, uint32_t * seed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // 1. Increment pos and n_past
        if (pos) (*pos)++;
        if (n_past) (*n_past)++;
        
        // 2. Shift history and push new token
        if (history && n_history > 0 && token) {
            for (int i = 0; i < n_history - 1; ++i) {
                history[i] = history[i+1];
            }
            history[n_history - 1] = *token;
        }

        // 3. Evolve seed
        if (seed) {
            *seed = *seed * 1103515245 + 12345;
        }
    }
}

void ggml_cuda_update_state(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0_pos = dst->src[0];
    const ggml_tensor * src1_past = dst->src[1];
    const ggml_tensor * src2_token = dst->src[2];
    const ggml_tensor * src3_history = dst->src[3];
    const ggml_tensor * src4_seed = dst->src[4];

    int32_t * d_pos = (int32_t *) src0_pos->data;
    int32_t * d_past = (int32_t *) src1_past->data;
    const int32_t * d_token = (const int32_t *) src2_token->data;
    int32_t * d_history = src3_history ? (int32_t *) src3_history->data : nullptr;
    int32_t n_history = src3_history ? (int32_t) src3_history->ne[0] : 0;
    uint32_t * d_seed = src4_seed ? (uint32_t *) src4_seed->data : nullptr;

    cudaStream_t stream = ctx.stream();
    update_state_kernel<<<1, 1, 0, stream>>>(d_pos, d_past, d_history, d_token, n_history, d_seed);
}

// ============================================================================
// Penalty Kernels
// ============================================================================

__global__ void apply_penalties_optimized_kernel(float *         logits,
                                                 const int32_t * last_tokens,
                                                 int32_t         n_vocab,
                                                 int32_t         n_last,
                                                 float           repeat_penalty,
                                                 float           alpha_frequency,
                                                 float           alpha_presence) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_last) {
        return;
    }

    int32_t token = last_tokens[tid];
    if (token < 0 || token >= n_vocab) {
        return;
    }

    // Only the first occurrence maps to an update
    for (int i = 0; i < tid; ++i) {
        if (last_tokens[i] == token) {
            return;
        }
    }

    int count = 0;
    for (int i = 0; i < n_last; ++i) {
        if (last_tokens[i] == token) {
            count++;
        }
    }

    float logit = logits[token];
    if (logit <= 0.0f) {
        logit *= repeat_penalty;
    } else {
        logit /= repeat_penalty;
    }

    logit -= (float) count * alpha_frequency + (float) (count > 0) * alpha_presence;
    logits[token] = logit;
}

void ggml_cuda_penalties(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    const int64_t n_vocab = src0->ne[0];
    const int64_t nrows   = ggml_nrows(src0);
    const int64_t n_last  = src1->ne[0];

    const float repeat_penalty  = ggml_get_op_params_f32(dst, 0);
    const float alpha_frequency = ggml_get_op_params_f32(dst, 1);
    const float alpha_presence  = ggml_get_op_params_f32(dst, 2);

    float *         logits_d      = (float *) src0->data;
    const int32_t * last_tokens_d = (const int32_t *) src1->data;

    cudaStream_t stream = ctx.stream();

    if (n_last > 0) {
        int block_size = 128;
        int num_blocks = (n_last + block_size - 1) / block_size;

        for (int64_t i = 0; i < nrows; ++i) {
            apply_penalties_optimized_kernel<<<num_blocks, block_size, 0, stream>>>(
                logits_d + i * n_vocab, last_tokens_d, n_vocab, n_last, repeat_penalty, alpha_frequency,
                alpha_presence);
        }
    }
}

// ============================================================================
// Fused Sampling Implementation
// ============================================================================

__global__ void sample_multinomial_simple(const float * probs, int * out_idx, int n, uint64_t seed_val, const uint32_t * d_seed) {
    extern __shared__ float s_cdf[];
    int tid = threadIdx.x;
    
    // 1. Cooperative load
    float val = (tid < n) ? probs[tid] : 0.0f;
    s_cdf[tid] = val;
    __syncthreads();
    
    // 2. Serial Prefix sum (thread 0) - reliable for MVP
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += s_cdf[i];
            s_cdf[i] = sum; // In-place prefix sum
        }
        
        // 3. Random
        // Simple PCG
        uint64_t s = d_seed ? (uint64_t)*d_seed : seed_val;
        uint64_t state = s + 0x853c49e6748fea9bULL;
        uint64_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        uint32_t res = (uint32_t)((word >> 22u) ^ word);
        float r = (float)res / (float)UINT32_MAX; 
        
        float target = r * sum;
        int idx = n - 1;
        
        // 4. Search
        for (int i = 0; i < n; ++i) {
            if (target <= s_cdf[i]) {
                idx = i;
                break;
            }
        }
        *out_idx = idx;
    }
}

__global__ void gather_result(int * dst, const int * local_idx, const int * candidates) {
    if (candidates) {
        *dst = candidates[*local_idx];
    } else {
        *dst = *local_idx;
    }
}

void ggml_cuda_sample_candidates(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0_logits = dst->src[0];
    
    int32_t params[16];
    memcpy(params, dst->op_params, sizeof(params));

    int32_t   k    = params[0];
    float     temp = 0.0f; memcpy(&temp, &params[1], sizeof(float));
    uint32_t  seed = (uint32_t)params[2];
    const ggml_tensor * src2_seed = dst->src[2];
    const uint32_t * d_seed = src2_seed ? (const uint32_t *) src2_seed->data : nullptr;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    int64_t n_vocab = src0_logits->ne[0];
    
    // 1. Alloc temp logits
    ggml_cuda_pool_alloc<float> temp_logits_alloc(pool, n_vocab * sizeof(float));
    float * d_logits = temp_logits_alloc.get();
    
    CUDA_CHECK(cudaMemcpyAsync(d_logits, src0_logits->data, n_vocab * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    
    // 2. Temp Scale
    if (temp > 0.0f && temp != 1.0f) {
        cuda_temperature_scale_kernel(d_logits, temp, 0, (int)n_vocab, stream);
    }
    
    float * d_probs = d_logits;
    int *   d_candidates = nullptr;
    int     n_probs = (int)n_vocab;
    
    // 3. Top K
    if (k > 0 && k < n_vocab) {
        ggml_cuda_pool_alloc<float> top_k_vals_alloc(pool, k * sizeof(float));
        ggml_cuda_pool_alloc<int>   top_k_inds_alloc(pool, k * sizeof(int));
        float * d_vals = top_k_vals_alloc.get();
        int *   d_inds = top_k_inds_alloc.get();
        
        ggml_cuda_top_k_values_indices(ctx, d_logits, d_vals, d_inds, (int)n_vocab, k);
        
        d_probs = d_vals;
        d_candidates = d_inds;
        n_probs = k;
    }
    
    // 4. Softmax
    ggml_cuda_pool_alloc<float> scratch_alloc(pool, 4096); 
    cuda_softmax_kernel(d_probs, d_probs, n_probs, scratch_alloc.get(), stream);
    
    // 5. Sample
    ggml_cuda_pool_alloc<int> out_idx_alloc(pool, sizeof(int));
    int * d_out_idx = out_idx_alloc.get();
    
    if (n_probs <= 1024) {
        sample_multinomial_simple<<<1, 256, n_probs * sizeof(float), stream>>>(d_probs, d_out_idx, n_probs, (uint64_t)seed, d_seed);
    } else {
        // Fallback for very large k (not recommended but for safety)
        cuda_argmax_kernel(d_probs, d_out_idx, n_probs, scratch_alloc.get(), stream);
    }
    
    // 6. Write result
    int * dst_ptr = (int *)dst->data;
    gather_result<<<1, 1, 0, stream>>>(dst_ptr, d_out_idx, d_candidates);
}
