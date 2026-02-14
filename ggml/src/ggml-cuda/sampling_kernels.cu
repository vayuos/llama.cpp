/*
 * sampling_kernels.cu
 *
 * GPU implementations (and safe fallbacks) for sampling helper kernels:
 *  - cuda_argmax_kernel
 *  - cuda_apply_penalties_kernel
 *  - cuda_softmax_kernel
 *
 * These are conservative, correct implementations intended as working
 * replacements until highly-optimized device-only kernels are added.
 */

#include "sampling.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

extern "C" {

// ----------------------------------------------------------------------------
// Argmax Implementation
// ----------------------------------------------------------------------------

// Stage 1: Block-level reduction
// Writes to scratch: [0..gridDim.x-1] = max_val, [gridDim.x..2*gridDim.x-1] = max_idx
__global__ void argmax_block_reduce(const float * logits,
                                    float *       scratch_val,
                                    int32_t *     scratch_idx,
                                    int32_t       vocab_size) {
    extern __shared__ char s_mem[];
    float *                s_val = (float *) s_mem;
    int32_t *              s_idx = (int32_t *) &s_val[blockDim.x];

    int tid       = threadIdx.x;
    int gid       = blockIdx.x * blockDim.x + tid;
    int grid_size = blockDim.x * gridDim.x;

    float   local_max = -INFINITY;
    int32_t local_idx = -1;

    // Grid stride loop
    for (int i = gid; i < vocab_size; i += grid_size) {
        float v = logits[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Block reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float v1 = s_val[tid];
            float v2 = s_val[tid + offset];
            if (v2 > v1) {
                s_val[tid] = v2;
                s_idx[tid] = s_idx[tid + offset];
            }
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        scratch_val[blockIdx.x] = s_val[0];
        scratch_idx[blockIdx.x] = s_idx[0];
    }
}

// Stage 2: Final reduction of partial results (single block)
__global__ void argmax_final_reduce(const float *   scratch_val,
                                    const int32_t * scratch_idx,
                                    int32_t *       out_token,
                                    int32_t         n_blocks) {
    extern __shared__ char s_mem[];
    float *                s_val = (float *) s_mem;
    int32_t *              s_idx = (int32_t *) &s_val[blockDim.x];

    int tid = threadIdx.x;

    float   local_max = -INFINITY;
    int32_t local_idx = -1;

    for (int i = tid; i < n_blocks; i += blockDim.x) {
        float v = scratch_val[i];
        if (v > local_max) {
            local_max = v;
            local_idx = scratch_idx[i];
        }
    }

    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Block reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float v1 = s_val[tid];
            float v2 = s_val[tid + offset];
            if (v2 > v1) {
                s_val[tid] = v2;
                s_idx[tid] = s_idx[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_token = s_idx[0];
    }
}

int cuda_argmax_kernel(const float * d_logits,
                       int32_t *     d_out_token,
                       int32_t       vocab_size,
                       float *       d_scratch,
                       void *        cuda_stream) {
    if (!d_logits || !d_out_token || vocab_size <= 0) {
        return -1;
    }

    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) {
        stream = 0;
    }

    const int threads    = 256;
    // Cap blocks to ensure we fit in scratch (4096 bytes default -> 1024 floats)
    // We need 2 * blocks items (val + idx)
    // d_scratch is assumed to be at least 4096 bytes (1024 floats)
    // So safe max blocks is 512.
    // For 128k vocab, 128 blocks covers it with 1024 items/thread if needed, but 128*256 = 32k threads.
    // Each thread does ~4 items. This is fine.
    const int max_blocks = 256;
    int       blocks     = std::min(max_blocks, (vocab_size + threads - 1) / threads);
    size_t    shared_mem = threads * (sizeof(float) + sizeof(int32_t));

    // If scratch is missing, we must fallback or fail.
    // However, if vocab is very small (1 block), we can skip scratch usage for Stage 1?
    // The previous implementation did single block direct write.
    // Let's use scratch if available, otherwise fail.
    if (!d_scratch && blocks > 1) {
        return -1;
    }

    // For small vocab, single stage
    if (blocks == 1) {
        // We can reuse the block reduce kernel but we need to write to d_out_token directly.
        // Or just map d_scratch to d_out_token? No, type mismatch.
        // Let's keep a dedicated single-pass kernel?
        // Actually, Step 1 writes to scratch. Step 2 reads scratch.
        // If blocks=1, Step 1 writes to scratch[0]. Step 2 reads scratch[0] -> out.
        // This works fine. overhead is 1 extra kernel launch for 1 block. Negligible.
    }

    // Launch Stage 1
    // scratch buffer layout: [0..blocks-1] floats, [blocks..2*blocks-1] ints
    float *   d_s_val = d_scratch;
    int32_t * d_s_idx = (int32_t *) &d_s_val[blocks];  // align? int32 and float are both 4 bytes. OK.

    argmax_block_reduce<<<blocks, threads, shared_mem, stream>>>(d_logits, d_s_val, d_s_idx, vocab_size);

    // Launch Stage 2
    // Reduces 'blocks' elements to 1
    argmax_final_reduce<<<1, threads, shared_mem, stream>>>(d_s_val, d_s_idx, d_out_token, blocks);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Apply penalties: logits[i] -= alpha * penalties[i]
__global__ static void apply_penalties_simple_kernel(float *       logits,
                                                     const float * penalties,
                                                     float         alpha,
                                                     int32_t       vocab_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < vocab_size; i += blockDim.x * gridDim.x) {
        logits[i] -= alpha * penalties[i];
    }
}

int cuda_apply_penalties_kernel(float *       d_logits,
                                const float * d_penalties,
                                float         alpha,
                                int32_t       vocab_size,
                                void *        cuda_stream) {
    if (!d_logits || !d_penalties || vocab_size <= 0) {
        return -1;
    }
    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) {
        stream = 0;
    }

    int threads = 256;
    int blocks  = (vocab_size + threads - 1) / threads;
    blocks      = std::min(blocks, 1024);

    apply_penalties_simple_kernel<<<blocks, threads, 0, stream>>>(d_logits, d_penalties, alpha, vocab_size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ----------------------------------------------------------------------------
// Softmax Implementation
// ----------------------------------------------------------------------------

// 1. Find Max (for numerical stability)
__global__ void reduce_max_block_kernel(const float * x, float * scratch, int n) {
    extern __shared__ float sdata[];
    int                     tid       = threadIdx.x;
    int                     gid       = blockIdx.x * blockDim.x + tid;
    int                     grid_size = blockDim.x * gridDim.x;

    float local_max = -INFINITY;
    for (int i = gid; i < n; i += grid_size) {
        float v = x[i];
        if (v > local_max) {
            local_max = v;
        }
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (sdata[tid + offset] > sdata[tid]) {
                sdata[tid] = sdata[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        scratch[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_max_final_kernel(const float * scratch_in, float * out_max, int n_blocks) {
    extern __shared__ float sdata[];
    int                     tid       = threadIdx.x;
    float                   local_max = -INFINITY;

    for (int i = tid; i < n_blocks; i += blockDim.x) {
        float v = scratch_in[i];
        if (v > local_max) {
            local_max = v;
        }
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (sdata[tid + offset] > sdata[tid]) {
                sdata[tid] = sdata[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_max = sdata[0];
    }
}

// 2. Compute Sum of Exps
__global__ void reduce_sum_exp_block_kernel(const float * x, float * scratch, const float * max_val_ptr, int n) {
    extern __shared__ float sdata[];
    int                     tid       = threadIdx.x;
    int                     gid       = blockIdx.x * blockDim.x + tid;
    int                     grid_size = blockDim.x * gridDim.x;
    float                   max_val   = *max_val_ptr;

    float local_sum = 0.0f;
    for (int i = gid; i < n; i += grid_size) {
        local_sum += expf(x[i] - max_val);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        scratch[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sum_final_kernel(const float * scratch_in, float * out_sum, int n_blocks) {
    extern __shared__ float sdata[];
    int                     tid       = threadIdx.x;
    float                   local_sum = 0.0f;

    for (int i = tid; i < n_blocks; i += blockDim.x) {
        local_sum += scratch_in[i];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_sum = sdata[0];
    }
}

// 3. Scale and Output
__global__ void softmax_scale_kernel(const float * logits,
                                     float *       probs,
                                     const float * max_val_ptr,
                                     const float * sum_ptr,
                                     int           n) {
    int   gid       = blockIdx.x * blockDim.x + threadIdx.x;
    int   grid_size = blockDim.x * gridDim.x;
    float max_val   = *max_val_ptr;
    float sum       = *sum_ptr;

    // Safety check for NaN/Inf sum
    if (sum == 0.0f) {
        sum = 1.0f;
    }
    float scale = 1.0f / sum;

    for (int i = gid; i < n; i += grid_size) {
        probs[i] = expf(logits[i] - max_val) * scale;
    }
}

int cuda_softmax_kernel(const float * d_logits,
                        float *       d_probs,
                        int32_t       vocab_size,
                        float *       d_scratch,
                        void *        cuda_stream) {
    if (!d_logits || !d_probs || vocab_size <= 0) {
        return -1;
    }
    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) {
        stream = 0;
    }

    const int threads    = 256;
    const int max_blocks = 256;
    int       blocks     = std::min(max_blocks, (vocab_size + threads - 1) / threads);
    size_t    shared_mem = threads * sizeof(float);

    // Scratch layout:
    // [0..blocks-1]: partial max / partial sum
    // [blocks]: global max
    // [blocks+1]: global sum

    // We need d_scratch to hold at least blocks + 2 floats.
    // Default scratch is 4096 bytes (1024 floats). > 256+2. Safe.
    if (!d_scratch) {
        return -1;  // Mandatory for this implementation
    }

    float * d_partial    = d_scratch;
    float * d_global_max = d_scratch + blocks;
    float * d_global_sum = d_scratch + blocks + 1;

    // 1. Find Max
    reduce_max_block_kernel<<<blocks, threads, shared_mem, stream>>>(d_logits, d_partial, vocab_size);
    reduce_max_final_kernel<<<1, threads, shared_mem, stream>>>(d_partial, d_global_max, blocks);

    // 2. Compute Sum of Exps
    reduce_sum_exp_block_kernel<<<blocks, threads, shared_mem, stream>>>(d_logits, d_partial, d_global_max, vocab_size);
    reduce_sum_final_kernel<<<1, threads, shared_mem, stream>>>(d_partial, d_global_sum, blocks);

    // 3. Scale
    softmax_scale_kernel<<<blocks, threads, 0, stream>>>(d_logits, d_probs, d_global_max, d_global_sum, vocab_size);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Temperature scaling kernel: logits[i] /= temperature
__global__ static void temperature_scale_kernel(float * logits, float temperature, int32_t vocab_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < vocab_size; i += blockDim.x * gridDim.x) {
        if (temperature == 0.0f) {
            continue;
        }
        logits[i] = logits[i] / temperature;
    }
}

int cuda_temperature_scale_kernel(float * d_logits,
                                  float   temperature,
                                  int32_t top_k,
                                  int32_t vocab_size,
                                  void *  cuda_stream) {
    (void) top_k;  // top_k filtering is now handled in cuda_sampling_sample_specialized via cuda_topk_kernel
    if (!d_logits || vocab_size <= 0) {
        return -1;
    }
    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) {
        stream = 0;
    }

    // apply simple in-place temperature scaling
    int threads = 256;
    int blocks  = (vocab_size + threads - 1) / threads;
    blocks      = std::max(1, std::min(blocks, 1024));
    temperature_scale_kernel<<<blocks, threads, 0, stream>>>(d_logits, temperature, vocab_size);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

}  // extern "C"
