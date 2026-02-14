/*
 * sampling-topp-kernel.cu
 *
 * GPU-native nucleus (top-p) filtering kernel for efficient token sampling
 *
 * Implements complete top-p pipeline on GPU:
 *  1. Softmax probabilities from logits
 *  2. Bitonic sort to order probabilities descending
 *  3. Parallel prefix sum (cumulative probabilities)
 *  4. Threshold detection (find cutoff where cumsum >= p)
 *  5. Mask and selection
 *
 * All operations GPU-resident; no PCIe transfer of probability arrays.
 */

#include "sampling.h"
#include "common.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

extern "C" {

// ============================================================================
// Bitonic Sort (for sorting probabilities)
// ============================================================================

__device__ static void bitonic_compare_swap(float &a, int32_t &a_idx, float &b, int32_t &b_idx, bool dir) {
    if ((a > b) != dir) {
        float tmp_a = a;
        int32_t tmp_idx = a_idx;
        a = b;
        a_idx = b_idx;
        b = tmp_a;
        b_idx = tmp_idx;
    }
}

__global__ void bitonic_sort_step(float *values, int32_t *indices, int32_t n, int32_t stage, int32_t step) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t cmp_idx = idx ^ step;
    if (cmp_idx <= idx) return;  // Only compare once per pair

    bool dir = ((idx >> stage) & 1) == 0;
    bitonic_compare_swap(values[idx], indices[idx], values[cmp_idx], indices[cmp_idx], dir);
}

// Bitonic sort for probability array (descending order)
int cuda_bitonic_sort_desc(float *d_values,
                           int32_t *d_indices,
                           int32_t n,
                           void *cuda_stream) {
    if (!d_values || !d_indices || n <= 0) {
        return -1;
    }

    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) {
        stream = 0;
    }

    // Ensure n is power of 2
    int32_t padded_n = 1;
    while (padded_n < n) padded_n *= 2;

    int threads = 256;
    for (int32_t stage = 0; stage < 32 && (1 << stage) < padded_n; stage++) {
        for (int32_t step = stage; step >= 0; step--) {
            int blocks = (padded_n + threads - 1) / threads;
            bitonic_sort_step<<<blocks, threads, 0, stream>>>(d_values, d_indices, n, stage, 1 << step);
        }
    }

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ============================================================================
// Parallel Prefix Sum (Blelloch scan algorithm)
// ============================================================================

__global__ void prefix_sum_up_sweep(float *data, int32_t n, int32_t stride) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx >= stride) {
        data[idx] += data[idx - stride];
    }
}

int cuda_prefix_sum(float *d_data, int32_t n, void *cuda_stream) {
    if (!d_data || n <= 0) {
        return -1;
    }

    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) {
        stream = 0;
    }

    int threads = 256;

    // Blelloch scan: up-sweep phase
    for (int32_t stride = 1; stride < n; stride *= 2) {
        int blocks = (n + threads - 1) / threads;
        prefix_sum_up_sweep<<<blocks, threads, 0, stream>>>(d_data, n, stride);
    }

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ============================================================================
// Top-P Threshold Detection & Masking
// ============================================================================

__global__ void find_topp_cutoff_kernel(const float *probs,  /* sorted, cumsum computed */
                                        int32_t *            mask,  /* output: 0=keep, 1=mask */
                                        int32_t *            n_keep,  /* output: count kept */
                                        float                p,
                                        int32_t              vocab_size) {
    int32_t idx = threadIdx.x;
    int32_t stride = blockDim.x;

    // Find cutoff index where probs[i] >= p (exclusive so probs[i-1] < p, probs[i] >= p]
    int32_t cutoff = vocab_size;
    for (int32_t i = idx; i < vocab_size; i += stride) {
        if (probs[i] >= p) {
            cutoff = i + 1;  // Include this element and all before it
            break;
        }
    }

    // Reduce to find minimum cutoff across threads
    __shared__ int32_t s_cutoff;
    if (idx == 0) s_cutoff = vocab_size;
    __syncthreads();

    // Atomic min to find earliest cutoff
    atomicMin(&s_cutoff, cutoff);
    __syncthreads();

    cutoff = s_cutoff;

    // Create mask: keep [0, cutoff), mask [cutoff, vocab_size)
    for (int32_t i = idx; i < vocab_size; i += stride) {
        mask[i] = (i < cutoff) ? 0 : 1;
    }

    if (idx == 0) {
        *n_keep = cutoff;
    }
}

// ============================================================================
// Fused Softmax + Prefix Sum Kernel
// ============================================================================

__global__ void softmax_and_cumsum_kernel(const float *logits,   /* [vocab_size] */
                                          float *       probs,   /* output: cumsum */
                                          int32_t       vocab_size,
                                          float         temperature) {
    // Each block processes entire vocabulary
    extern __shared__ float s_mem[];
    float *s_probs = s_mem;
    float *s_temp = &s_mem[vocab_size];

    int tid = threadIdx.x;
    int threads = blockDim.x;

    // ---- Stage 1: Find max logit for stability ----
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += threads) {
        float val = logits[i];
        if (temperature != 1.0f) {
            val = val / temperature;
        }
        s_temp[i] = val;
        local_max = fmaxf(local_max, val);
    }
    __syncthreads();

    // Reduce to find global max
    __shared__ float s_max;
    for (int offset = threads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
        }
    }
    if (tid == 0) s_max = local_max;
    __syncthreads();

    float max_logit = s_max;

    // ---- Stage 2: Compute softmax ----
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += threads) {
        float val = expf(s_temp[i] - max_logit);
        s_probs[i] = val;
        local_sum += val;
    }
    __syncthreads();

    // Reduce to find total sum
    __shared__ float s_sum;
    for (int offset = threads / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (tid == 0) s_sum = local_sum;
    __syncthreads();

    float total_sum = s_sum;

    // ---- Stage 3: Normalize and compute cumsum ----
    for (int i = tid; i < vocab_size; i += threads) {
        s_probs[i] = s_probs[i] / total_sum;
    }
    __syncthreads();

    // Serial prefix sum (thread 0 does it - for correctness)
    if (tid == 0) {
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += s_probs[i];
            s_probs[i] = cumsum;
        }
    }
    __syncthreads();

    // ---- Stage 4: Copy to global memory ----
    for (int i = tid; i < vocab_size; i += threads) {
        probs[i] = s_probs[i];
    }
}

// ============================================================================
// GPU Top-P Entry Point
// ============================================================================

int cuda_topp_kernel(const float * d_logits,      /* [vocab_size] logits */
                     float *       d_probs,       /* output: sorted cumsum probs */
                     int32_t *     d_sorted_inds, /* output: indices after sort */
                     int32_t *     d_mask,        /* output: 0=keep, 1=mask */
                     int32_t *     d_n_keep,      /* output: count of kept tokens */
                     float         p,             /* nucleus threshold (0.0-1.0) */
                     float         temperature,   /* temperature for softmax */
                     int32_t       vocab_size,
                     void *        cuda_stream) {
    if (!d_logits || !d_probs || !d_sorted_inds || !d_mask || !d_n_keep ||
        vocab_size <= 0 || p <= 0.0f || p > 1.0f) {
        return -1;
    }

    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) {
        stream = 0;
    }

    // ---- Stage 1: Fused softmax + prefix sum ----
    int shared_mem = vocab_size * (sizeof(float) + sizeof(float));
    if (shared_mem > 48000) {  // Shared memory limit
        return -1;  // Vocab too large for single block
    }

    softmax_and_cumsum_kernel<<<1, 256, shared_mem, stream>>>(
        d_logits, d_probs, vocab_size, temperature);

    if (cudaGetLastError() != cudaSuccess) {
        return -1;
    }

    // ---- Stage 2: Initialize indices array (0..vocab_size-1) ----
    // (This would typically be done once per context, not per-token)
    // For now, assume d_sorted_inds is pre-initialized

    // ---- Stage 3: Bitonic sort (sort probabilities and track indices) ----
    if (cuda_bitonic_sort_desc(d_probs, d_sorted_inds, vocab_size, (void *) stream) != 0) {
        return -1;
    }

    // ---- Stage 4: Find cutoff and create mask ----
    int threads = 256;
    find_topp_cutoff_kernel<<<1, threads, sizeof(int32_t), stream>>>(
        d_probs, d_mask, d_n_keep, p, vocab_size);

    if (cudaGetLastError() != cudaSuccess) {
        return -1;
    }

    return 0;
}

}  // extern "C"
