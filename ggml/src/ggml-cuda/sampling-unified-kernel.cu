/*
 * sampling-unified-kernel.cu
 *
 * Unified GPU token selection kernel - THE SOLE AUTHORITY for next-token decision
 *
 * Enforces GPU-only token selection during decode phase:
 *  - No CPU decision-making
 *  - No CPU tie-breaking
 *  - No CPU randomness
 *  - Deterministic and seeded if configured
 *
 * Output: Single token ID (4 bytes across PCIe)
 */

#include "sampling.h"
#include "common.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" {

// ============================================================================
// Argmax Selection (Greedy Decoding)
// ============================================================================

__global__ void argmax_selection_kernel(const float *d_logits,
                                       int32_t *    d_selected_token,
                                       int32_t      vocab_size) {
    // Parallel reduction to find max logit and its index
    extern __shared__ char s_mem[];
    float *s_max = (float *) s_mem;
    int32_t *s_idx = (int32_t *) &s_mem[blockDim.x * sizeof(float)];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Load max into shared memory
    float local_max = -INFINITY;
    int32_t local_idx = 0;

    for (int i = tid; i < vocab_size; i += stride) {
        if (d_logits[i] > local_max) {
            local_max = d_logits[i];
            local_idx = i;
        }
    }

    s_max[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Reduce within block
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (s_max[tid + offset] > s_max[tid]) {
                s_max[tid] = s_max[tid + offset];
                s_idx[tid] = s_idx[tid + offset];
            } else if (s_max[tid + offset] == s_max[tid] && s_idx[tid + offset] < s_idx[tid]) {
                // Tie-breaking: prefer lower index
                s_idx[tid] = s_idx[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_selected_token = s_idx[0];
    }
}

int cuda_select_token_argmax(const float *d_logits,
                             int32_t *    d_selected_token,
                             int32_t      vocab_size,
                             void *       cuda_stream) {
    if (!d_logits || !d_selected_token || vocab_size <= 0) {
        return -1;
    }

    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) {
        stream = 0;
    }

    int threads = 256;
    size_t shared_mem = threads * (sizeof(float) + sizeof(int32_t));

    argmax_selection_kernel<<<1, threads, shared_mem, stream>>>(d_logits, d_selected_token, vocab_size);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ============================================================================
// Categorical Sampling (Stochastic Decoding with Deterministic Seed)
// ============================================================================

__global__ void categorical_selection_kernel(const float *d_cumsum_probs,
                                             const int32_t *d_sorted_indices,
                                             int32_t *      d_selected_token,
                                             int32_t        vocab_size,
                                             uint64_t       seed) {
    // Single thread determines selected token using pre-computed cumsum
    // d_cumsum_probs: cumulative sum probabilities (sorted by probability)
    // d_sorted_indices: mapping to original vocabulary indices
    // seed: deterministic RNG seed

    if (threadIdx.x == 0) {
        // Deterministic random value from seed [0.0, 1.0)
        curandState state;
        curand_init(seed, 0, 0, &state);
        float rnd = curand_uniform(&state);

        // Binary search to find cutoff (where cumsum >= rnd)
        int32_t left = 0, right = vocab_size;
        while (left < right) {
            int32_t mid = (left + right) / 2;
            if (d_cumsum_probs[mid] < rnd) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        int32_t selected_idx = left < vocab_size ? left : vocab_size - 1;
        *d_selected_token = d_sorted_indices[selected_idx];
    }
}

int cuda_select_token_categorical(const float *    d_cumsum_probs,
                                  const int32_t *  d_sorted_indices,
                                  int32_t *        d_selected_token,
                                  int32_t          vocab_size,
                                  uint64_t         seed,
                                  void *           cuda_stream) {
    if (!d_cumsum_probs || !d_sorted_indices || !d_selected_token || vocab_size <= 0) {
        return -1;
    }

    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) {
        stream = 0;
    }

    categorical_selection_kernel<<<1, 1, 0, stream>>>(
        d_cumsum_probs, d_sorted_indices, d_selected_token, vocab_size, seed);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ============================================================================
// Unified Token Selection Kernel
// ============================================================================

/**
 * GPU-ONLY Token Selection - THE SOLE AUTHORITY FOR NEXT-TOKEN DECISION
 *
 * This kernel is the exclusive decision-maker during decode phase.
 * CPU is forbidden from influencing token selection.
 *
 * Execution flow:
 *  1. (Upstream) Temperature scaling applied (GPU)
 *  2. (Upstream) Penalties applied (GPU)
 *  3. (Upstream) Top-k/top-p filtering applied (GPU)
 *  4. (HERE) Final selection (argmax or categorical) (GPU)
 *  5. (Downstream) Token ID transferred to CPU (4 bytes only)
 *
 * @param d_logits              Input logits (after penalties/temp/filtering)
 * @param d_selected_token      Output: selected token ID [1]
 * @param vocab_size            Vocabulary size
 * @param selection_mode        0=argmax (greedy), 1=categorical (stochastic)
 * @param d_cumsum_probs        Cumulative probabilities (for mode=1, NULL for mode=0)
 * @param d_sorted_indices      Sorted indices (for mode=1, NULL for mode=0)
 * @param seed                  RNG seed (for mode=1, ignored for mode=0)
 * @param cuda_stream           CUDA stream for async execution
 *
 * @return                      0 on success, negative on error
 *
 * **INVARIANT:** This is the ONLY legitimate CPU→GPU token selection call.
 * All other CPU sampling code is disabled during decode.
 */
int cuda_unified_select_token(const float *    d_logits,
                              int32_t *        d_selected_token,
                              int32_t          vocab_size,
                              int32_t          selection_mode,
                              const float *    d_cumsum_probs,
                              const int32_t *  d_sorted_indices,
                              uint64_t         seed,
                              void *           cuda_stream) {
    if (!d_logits || !d_selected_token || vocab_size <= 0) {
        return -1;
    }

    if (selection_mode == 0) {
        // Argmax (deterministic greedy)
        return cuda_select_token_argmax(d_logits, d_selected_token, vocab_size, cuda_stream);
    } else if (selection_mode == 1) {
        // Categorical (deterministic seeded sampling)
        if (!d_cumsum_probs || !d_sorted_indices) {
            return -1;
        }
        return cuda_select_token_categorical(d_cumsum_probs, d_sorted_indices, d_selected_token,
                                            vocab_size, seed, cuda_stream);
    } else {
        return -1;  // Invalid mode
    }
}

}  // extern "C"
