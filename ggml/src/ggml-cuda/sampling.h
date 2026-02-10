/*
 * ggml-cuda/sampling.h - CUDA kernels for efficient token sampling on GPU
 *
 * This header defines the interface for GPU-accelerated sampling operations:
 *  - cuda_argmax_kernel: Find token with highest logits
 *  - cuda_apply_penalties_kernel: Apply repetition/frequency penalties
 *  - cuda_softmax_kernel: Compute probability distribution
 *
 * These kernels eliminate the need to copy logits from GPU to CPU for sampling,
 * reducing latency by 5-10ms per token.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Data Structures
// ============================================================================

/**
 * GPU-side sampling context
 *
 * Holds allocated GPU buffers and configuration for sampling operations
 */
typedef struct {
    // Allocated device memory
    float * d_logits;           // [vocab_size] - token logits
    float * d_penalties;        // [vocab_size] - frequency/repetition penalties
    float * d_probs;            // [vocab_size] - softmax probabilities
    float * d_scratch;          // scratch space for reductions
    int32_t * d_sampled_token;  // [1] - output token ID
    
    // Configuration
    int32_t vocab_size;
    int32_t cuda_device;
    void * cuda_stream;         // optional CUDA stream (opaque)
    
} cuda_sampling_context_t;

// ============================================================================
// Kernel Functions (CUDA implementations)
// ============================================================================

/**
 * Find token with maximum logit value (greedy sampling)
 *
 * @param d_logits      [vocab_size] float array of logits
 * @param d_out_token   [1] output token ID (device pointer)
 * @param vocab_size    Size of vocabulary
 * @param cuda_stream   CUDA stream for async execution (optional)
 *
 * @return              CUDA error code
 *
 * Performance: ~100-200 us depending on vocab size
 * Uses parallel reduction (register-level tiling)
 */
int cuda_argmax_kernel(
    float * d_logits,
    int32_t * d_out_token,
    int32_t vocab_size,
    void * cuda_stream
);

/**
 * Apply temperature scaling and top-k filtering to logits
 *
 * @param d_logits      [vocab_size] float array of logits
 * @param temperature   Temperature factor (1.0 = no scaling)
 * @param top_k         Keep only top-k tokens (0 = no filtering)
 * @param cuda_stream   CUDA stream for async execution
 *
 * @return              CUDA error code
 *
 * Performance: ~200-500 us
 * In-place modification of logits
 */
int cuda_temperature_scale_kernel(
    float * d_logits,
    float temperature,
    int32_t top_k,
    int32_t vocab_size,
    void * cuda_stream
);

/**
 * Apply frequency penalties to logits based on token history
 *
 * Penalties are applied as:
 *   logits[i] -= alpha * count[i]  (where count[i] = frequency of token i)
 *
 * @param d_logits          [vocab_size] logit array
 * @param d_penalties       [vocab_size] pre-computed penalty offsets
 * @param alpha             Penalty strength coefficient
 * @param vocab_size        Size of vocabulary
 * @param cuda_stream       CUDA stream for async execution
 *
 * @return                  CUDA error code
 *
 * Performance: ~100-300 us
 * In-place modification of logits
 */
int cuda_apply_penalties_kernel(
    float * d_logits,
    const float * d_penalties,
    float alpha,
    int32_t vocab_size,
    void * cuda_stream
);

/**
 * Compute softmax probabilities from logits
 *
 * softmax[i] = exp(logits[i]) / sum(exp(logits[j]) for all j)
 *
 * Uses numerically stable algorithm (subtract max before exp)
 *
 * @param d_logits      [vocab_size] float array of logits
 * @param d_probs       [vocab_size] output probability array
 * @param vocab_size    Size of vocabulary
 * @param cuda_stream   CUDA stream for async execution
 *
 * @return              CUDA error code
 *
 * Performance: ~500-1000 us
 * Includes two-pass reduction for numerical stability
 */
int cuda_softmax_kernel(
    const float * d_logits,
    float * d_probs,
    int32_t vocab_size,
    void * cuda_stream
);

/**
 * Sample from categorical distribution using precomputed cumulative probabilities
 *
 * Uses inverse transform sampling with GPU random number generation
 *
 * @param d_probs       [vocab_size] normalized probability distribution
 * @param d_out_token   [1] output token ID
 * @param vocab_size    Size of vocabulary
 * @param seed          Random seed for generator
 * @param cuda_stream   CUDA stream for async execution
 *
 * @return              CUDA error code
 *
 * Performance: ~300-500 us
 */
int cuda_sample_categorical_kernel(
    const float * d_probs,
    int32_t * d_out_token,
    int32_t vocab_size,
    uint64_t seed,
    void * cuda_stream
);

// ============================================================================
// Host-Side Interface Functions
// ============================================================================

/**
 * Allocate GPU sampling context
 *
 * @param ctx           Output context pointer
 * @param vocab_size    Size of token vocabulary
 * @param cuda_device   GPU device ID
 *
 * @return              0 on success, negative on error
 */
int cuda_sampling_init_gpu(
    cuda_sampling_context_t ** ctx,
    int32_t vocab_size,
    int32_t cuda_device
);

/**
 * Free GPU sampling context and associated device memory
 *
 * @param ctx           Context to free (set to NULL after)
 *
 * @return              0 on success, negative on error
 */
int cuda_sampling_free_gpu(cuda_sampling_context_t * ctx);

/**
 * Copy logits from GPU to sampling context (or use existing device pointer)
 *
 * @param ctx           Sampling context
 * @param d_logits      Device pointer to logits
 * @param size_bytes    Size in bytes
 * @param copy          If true, perform D2D copy; if false, use pointer directly
 *
 * @return              0 on success, negative on error
 */
int cuda_sampling_set_logits(
    cuda_sampling_context_t * ctx,
    float * d_logits,
    size_t size_bytes,
    int copy
);

/**
 * Perform greedy sampling (return token with highest logit)
 *
 * @param ctx           Sampling context (must have logits set)
 * @param token_out     [1] Host pointer to output token ID
 * @param cuda_stream   Optional CUDA stream for async execution
 *
 * @return              0 on success, negative on error
 *
 * This function:
 *  1. Calls cuda_argmax_kernel on logits
 *  2. Copies result back to host (synchronous)
 */
int cuda_sampling_sample_greedy(
    cuda_sampling_context_t * ctx,
    int32_t * token_out,
    void * cuda_stream
);

/**
 * Perform top-k + temperature sampling
 *
 * @param ctx               Sampling context
 * @param token_out         [1] Host pointer to output token ID
 * @param temperature       Temperature for softmax scaling
 * @param top_k             Keep only top-k tokens (0 to disable)
 * @param penalty_alpha     Frequency penalty coefficient
 * @param seed              Random seed
 * @param cuda_stream       Optional CUDA stream for async execution
 *
 * @return                  0 on success, negative on error
 *
 * Process:
 *  1. Apply temperature scaling and top-k filtering
 *  2. Apply frequency penalties
 *  3. Compute softmax probabilities
 *  4. Sample from distribution
 *  5. Copy result to host
 */
int cuda_sampling_sample_specialized(
    cuda_sampling_context_t * ctx,
    int32_t * token_out,
    float temperature,
    int32_t top_k,
    float penalty_alpha,
    uint64_t seed,
    void * cuda_stream
);

/**
 * Synchronize GPU sampling operations
 *
 * @param ctx           Sampling context
 *
 * @return              0 on success, negative on error
 */
int cuda_sampling_synchronize(cuda_sampling_context_t * ctx);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get device name for given GPU
 *
 * @param device_id     GPU device ID
 * @param name_out      Output buffer for device name
 * @param name_len      Size of output buffer
 *
 * @return              0 on success, negative on error
 */
int cuda_get_device_name(int device_id, char * name_out, int name_len);

/**
 * Get available VRAM on device
 *
 * @param device_id     GPU device ID
 * @param free_bytes    Output free memory in bytes
 * @param total_bytes   Output total memory in bytes
 *
 * @return              0 on success, negative on error
 */
int cuda_get_device_memory(
    int device_id,
    size_t * free_bytes,
    size_t * total_bytes
);

#ifdef __cplusplus
}
#endif

