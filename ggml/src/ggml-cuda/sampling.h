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

#include <stddef.h>
#include <stdint.h>

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
 *
 * DECODE PHASE ENFORCEMENT:
 * - During decode, GPU is the SOLE authority for token selection
 * - CPU path forbidden (will abort)
 * - All decision logic (penalties, temp, top-k/p, selection) on GPU
 *
 * TRANSFER PROHIBITION INVARIANT:
 * - During decode, NO host↔device memory transfers permitted except token ID
 * - Only the final selected token ID (4-8 bytes) is allowed to cross PCIe
 * - All sampling intermediates (logits, probs, buffers) remain GPU-resident
 * - Logits must NEVER be copied to CPU during decode
 */
typedef struct {
    // Allocated device memory
    float *   d_logits;         // [vocab_size] - token logits
    float *   d_penalties;      // [vocab_size] - frequency/repetition penalties
    float *   d_probs;          // [vocab_size] - softmax probabilities / cumsum probs (top-p)
    float *   d_scratch;        // scratch space for reductions
    int32_t * d_sampled_token;  // [1] - output token ID
    
    // Top-p GPU filters (allocated in context init)
    int32_t * d_sorted_inds;    // [vocab_size] - sorted indices for top-p
    int32_t * d_mask;           // [vocab_size] - filter mask (0=keep, 1=filter)
    int32_t * d_n_keep;         // [1] - count of kept tokens after filtering

    // Decode phase enforcement (GPU token selection authority)
    int32_t   in_decode_phase;  // 0=prefill/disabled, 1=decode/GPU-only
    int32_t   token_selection_locked;  // 1 if GPU locked as selection authority

    // ========================================================================
    // TRANSFER PROHIBITION INSTRUMENTATION
    // ========================================================================
    // During decode: NO host↔device transfers except final token ID (4 bytes)
    
    uint64_t  transfer_guard_counter;  // Count of transfers during decode
    uint64_t  max_allowed_transfer;    // Max bytes per transfer (sizeof(int32_t)=4)
    int32_t   logits_copied_to_host;   // ERROR: 1 if logits illegally copied to host
    int32_t   bulk_transfer_attempted; // ERROR: 1 if non-token transfer attempted

    // Configuration
    int32_t vocab_size;
    int32_t cuda_device;
    void *  cuda_stream;  // optional CUDA stream (opaque)
    void *  token_event;  // optional CUDA event (opaque) for host-visible token completion

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
int cuda_argmax_kernel(const float * d_logits,
                       int32_t *     d_out_token,
                       int32_t       vocab_size,
                       float *       d_scratch,
                       void *        cuda_stream);

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
int cuda_temperature_scale_kernel(float * d_logits,
                                  float   temperature,
                                  int32_t top_k,
                                  int32_t vocab_size,
                                  void *  cuda_stream);

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
int cuda_apply_penalties_kernel(float *       d_logits,
                                const float * d_penalties,
                                float         alpha,
                                int32_t       vocab_size,
                                void *        cuda_stream);

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
int cuda_softmax_kernel(const float * d_logits,
                        float *       d_probs,
                        int32_t       vocab_size,
                        float *       d_scratch,
                        void *        cuda_stream);

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
int cuda_sample_categorical_kernel(const float * d_probs,
                                   int32_t *     d_out_token,
                                   int32_t       vocab_size,
                                   uint64_t      seed,
                                   void *        cuda_stream);

/**
 * GPU-native top-k selection kernel
 *
 * Selects the k largest values from logits array and returns their indices
 * and values entirely on GPU, eliminating PCIe transfer of full logits array.
 *
 * Avoids full-device sort by using:
 *  - Warp-level reduction (k <= 32, vocab_size <= 1024)
 *  - Block-level heap selection (medium k)
 *  - CUB top-k (if available, large k)
 *
 * @param d_logits      [n_vocab] float array of logits
 * @param d_topk_vals   [k] output array of top-k values (sorted descending)
 * @param d_topk_inds   [k] output array of corresponding token indices
 * @param n_vocab       Size of vocabulary
 * @param k             Number of top tokens to select
 * @param cuda_stream   CUDA stream for async execution (optional)
 *
 * @return              0 on success, negative on error
 *
 * Performance:
 *  - Small k (<=32): ~500-1000 us
 *  - Medium k: ~1-2 ms
 *  - Large k (with CUB): ~2-5 ms
 *
 * Memory:
 *  - O(k) shared memory
 *  - No temporary device allocations
 *
 * Determinism: Stable sorting for equal values; matches CPU implementation
 */
int cuda_topk_kernel(const float * d_logits,
                     float *       d_topk_vals,
                     int32_t *     d_topk_inds,
                     int32_t       n_vocab,
                     int32_t       k,
                     void *        cuda_stream);

/**
 * GPU-native nucleus (top-p) filtering kernel
 *
 * Complete nucleus filtering pipeline on GPU:
 *  1. Softmax normalization with temperature scaling
 *  2. Bitonic sort (descending probability order)
 *  3. Prefix sum (cumulative probabilities)
 *  4. Threshold detection (find cutoff where cumsum >= p)
 *  5. Mask generation (0=keep, 1=filter)
 *
 * All operations GPU-resident with no PCIe transfer of probability arrays.
 * Only vocabulary size and threshold parameter sent to GPU; result is mask.
 *
 * @param d_logits      [vocab_size] float array of logits
 * @param d_probs       [vocab_size] output cumulative probabilities (after sort)
 * @param d_sorted_inds [vocab_size] output indices after sort
 * @param d_mask        [vocab_size] output mask (0=keep, 1=filter out)
 * @param d_n_keep      [1] output: count of tokens kept (< cutoff)
 * @param p             Nucleus threshold (0.0 < p <= 1.0)
 * @param temperature   Temperature for softmax scaling (1.0 = no scaling)
 * @param vocab_size    Size of vocabulary
 * @param cuda_stream   CUDA stream for async execution (optional)
 *
 * @return              0 on success, negative on error
 *
 * Performance: ~5-12 ms (dominated by bitonic sort for large vocab)
 *
 * Memory:
 *  - Shared memory: 2 * vocab_size * sizeof(float)
 *  - No temporary device allocations beyond d_probs, d_sorted_inds, d_mask
 *
 * Determinism:
 *  - Bitonic sort is stable and deterministic
 *  - Prefix sum is serial (single-threaded within kernel)
 *  - Matches CPU implementation with bit-exact probabilities (within float precision)
 *
 * Limitations:
 *  - Shared memory limit: vocab_size * (2*sizeof(float)) <= 48KB
 *  - For vocab > 6000: may exceed shared memory; split into tile-based approach
 */
int cuda_topp_kernel(const float * d_logits,
                     float *       d_probs,
                     int32_t *     d_sorted_inds,
                     int32_t *     d_mask,
                     int32_t *     d_n_keep,
                     float         p,
                     float         temperature,
                     int32_t       vocab_size,
                     void *        cuda_stream);

/**
 * Bitonic sort for probability arrays (descending order)
 *
 * Sorts values array while maintaining index array synchronization.
 * Output: values in descending order, indices map to original positions.
 *
 * @param d_values      [n] float array to sort (in-place)
 * @param d_indices     [n] index array to permute (in-place)
 * @param n             Array size (must be <= 2^30)
 * @param cuda_stream   CUDA stream for async execution
 *
 * @return              0 on success, negative on error
 *
 * Performance: O(log²n) compares, ~5-10 ms for vocab_size
 *
 * Stability: Stable for equal values (preserves lower index)
 */
int cuda_bitonic_sort_desc(float *       d_values,
                           int32_t *     d_indices,
                           int32_t       n,
                           void *        cuda_stream);

/**
 * Parallel prefix sum (cumulative sum) operation
 *
 * Transforms array in-place: out[i] = sum(in[0..i])
 *
 * Uses Blelloch scan algorithm (log-linear complexity)
 *
 * @param d_data        [n] array to scan (in-place)
 * @param n             Array size
 * @param cuda_stream   CUDA stream for async execution
 *
 * @return              0 on success, negative on error
 *
 * Performance: ~1-3 ms for vocab_size
 *
 * Numerical stability: Accumulates from left to right; no compensation
 */
int cuda_prefix_sum(float * d_data, int32_t n, void * cuda_stream);

/**
 * UNIFIED GPU TOKEN SELECTION KERNEL - GPU-ONLY AUTHORITY FOR DECODE
 *
 * This is the EXCLUSIVE decision-maker for next-token selection during decode phase.
 * CPU is completely forbidden from influencing token choice.
 *
 * The kernel performs final token selection after all upstream processing (penalties,
 * temperature, top-k/top-p filtering) has been applied on GPU.
 *
 * Supports two selection modes:
 *  1. Argmax (deterministic greedy): Select token with highest logit
 *  2. Categorical (deterministic seeded): Sample from probability distribution
 *
 * **INVARIANT:** This MUST be the ONLY GPU→CPU token selection call.
 * All other CPU sampling code is disabled during decode phase.
 *
 * @param d_logits              Input logits (after penalties, temp, filtering)
 * @param d_selected_token      [1] Output: selected token ID
 * @param vocab_size            Vocabulary size
 * @param selection_mode        0=argmax, 1=categorical
 * @param d_cumsum_probs        [vocab_size] Cumulative probabilities (mode=1 only)
 * @param d_sorted_indices      [vocab_size] Sorted indices (mode=1 only)
 * @param seed                  RNG seed (mode=1 only)
 * @param cuda_stream           CUDA stream for async execution
 *
 * @return                      0 on success, negative on error
 *
 * Determinism:
 *  - Argmax: ties broken by lower index (stable)
 *  - Categorical: seeded RNG, identical output with same seed
 *  - Matches CPU reference implementation
 *
 * GPU Authority:
 *  - No CPU post-processing
 *  - No CPU tie-breaking
 *  - No CPU verification
 *  - Token ID is final and binding
 */
int cuda_unified_select_token(const float *    d_logits,
                              int32_t *        d_selected_token,
                              int32_t          vocab_size,
                              int32_t          selection_mode,
                              const float *    d_cumsum_probs,
                              const int32_t *  d_sorted_indices,
                              uint64_t         seed,
                              void *           cuda_stream);

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
int cuda_sampling_init_gpu(cuda_sampling_context_t ** ctx, int32_t vocab_size, int32_t cuda_device);

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
int cuda_sampling_set_logits(cuda_sampling_context_t * ctx, float * d_logits, size_t size_bytes, int copy);

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
int cuda_sampling_sample_greedy(cuda_sampling_context_t * ctx, int32_t * token_out, void * cuda_stream);

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
int cuda_sampling_sample_specialized(cuda_sampling_context_t * ctx,
                                     int32_t *                 token_out,
                                     float                     temperature,
                                     int32_t                   top_k,
                                     float                     penalty_alpha,
                                     uint64_t                  seed,
                                     void *                    cuda_stream);

/**
 * Perform nucleus (top-p) + temperature sampling
 *
 * GPU-native nucleus filtering pipeline:
 *  1. Apply temperature scaling
 *  2. Apply frequency penalties
 *  3. GPU softmax + sort + prefix sum + nucleus cutoff detection
 *  4. Sample from nucleus-filtered vocabulary
 *
 * All filtering operations are GPU-resident; only output token ID crosses PCIe.
 *
 * @param ctx               Sampling context
 * @param token_out         [1] Host pointer to output token ID
 * @param temperature       Temperature for softmax scaling
 * @param p                 Nucleus threshold (0.0 < p <= 1.0)
 * @param penalty_alpha     Frequency penalty coefficient
 * @param seed              Random seed
 * @param cuda_stream       Optional CUDA stream for async execution
 *
 * @return                  0 on success, negative on error
 *
 * Performance: ~5-12 ms depending on vocabulary size (dominated by sort)
 *
 * Determinism: Matches CPU nucleus filtering with bit-exact probabilities
 */
int cuda_sampling_sample_topk_topp(cuda_sampling_context_t * ctx,
                                   int32_t *                 token_out,
                                   float                     temperature,
                                   float                     p,
                                   float                     penalty_alpha,
                                   uint64_t                  seed,
                                   void *                    cuda_stream);

/**
 * Lock GPU as the exclusive token selection authority for decode phase
 *
 * Enforces that during token generation, GPU is the SOLE decision-maker.
 * No CPU sampling code will be permitted to execute.
 *
 * Must be called at decode start before any token generation.
 *
 * @param ctx           Sampling context
 *
 * @return              0 on success, -1 if already locked or error
 */
int cuda_sampling_lock_decode_phase(cuda_sampling_context_t * ctx);

/**
 * Unlock GPU authority and return to prefill/initialization phase
 *
 * Permits CPU sampling code to execute (for debugging or CPU-only inference).
 *
 * @param ctx           Sampling context
 *
 * @return              0 on success, -1 if not locked or error
 */
int cuda_sampling_unlock_decode_phase(cuda_sampling_context_t * ctx);

/**
 * Check if decode phase is active (GPU-only authority)
 *
 * @param ctx           Sampling context
 *
 * @return              1 if in decode phase, 0 if not
 */
int cuda_sampling_in_decode_phase(const cuda_sampling_context_t * ctx);

/**
 * Synchronize GPU sampling operations
 *
 * @param ctx           Sampling context
 *
 * @return              0 on success, negative on error
 */
int cuda_sampling_synchronize(cuda_sampling_context_t * ctx);

// ============================================================================
// TRANSFER GUARD ENFORCEMENT - Prohibition of host↔device transfers during decode
// ============================================================================

/**
 * Check if host↔device transfer is allowed during decode
 *
 * INVARIANT: During decode phase, only transfers for final token ID are permitted.
 * If a transfer larger than 4 bytes (sizeof(int32_t)) is attempted, this returns -1.
 *
 * @param ctx           Sampling context
 * @param transfer_size Proposed transfer size in bytes
 *
 * @return              0 if allowed, -1 if forbidden (ABORT transfer)
 *
 * Restricted transfers during decode:
 *  - Logits D2H: FORBIDDEN (abort)
 *  - Probabilities D2H: FORBIDDEN (abort)
 *  - Top-k buffers D2H: FORBIDDEN (abort)
 *  - Top-p buffers D2H: FORBIDDEN (abort)
 *  - Token ID D2H: ALLOWED (4 bytes)
 *  - Token ID H2D: ALLOWED (4 bytes)
 */
int cuda_check_transfer_guard(const cuda_sampling_context_t * ctx, size_t transfer_size);

/**
 * Reset transfer guard counters (called at decode phase start)
 *
 * @param ctx           Sampling context
 *
 * @return              0 on success
 */
int cuda_reset_transfer_guard(cuda_sampling_context_t * ctx);

/**
 * Get transfer violation details for debugging
 *
 * @param ctx           Sampling context
 * @param logits_D2H    Output: 1 if logits were illegally copied to host
 * @param bulk_transfer Output: 1 if non-token transfer was attempted
 *
 * @return              0 on success, -1 if no violations
 */
int cuda_get_transfer_violations(const cuda_sampling_context_t * ctx,
                                 int32_t * logits_D2H,
                                 int32_t * bulk_transfer);

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
int cuda_get_device_memory(int device_id, size_t * free_bytes, size_t * total_bytes);

#ifdef __cplusplus
}
#endif
