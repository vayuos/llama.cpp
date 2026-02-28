/**
 * GPU-EXCLUSIVE DECODE ENGINE
 * Header: Unified orchestration of GPU-exclusive components
 *
 * Provides a single coherent API for:
 * - CUDA graph capture/replay
 * - GPU RNG state management
 * - Memory residency verification
 * - Persistent kernel execution
 * - Decode boundary enforcement
 */

#pragma once

#include "llama.h"
#include <cstdint>

// Forward declaration for async pipelining scheduler
struct llama_stream_scheduler;

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ENGINE LIFECYCLE
// ============================================================================

/**
 * Initialize GPU-exclusive decode engine.
 * Must be called once before any decoding.
 *
 * @param ctx Llama context
 * @param rng_seed Random seed for GPU RNG
 * @return 0 on success, -1 on error
 */
int llama_gpu_exclusive_engine_init(
    const llama_context * ctx,
    uint32_t rng_seed);

/**
 * Prepare decode with graph capture.
 * Should be called once per context before first decode.
 *
 * @param ctx Llama context
 * @param max_tokens Maximum tokens to generate
 * @return 0 on success, -1 on error
 */
int llama_gpu_exclusive_engine_prepare_decode(
    const llama_context * ctx,
    int max_tokens);

/**
 * Start decode session (transition to decoding state).
 */
int llama_gpu_exclusive_engine_start_decode();

/**
 * Stop decode session.
 */
int llama_gpu_exclusive_engine_stop_decode();

/**
 * Cleanup engine and free all resources.
 * Called at shutdown.
 */
void llama_gpu_exclusive_engine_cleanup();

// ============================================================================
// RUNTIME EXECUTION
// ============================================================================

/**
 * Execute single decode step.
 * Updates GPU-resident state, launches CUDA graph, retrieves output token.
 * Uses async pipelining: CPU and GPU process different tokens concurrently.
 *
 * @param token Current token to process
 * @return >=0 output token ready, -1 error, -2 no output yet (still in pipeline)
 */
int llama_gpu_exclusive_engine_decode_step(int token);

// ============================================================================
// STREAM SCHEDULER ACCESSORS (Async Pipelining)
// ============================================================================

/**
 * Get the global stream scheduler instance.
 * Used for CPU-GPU async pipelining coordination.
 * Returns NULL if not initialized.
 */
struct llama_stream_scheduler * llama_gpu_exclusive_engine_get_scheduler();

/**
 * Get CPU compute stream from scheduler.
 * Used to queue layer 0-66 compute on CPU_COMPUTE stream.
 */
void * llama_gpu_exclusive_engine_get_cpu_stream();

/**
 * Get GPU compute stream from scheduler.
 * Used to queue layer 36-49 compute on GPU_COMPUTE stream.
 */
void * llama_gpu_exclusive_engine_get_gpu_stream();

/**
 * Print stream scheduler diagnostics for debugging async pipelining.
 */
void llama_gpu_exclusive_engine_print_scheduler_diagnostics();

// ============================================================================
// STATISTICS AND DIAGNOSTICS
// ============================================================================

struct llama_gpu_engine_stats {
    int state;  // Current engine state
    bool rng_initialized;
    bool memory_verified;
    bool graph_ready;
    uint64_t total_decodes;
    uint64_t total_tokens;
    uint64_t total_time_ns;
    int total_errors;
};

/**
 * Get engine statistics.
 */
struct llama_gpu_engine_stats llama_gpu_exclusive_engine_get_stats();

/**
 * Print comprehensive diagnostics (for debugging).
 */
void llama_gpu_exclusive_engine_print_diagnostics();

// ============================================================================
// GLOBAL CONTROL
// ============================================================================

/**
 * Enable/disable the GPU-exclusive engine.
 * Default: enabled
 */
void llama_gpu_exclusive_engine_set_enabled(bool enabled);

/**
 * Check if engine is enabled.
 */
bool llama_gpu_exclusive_engine_is_enabled();

#ifdef __cplusplus
}
#endif
