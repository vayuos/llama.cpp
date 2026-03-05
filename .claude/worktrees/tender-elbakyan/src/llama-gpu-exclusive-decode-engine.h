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
#include "llama-kernel-fusion-enforce.h"
#include <cstdint>

// Use LLAMA_API macro for proper symbol export on all platforms
// (Windows DLL, Linux ELF visibility, macOS, etc.)
#ifndef LLAMA_API
#    if defined(_WIN32) && !defined(__MINGW32__)
#        define LLAMA_API __declspec(dllexport)
#    elif defined(__MINGW32__)
#        define LLAMA_API __declspec(dllexport)
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#endif

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
LLAMA_API int llama_gpu_exclusive_engine_init(
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
LLAMA_API int llama_gpu_exclusive_engine_prepare_decode(
    const llama_context * ctx,
    int max_tokens);

/**
 * Start decode session (transition to decoding state).
 */
LLAMA_API int llama_gpu_exclusive_engine_start_decode();

/**
 * Stop decode session.
 */
LLAMA_API int llama_gpu_exclusive_engine_stop_decode();

/**
 * Cleanup engine and free all resources.
 * Called at shutdown.
 */
LLAMA_API void llama_gpu_exclusive_engine_cleanup();

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
LLAMA_API int llama_gpu_exclusive_engine_decode_step(int token);

// ============================================================================
// STREAM SCHEDULER ACCESSORS (Async Pipelining)
// ============================================================================

/**
 * Get the global stream scheduler instance.
 * Used for CPU-GPU async pipelining coordination.
 * Returns NULL if not initialized.
 */
LLAMA_API struct llama_stream_scheduler * llama_gpu_exclusive_engine_get_scheduler();

/**
 * Get CPU compute stream from scheduler.
 * Used to queue layer 0-66 compute on CPU_COMPUTE stream.
 */
LLAMA_API void * llama_gpu_exclusive_engine_get_cpu_stream();

/**
 * Get GPU compute stream from scheduler.
 * Used to queue layer 36-49 compute on GPU_COMPUTE stream.
 */
LLAMA_API void * llama_gpu_exclusive_engine_get_gpu_stream();

/**
 * Print stream scheduler diagnostics for debugging async pipelining.
 */
LLAMA_API void llama_gpu_exclusive_engine_print_scheduler_diagnostics();

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

    // Per-token timing (Phase B4+)
    uint64_t last_token_time_ns;   // Most recent token latency
    uint64_t min_token_time_ns;    // Minimum token latency
    uint64_t max_token_time_ns;    // Maximum token latency
    uint64_t avg_token_time_ns;    // Average token latency (total_time_ns / total_tokens)
};

/**
 * Get engine statistics.
 */
LLAMA_API struct llama_gpu_engine_stats llama_gpu_exclusive_engine_get_stats();

/**
 * Get kernel fusion metrics (Phase C5+).
 * Returns current kernel fusion enforcement status and metrics.
 */
LLAMA_API llama_kernel_metrics llama_gpu_exclusive_engine_get_fusion_metrics();

/**
 * Print comprehensive diagnostics (for debugging).
 */
LLAMA_API void llama_gpu_exclusive_engine_print_diagnostics();

// ============================================================================
// GLOBAL CONTROL
// ============================================================================

/**
 * Enable/disable the GPU-exclusive engine.
 * Default: enabled
 */
LLAMA_API void llama_gpu_exclusive_engine_set_enabled(bool enabled);

/**
 * Check if engine is enabled.
 */
LLAMA_API bool llama_gpu_exclusive_engine_is_enabled();

// ============================================================================
// INFRASTRUCTURE STUBS (Phase 2.4+ implementations)
// ============================================================================

/**
 * Verify that all decode tensors reside on GPU.
 * Phase 2.3: Stub returns success
 * Phase 2.4+: Real implementation checking tensor placement
 */
LLAMA_API int llama_verify_decode_memory_residency(const struct llama_context * ctx);

/**
 * Print memory residency diagnostics.
 * Phase 2.3: Stub (silent)
 * Phase 2.4+: Real implementation printing residency statistics
 */
LLAMA_API void llama_residency_print_report();

/**
 * Initialize persistent kernel infrastructure.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation setting up persistent kernels
 */
LLAMA_API int llama_persistent_kernel_init(int max_tokens);

/**
 * Launch persistent kernel for token decoding.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation launching GPU kernels
 */
LLAMA_API int llama_persistent_kernel_launch(const struct llama_context * ctx, int max_tokens);

/**
 * Stop persistent kernel execution.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation stopping kernels
 */
LLAMA_API int llama_persistent_kernel_stop();

/**
 * Wait for persistent kernel to complete.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation waiting for kernel completion
 */
LLAMA_API int llama_persistent_kernel_wait(int timeout_ms);

/**
 * Cleanup persistent kernel resources.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation cleaning up kernel resources
 */
LLAMA_API void llama_persistent_kernel_cleanup();

/**
 * Initialize CUDA RNG state for GPU-side operations.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation initializing CUDA RNG
 */
LLAMA_API int ggml_cuda_rng_init(uint32_t seed);

/**
 * Cleanup CUDA RNG state.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation cleaning up CUDA RNG
 */
LLAMA_API int ggml_cuda_rng_cleanup();

/**
 * Check if CUDA RNG is initialized.
 * Phase 2.3: Stub returns false
 * Phase 2.4+: Real implementation checking CUDA RNG state
 */
LLAMA_API bool ggml_cuda_rng_is_initialized();

/**
 * Begin CUDA graph capture on a stream.
 * Phase 2.3: Stub returns 0
 * Phase 2.4+: Real implementation capturing CUDA graphs
 */
LLAMA_API uint64_t ggml_cuda_graph_capture_begin(void * stream);

/**
 * End CUDA graph capture.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation ending graph capture
 */
LLAMA_API int ggml_cuda_graph_capture_end(uint64_t graph_id, void * stream);

/**
 * Instantiate a captured CUDA graph.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation instantiating CUDA graphs
 */
LLAMA_API int ggml_cuda_graph_instantiate(uint64_t graph_id, void * stream);

/**
 * Launch an instantiated CUDA graph.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation launching CUDA graphs
 */
LLAMA_API int ggml_cuda_graph_launch(uint64_t graph_id, void * stream);

/**
 * Check if CUDA graph support is enabled.
 * Phase 2.3: Stub returns false
 * Phase 2.4+: Real implementation checking CUDA graph support
 */
LLAMA_API bool ggml_cuda_graph_is_enabled();

#ifdef __cplusplus
}
#endif
