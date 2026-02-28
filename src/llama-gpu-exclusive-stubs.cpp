/**
 * GPU-EXCLUSIVE DECODE ENGINE STUBS
 * Placeholder implementations for Phase 2.3
 *
 * These functions are declared by the GPU-exclusive decode engine but not yet
 * implemented. This file provides stub implementations to allow compilation
 * and linking. Full implementations will be added in later phases.
 *
 * Phase 2.3 Status: Declarations only (stubs provided here)
 * Phase 2.4+: Full implementations of:
 * - CUDA RNG state management
 * - Memory residency verification
 * - Persistent kernel framework
 * - Graph capture/replay (GGML CUDA backend integration)
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>

// ============================================================================
// MEMORY RESIDENCY VERIFICATION STUBS
// ============================================================================

/**
 * Verify that decode tensors reside on GPU.
 * Phase 2.3: Stub (always returns success)
 * Phase 2.4+: Real implementation checking tensor placement
 */
extern "C" {

int llama_verify_decode_memory_residency(const struct llama_context * ctx) {
    (void)ctx;  // Unused in stub
    // Phase 2.3: Assume success (all tensors OK)
    // Phase 2.4: Actually verify tensor placement on GPU
    return 0;
}

/**
 * Print memory residency diagnostics.
 * Phase 2.3: Stub (silent)
 * Phase 2.4+: Real implementation printing residency report
 */
void llama_residency_print_report() {
    // Phase 2.3: Silent (no diagnostics yet)
    // Phase 2.4: Print actual residency statistics
}

// ============================================================================
// PERSISTENT KERNEL FRAMEWORK STUBS
// ============================================================================

/**
 * Initialize persistent kernel infrastructure.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation setting up persistent kernels
 */
int llama_persistent_kernel_init(int max_tokens) {
    (void)max_tokens;  // Unused in stub
    // Phase 2.3: Not implemented
    // Phase 2.4: Initialize persistent kernel context
    return 0;
}

/**
 * Launch persistent kernel for token decoding.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation launching GPU kernels
 */
int llama_persistent_kernel_launch(const struct llama_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;  // Unused in stub
    // Phase 2.3: Not implemented
    // Phase 2.4: Launch actual GPU kernels
    return 0;
}

/**
 * Stop persistent kernel execution.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation stopping kernels
 */
int llama_persistent_kernel_stop() {
    // Phase 2.3: Not implemented
    // Phase 2.4: Stop GPU kernel execution
    return 0;
}

/**
 * Wait for persistent kernel completion.
 * Phase 2.3: Stub (immediate return)
 * Phase 2.4+: Real implementation with timeout handling
 */
int llama_persistent_kernel_wait(int timeout_ms) {
    (void)timeout_ms;  // Unused in stub
    // Phase 2.3: Immediate return (no kernels running)
    // Phase 2.4: Actually wait with timeout
    return 0;
}

/**
 * Cleanup persistent kernel framework.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real implementation cleaning up resources
 */
void llama_persistent_kernel_cleanup() {
    // Phase 2.3: Not implemented
    // Phase 2.4: Clean up persistent kernel resources
}

// ============================================================================
// CUDA RNG STATE MANAGEMENT STUBS
// ============================================================================

/**
 * Initialize CUDA RNG with seed.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real CUDA RNG initialization
 */
int ggml_cuda_rng_init(uint32_t seed) {
    (void)seed;  // Unused in stub
    // Phase 2.3: Not implemented
    // Phase 2.4: Initialize cuRNG or CUDA RNG state
    return 0;
}

/**
 * Cleanup CUDA RNG state.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real CUDA RNG cleanup
 */
int ggml_cuda_rng_cleanup() {
    // Phase 2.3: Not implemented
    // Phase 2.4: Clean up cuRNG state
    return 0;
}

/**
 * Check if CUDA RNG is initialized.
 * Phase 2.3: Stub (always false)
 * Phase 2.4+: Real status check
 */
bool ggml_cuda_rng_is_initialized() {
    // Phase 2.3: Not initialized yet
    // Phase 2.4: Return actual RNG status
    return false;
}

// ============================================================================
// CUDA GRAPH MANAGEMENT STUBS
// ============================================================================

/**
 * Begin CUDA graph capture.
 * Phase 2.3: Stub (returns dummy ID)
 * Phase 2.4+: Real graph capture
 */
uint64_t ggml_cuda_graph_capture_begin(void * stream) {
    (void)stream;  // Unused in stub
    // Phase 2.3: Return dummy graph ID
    // Phase 2.4: Start actual graph capture
    return 0;
}

/**
 * End CUDA graph capture.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real graph capture completion
 */
int ggml_cuda_graph_capture_end(uint64_t graph_id, void * stream) {
    (void)graph_id;
    (void)stream;  // Unused in stub
    // Phase 2.3: Not implemented
    // Phase 2.4: Complete graph capture
    return 0;
}

/**
 * Instantiate (compile) CUDA graph.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real graph instantiation
 */
int ggml_cuda_graph_instantiate(uint64_t graph_id, void * stream) {
    (void)graph_id;
    (void)stream;  // Unused in stub
    // Phase 2.3: Not implemented
    // Phase 2.4: Compile graph to GPU executable
    return 0;
}

/**
 * Launch CUDA graph.
 * Phase 2.3: Stub (no-op)
 * Phase 2.4+: Real graph launch
 */
int ggml_cuda_graph_launch(uint64_t graph_id, void * stream) {
    (void)graph_id;
    (void)stream;  // Unused in stub
    // Phase 2.3: Not implemented
    // Phase 2.4: Execute compiled graph on GPU
    return 0;
}

/**
 * Check if CUDA graph support is enabled.
 * Phase 2.3: Stub (always false)
 * Phase 2.4+: Real capability check
 */
bool ggml_cuda_graph_is_enabled() {
    // Phase 2.3: Graph support not enabled yet
    // Phase 2.4: Return actual CUDA graph capability
    return false;
}

}  // extern "C"
