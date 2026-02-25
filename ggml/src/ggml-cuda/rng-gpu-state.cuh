/**
 * SECTION 6: GPU-Resident RNG State
 * Header: GPU RNG initialization and sampling API
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// RNG STATE STRUCTURE
// ============================================================================

struct ggml_cuda_rng_state_t {
    uint32_t seed;
    uint32_t state_a;
    uint32_t state_b;
    uint32_t state_c;
    uint32_t state_d;
};

// ============================================================================
// INITIALIZATION AND CLEANUP
// ============================================================================

/**
 * Initialize GPU RNG with seed.
 * Allocates device memory and initializes state.
 * Must be called once before any sampling operations.
 */
int ggml_cuda_rng_init(uint32_t seed);

/**
 * Cleanup GPU RNG (free device memory).
 * Called at shutdown.
 */
int ggml_cuda_rng_cleanup();

// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

/**
 * Generate n uniform random floats in [0, 1) on device.
 * Output buffer must be device-resident (GPU memory).
 * Used by GPU sampling kernels (penalties, top-k, top-p, selection).
 */
int ggml_cuda_rng_generate_uniform(
    float * d_output,      // Device output buffer
    int32_t n,              // Number of floats to generate
    cudaStream_t stream);   // Stream to launch kernel on

// ============================================================================
// STATE MANAGEMENT (for checkpointing/resuming)
// ============================================================================

/**
 * Get current RNG state from device.
 * Synchronizes to copy state back to host.
 * Used for saving checkpoints.
 */
int ggml_cuda_rng_get_state(struct ggml_cuda_rng_state_t * state);

/**
 * Set RNG state on device.
 * Used for resuming from checkpoints.
 */
int ggml_cuda_rng_set_state(const struct ggml_cuda_rng_state_t * state);

/**
 * Reseed RNG with new value.
 * Useful for non-deterministic sampling.
 */
int ggml_cuda_rng_reseed(uint32_t seed);

// ============================================================================
// QUERY API
// ============================================================================

/**
 * Check if RNG is initialized.
 */
bool ggml_cuda_rng_is_initialized();

#ifdef __cplusplus
}
#endif
