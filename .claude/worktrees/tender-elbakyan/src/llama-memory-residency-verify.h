/**
 * SECTION 17: Memory Residency Guarantee
 * Header: Pre-decode verification API
 */

#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// RESIDENCY STATISTICS
// ============================================================================

struct llama_residency_stats {
    int total_checks;
    int total_failures;
    size_t num_requirements;
    bool last_passed;
};

// ============================================================================
// VERIFICATION API
// ============================================================================

/**
 * Verify that all critical data structures are GPU-resident.
 * Should be called before llama_decode to enforce the GPU-exclusive invariant.
 *
 * Checks:
 * 1. All model layers (weights) are in VRAM
 * 2. KV cache is GPU-resident
 * 3. Sampling state (logits, RNG) is on GPU
 *
 * In strict mode, returns -1 if any requirement fails (triggers abort).
 * In lenient mode, logs warnings but returns 0.
 *
 * @param ctx The llama context to verify
 * @return 0 if all checks pass or in lenient mode, -1 if strict mode fails
 */
int llama_verify_decode_memory_residency(const llama_context * ctx);

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Enable/disable residency verification.
 * Default: enabled
 */
void llama_residency_set_enabled(bool enabled);

/**
 * Set strict mode.
 * In strict mode: abort if any requirement fails
 * In lenient mode: log warnings but continue
 * Default: strict mode ON
 */
void llama_residency_set_strict(bool strict);

// ============================================================================
// QUERY API
// ============================================================================

/**
 * Get result of last verification.
 */
bool llama_residency_get_last_result();

/**
 * Get count of failed verifications.
 */
int llama_residency_get_failure_count();

/**
 * Get comprehensive statistics.
 */
struct llama_residency_stats llama_residency_get_stats();

/**
 * Print detailed report (for debugging).
 */
void llama_residency_print_report();

#ifdef __cplusplus
}
#endif
