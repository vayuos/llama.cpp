/**
 * Decode Invariant Enforcement
 * 
 * Enforces GPU-exclusive execution for decode-critical operations.
 */

#pragma once

#include "../ggml/include/ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_decode_invariant {
    bool enabled;
    bool allow_hybrid_mode;  // FIX: Allow hybrid CPU/GPU mode in addition to GPU-exclusive
    // Add additional state fields as needed
};

void llama_decode_invariant_init(void);

/**
 * Enforce GPU exclusive invariant on the graph.
 * Returns 0 on success, non-zero on failure.
 */
int llama_enforce_gpu_exclusive_invariant(struct ggml_cgraph * graph, ggml_backend_sched_t sched, struct llama_decode_invariant * invariant);

#ifdef __cplusplus
}
#endif
