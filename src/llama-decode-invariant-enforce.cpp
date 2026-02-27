/**
 * Decode Invariant Enforcement Implementation
 */

#include "llama-decode-invariant-enforce.h"
#include "llama-impl.h"
#include "../ggml/src/ggml-impl.h"

void llama_decode_invariant_init(void) {
    // Initialization logic
    // Currently a stub
}

int llama_enforce_gpu_exclusive_invariant(struct ggml_cgraph * graph, ggml_backend_sched_t sched, struct llama_decode_invariant * invariant) {
    if (!graph || !sched || !invariant) {
        return 1;
    }

    // ========================================================================
    // FIX: Allow hybrid mode by relaxing strict GPU-exclusive enforcement
    // ========================================================================
    bool has_cpu_ops = false;
    int cpu_op_count = 0;

    // First pass: detect if we have CPU operations (hybrid mode)
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // We only care about decode-critical operations
        if (!(node->flags & GGML_TENSOR_FLAG_DECODE_CRITICAL)) {
            continue;
        }

        // Get the backend assigned to this node
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, node);
        if (!backend) {
            continue; // Not assigned yet?
        }

        // Check if the backend is a GPU device
        bool is_gpu = (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU);

        if (!is_gpu) {
            has_cpu_ops = true;
            cpu_op_count++;
        }
    }

    // If we detected CPU operations, we're in hybrid mode - allow it with warnings
    if (has_cpu_ops) {
        fprintf(stdout, "[INVARIANT ENFORCEMENT] Hybrid mode detected (%d decode nodes on CPU)\n", cpu_op_count);
        fprintf(stdout, "[INVARIANT ENFORCEMENT] Relaxing strict GPU-exclusive enforcement for hybrid execution\n");
        fprintf(stdout, "[INVARIANT ENFORCEMENT] Performance will be reduced due to CPU-GPU synchronization overhead\n");
        fprintf(stdout, "[INVARIANT ENFORCEMENT] For optimal performance, use -ngl 999 for full GPU offloading\n");

        // Allow hybrid mode to proceed (don't fatally reject)
        invariant->enabled = true;
        invariant->allow_hybrid_mode = true;
        return 0;
    }

    // GPU-exclusive path: all decode-critical ops are on GPU
    invariant->enabled = true;
    invariant->allow_hybrid_mode = false;
    return 0;
}
