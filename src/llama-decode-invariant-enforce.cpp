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

    // Iterate through all nodes in the decode graph
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
            // [STRICT] Fatal violation: decode-critical node on CPU
            fprintf(stderr, "FATAL: [HIERARCHICAL POLICY VIOLATION] Critical node %s (op: %s) scheduled on CPU.\n",
                    node->name, ggml_op_name(node->op));
            fprintf(stderr, "  - Policy: Step 3 (Attention, MLP, KV Updates, Logits) MUST NOT run on CPU.\n");
            fprintf(stderr, "  - Remedy: Follow ADMISSION ADVICE hierarchy (reduce -c, -b, or offload more layers).\n");
            return -1;
        }
    }
    
    invariant->enabled = true;
    return 0;
}
