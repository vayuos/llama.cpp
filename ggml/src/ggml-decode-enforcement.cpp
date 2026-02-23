/**
 * GGML Decode Enforcement Implementation
 *
 * Low-level backend dispatch enforcement to prevent implicit device transfers
 * and ensure GPU-exclusive decode execution.
 */

#include "ggml-decode-enforcement.h"
#include "ggml-impl.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <string.h>

/**
 * [CRITICAL] Check for implicit tensor copy during backend mismatch
 *
 * When a tensor is on CPU and op is on GPU (or vice versa),
 * implicit copy is normally performed. In decode mode, this is forbidden.
 */
bool ggml_decode_check_implicit_copy(int tensor_backend, int op_backend, bool in_decode_mode) {
    // If backends match, no copy needed
    if (tensor_backend == op_backend) {
        return true;
    }

    // In decode mode, any backend mismatch is fatal
    if (in_decode_mode) {
        GGML_LOG_ERROR(
            "GGML DECODE ENFORCEMENT: Implicit tensor copy detected!\n"
            "  Tensor backend: %d\n"
            "  Op backend: %d\n"
            "  In decode mode, all tensors must be on the same backend as their ops.\n"
            "  No implicit device transfers allowed.\n",
            tensor_backend, op_backend);
        GGML_ABORT("Implicit tensor copy in decode mode\n");
        return false;
    }

    // In normal mode, copy is allowed
    return true;
}

/**
 * [CRITICAL] Enforce backend assignment is immutable
 *
 * Once a node's backend is determined during graph freeze,
 * it cannot be reassigned. Any change during execution is a violation.
 */
bool ggml_decode_check_backend_immutable(
    int node_id,
    int previous_backend,
    int new_backend,
    bool in_decode_mode) {

    if (!in_decode_mode) {
        return true;  // Not in decode, reassignments allowed
    }

    // Check if backend changed
    if (previous_backend != new_backend) {
        GGML_LOG_ERROR(
            "GGML DECODE ENFORCEMENT: Backend reassignment detected!\n"
            "  Node: %d\n"
            "  Previous backend: %d\n"
            "  New backend: %d\n"
            "  Decode graphs are immutable after freeze.\n",
            node_id, previous_backend, new_backend);
        GGML_ABORT("Backend reassignment in decode mode\n");
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Validate tensor buffer residency
 *
 * A tensor assigned to GPU backend must actually be in GPU memory.
 * This catches host-aliased tensors and allocation errors.
 */
bool ggml_decode_check_tensor_buffer_residency(
    struct ggml_tensor * tensor,
    int expected_backend,
    bool in_decode_mode) {

    if (!tensor || !in_decode_mode) {
        return true;
    }

    // Only validate GPU-assigned tensors
    if (expected_backend == 0) {  // Assuming 0 is GPU
        // Check that tensor has a buffer
        if (!tensor->buffer) {
            GGML_LOG_ERROR(
                "GGML DECODE ENFORCEMENT: Tensor '%s' assigned to GPU but has no buffer!\n",
                tensor->name ? tensor->name : "unnamed");
            return false;
        }

        // Check that buffer is not host memory
        if (tensor->buffer->buft && ggml_backend_buft_is_host(tensor->buffer->buft)) {
            GGML_LOG_ERROR(
                "GGML DECODE ENFORCEMENT: Tensor '%s' assigned to GPU but is CPU-resident!\n",
                tensor->name ? tensor->name : "unnamed");
            GGML_ABORT("GPU-scheduled tensor is CPU-resident\n");
            return false;
        }
    }

    return true;
}

/**
 * Operations that MUST have GPU implementations during decode
 */
static const struct {
    const char * op_name;
    bool is_critical;
} GGML_DECODE_CRITICAL_OPS[] = {
    // Matrix operations
    {"mul_mat",         true},
    {"mul_mat_q",       true},
    {"mul_mat_id",      true},
    {"mul_mat_id_q",    true},
    {"mul_mat_vec",     true},

    // Normalization
    {"rms_norm",        true},
    {"norm",            true},
    {"group_norm",      true},

    // Softmax and activations
    {"soft_max",        true},
    {"gelu",            true},
    {"silu",            true},
    {"relu",            true},
    {"relu_sqr",        true},
    {"gelu_new",        true},

    // Other critical ops
    {"get_rows",        true},
    {"set_rows",        true},
    {"add",             true},
    {"mul",             true},

    // Sampling (Phase 3)
    {"sample_candidates", true},
    {"penalties",         true},
    {"update_state",      true},

    {nullptr, false}
};

/**
 * Check if operation is critical for decode
 */
static bool is_ggml_decode_critical_op(const char * op_name) {
    if (!op_name) return false;

    for (int i = 0; GGML_DECODE_CRITICAL_OPS[i].op_name != nullptr; i++) {
        if (strcmp(op_name, GGML_DECODE_CRITICAL_OPS[i].op_name) == 0) {
            return GGML_DECODE_CRITICAL_OPS[i].is_critical;
        }
    }
    return false;
}

/**
 * [CRITICAL] Reject CPU implementations of GPU-critical ops
 *
 * If an operation is critical for decode and GPU backend doesn't support it,
 * abort at graph build time rather than silently falling back to CPU.
 */
bool ggml_decode_check_op_has_gpu_impl(
    const char * op_name,
    int backend_available,
    bool in_decode_mode) {

    if (!op_name || !in_decode_mode) {
        return true;  // Not in decode or no op name, allow
    }

    // Check if this is a critical op
    if (!is_ggml_decode_critical_op(op_name)) {
        return true;  // Not critical, allow on any backend
    }

    // Critical op - GPU backend required
    if (backend_available != 0) {  // 0 = GPU
        GGML_LOG_ERROR(
            "GGML DECODE ENFORCEMENT: Critical op '%s' not available on GPU!\n"
            "  Decode requires all critical ops to have GPU implementations.\n"
            "  Backend: %d (expected 0 for GPU)\n"
            "  CPU fallback forbidden during decode.\n",
            op_name, backend_available);
        GGML_ABORT("Critical decode op lacks GPU implementation\n");
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Check that no CPU nodes exist in decode graph
 *
 * Scans all nodes and aborts if any are assigned to CPU backend.
 */
bool ggml_decode_check_no_cpu_nodes(
    struct ggml_cgraph * graph,
    const int * backend_ids,
    int n_nodes,
    bool in_decode_mode) {

    if (!graph || !backend_ids || !in_decode_mode) {
        return true;
    }

    for (int i = 0; i < n_nodes; i++) {
        int backend = backend_ids[i];

        // 0 = GPU, anything else in decode mode is an error
        if (backend != 0) {
            struct ggml_tensor * node = nullptr;
            if (i < graph->n_nodes) {
                node = graph->nodes[i];
            }

            const char * op_name = node ? ggml_op_name(node->op) : "unknown";
            const char * node_name = node && node->name ? node->name : "unnamed";

            GGML_LOG_ERROR(
                "GGML DECODE ENFORCEMENT: CPU node detected in decode graph!\n"
                "  Node %d: %s (%s)\n"
                "  Backend: %d (expected 0 for GPU)\n"
                "  Decode graphs must be 100%% GPU-resident.\n",
                i, op_name, node_name, backend);

            GGML_ABORT("CPU node in GPU decode graph\n");
            return false;
        }
    }

    return true;
}

/**
 * [CRITICAL] Disable dynamic backend selection during decode
 *
 * Once backends are assigned and graph is frozen, they cannot change.
 * This prevents adaptive re-optimization during execution.
 */
bool ggml_decode_check_backend_stable(
    int current_backend,
    int previous_backend,
    bool in_decode_mode) {

    if (!in_decode_mode) {
        return true;  // Not in decode, backend can change
    }

    // In decode, backends must remain stable
    if (current_backend != previous_backend) {
        GGML_LOG_ERROR(
            "GGML DECODE ENFORCEMENT: Backend changed during decode execution!\n"
            "  Previous: %d\n"
            "  Current: %d\n"
            "  Decode requires stable backend selection.\n",
            previous_backend, current_backend);
        GGML_ABORT("Backend selection changed in decode mode\n");
        return false;
    }

    return true;
}
