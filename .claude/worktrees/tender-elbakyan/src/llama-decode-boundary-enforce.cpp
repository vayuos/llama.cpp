/**
 * Decode Critical Boundary Enforcement Implementation
 *
 * Prevents CPU↔GPU op boundary splitting during autonomous token decode.
 * All decode-critical operations execute entirely on GPU.
 */

#include "llama-decode-boundary-enforce.h"
#include "llama-impl.h"

#include "../ggml/src/ggml-impl.h"
#include "../ggml/include/ggml-backend.h"

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cstring>

#ifndef LLAMA_ABORT
#define LLAMA_ABORT(msg) do { fprintf(stderr, "LLAMA_ABORT: %s\n", msg); abort(); } while(0)
#endif

// Global enforcement state (thread-local would be better for multi-context scenarios)
// Note: Reserved for future use - decode boundary state tracking
// static llama_decode_boundary_state g_decode_boundary_state = {
//     false,    // is_decode_active
//     false,    // graph_frozen
//     0,        // primary_gpu_backend
//     0,        // frozen_graph_hash
//     nullptr,  // frozen_backend_assignments
//     0         // n_frozen_nodes
// };

/**
 * Initialize decode boundary enforcement state
 */
void llama_decode_boundary_init(llama_decode_boundary_state * state) {
    if (!state) return;

    state->is_decode_active = false;
    state->graph_frozen = false;
    state->primary_gpu_backend = 0;
    state->frozen_graph_hash = 0;
    state->frozen_backend_assignments = nullptr;
    state->n_frozen_nodes = 0;
}

/**
 * Activate decode mode
 */
void llama_decode_boundary_activate(llama_decode_boundary_state * state, int gpu_backend_id) {
    if (!state) return;

    state->is_decode_active = true;
    state->primary_gpu_backend = gpu_backend_id;
    state->graph_frozen = false;
    state->frozen_backend_assignments = nullptr;
    state->n_frozen_nodes = 0;

    LLAMA_LOG_INFO("DECODE BOUNDARY: Activated (GPU backend = %d)\n", gpu_backend_id);
}

/**
 * Deactivate decode mode
 */
void llama_decode_boundary_deactivate(llama_decode_boundary_state * state) {
    if (!state) return;

    state->is_decode_active = false;
    state->graph_frozen = false;
    state->frozen_backend_assignments = nullptr;
    state->n_frozen_nodes = 0;

    LLAMA_LOG_INFO("DECODE BOUNDARY: Deactivated\n");
}

/**
 * [CRITICAL] Enforce all nodes are GPU-assigned during decode
 *
 * If decode is active and any node is not on GPU backend, abort immediately.
 * This prevents silent reassignments to CPU during scheduling.
 */
bool llama_decode_boundary_enforce_all_gpu(
    const llama_decode_boundary_state * state,
    const int * node_backend_ids,
    int n_nodes) {

    if (!state || !node_backend_ids || !state->is_decode_active) {
        return true;
    }

    for (int i = 0; i < n_nodes; i++) {
        int backend_id = node_backend_ids[i];

        // In decode mode, ONLY GPU backend (primary_gpu_backend) is allowed
        if (backend_id != state->primary_gpu_backend) {
            LLAMA_LOG_ERROR(
                "DECODE BOUNDARY ERROR: Node %d assigned to backend %d (expected %d - GPU only).\n"
                "  Decode graphs must be 100%% GPU-resident.\n"
                "  No CPU ops, no fallbacks, no exceptions.\n",
                i, backend_id, state->primary_gpu_backend);

            LLAMA_ABORT("Decode node CPU assignment detected\n");
            return false;
        }
    }

    return true;
}

/**
 * [CRITICAL] Freeze graph backend topology
 *
 * Once frozen, no reassignments are allowed.
 * Validates pre-conditions:
 * - All nodes are GPU-assigned
 * - Graph topology hasn't changed
 */
bool llama_decode_boundary_freeze_graph(
    llama_decode_boundary_state * state,
    const int * node_backend_ids,
    int n_nodes,
    uint64_t graph_hash) {

    if (!state || !node_backend_ids) {
        LLAMA_LOG_ERROR("DECODE BOUNDARY: Null state/assignments on freeze\n");
        return false;
    }

    if (!state->is_decode_active) {
        // Not in decode mode, no freezing needed
        return true;
    }

    // Pre-condition: all nodes must be GPU-assigned
    if (!llama_decode_boundary_enforce_all_gpu(state, node_backend_ids, n_nodes)) {
        LLAMA_LOG_ERROR("DECODE BOUNDARY: Cannot freeze graph with non-GPU nodes\n");
        return false;
    }

    // Store frozen state
    state->frozen_backend_assignments = node_backend_ids;
    state->n_frozen_nodes = n_nodes;
    state->frozen_graph_hash = graph_hash;
    state->graph_frozen = true;

    LLAMA_LOG_INFO(
        "DECODE BOUNDARY: Graph frozen with %d GPU-exclusive nodes (hash=%lx)\n",
        n_nodes, graph_hash);

    return true;
}

/**
 * [CRITICAL] Validate immutability after freeze
 *
 * Aborts if any node's backend assignment has changed since freeze.
 * This prevents the graph from being silently re-optimized or re-scheduled.
 */
bool llama_decode_boundary_validate_immutable(
    const llama_decode_boundary_state * state,
    const int * current_backend_ids,
    int n_nodes) {

    if (!state || !state->graph_frozen || !state->frozen_backend_assignments) {
        return true;  // Not frozen or not in decode, no validation needed
    }

    if (n_nodes != state->n_frozen_nodes) {
        LLAMA_LOG_ERROR(
            "DECODE BOUNDARY: Graph topology changed! Frozen %d nodes, now %d\n",
            state->n_frozen_nodes, n_nodes);
        LLAMA_ABORT("Decode graph topology mutation detected\n");
        return false;
    }

    // Check each node's backend assignment
    for (int i = 0; i < n_nodes; i++) {
        int frozen = state->frozen_backend_assignments[i];
        int current = current_backend_ids[i];

        if (frozen != current) {
            LLAMA_LOG_ERROR(
                "DECODE BOUNDARY: Node %d backend changed from %d to %d!\n"
                "  Decode graphs are immutable after freeze.\n"
                "  No re-optimization, no re-scheduling allowed.\n",
                i, frozen, current);
            LLAMA_ABORT("Decode graph backend reassignment detected\n");
            return false;
        }
    }

    return true;
}

/**
 * [CRITICAL] Prevent implicit backend bridging
 *
 * Implicit bridging = automatic device transfer when tensor and op are on different backends.
 * In decode mode, this is a hard error - no silent copies allowed.
 */
bool llama_decode_boundary_check_no_bridging(
    const llama_decode_boundary_state * state,
    int tensor_backend,
    int op_backend) {

    if (!state || !state->is_decode_active) {
        return true;  // Not in decode, bridging allowed
    }

    // In decode, tensor and op backends must match
    if (tensor_backend != op_backend) {
        LLAMA_LOG_ERROR(
            "DECODE BOUNDARY: Implicit bridging detected!\n"
            "  Tensor backend: %d\n"
            "  Op backend: %d\n"
            "  Decode forbids automatic device transfers.\n",
            tensor_backend, op_backend);
        LLAMA_ABORT("Implicit backend bridging in decode mode\n");
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Validate tensor residency for GPU-scheduled ops
 *
 * If a tensor is assigned to GPU backend, its buffer must be GPU-allocated.
 * Catches host-aliased tensors and implicit host materialization.
 */
bool llama_decode_boundary_validate_tensor_residency(
    const llama_decode_boundary_state * state,
    struct ggml_tensor * tensor,
    int assigned_backend) {

    if (!tensor || !state || !state->is_decode_active) {
        return true;
    }

    // Only validate GPU-assigned ops
    if (assigned_backend != state->primary_gpu_backend) {
        return true;
    }

    // Check that tensor buffer is GPU-allocated
    if (!tensor->buffer) {
        LLAMA_LOG_ERROR(
            "DECODE BOUNDARY: Tensor '%s' has no buffer but assigned to GPU!\n"
            "  Possible host-aliasing or uninitialized allocation.\n",
            tensor->name ? tensor->name : "unnamed");
        return false;
    }

    if (!tensor->buffer) {
        LLAMA_LOG_ERROR(
            "DECODE BOUNDARY: Tensor '%s' has no buffer!\n",
            tensor->name ? tensor->name : "unnamed");
        return false;
    }

    // Check buffer is not host memory
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        LLAMA_LOG_ERROR(
            "DECODE BOUNDARY: Tensor '%s' is GPU-scheduled but host-resident!\n"
            "  Implicit CPU materialization detected.\n",
            tensor->name ? tensor->name : "unnamed");
        LLAMA_ABORT("GPU-scheduled tensor is CPU-resident\n");
        return false;
    }

    return true;
}

/**
 * Operations that must have GPU implementations during decode
 */
static const struct {
    const char * op_name;
    bool is_critical;
} DECODE_CRITICAL_OPS[] = {
    // Attention - absolutely critical
    {"mul_mat",         true},    // Dense matmul in attention
    {"mul_mat_q",       true},    // Quantized matmul in attention
    {"mul_mat_id_q",    true},    // Quantized MoE matmul
    {"soft_max",        true},    // Attention softmax

    // Normalization - critical
    {"rms_norm",        true},    // RMSNorm layers
    {"norm",            true},    // Layer normalization
    {"group_norm",      true},    // Group norm

    // FFN - critical
    {"gelu",            true},    // Activation function
    {"silu",            true},    // SwiGLU/SiLU
    {"relu",            true},    // ReLU
    {"relu_sqr",        true},    // Squared ReLU

    // KV cache - critical
    {"get_rows",        true},    // KV cache retrieval
    {"set_rows",        true},    // KV cache updates

    // Sampling - critical
    {"sample_top_k",    true},
    {"sample_top_p",    true},
    {"sample_temperature", true},

    {nullptr, false}
};

/**
 * Check if operation is decode-critical
 */
static bool is_decode_critical_op(const char * op_name) {
    if (!op_name) return false;

    for (int i = 0; DECODE_CRITICAL_OPS[i].op_name != nullptr; i++) {
        if (strcmp(op_name, DECODE_CRITICAL_OPS[i].op_name) == 0) {
            return DECODE_CRITICAL_OPS[i].is_critical;
        }
    }
    return false;
}

/**
 * [CRITICAL] Reject CPU fallback micro-ops
 *
 * If an operation doesn't have GPU support, we must fail at graph build time.
 * No "fallback to CPU" allowed during decode.
 */
bool llama_decode_boundary_reject_fallback_ops(
    enum ggml_op op_type,
    int requested_backend,
    const llama_decode_boundary_state * state) {

    if (!state || !state->is_decode_active) {
        return true;  // Not in decode, fallbacks allowed
    }

    const char * op_name = ggml_op_name(op_type);
    if (!op_name) {
        return true;
    }

    // Check if this is a decode-critical op
    if (!is_decode_critical_op(op_name)) {
        return true;  // Not critical, fallback allowed
    }

    // Decode-critical ops MUST run on GPU
    if (requested_backend != state->primary_gpu_backend) {
        LLAMA_LOG_ERROR(
            "DECODE BOUNDARY: Critical op '%s' cannot execute on backend %d\n"
            "  Required: GPU backend %d\n"
            "  Decode forbids CPU fallback for critical operations.\n",
            op_name, requested_backend, state->primary_gpu_backend);
        LLAMA_ABORT("CPU fallback for decode-critical op\n");
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Audit all tensors are GPU-resident
 *
 * Starting from the decode output tensor, traverse the full computation graph
 * and validate every tensor is GPU-allocated.
 */
bool llama_decode_boundary_audit_all_gpu_resident(
    const llama_decode_boundary_state * state,
    struct ggml_tensor * root_tensor) {

    if (!root_tensor || !state || !state->is_decode_active) {
        return true;
    }

    // BFS to find all reachable tensors
    std::queue<struct ggml_tensor *> q;
    std::unordered_set<struct ggml_tensor *> visited;

    q.push(root_tensor);
    visited.insert(root_tensor);

    while (!q.empty()) {
        struct ggml_tensor * t = q.front();
        q.pop();

        // Validate this tensor is GPU-resident
        if (t->buffer) {
            if (ggml_backend_buffer_is_host(t->buffer)) {
                LLAMA_LOG_ERROR(
                    "DECODE BOUNDARY: CPU-resident tensor found in decode graph!\n"
                    "  Tensor: '%s'\n"
                    "  All decode tensors must be GPU-allocated.\n",
                    t->name ? t->name : "unnamed");
                LLAMA_ABORT("CPU tensor in GPU decode graph\n");
                return false;
            }
        } else if (t->data != NULL) {
            // Tensor has data but no buffer - suspicious (host-aliased?)
            LLAMA_LOG_WARN(
                "DECODE BOUNDARY: Tensor '%s' is host-aliased (data without buffer)!\n",
                t->name ? t->name : "unnamed");
        }

        // Enqueue source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            struct ggml_tensor * src = t->src[i];
            if (src && visited.find(src) == visited.end()) {
                visited.insert(src);
                q.push(src);
            }
        }
    }

    LLAMA_LOG_INFO("DECODE BOUNDARY: Audited %zu tensors - all GPU-resident\n", visited.size());
    return true;
}

/**
 * [CRITICAL] Audit no implicit device transfers
 *
 * Device transfers (ggml_backend_tensor_copy) should only happen before decode,
 * not during token generation. This validates the tensor hasn't been subject to
 * post-allocation transfers.
 */
bool llama_decode_boundary_audit_no_implicit_transfers(
    const llama_decode_boundary_state * state,
    struct ggml_tensor * tensor) {

    if (!tensor || !state || !state->is_decode_active) {
        return true;
    }

    // Check for host-aliased tensors (data without buffer = manual management)
    if (tensor->data != NULL && tensor->buffer == NULL) {
        LLAMA_LOG_WARN(
            "DECODE BOUNDARY: Tensor '%s' appears host-aliased (no buffer).\n"
            "  This may indicate manual device transfer management.\n",
            tensor->name ? tensor->name : "unnamed");
        return false;
    }

    return true;
}

/**
 * [DEBUG] Dump all backend assignments
 */
void llama_decode_boundary_dump_assignments(
    const llama_decode_boundary_state * state,
    struct ggml_cgraph * graph,
    const int * backend_assignments,
    int n_nodes) {

    if (!state || !graph || !backend_assignments) {
        return;
    }

    if (!state->is_decode_active) {
        return;
    }

    LLAMA_LOG_INFO("DECODE BOUNDARY: Backend assignments (%d nodes):\n", n_nodes);

    for (int i = 0; i < n_nodes && i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const char * op_name = ggml_op_name(node->op);
        int backend = backend_assignments[i];
        bool is_critical = is_decode_critical_op(op_name);

        LLAMA_LOG_INFO(
            "  [%3d] %-20s (backend=%d, critical=%d)\n",
            i, op_name, backend, is_critical ? 1 : 0);
    }
}
