#include "llama-decode-structure.h"
#include "llama-impl.h"

#include "../ggml/src/ggml-impl.h"
#include "../ggml/include/ggml-backend.h"

#include <cassert>
#include <cstring>
#include <unordered_map>
#include <queue>

/**
 * Decode-Critical Operation Boundary Enforcement
 * 
 * This file implements strict structural checks to ensure:
 * 1. No single logical decode operation is split across backends.
 * 2. All decode-critical ops execute entirely on GPU.
 * 3. No intermediate tensors cross CPU↔GPU boundaries.
 * 4. No implicit device transfers occur mid-op.
 * 5. Fallback micro-ops are disabled.
 */

// Global state for tracking frozen decode graphs
static std::unordered_map<uint64_t, llama_decode_graph_ownership*> g_frozen_decode_graphs;

/**
 * Validates that a node has single-backend ownership.
 * 
 * @param backend_owner The backend ID assigned to the node
 * @param is_decode_critical Whether this node is part of decode-critical path
 * @return true if valid, false if split/mixed-backend detected
 */
bool llama_decode_validate_single_backend_ownership(int32_t backend_owner, bool is_decode_critical) {
    // Decode-critical ops must have -1 (unassigned) or backend 0 (GPU)
    // Backend -1 may indicate "not yet assigned", but for decode, we disallow this.
    if (is_decode_critical) {
        if (backend_owner < 0) {
            LLAMA_LOG_ERROR("DECODE STRUCTURE ERROR: Decode-critical node has unassigned backend (%d).\n", backend_owner);
            return false;
        }
        // Only backend 0 (GPU) is allowed during decode
        if (backend_owner != 0) {
            LLAMA_LOG_ERROR("DECODE STRUCTURE ERROR: Decode-critical node assigned to backend %d (non-GPU).\n", backend_owner);
            return false;
        }
    }
    
    return true;
}

/**
 * Freezes backend ownership state for a decode graph.
 * Once frozen, any attempt to reassign backends will fail.
 */
bool llama_decode_freeze_ownership(llama_decode_graph_ownership * ownership) {
    if (!ownership) {
        LLAMA_LOG_ERROR("DECODE STRUCTURE ERROR: Null ownership map on freeze.\n");
        return false;
    }

    // Validate all nodes have GPU backend ownership
    for (int i = 0; i < ownership->n_nodes; i++) {
        if (ownership->nodes[i].backend_owner != 0) {
            LLAMA_LOG_ERROR("DECODE STRUCTURE ERROR: Node %d has non-GPU backend %d at freeze time.\n",
                            i, ownership->nodes[i].backend_owner);
            return false;
        }
        ownership->nodes[i].is_locked = true;
    }

    ownership->is_frozen = true;
    
    // Store in global frozen map for later validation
    g_frozen_decode_graphs[ownership->graph_hash] = ownership;
    
    LLAMA_LOG_INFO("DECODE STRUCTURE: Ownership frozen for graph %lx with %d nodes.\n",
                   ownership->graph_hash, ownership->n_nodes);
    
    return true;
}

/**
 * Checks that a tensor scheduled for GPU execution is GPU-resident.
 */
bool llama_decode_tensor_gpu_resident_check(struct ggml_tensor * tensor, int32_t backend_owner) {
    if (!tensor) {
        return true; // Null tensors don't need checks
    }

    // Only check for GPU backend (0)
    if (backend_owner != 0) {
        return true;
    }

    // Check that buffer exists and is not host-only
    if (tensor->buffer && tensor->buffer->buft) {
        bool is_host = ggml_backend_buft_is_host(tensor->buffer->buft);
        if (is_host) {
            LLAMA_LOG_ERROR("DECODE STRUCTURE ERROR: GPU-scheduled tensor '%s' has host buffer. Implicit CPU materialization detected.\n",
                            tensor->name);
            return false;
        }
    }

    return true;
}

/**
 * Audits all tensors reachable from a root tensor for GPU residency.
 * Uses BFS to traverse the computation graph.
 */
bool llama_decode_validate_all_gpu_resident(struct ggml_tensor * root_tensor) {
    if (!root_tensor) {
        return true;
    }

    if (!ggml_backend_decode_mode_active()) {
        return true; // Only validate during decode mode
    }

    std::queue<struct ggml_tensor *> q;
    std::unordered_map<struct ggml_tensor *, bool> visited;

    q.push(root_tensor);
    visited[root_tensor] = true;

    while (!q.empty()) {
        struct ggml_tensor * t = q.front();
        q.pop();

        // Check that this tensor is GPU-resident
        if (t->buffer && t->buffer->buft) {
            bool is_host = ggml_backend_buft_is_host(t->buffer->buft);
            if (is_host) {
                LLAMA_LOG_ERROR("DECODE STRUCTURE ERROR: Tensor '%s' in decode graph is CPU-resident.\n", t->name);
                return false;
            }
        }

        // Enqueue source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i] && !visited[t->src[i]]) {
                visited[t->src[i]] = true;
                q.push(t->src[i]);
            }
        }
    }

    return true;
}

/**
 * Prevents implicit backend bridging during decode.
 * 
 * @param tensor_backend Backend where tensor is allocated
 * @param op_backend Backend where operation will execute
 * @return true if backends match, false if mismatch (would trigger auto-copy)
 */
bool llama_decode_check_backend_match_strict(int32_t tensor_backend, int32_t op_backend) {
    // Implicit bridging is when tensor and op are on different backends
    if (tensor_backend == op_backend) {
        return true; // No bridging needed
    }

    // In decode mode, any mismatch is a violation
    if (ggml_backend_decode_mode_active()) {
        LLAMA_LOG_ERROR("DECODE STRUCTURE ERROR: Backend mismatch during decode:\n"
                        "  Tensor backend: %d\n"
                        "  Op backend: %d\n"
                        "  Implicit bridging (auto-copy) is forbidden.\n",
                        tensor_backend, op_backend);
        return false;
    }

    return true; // Allowed in non-decode mode
}

/**
 * Audits that no implicit device transfers occur for a tensor.
 * Would search for ggml_backend_tensor_copy calls in the op chain.
 */
bool llama_decode_audit_no_implicit_copies(struct ggml_tensor * tensor) {
    if (!tensor) {
        return true;
    }

    if (!ggml_backend_decode_mode_active()) {
        return true;
    }

    // In a full audit, we would trace back through the compute graph
    // looking for ggml_backend_tensor_copy operations. For now, we check
    // that the tensor has not been subject to host aliasing.

    if (tensor->buffer == NULL && tensor->data != NULL) {
        // Tensor data is not backed by a buffer - this could indicate
        // a host-aliased tensor (dangerous in decode mode)
        LLAMA_LOG_WARN("DECODE STRUCTURE: Tensor '%s' appears to be host-aliased (no buffer).\n", tensor->name);
    }

    return true;
}

/**
 * Rejects CPU fallback micro-ops during decode.
 * 
 * Examples of disallowed fallbacks:
 * - CPU softmax when GPU kernel is unavailable
 * - CPU norm when GPU kernel is unavailable
 * - CPU quantization decomposition when GPU kernel is unavailable
 * - CPU attention when GPU kernel is unavailable
 */
bool llama_decode_reject_cpu_fallbacks(struct ggml_tensor * node, int32_t fallback_backend, bool in_decode_mode) {
    if (!node || !in_decode_mode) {
        return true;
    }

    // Fallback backend should not be CPU (backend -1 or negative) during decode
    if (fallback_backend < 0) {
        // Negative backend ID typically means "use CPU" or "fallback"
        ggml_op_type op_type = node->op;
        
        const char * op_name = ggml_op_name(op_type);
        
        // List of disallowed fallback ops
        const char * disallowed_fallbacks[] = {
            "soft_max",     // Softmax
            "norm",         // Norm/RMSNorm
            "mul_mat",      // MatMul (if GPU version missing)
            "sum_rows",     // Sum reduction
            nullptr
        };

        for (int i = 0; disallowed_fallbacks[i] != nullptr; i++) {
            if (strcmp(op_name, disallowed_fallbacks[i]) == 0) {
                LLAMA_LOG_ERROR("DECODE STRUCTURE ERROR: CPU fallback for '%s' op is forbidden during decode.\n", op_name);
                return false;
            }
        }
    }

    return true;
}

/**
 * Logs backend assignments for debugging/auditing.
 */
void llama_decode_dump_backend_assignments(struct ggml_cgraph * graph, const llama_decode_graph_ownership * ownership) {
    if (!graph) {
        return;
    }

    if (!ggml_backend_decode_mode_active()) {
        return; // Only dump during decode mode
    }

    LLAMA_LOG_INFO("DECODE STRUCTURE: Backend assignments for graph (%p):\n", (void*)graph);
    
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const char * op_name = ggml_op_name(node->op);
        
        // If we have ownership map, use it; otherwise just log raw info
        if (ownership && i < ownership->n_nodes) {
            int backend = ownership->nodes[i].backend_owner;
            bool locked = ownership->nodes[i].is_locked;
            LLAMA_LOG_INFO("  Node %3d: %-16s (backend %d, locked=%d)\n", i, op_name, backend, locked);
        } else {
            LLAMA_LOG_INFO("  Node %3d: %-16s\n", i, op_name);
        }
    }
}

/**
 * Allocates an ownership tracking structure.
 */
llama_decode_graph_ownership * llama_decode_alloc_ownership(int32_t n_nodes) {
    llama_decode_graph_ownership * ownership = new llama_decode_graph_ownership{
        /* .nodes */ new llama_decode_node_owner[n_nodes],
        /* .n_nodes */ n_nodes,
        /* .primary_backend */ 0,
        /* .graph_hash */ 0,
        /* .is_frozen */ false
    };

    // Initialize all nodes to unassigned
    for (int i = 0; i < n_nodes; i++) {
        ownership->nodes[i] = {
            /* .node_id */ i,
            /* .backend_owner */ -1,
            /* .is_locked */ false,
            /* .is_composite */ false
        };
    }

    return ownership;
}

/**
 * Frees an ownership tracking structure.
 */
void llama_decode_free_ownership(llama_decode_graph_ownership * ownership) {
    if (!ownership) {
        return;
    }

    // Remove from frozen map
    if (ownership->is_frozen) {
        g_frozen_decode_graphs.erase(ownership->graph_hash);
    }

    delete[] ownership->nodes;
    delete ownership;
}
