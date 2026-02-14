#pragma once

/**
 * Decode-Critical Structural Isolation Header
 * 
 * This header exposes enforcement mechanisms to prevent CPU↔GPU op boundary splitting
 * during autonomous token decode. The goal is to ensure:
 * 
 * 1. No single logical decode operation is split across CPU and GPU backends.
 * 2. All intermediate tensors remain GPU-resident.
 * 3. No implicit device transfers occur within an op execution.
 * 4. No fallback micro-ops execute on CPU during decode.
 * 5. The decode graph remains immutable after freezing.
 */

#include <cstdint>
#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Node backend ownership metadata.
     * Stores the single backend that owns a given node during decode.
     */
    typedef struct {
        int32_t node_id;         ///< Index of the node in the graph
        int32_t backend_owner;   ///< Single backend ID that owns this node (typically 0 for GPU)
        bool    is_locked;       ///< True if backend ownership cannot be reassigned
        bool    is_composite;    ///< True if this is a composite op that requires all sub-kernels on GPU
    } llama_decode_node_owner;

    /**
     * Holds frozen backend ownership state for a decode graph.
     * This structure is populated when the decode graph is frozen and used to
     * validate that no backend reassignments occur during decode execution.
     */
    typedef struct {
        llama_decode_node_owner * nodes;
        int32_t n_nodes;
        int32_t primary_backend;  ///< Expected GPU backend ID (typically 0)
        uint64_t graph_hash;      ///< Hash of the graph to detect mutations
        bool is_frozen;           ///< True if this ownership map is locked/permanent
    } llama_decode_graph_ownership;

    /**
     * Validates that a node will execute on a single backend (GPU).
     * Must be called during graph construction for decode ops.
     * 
     * @param backend_owner The backend assigned to this node
     * @param is_decode_critical True if this is a decode-critical op
     * @return True if valid, false if mixed-backend op detected
     */
    bool llama_decode_validate_single_backend_ownership(int32_t backend_owner, bool is_decode_critical);

    /**
     * Freezes the backend ownership state for a decode graph.
     * After this call, any attempt to reassign node backends will fail.
     * 
     * @param ownership Ownership map to freeze
     * @return True if freeze successful, false if inconsistencies detected
     */
    bool llama_decode_freeze_ownership(llama_decode_graph_ownership * ownership);

    /**
     * Checks that a tensor scheduled for GPU execution will remain GPU-resident.
     * Aborts if implicit CPU materialization is detected.
     * 
     * @param tensor The tensor to validate
     * @param backend_owner Expected backend owner
     * @return True if tensor is GPU-resident and has proper ownership
     */
    bool llama_decode_tensor_gpu_resident_check(struct ggml_tensor * tensor, int32_t backend_owner);

    /**
     * Audits all tensors reachable from the decode root for GPU residency.
     * Computes closure and validates all tensors are GPU-allocated.
     * 
     * @param root_tensor The output tensor of the decode graph
     * @return True if all reachable tensors are GPU-resident
     */
    bool llama_decode_validate_all_gpu_resident(struct ggml_tensor * root_tensor);

    /**
     * Prevents implicit backend bridging (auto-copy on backend mismatch).
     * When in decode mode, any backend mismatch will abort instead of auto-copying.
     * 
     * @param tensor_backend Backend where tensor is located
     * @param op_backend Backend where operation will execute
     * @return True if backends match (no bridging needed), false otherwise (abort in decode mode)
     */
    bool llama_decode_check_backend_match_strict(int32_t tensor_backend, int32_t op_backend);

    /**
     * Validates that no per-node device transfer logic executes for a tensor.
     * Searches for implicit ggml_backend_tensor_copy calls and fails if found.
     * 
     * @param tensor The tensor to audit
     * @return True if no implicit transfers found
     */
    bool llama_decode_audit_no_implicit_copies(struct ggml_tensor * tensor);

    /**
     * Enforces that fallback micro-ops do not execute on CPU during decode.
     * Fails if:
     * - CPU softmax fallback would be used
     * - CPU norm fallback would be used
     * - CPU quant decompose fallback would be used
     * - CPU attention fallback would be used
     * 
     * @param node The node being scheduled
     * @param fallback_backend Fallback backend if primary is unavailable
     * @param in_decode_mode Whether we are currently in decode mode
     * @return True if no problematic fallback, false if fallback detected (abort in decode mode)
     */
    bool llama_decode_reject_cpu_fallbacks(struct ggml_tensor * node, int32_t fallback_backend, bool in_decode_mode);

    /**
     * Logs all backend assignments for debugging/audit purposes.
     * At INFO log level, dumps the backend assigned to each node.
     * 
     * @param graph The graph to audit
     * @param ownership Frozen ownership map (may be NULL)
     */
    void llama_decode_dump_backend_assignments(struct ggml_cgraph * graph, const llama_decode_graph_ownership * ownership);

    /**
     * Allocates and initializes a new ownership tracking structure.
     * 
     * @param n_nodes Number of nodes in the graph
     * @return New ownership map (must be freed with llama_decode_free_ownership)
     */
    llama_decode_graph_ownership * llama_decode_alloc_ownership(int32_t n_nodes);

    /**
     * Frees an ownership tracking structure.
     * 
     * @param ownership Ownership map to free
     */
    void llama_decode_free_ownership(llama_decode_graph_ownership * ownership);

#ifdef __cplusplus
}
#endif
