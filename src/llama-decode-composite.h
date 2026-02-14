#pragma once

/**
 * Decode Composite Operation Enforcement Header
 * 
 * Ensures that composite decode operations (attention, matmul, norm, softmax, etc.)
 * execute entirely on GPU with no mixed CPU/GPU execution paths.
 */

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Enforces single-backend (GPU) ownership for composite operations.
     * 
     * Called during graph building to ensure that:
     * 1. The operation will execute on GPU
     * 2. All sub-components have GPU implementations
     * 3. No CPU fallback is allowed
     * 
     * @param op The tensor operation node
     * @param backend The backend that will execute this op
     * @return true if op is valid for GPU execution, false otherwise (will abort)
     */
    bool ggml_composite_op_enforce_gpu_only(struct ggml_tensor * op, ggml_backend_t backend);

    /**
     * Audits decode graph for CPU fallback micro-ops.
     * 
     * Warns/aborts if:
     * - GPU softmax fallback would be used
     * - GPU norm fallback would be used
     * - GPU quantization decomposition fallback would be used
     * - GPU attention fallback would be used
     * 
     * @param graph The decode computation graph
     * @return true if no problematic fallbacks found
     */
    bool ggml_audit_no_cpu_fallbacks_in_decode(struct ggml_cgraph * graph);

    /**
     * Validates that all nodes in a decode graph are assigned to GPU backend.
     * 
     * @param graph The computation graph
     * @param node_backend_ids Array of backend assignments per node
     * @return true if all nodes are GPU-assigned, false/aborts otherwise
     */
    bool ggml_validate_decode_graph_all_gpu(struct ggml_cgraph * graph, const int * node_backend_ids);

    /**
     * Validates that decode graph backend assignments remain immutable after freeze.
     * 
     * @param graph The computation graph
     * @param current_backend_ids Current backend assignments
     * @param frozen_backend_ids Frozen backend assignments from when graph was frozen
     * @param n_nodes Number of nodes in graph
     * @return true if assignments match, false/aborts if changed
     */
    bool ggml_validate_decode_graph_immutable(struct ggml_cgraph * graph,
                                             const int * current_backend_ids,
                                             const int * frozen_backend_ids,
                                             int n_nodes);

#ifdef __cplusplus
}
#endif
