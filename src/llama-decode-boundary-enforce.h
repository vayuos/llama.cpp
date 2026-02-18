#pragma once

/**
 * Decode Critical Boundary Enforcement
 *
 * Strict structural enforcement to prevent CPU↔GPU op boundary splitting.
 * This ensures all decode-critical operations execute entirely on GPU
 * with no intermediate tensors crossing device boundaries.
 */

#include "../ggml/include/ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * State tracking for decode boundary enforcement
 */
typedef struct {
    bool            is_decode_active;          // True when in autonomous decode phase
    bool            graph_frozen;              // True after decode graph is frozen
    int             primary_gpu_backend;       // GPU backend ID (typically 0)
    uint64_t        frozen_graph_hash;         // Hash of frozen graph for mutation detection
    const int *     frozen_backend_assignments; // Frozen backend assignments per node
    int             n_frozen_nodes;            // Number of nodes in frozen graph
} llama_decode_boundary_state;

/**
 * Initialize decode boundary enforcement state
 */
void llama_decode_boundary_init(llama_decode_boundary_state * state);

/**
 * Activate decode mode and set primary GPU backend
 * After this, strict enforcement begins
 */
void llama_decode_boundary_activate(llama_decode_boundary_state * state, int gpu_backend_id);

/**
 * Deactivate decode mode (end of token generation)
 */
void llama_decode_boundary_deactivate(llama_decode_boundary_state * state);

/**
 * [CRITICAL] Enforce single-backend ownership during graph scheduling
 *
 * When scheduling operations in decode mode:
 * - All ops must be assigned to GPU backend (0)
 * - No CPU reassignment allowed mid-decode
 * - Aborts if backend mismatch detected
 *
 * @param state Boundary state
 * @param node_backend_ids Array of backend assignments (one per node)
 * @param n_nodes Number of nodes
 * @return true if all nodes GPU-assigned, false/abort otherwise
 */
bool llama_decode_boundary_enforce_all_gpu(
    const llama_decode_boundary_state * state,
    const int * node_backend_ids,
    int n_nodes);

/**
 * [CRITICAL] Freeze graph backend topology to prevent reassignments
 *
 * Once called, no node can change backends until decode completes.
 * Validates all nodes are GPU-resident before freezing.
 *
 * @param state Boundary state (will mark graph_frozen = true)
 * @param node_backend_ids Current backend assignments
 * @param n_nodes Number of nodes
 * @param graph_hash Hash of the graph topology for mutation detection
 * @return true if freeze successful, false/abort if inconsistencies
 */
bool llama_decode_boundary_freeze_graph(
    llama_decode_boundary_state * state,
    const int * node_backend_ids,
    int n_nodes,
    uint64_t graph_hash);

/**
 * [CRITICAL] Validate backend assignments haven't changed since freeze
 *
 * Called before each graph execution to ensure immutability.
 * Aborts if any node's backend has been reassigned.
 *
 * @param state Boundary state with frozen assignments
 * @param current_backend_ids Current backend assignments
 * @param n_nodes Number of nodes
 * @return true if all assignments match frozen state
 */
bool llama_decode_boundary_validate_immutable(
    const llama_decode_boundary_state * state,
    const int * current_backend_ids,
    int n_nodes);

/**
 * [CRITICAL] Prevent implicit backend bridging (auto-copy on mismatch)
 *
 * When tensor is on one backend and op runs on another:
 * - In normal mode: implicit copy is allowed
 * - In decode mode: immediately abort
 *
 * No silent device transfers during decode!
 *
 * @param state Boundary state
 * @param tensor_backend Backend where tensor is allocated
 * @param op_backend Backend where operation will execute
 * @return true if backends match (no copy needed), false/abort if mismatch in decode
 */
bool llama_decode_boundary_check_no_bridging(
    const llama_decode_boundary_state * state,
    int tensor_backend,
    int op_backend);

/**
 * [CRITICAL] Validate tensor is GPU-resident for GPU-scheduled operations
 *
 * If an op is scheduled for GPU backend but its input tensor is CPU-resident,
 * this catches the violation immediately.
 *
 * @param state Boundary state
 * @param tensor Tensor to validate
 * @param assigned_backend Backend this tensor's op is assigned to
 * @return true if tensor is properly allocated for assigned backend
 */
bool llama_decode_boundary_validate_tensor_residency(
    const llama_decode_boundary_state * state,
    struct ggml_tensor * tensor,
    int assigned_backend);

/**
 * [CRITICAL] Reject CPU fallback micro-ops during decode
 *
 * Operations like softmax, norm, matmul must have GPU implementations.
 * If an op lacks GPU support, graph construction must fail immediately.
 * No "fallback to CPU" allowed.
 *
 * @param op_type The operation type
 * @param requested_backend Backend requested for this op
 * @param state Boundary state (to check if decode active)
 * @return true if op can execute on requested backend, false/abort otherwise
 */
bool llama_decode_boundary_reject_fallback_ops(
    enum ggml_op op_type,
    int requested_backend,
    const llama_decode_boundary_state * state);

/**
 * [CRITICAL] Audit all tensors in decode path are GPU-resident
 *
 * Computes transitive closure of all tensors reachable from decode root.
 * Validates each one is allocated on GPU buffer.
 *
 * @param state Boundary state
 * @param root_tensor Output tensor of decode graph
 * @return true if all reachable tensors are GPU-allocated
 */
bool llama_decode_boundary_audit_all_gpu_resident(
    const llama_decode_boundary_state * state,
    struct ggml_tensor * root_tensor);

/**
 * [CRITICAL] Prevent per-node device transfers (ggml_backend_tensor_copy)
 *
 * Device transfers should only happen during pre-decode phase, not during compute.
 * This validates no implicit transfer operations are inserted.
 *
 * @param state Boundary state
 * @param tensor Tensor to validate
 * @return true if no transfer operations found
 */
bool llama_decode_boundary_audit_no_implicit_transfers(
    const llama_decode_boundary_state * state,
    struct ggml_tensor * tensor);

/**
 * [DEBUG] Dump all backend assignments for auditing
 */
void llama_decode_boundary_dump_assignments(
    const llama_decode_boundary_state * state,
    struct ggml_cgraph * graph,
    const int * backend_assignments,
    int n_nodes);

#ifdef __cplusplus
}
#endif
