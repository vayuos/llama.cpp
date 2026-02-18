#pragma once

/**
 * GGML Decode Enforcement Layer
 *
 * Low-level enforcement hooks for the ggml backend dispatch system.
 * Prevents implicit device transfers and ensures GPU-exclusive execution.
 */

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * [CRITICAL] Check for implicit tensor copy during backend mismatch
 *
 * When tensor is on backend A and operation runs on backend B:
 * - Normal mode: copy is performed implicitly
 * - Decode mode: immediately abort
 *
 * Call before ggml_backend_tensor_copy to enforce decode restrictions.
 *
 * @param tensor_backend Backend where tensor is allocated
 * @param op_backend Backend where operation will execute
 * @param in_decode_mode Whether autonomous decode is active
 * @return true if copy is allowed, false if forbidden in decode mode
 */
bool ggml_decode_check_implicit_copy(int tensor_backend, int op_backend, bool in_decode_mode);

/**
 * [CRITICAL] Enforce backend assignment is immutable during decode
 *
 * Once a node's backend is assigned during decode graph freeze,
 * it cannot be changed. Any re-assignment attempt aborts.
 *
 * @param node_id Index of node in graph
 * @param previous_backend Backend assigned at freeze time
 * @param new_backend Backend being reassigned now
 * @param in_decode_mode Whether decode is active
 * @return true if assignment is valid, false/abort if changed
 */
bool ggml_decode_check_backend_immutable(
    int node_id,
    int previous_backend,
    int new_backend,
    bool in_decode_mode);

/**
 * [CRITICAL] Validate tensor buffer residency
 *
 * GPU-scheduled tensor must be in GPU buffer, not CPU/host memory.
 *
 * @param tensor Tensor to validate
 * @param expected_backend Backend tensor is assigned to
 * @param in_decode_mode Whether decode is active
 * @return true if tensor buffer matches expected backend
 */
bool ggml_decode_check_tensor_buffer_residency(
    struct ggml_tensor * tensor,
    int expected_backend,
    bool in_decode_mode);

/**
 * [CRITICAL] Reject CPU implementations of GPU-critical ops
 *
 * During decode, these operations MUST have GPU implementations:
 * - mul_mat, mul_mat_q (matmul)
 * - rms_norm, norm (normalization)
 * - soft_max (attention softmax)
 * - gelu, silu (activation)
 * - get_rows, set_rows (KV cache access)
 *
 * If GPU implementation missing → abort at graph build time.
 *
 * @param op_name Name of operation
 * @param backend_available Backend that would execute this op
 * @param in_decode_mode Whether decode is active
 * @return true if GPU implementation exists or not critical
 */
bool ggml_decode_check_op_has_gpu_impl(
    const char * op_name,
    int backend_available,
    bool in_decode_mode);

/**
 * [CRITICAL] Log all nodes scheduled for CPU during decode
 *
 * For debugging: identifies which ops were incorrectly scheduled to CPU.
 * Aborts if any CPU nodes found in decode mode.
 *
 * @param graph The computation graph
 * @param backend_ids Array of backend assignments per node
 * @param n_nodes Number of nodes
 * @param in_decode_mode Whether decode is active
 * @return false if any CPU nodes found in decode mode
 */
bool ggml_decode_check_no_cpu_nodes(
    struct ggml_cgraph * graph,
    const int * backend_ids,
    int n_nodes,
    bool in_decode_mode);

/**
 * [CRITICAL] Disable dynamic backend selection during decode
 *
 * During decode graph freeze, backends are fixed.
 * No adaptive re-optimization allowed during execution.
 *
 * @param current_backend Current backend selection
 * @param previous_backend Backend selected at freeze time
 * @param in_decode_mode Whether decode is active
 * @return false if backend selection changed and decode is active
 */
bool ggml_decode_check_backend_stable(
    int current_backend,
    int previous_backend,
    bool in_decode_mode);

#ifdef __cplusplus
}
#endif
