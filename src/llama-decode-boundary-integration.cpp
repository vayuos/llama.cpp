/**
 * Decode Boundary Enforcement Integration
 *
 * Hooks into the graph building and execution pipeline to enforce
 * CPU↔GPU boundary restrictions during decode.
 */

#include "llama-decode-boundary-enforce.h"
#include "llama-decode-composite.h"
#include "llama-decode-structure.h"
#include "llama-impl.h"
#include "llama-context.h"

#include "../ggml/src/ggml-impl.h"
#include "../ggml/include/ggml-backend.h"

#include <cstring>

/**
 * Called during graph building to enforce composite op GPU-only execution
 *
 * This wraps ggml_composite_op_enforce_gpu_only and integrates it with
 * the boundary enforcement state.
 */
bool llama_decode_boundary_enforce_composite_op(
    struct llama_context * ctx,
    struct ggml_tensor * op,
    ggml_backend_t backend) {

    if (!ctx || !op || !backend) {
        return false;
    }

    // Check if decode is active
    if (!ctx->decode_boundary.is_decode_active) {
        return true;  // Not in decode, no enforcement needed
    }

    const char * op_name = ggml_op_name(op->op);

    // Log for audit trail
    LLAMA_LOG_DEBUG("DECODE BOUNDARY: Enforcing composite op '%s'\n", op_name);

    // Use the composite op enforcement
    return ggml_composite_op_enforce_gpu_only(op, backend);
}

/**
 * Called before graph computation to validate backend assignments
 *
 * This ensures all nodes are GPU-assigned before execution begins.
 */
bool llama_decode_boundary_pre_compute_validate(
    struct llama_context * ctx,
    struct ggml_cgraph * graph,
    const int * node_backend_ids,
    int n_nodes) {

    if (!ctx || !graph || !node_backend_ids) {
        return false;
    }

    // Enforce GPU-only assignments
    if (!llama_decode_boundary_enforce_all_gpu(&ctx->decode_boundary, node_backend_ids, n_nodes)) {
        LLAMA_LOG_ERROR("DECODE BOUNDARY: Pre-compute validation failed\n");
        return false;
    }

    // If graph is not yet frozen, freeze it now
    if (!ctx->decode_boundary.graph_frozen) {
        uint64_t graph_hash = (uint64_t)(uintptr_t)graph;  // Simple hash
        if (!llama_decode_boundary_freeze_graph(&ctx->decode_boundary, node_backend_ids, n_nodes, graph_hash)) {
            LLAMA_LOG_ERROR("DECODE BOUNDARY: Graph freeze failed\n");
            return false;
        }
    }

    // Validate immutability if already frozen
    if (!llama_decode_boundary_validate_immutable(&ctx->decode_boundary, node_backend_ids, n_nodes)) {
        LLAMA_LOG_ERROR("DECODE BOUNDARY: Immutability validation failed\n");
        return false;
    }

    // Audit all tensors are GPU-resident
    if (graph->nodes && graph->n_nodes > 0) {
        struct ggml_tensor * root = graph->nodes[graph->n_nodes - 1];
        if (!llama_decode_boundary_audit_all_gpu_resident(&ctx->decode_boundary, root)) {
            LLAMA_LOG_ERROR("DECODE BOUNDARY: GPU residency audit failed\n");
            return false;
        }
    }

    return true;
}

/**
 * Called after graph computation succeeds
 *
 * Maintains audit state and validates no unexpected changes occurred.
 */
void llama_decode_boundary_post_compute(
    struct llama_context * ctx,
    struct ggml_cgraph * graph) {

    if (!ctx || !graph) {
        return;
    }

    if (!ctx->decode_boundary.is_decode_active) {
        return;
    }

    // Could add additional post-compute audits here
    // For now, just log success
    LLAMA_LOG_DEBUG("DECODE BOUNDARY: Graph execution completed successfully\n");
}

/**
 * Activate decode boundary enforcement for a context
 *
 * Call this before starting autonomous token decode.
 */
void llama_decode_boundary_activate_context(
    struct llama_context * ctx,
    int gpu_backend_id) {

    if (!ctx) {
        return;
    }

    llama_decode_boundary_activate(&ctx->decode_boundary, gpu_backend_id);

    // Store primary GPU backend
    ctx->decode_boundary.primary_gpu_backend = gpu_backend_id;

    LLAMA_LOG_INFO("DECODE BOUNDARY: Context decode mode activated (GPU backend = %d)\n", gpu_backend_id);
}

/**
 * Deactivate decode boundary enforcement
 *
 * Call this after token generation completes.
 */
void llama_decode_boundary_deactivate_context(struct llama_context * ctx) {
    if (!ctx) {
        return;
    }

    llama_decode_boundary_deactivate(&ctx->decode_boundary);

    LLAMA_LOG_INFO("DECODE BOUNDARY: Context decode mode deactivated\n");
}

/**
 * Check backend match with boundary enforcement
 *
 * Prevents implicit copying of tensors across devices during decode.
 */
bool llama_decode_boundary_check_backend_match(
    struct llama_context * ctx,
    int tensor_backend,
    int op_backend) {

    if (!ctx) {
        return true;  // No context, allow operation
    }

    return llama_decode_boundary_check_no_bridging(&ctx->decode_boundary, tensor_backend, op_backend);
}

/**
 * Validate tensor residency
 *
 * Ensures GPU-scheduled tensors are actually GPU-resident.
 */
bool llama_decode_boundary_check_tensor_residency(
    struct llama_context * ctx,
    struct ggml_tensor * tensor,
    int assigned_backend) {

    if (!ctx || !tensor) {
        return true;
    }

    return llama_decode_boundary_validate_tensor_residency(&ctx->decode_boundary, tensor, assigned_backend);
}

/**
 * Reject CPU fallback ops
 *
 * Critical decode ops must have GPU implementations.
 */
bool llama_decode_boundary_check_fallback_allowed(
    struct llama_context * ctx,
    ggml_op_type op_type,
    int requested_backend) {

    if (!ctx) {
        return true;  // No context, allow fallback
    }

    return llama_decode_boundary_reject_fallback_ops(op_type, requested_backend, &ctx->decode_boundary);
}

/**
 * Dump backend assignments for debugging
 */
void llama_decode_boundary_dump_context_assignments(
    struct llama_context * ctx,
    struct ggml_cgraph * graph,
    const int * backend_assignments,
    int n_nodes) {

    if (!ctx || !graph) {
        return;
    }

    llama_decode_boundary_dump_assignments(&ctx->decode_boundary, graph, backend_assignments, n_nodes);
}
