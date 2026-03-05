#pragma once

/**
 * Decode Boundary Enforcement Integration
 *
 * Context-aware wrappers for boundary enforcement that integrate with
 * the graph building and execution pipeline.
 */

#include "../ggml/include/ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_context;
struct ggml_tensor;
struct ggml_cgraph;

/**
 * Enforce composite op GPU-only execution during graph building
 */
bool llama_decode_boundary_enforce_composite_op(
    struct llama_context * ctx,
    struct ggml_tensor * op,
    ggml_backend_t backend);

/**
 * Pre-compute validation: ensure all nodes are GPU-assigned and graph is frozen
 */
bool llama_decode_boundary_pre_compute_validate(
    struct llama_context * ctx,
    struct ggml_cgraph * graph,
    const int * node_backend_ids,
    int n_nodes);

/**
 * Post-compute actions after successful graph execution
 */
void llama_decode_boundary_post_compute(
    struct llama_context * ctx,
    struct ggml_cgraph * graph);

/**
 * Activate decode boundary enforcement for a context
 */
void llama_decode_boundary_activate_context(
    struct llama_context * ctx,
    int gpu_backend_id);

/**
 * Deactivate decode boundary enforcement
 */
void llama_decode_boundary_deactivate_context(struct llama_context * ctx);

/**
 * Check backend match during tensor-op scheduling
 */
bool llama_decode_boundary_check_backend_match(
    struct llama_context * ctx,
    int tensor_backend,
    int op_backend);

/**
 * Validate tensor is properly allocated for its assigned backend
 */
bool llama_decode_boundary_check_tensor_residency(
    struct llama_context * ctx,
    struct ggml_tensor * tensor,
    int assigned_backend);

/**
 * Check if CPU fallback is allowed for this op
 */
bool llama_decode_boundary_check_fallback_allowed(
    struct llama_context * ctx,
    ggml_op_type op_type,
    int requested_backend);

/**
 * Dump backend assignments for debugging
 */
void llama_decode_boundary_dump_context_assignments(
    struct llama_context * ctx,
    struct ggml_cgraph * graph,
    const int * backend_assignments,
    int n_nodes);

#ifdef __cplusplus
}
#endif
