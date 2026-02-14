/**
 * Decode Composite Op Enforcement
 * 
 * This file enforces that composite decode operations (attention, matmul, etc.)
 * execute entirely on GPU with no mixed CPU/GPU execution paths.
 */

#include "../ggml/src/ggml-impl.h"
#include "../ggml/include/ggml-backend.h"

#include <cstring>

/**
 * List of composite decode operations that must execute with all sub-kernels on GPU.
 * These operations are normally composed of multiple kernels (e.g., matmul + bias + activation).
 * If any sub-kernel lacks GPU implementation, the entire operation must fail at graph construction time.
 */
static const struct {
    const char * name;
    bool is_composite;
} DECODE_COMPOSITE_OPS[] = {
    // Attention operations
    {"mul_mat_q",       true},   // Quantized matrix multiply (used in attention)
    {"mul_mat_id_q",    true},   // Quantized MoE matrix multiply
    {"mul_mat",         true},   // General matrix multiply (critical in attention)
    
    // Norm operations
    {"norm",            true},   // RMSNorm, LayerNorm must execute on GPU
    {"rms_norm",        true},   // Explicit RMSNorm
    {"group_norm",      true},   // Group normalization
    
    // Softmax (always GPU-resident in decode)
    {"soft_max",        true},   // Softmax in attention
    
    // FFN operations (feedforward network)
    {"gelu",            true},   // Activation must be GPU-resident
    {"silu",            true},   // SwiGLU activation
    {"relu",            true},   // ReLU activation
    
    // Fallback-prone operations 
    {"sum_rows",        true},   // Row-wise sum (used in various places)
    {"repeat",          true},   // Tensor repetition
    
    {nullptr, false}               // Sentinel
};

/**
 * Checks if an operation is in the composite decode list.
 */
static bool is_composite_decode_op(const char * op_name) {
    if (!op_name) return false;
    
    for (int i = 0; DECODE_COMPOSITE_OPS[i].name != nullptr; i++) {
        if (strcmp(op_name, DECODE_COMPOSITE_OPS[i].name) == 0) {
            return DECODE_COMPOSITE_OPS[i].is_composite;
        }
    }
    
    return false;
}

/**
 * Enforces single-backend ownership for composite ops.
 * 
 * Called during graph building to ensure that:
 * 1. The operation will execute on GPU (backend 0)
 * 2. All sub-components have GPU implementations
 * 3. No CPU fallback is allowed
 */
extern "C" bool ggml_composite_op_enforce_gpu_only(struct ggml_tensor * op, ggml_backend_t backend) {
    if (!op || !backend) {
        return false;
    }
    
    // Only enforce for decode mode
    if (!ggml_get_decode_mode()) {
        return true;
    }
    
    const char * op_name = ggml_op_name(op->op);
    
    // Check if this is a composite op that needs strict enforcement
    if (!is_composite_decode_op(op_name)) {
        return true; // Not a composite op, no special enforcement
    }
    
    // For composite ops in decode mode, verify GPU support
    if (!ggml_backend_supports_op(backend, op)) {
        GGML_ABORT("DECODE STRUCTURE ERROR: Composite op '%s' has no GPU implementation.\n"
                   "  Backend: %s\n"
                   "  Composite operations must execute entirely on GPU during decode.\n"
                   "  CPU fallback is forbidden.\n",
                   op_name, ggml_backend_name(backend));
    }
    
    // Verify all input tensors are GPU-resident
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * src = op->src[i];
        if (!src) continue;
        
        // Check buffer residency
        if (src->buffer && ggml_backend_buffer_is_host(src->buffer)) {
            GGML_ABORT("DECODE STRUCTURE ERROR: Composite op '%s' has host-resident input '%s'.\n"
                       "  All inputs to GPU composite ops must be GPU-resident during decode.\n",
                       op_name, src->name ? src->name : "unnamed");
        }
    }
    
    return true;
}

/**
 * Audit for CPU fallback micro-ops during decode.
 * 
 * Searches for problematic fallback patterns:
 * 1. Is there a CPU softmax fallback when GPU version is unavailable?
 * 2. Is there a CPU norm fallback?
 * 3. Is there a CPU quantization decomposition?
 * 4. Is there a CPU attention fallback?
 */
extern "C" bool ggml_audit_no_cpu_fallbacks_in_decode(struct ggml_cgraph * graph) {
    if (!graph || !ggml_get_decode_mode()) {
        return true;
    }
    
    // Fallback-prone ops that should have GPU implementations
    const struct {
        const char * op_name;
        const char * description;
    } FALLBACK_RISKS[] = {
        {"soft_max",    "Softmax fallback"},
        {"norm",        "Norm/RMSNorm fallback"},
        {"mul_mat_q",   "Quantized GEMM decompose fallback"},
        {"gelu",        "GELU/Activation fallback"},
        {nullptr, nullptr}
    };
    
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const char * op_name = ggml_op_name(node->op);
        
        // Check if this node is a known fallback risk
        for (int j = 0; FALLBACK_RISKS[j].op_name != nullptr; j++) {
            if (strcmp(op_name, FALLBACK_RISKS[j].op_name) == 0) {
                // This node could potentially fallback to CPU
                // We need to verify that a GPU backend explicitly supports it
                
                // Since we can't directly query the current backend assignment here,
                // we at least warn in debug builds
                GGML_LOG_WARN("DECODE AUDIT: Op '%s' (%s) marked as potential fallback risk.\n",
                              op_name, FALLBACK_RISKS[j].description);
                break;
            }
        }
    }
    
    return true;
}

/**
 * Validates that all nodes in a decode graph have been assigned to GPU backend.
 */
extern "C" bool ggml_validate_decode_graph_all_gpu(struct ggml_cgraph * graph, 
                                                   const int * node_backend_ids) {
    if (!graph || !node_backend_ids || !ggml_get_decode_mode()) {
        return true;
    }
    
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int backend_id = node_backend_ids[i];
        
        // All nodes should be on backend 0 (GPU) during decode
        if (backend_id != 0) {
            const char * op_name = ggml_op_name(node->op);
            GGML_ABORT("DECODE STRUCTURE ERROR: Non-GPU node detected in decode graph.\n"
                       "  Node: '%s' (%s)\n"
                       "  Assigned backend: %d (expected 0 for GPU)\n"
                       "  Decode graphs must be GPU-exclusive.\n",
                       node->name ? node->name : "unnamed",
                       op_name,
                       backend_id);
        }
    }
    
    return true;
}

/**
 * Validates graph immutability after freeze.
 * Aborts if the graph topology or backend assignments have changed.
 */
extern "C" bool ggml_validate_decode_graph_immutable(struct ggml_cgraph * graph,
                                                     const int * current_backend_ids,
                                                     const int * frozen_backend_ids,
                                                     int n_nodes) {
    if (!ggml_get_decode_mode() || !frozen_backend_ids) {
        return true;  // Not in decode mode or no frozen state to compare
    }
    
    // Check for changes in backend assignments
    for (int i = 0; i < n_nodes; i++) {
        if (current_backend_ids[i] != frozen_backend_ids[i]) {
            struct ggml_tensor * node = graph->nodes[i];
            GGML_ABORT("DECODE STRUCTURE VIOLATION: Backend assignment changed for node %d ('%s').\n"
                       "  Previous backend: %d\n"
                       "  Current backend: %d\n"
                       "  Decode graphs are immutable after freeze.\n",
                       i,
                       node->name ? node->name : "unnamed",
                       frozen_backend_ids[i],
                       current_backend_ids[i]);
        }
    }
    
    return true;
}
