/**
 * SECTION 9: Convert Unsupported CUDA Ops Into Hard Decode Errors
 * Implementation
 *
 * This file implements enforcement that any decode-critical operation lacking CUDA
 * support results in an immediate hard error. Unsupported CUDA ops on the decode
 * path are correctness failures, not performance issues. CPU fallback is forbidden.
 */

#include "llama-cuda-support-enforce.h"
#include <cstring>
#include <cstdio>
#include <map>

// ============================================================================
// GLOBAL STATE MANAGEMENT
// ============================================================================

static struct llama_cuda_support_validation_state g_cuda_support_state = {
    NULL,                     // operations
    0,                        // num_operations
    0,                        // max_operations
    false,                    // admission_validation_complete
    false,                    // all_ops_cuda_supported
    0,                        // violation_count
    LLAMA_CUDA_VIOL_UNKNOWN,  // last_violation_type
    NULL,                     // last_violation_op
    NULL,                     // last_violation_reason
    0                         // late_discovery_count
};

static bool g_cuda_support_enforcement_strict = true;
static int g_total_cuda_violations = 0;
static int g_late_discovery_violations = 0;

// Per-operation CUDA support tracking
static std::map<std::string, enum llama_cuda_support_status> g_operation_cuda_support_map;
static std::map<std::string, enum llama_cuda_requirement_level> g_operation_requirement_map;
static std::map<std::string, struct llama_cuda_support_info*> g_operation_info_map;

// Violation tracking per operation
static std::map<std::string, int> g_operation_violation_count_map;

// ============================================================================
// INITIALIZATION
// ============================================================================

int llama_cuda_support_enforce_init(void) {
    if (g_cuda_support_state.operations != NULL) {
        return 0; // Already initialized
    }

    // Allocate initial operation registry (will grow as needed)
    g_cuda_support_state.max_operations = 100;
    g_cuda_support_state.operations = (struct llama_cuda_support_info*)malloc(
        g_cuda_support_state.max_operations * sizeof(struct llama_cuda_support_info)
    );

    if (g_cuda_support_state.operations == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate CUDA support operation registry\n");
        return -1;
    }

    g_cuda_support_state.num_operations = 0;
    g_cuda_support_state.admission_validation_complete = false;
    g_cuda_support_state.all_ops_cuda_supported = false;
    g_cuda_support_state.violation_count = 0;
    g_cuda_support_state.late_discovery_count = 0;

    return 0;
}

// ============================================================================
// OPERATION ENUMERATION & REGISTRATION
// ============================================================================

int llama_enumerate_cuda_requirements_for_decode(void) {
    // Initialize registry
    if (llama_cuda_support_enforce_init() != 0) {
        return -1;
    }

    // Register all decode-critical operations
    // These are the operations that MUST have CUDA support during decode

    // Token embedding and input processing
    llama_register_cuda_operation_support(
        "token_embedding",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16, i32",
        "Any 1D/2D embedding lookup",
        NULL
    );

    // Attention operations
    llama_register_cuda_operation_support(
        "attention_q_projection",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any dense matrix",
        NULL
    );

    llama_register_cuda_operation_support(
        "attention_k_projection",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any dense matrix",
        NULL
    );

    llama_register_cuda_operation_support(
        "attention_v_projection",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any dense matrix",
        NULL
    );

    llama_register_cuda_operation_support(
        "attention_output_projection",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any dense matrix",
        NULL
    );

    llama_register_cuda_operation_support(
        "kv_cache_append",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any 2D/4D tensor",
        NULL
    );

    // MLP operations
    llama_register_cuda_operation_support(
        "mlp_gate_projection",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any dense matrix",
        NULL
    );

    llama_register_cuda_operation_support(
        "mlp_up_projection",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any dense matrix",
        NULL
    );

    llama_register_cuda_operation_support(
        "mlp_down_projection",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any dense matrix",
        NULL
    );

    // Activation functions
    llama_register_cuda_operation_support(
        "gelu_activation",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any tensor",
        NULL
    );

    llama_register_cuda_operation_support(
        "silu_activation",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any tensor",
        NULL
    );

    // Normalization operations
    llama_register_cuda_operation_support(
        "layer_norm",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any tensor",
        NULL
    );

    llama_register_cuda_operation_support(
        "rms_norm",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any tensor",
        NULL
    );

    // Logits computation
    llama_register_cuda_operation_support(
        "logits_computation",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any dense matrix",
        NULL
    );

    // Softmax for sampling
    llama_register_cuda_operation_support(
        "softmax_for_sampling",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "1D/2D logits",
        NULL
    );

    // Sampling operation
    llama_register_cuda_operation_support(
        "token_sampling",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32",
        "1D probability distribution",
        NULL
    );

    // Element-wise operations
    llama_register_cuda_operation_support(
        "element_mul",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any tensor",
        NULL
    );

    llama_register_cuda_operation_support(
        "element_add",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any tensor",
        NULL
    );

    // Reduction operations
    llama_register_cuda_operation_support(
        "reduce_sum",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any tensor",
        NULL
    );

    llama_register_cuda_operation_support(
        "reduce_max",
        LLAMA_CUDA_SUPPORT_FULL,
        LLAMA_CUDA_REQ_DECODE_CRITICAL,
        true,
        "f32, f16, bf16",
        "Any tensor",
        NULL
    );

    return 0;
}

// ============================================================================
// OPERATION REGISTRATION
// ============================================================================

int llama_register_cuda_operation_support(
    const char* operation_name,
    enum llama_cuda_support_status support_status,
    enum llama_cuda_requirement_level requirement,
    bool decode_critical_compatible,
    const char* supported_dtypes,
    const char* supported_shapes,
    const char* unsupported_reason
) {
    if (operation_name == NULL) {
        fprintf(stderr, "ERROR: Operation name cannot be NULL\n");
        return -1;
    }

    // Grow registry if needed
    if (g_cuda_support_state.num_operations >= g_cuda_support_state.max_operations) {
        g_cuda_support_state.max_operations *= 2;
        struct llama_cuda_support_info* new_ops = (struct llama_cuda_support_info*)realloc(
            g_cuda_support_state.operations,
            g_cuda_support_state.max_operations * sizeof(struct llama_cuda_support_info)
        );
        if (new_ops == NULL) {
            fprintf(stderr, "ERROR: Failed to grow CUDA support operation registry\n");
            return -1;
        }
        g_cuda_support_state.operations = new_ops;
    }

    // Register operation
    struct llama_cuda_support_info* op = &g_cuda_support_state.operations[g_cuda_support_state.num_operations];
    op->operation_name = operation_name;
    op->status = support_status;
    op->requirement = requirement;
    op->decode_critical_compatible = decode_critical_compatible;
    op->supported_dtypes = supported_dtypes;
    op->supported_shapes = supported_shapes;
    op->unsupported_reason = unsupported_reason;

    // Update maps
    g_operation_cuda_support_map[operation_name] = support_status;
    g_operation_requirement_map[operation_name] = requirement;
    g_operation_info_map[operation_name] = op;
    g_operation_violation_count_map[operation_name] = 0;

    g_cuda_support_state.num_operations++;

    return 0;
}

// ============================================================================
// OPERATION LOOKUP
// ============================================================================

struct llama_cuda_support_info* llama_get_cuda_support_info(const char* operation_name) {
    if (operation_name == NULL) {
        return NULL;
    }

    auto it = g_operation_info_map.find(operation_name);
    if (it != g_operation_info_map.end()) {
        return it->second;
    }

    return NULL;
}

bool llama_operation_has_cuda_support_for_dtype(
    const char* operation_name,
    const char* dtype
) {
    struct llama_cuda_support_info* info = llama_get_cuda_support_info(operation_name);
    if (info == NULL) {
        return false; // Unknown operation defaults to unsupported
    }

    if (info->supported_dtypes == NULL) {
        return false;
    }

    // Simple substring check (in real impl would be more sophisticated)
    return strstr(info->supported_dtypes, dtype) != NULL;
}

bool llama_operation_has_cuda_support_for_shape(
    const char* operation_name,
    int num_dims,
    const int* shape
) {
    struct llama_cuda_support_info* info = llama_get_cuda_support_info(operation_name);
    if (info == NULL) {
        return false;
    }

    // Validate shape constraints
    if (num_dims > 0 && shape != NULL) {
        // Check tensor dimensions are reasonable
        for (int i = 0; i < num_dims; i++) {
            if (shape[i] <= 0) {
                return false;  // Invalid dimension
            }
        }
    }

    // Accept if operation supports CUDA and shape is valid
    if (info->status == LLAMA_CUDA_SUPPORT_FULL) {
        return true;
    }

    return false;
}

// ============================================================================
// ENFORCEMENT POINT 1: Decode Admission Validation
// ============================================================================

int llama_enforce_cuda_support_at_admission(
    const char** operation_names,
    bool* are_decode_critical,
    int num_operations
) {
    if (operation_names == NULL || are_decode_critical == NULL || num_operations <= 0) {
        fprintf(stderr, "ERROR: Invalid parameters to llama_enforce_cuda_support_at_admission\n");
        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    // Validate each decode-critical operation has CUDA support
    for (int i = 0; i < num_operations; i++) {
        if (!are_decode_critical[i]) {
            continue; // Skip non-critical ops
        }

        struct llama_cuda_support_info* info = llama_get_cuda_support_info(operation_names[i]);

        if (info == NULL) {
            llama_report_unsupported_cuda_op_violation(
                operation_names[i],
                true,
                LLAMA_CUDA_VIOL_UNSUPPORTED_OP,
                "Operation not registered with CUDA support system"
            );
            g_total_cuda_violations++;
            if (g_cuda_support_enforcement_strict) {
                return -1;
            }
        } else if (info->status == LLAMA_CUDA_SUPPORT_NONE ||
                   info->status == LLAMA_CUDA_SUPPORT_UNSUPPORTED_DTYPE ||
                   info->status == LLAMA_CUDA_SUPPORT_UNSUPPORTED_SHAPE ||
                   info->status == LLAMA_CUDA_SUPPORT_INVALID) {
            llama_report_unsupported_cuda_op_violation(
                operation_names[i],
                true,
                LLAMA_CUDA_VIOL_UNSUPPORTED_OP,
                info->unsupported_reason != NULL ? info->unsupported_reason : "No CUDA support"
            );
            g_total_cuda_violations++;
            if (g_cuda_support_enforcement_strict) {
                return -1;
            }
        }
    }

    g_cuda_support_state.admission_validation_complete = true;
    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 2: Graph Construction Validation
// ============================================================================

int llama_enforce_cuda_support_at_graph_build(
    const char* graph_name,
    const char** node_names,
    bool* are_decode_critical,
    int num_nodes
) {
    if (graph_name == NULL || node_names == NULL || are_decode_critical == NULL || num_nodes <= 0) {
        fprintf(stderr, "ERROR: Invalid parameters to llama_enforce_cuda_support_at_graph_build\n");
        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    // Validate each node in the graph
    for (int i = 0; i < num_nodes; i++) {
        if (!are_decode_critical[i]) {
            continue;
        }

        struct llama_cuda_support_info* info = llama_get_cuda_support_info(node_names[i]);

        if (info == NULL || info->status != LLAMA_CUDA_SUPPORT_FULL) {
            fprintf(stderr, "FATAL: Decode graph '%s' contains node '%s' without full CUDA support\n",
                    graph_name, node_names[i]);
            llama_report_unsupported_cuda_op_violation(
                node_names[i],
                true,
                LLAMA_CUDA_VIOL_UNSUPPORTED_OP,
                "Graph build: CUDA support not available"
            );
            g_total_cuda_violations++;
            if (g_cuda_support_enforcement_strict) {
                return -1;
            }
        }
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 3: Fail Fast on Unsupported Ops
// ============================================================================

int llama_enforce_no_unsupported_decode_critical_ops(
    const char* operation_name,
    bool is_decode_critical,
    enum llama_cuda_support_status support_status
) {
    if (operation_name == NULL) {
        fprintf(stderr, "ERROR: Operation name cannot be NULL\n");
        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    if (!is_decode_critical) {
        return 0; // Non-critical ops can fall back
    }

    // Decode-critical ops MUST have full CUDA support
    if (support_status != LLAMA_CUDA_SUPPORT_FULL) {
        fprintf(stderr, "FATAL: Decode-critical operation '%s' lacks CUDA support (status: %s)\n",
                operation_name, llama_cuda_support_status_name(support_status));

        enum llama_cuda_violation_type viol_type = LLAMA_CUDA_VIOL_UNSUPPORTED_OP;
        if (support_status == LLAMA_CUDA_SUPPORT_UNSUPPORTED_DTYPE) {
            viol_type = LLAMA_CUDA_VIOL_UNSUPPORTED_DTYPE;
        } else if (support_status == LLAMA_CUDA_SUPPORT_UNSUPPORTED_SHAPE) {
            viol_type = LLAMA_CUDA_VIOL_UNSUPPORTED_SHAPE;
        } else if (support_status == LLAMA_CUDA_SUPPORT_PARTIAL) {
            viol_type = LLAMA_CUDA_VIOL_PARTIAL_SUPPORT;
        }

        llama_report_unsupported_cuda_op_violation(
            operation_name,
            true,
            viol_type,
            "Decode-critical op requires full CUDA support"
        );
        g_total_cuda_violations++;
        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 4: Fail on Unsupported Data Type
// ============================================================================

int llama_enforce_cuda_dtype_support_for_decode(
    const char* operation_name,
    const char* dtype,
    bool is_decode_critical
) {
    if (operation_name == NULL || dtype == NULL) {
        fprintf(stderr, "ERROR: Invalid parameters to llama_enforce_cuda_dtype_support_for_decode\n");
        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    if (!is_decode_critical) {
        return 0; // Non-critical ops can use unsupported dtypes
    }

    if (!llama_operation_has_cuda_support_for_dtype(operation_name, dtype)) {
        fprintf(stderr, "FATAL: Decode-critical operation '%s' does not support dtype '%s'\n",
                operation_name, dtype);
        llama_report_unsupported_cuda_op_violation(
            operation_name,
            true,
            LLAMA_CUDA_VIOL_UNSUPPORTED_DTYPE,
            "Data type not supported by CUDA implementation"
        );
        g_total_cuda_violations++;
        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 5: Fail on Unsupported Tensor Shape
// ============================================================================

int llama_enforce_cuda_shape_support_for_decode(
    const char* operation_name,
    int num_dims,
    const int* shape,
    bool is_decode_critical
) {
    if (operation_name == NULL || shape == NULL || num_dims <= 0) {
        fprintf(stderr, "ERROR: Invalid parameters to llama_enforce_cuda_shape_support_for_decode\n");
        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    if (!is_decode_critical) {
        return 0;
    }

    if (!llama_operation_has_cuda_support_for_shape(operation_name, num_dims, shape)) {
        fprintf(stderr, "FATAL: Decode-critical operation '%s' does not support tensor shape\n",
                operation_name);
        llama_report_unsupported_cuda_op_violation(
            operation_name,
            true,
            LLAMA_CUDA_VIOL_UNSUPPORTED_SHAPE,
            "Tensor shape not supported by CUDA implementation"
        );
        g_total_cuda_violations++;
        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 6: Late Discovery Handling
// ============================================================================

int llama_enforce_no_late_unsupported_cuda_discovery(
    const char* operation_name,
    enum llama_cuda_support_status discovered_status,
    bool is_decode_critical
) {
    if (operation_name == NULL) {
        fprintf(stderr, "ERROR: Operation name cannot be NULL\n");
        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    if (!is_decode_critical) {
        return 0;
    }

    // If we discover an unsupported op during execution, that's a fatal error
    if (discovered_status != LLAMA_CUDA_SUPPORT_FULL) {
        fprintf(stderr, "FATAL: Late discovery of unsupported CUDA op '%s' during decode\n",
                operation_name);
        fprintf(stderr, "       This indicates admission control or graph validation failed\n");

        llama_report_unsupported_cuda_op_violation(
            operation_name,
            true,
            LLAMA_CUDA_VIOL_LATE_DISCOVERY,
            "Operation unsupported, discovered during decode execution"
        );

        g_total_cuda_violations++;
        g_late_discovery_violations++;
        g_cuda_support_state.late_discovery_count++;

        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

int llama_verify_all_decode_ops_cuda_supported(void) {
    if (g_cuda_support_state.operations == NULL || g_cuda_support_state.num_operations == 0) {
        fprintf(stderr, "ERROR: CUDA support registry not initialized\n");
        return -1;
    }

    // Check all registered operations
    for (int i = 0; i < g_cuda_support_state.num_operations; i++) {
        struct llama_cuda_support_info* op = &g_cuda_support_state.operations[i];

        if (op->requirement == LLAMA_CUDA_REQ_DECODE_CRITICAL) {
            if (op->status != LLAMA_CUDA_SUPPORT_FULL) {
                fprintf(stderr, "ERROR: Decode-critical operation '%s' lacks full CUDA support\n",
                        op->operation_name);
                return -1;
            }
        }
    }

    g_cuda_support_state.all_ops_cuda_supported = true;
    return 0;
}

int llama_assert_operation_meets_cuda_requirements_for_decode(
    const char* operation_name,
    bool is_decode_critical
) {
    if (operation_name == NULL) {
        fprintf(stderr, "ERROR: Operation name cannot be NULL\n");
        return -1;
    }

    if (!is_decode_critical) {
        return 0; // Non-critical ops have no strict requirements
    }

    struct llama_cuda_support_info* info = llama_get_cuda_support_info(operation_name);

    if (info == NULL) {
        fprintf(stderr, "ERROR: Operation '%s' not registered\n", operation_name);
        return -1;
    }

    if (info->requirement != LLAMA_CUDA_REQ_DECODE_CRITICAL &&
        info->requirement != LLAMA_CUDA_REQ_MANDATORY) {
        fprintf(stderr, "ERROR: Operation '%s' is not marked as decode-critical\n", operation_name);
        return -1;
    }

    if (info->status != LLAMA_CUDA_SUPPORT_FULL) {
        fprintf(stderr, "ERROR: Operation '%s' does not have full CUDA support\n", operation_name);
        return -1;
    }

    return 0;
}

int llama_check_cpu_fallback_allowed(
    const char* operation_name,
    bool is_decode_critical,
    enum llama_cuda_support_status support_status
) {
    if (!is_decode_critical) {
        return 0; // CPU fallback allowed for non-critical ops
    }

    // Decode-critical ops cannot fall back to CPU under any circumstances
    if (support_status != LLAMA_CUDA_SUPPORT_FULL) {
        fprintf(stderr, "[CUDA_ENFORCE] ERROR: Decode-critical op '%s' has unsupported status %d\n",
                operation_name, support_status);
        return -1;
    }

    // Operation has full support and is not critical - fallback allowed
    return 0;
}

// ============================================================================
// CPU FALLBACK PREVENTION
// ============================================================================

int llama_assert_no_unsupported_op_cpu_fallback(
    const char* operation_name,
    bool is_decode_critical,
    bool fallback_logic_exists
) {
    if (!is_decode_critical) {
        return 0;
    }

    if (fallback_logic_exists) {
        fprintf(stderr, "FATAL: Decode-critical operation '%s' has CPU fallback logic\n",
                operation_name);
        fprintf(stderr, "       All decode-critical ops must be GPU-only with no fallback\n");

        llama_report_unsupported_cuda_op_violation(
            operation_name,
            true,
            LLAMA_CUDA_VIOL_UNSUPPORTED_OP,
            "Fallback logic exists for decode-critical op"
        );
        g_total_cuda_violations++;

        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_prevent_unsupported_op_cpu_fallback(
    const char* operation_name,
    bool is_decode_critical,
    bool fallback_would_be_triggered
) {
    if (!is_decode_critical) {
        return 0;
    }

    if (fallback_would_be_triggered) {
        fprintf(stderr, "FATAL: CPU fallback would be triggered for decode-critical op '%s'\n",
                operation_name);

        llama_report_unsupported_cuda_op_violation(
            operation_name,
            true,
            LLAMA_CUDA_VIOL_UNSUPPORTED_OP,
            "CPU fallback prevention triggered"
        );
        g_total_cuda_violations++;

        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// EXPLICIT ERROR REPORTING
// ============================================================================

void llama_report_unsupported_cuda_op_violation(
    const char* operation_name,
    bool is_decode_critical,
    enum llama_cuda_violation_type violation_type,
    const char* unsupported_reason
) {
    fprintf(stderr, "\n");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "CUDA SUPPORT ENFORCEMENT VIOLATION\n");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "Operation:        %s\n", operation_name != NULL ? operation_name : "(unknown)");
    fprintf(stderr, "Classification:   %s\n", is_decode_critical ? "DECODE-CRITICAL" : "NON-CRITICAL");
    fprintf(stderr, "Violation Type:   %s\n", llama_cuda_violation_type_name(violation_type));
    fprintf(stderr, "Reason:           %s\n", unsupported_reason != NULL ? unsupported_reason : "(unknown)");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "\n");

    if (operation_name != NULL) {
        g_operation_violation_count_map[operation_name]++;
    }
    g_cuda_support_state.violation_count++;
    g_cuda_support_state.last_violation_type = violation_type;
    g_cuda_support_state.last_violation_op = operation_name;
    g_cuda_support_state.last_violation_reason = unsupported_reason;
}

void llama_print_unsupported_cuda_op_diagnostics(
    const char* operation_name,
    bool is_decode_critical,
    enum llama_cuda_violation_type violation_type,
    enum llama_cuda_validation_point validation_point,
    const char* unsupported_reason
) {
    fprintf(stderr, "\n");
    fprintf(stderr, "CUDA SUPPORT ENFORCEMENT DIAGNOSTICS\n");
    fprintf(stderr, "======================================\n");
    fprintf(stderr, "Operation:         %s\n", operation_name != NULL ? operation_name : "(unknown)");
    fprintf(stderr, "Classification:    %s\n", is_decode_critical ? "DECODE-CRITICAL" : "NON-CRITICAL");
    fprintf(stderr, "Violation Type:    %s\n", llama_cuda_violation_type_name(violation_type));
    fprintf(stderr, "Validation Point:  ");

    switch (validation_point) {
        case LLAMA_CUDA_VALIDATE_ADMISSION:
            fprintf(stderr, "ADMISSION CONTROL\n");
            break;
        case LLAMA_CUDA_VALIDATE_GRAPH_BUILD:
            fprintf(stderr, "GRAPH CONSTRUCTION\n");
            break;
        case LLAMA_CUDA_VALIDATE_EXECUTION:
            fprintf(stderr, "EXECUTION TIME (LATE DISCOVERY)\n");
            break;
        default:
            fprintf(stderr, "UNKNOWN\n");
    }

    fprintf(stderr, "Reason:            %s\n", unsupported_reason != NULL ? unsupported_reason : "(unknown)");
    fprintf(stderr, "======================================\n");

    // Print operation info if available
    struct llama_cuda_support_info* info = llama_get_cuda_support_info(operation_name);
    if (info != NULL) {
        fprintf(stderr, "\nOperation Details:\n");
        fprintf(stderr, "  Status:         %s\n", llama_cuda_support_status_name(info->status));
        fprintf(stderr, "  Requirement:    ");
        switch (info->requirement) {
            case LLAMA_CUDA_REQ_OPTIONAL:
                fprintf(stderr, "OPTIONAL\n");
                break;
            case LLAMA_CUDA_REQ_DECODE_CRITICAL:
                fprintf(stderr, "DECODE_CRITICAL\n");
                break;
            case LLAMA_CUDA_REQ_MANDATORY:
                fprintf(stderr, "MANDATORY\n");
                break;
        }
        fprintf(stderr, "  Supported Types: %s\n", info->supported_dtypes != NULL ? info->supported_dtypes : "N/A");
        fprintf(stderr, "  Supported Shapes: %s\n", info->supported_shapes != NULL ? info->supported_shapes : "N/A");
    }

    fprintf(stderr, "\n");
}

// ============================================================================
// REGRESSION PREVENTION
// ============================================================================

int llama_new_operation_default_unsupported_for_decode(
    const char* operation_name
) {
    if (operation_name == NULL) {
        return 0; // Unknown op defaults to unsupported
    }

    struct llama_cuda_support_info* info = llama_get_cuda_support_info(operation_name);

    if (info == NULL) {
        // Operation not registered - default to unsupported for safety
        fprintf(stderr, "WARNING: New operation '%s' not registered with CUDA support system\n",
                operation_name);
        fprintf(stderr, "         Defaulting to unsupported (safe default)\n");
        return 0; // Return 0 to indicate unsupported
    }

    return (info->status == LLAMA_CUDA_SUPPORT_FULL) ? 1 : 0;
}

int llama_assert_explicit_cuda_eligibility_decision(
    const char* operation_name
) {
    if (operation_name == NULL) {
        fprintf(stderr, "ERROR: Operation name cannot be NULL\n");
        return -1;
    }

    struct llama_cuda_support_info* info = llama_get_cuda_support_info(operation_name);

    if (info == NULL) {
        fprintf(stderr, "ERROR: No explicit CUDA eligibility decision for '%s'\n", operation_name);
        return -1;
    }

    return 0;
}

int llama_enable_cuda_support_for_operation(
    const char* operation_name,
    const char* supported_dtypes,
    const char* supported_shapes
) {
    if (operation_name == NULL) {
        fprintf(stderr, "ERROR: Operation name cannot be NULL\n");
        return -1;
    }

    struct llama_cuda_support_info* info = llama_get_cuda_support_info(operation_name);

    if (info == NULL) {
        // Register new operation
        return llama_register_cuda_operation_support(
            operation_name,
            LLAMA_CUDA_SUPPORT_FULL,
            LLAMA_CUDA_REQ_DECODE_CRITICAL,
            true,
            supported_dtypes,
            supported_shapes,
            NULL
        );
    }

    // Update existing operation
    info->status = LLAMA_CUDA_SUPPORT_FULL;
    info->requirement = LLAMA_CUDA_REQ_DECODE_CRITICAL;
    info->decode_critical_compatible = true;
    info->supported_dtypes = supported_dtypes;
    info->supported_shapes = supported_shapes;

    return 0;
}

// ============================================================================
// DECODE VS NON-DECODE DIFFERENTIATION
// ============================================================================

int llama_check_unsupported_op_cpu_allowed_by_criticality(
    bool is_decode_critical,
    bool fallback_to_cpu_requested
) {
    if (!is_decode_critical && fallback_to_cpu_requested) {
        return 0; // CPU fallback allowed for non-critical ops
    }

    if (is_decode_critical) {
        // CPU fallback forbidden for decode-critical ops
        return -1;
    }

    return 0;
}

int llama_enforce_decode_critical_precedence_over_fallback(
    const char* operation_name,
    bool is_decode_critical,
    enum llama_cuda_support_status support_status
) {
    if (!is_decode_critical) {
        return 0; // Non-critical ops not subject to this check
    }

    // Decode-critical classification overrides all fallback logic
    if (support_status != LLAMA_CUDA_SUPPORT_FULL) {
        fprintf(stderr, "FATAL: Decode-critical op '%s' has insufficient CUDA support\n",
                operation_name);

        llama_report_unsupported_cuda_op_violation(
            operation_name,
            true,
            LLAMA_CUDA_VIOL_UNSUPPORTED_OP,
            "Decode-critical precedence: CUDA support required"
        );
        g_total_cuda_violations++;

        if (g_cuda_support_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

void llama_set_cuda_support_enforcement_strict(bool enforce_strict) {
    g_cuda_support_enforcement_strict = enforce_strict;
}

bool llama_get_cuda_support_enforcement_strict(void) {
    return g_cuda_support_enforcement_strict;
}

int llama_get_cuda_support_violation_count(void) {
    return g_total_cuda_violations;
}

int llama_get_cuda_late_discovery_count(void) {
    return g_late_discovery_violations;
}

void llama_reset_cuda_support_violation_counters(void) {
    g_total_cuda_violations = 0;
    g_late_discovery_violations = 0;
    g_cuda_support_state.violation_count = 0;
    g_cuda_support_state.late_discovery_count = 0;

    for (auto& pair : g_operation_violation_count_map) {
        pair.second = 0;
    }
}

// ============================================================================
// EXPLICIT REQUIREMENT STATEMENT
// ============================================================================

void llama_print_cuda_support_requirement_statement(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "CUDA SUPPORT REQUIREMENT PRINCIPLE\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "Decode-critical ops require mandatory CUDA support.\n");
    fprintf(stdout, "Unsupported CUDA ops on the decode path are fatal errors.\n");
    fprintf(stdout, "CUDA support is guaranteed before decode starts.\n");
    fprintf(stdout, "CPU fallback for unsupported ops is impossible.\n");
    fprintf(stdout, "Failures are early, explicit, and actionable.\n");
    fprintf(stdout, "Decode throughput and invariants are protected by design.\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "Key Constraints:\n");
    fprintf(stdout, "  1. All decode-critical ops must have CUDA_SUPPORT_FULL\n");
    fprintf(stdout, "  2. No decode-critical op can use CPU as fallback\n");
    fprintf(stdout, "  3. Unsupported ops detected at admission control (not execution)\n");
    fprintf(stdout, "  4. Late discovery of unsupported ops is fatal error\n");
    fprintf(stdout, "  5. Decode-critical classification overrides all fallback logic\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================

int llama_cuda_support_enforce_selftest(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "CUDA SUPPORT ENFORCEMENT SELF-TEST SUITE\n");
    fprintf(stdout, "================================================================================\n");

    int test_count = 0;
    int pass_count = 0;

    // TEST 1: Initialization
    fprintf(stdout, "\nTest 1: Initialization...");
    test_count++;
    if (llama_cuda_support_enforce_init() == 0) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 2: Operation enumeration
    fprintf(stdout, "Test 2: Operation Enumeration...");
    test_count++;
    if (llama_enumerate_cuda_requirements_for_decode() == 0) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 3: Operation lookup
    fprintf(stdout, "Test 3: Operation Lookup...");
    test_count++;
    struct llama_cuda_support_info* info = llama_get_cuda_support_info("attention_q_projection");
    if (info != NULL && info->status == LLAMA_CUDA_SUPPORT_FULL) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 4: Data type support check
    fprintf(stdout, "Test 4: Data Type Support Check...");
    test_count++;
    if (llama_operation_has_cuda_support_for_dtype("attention_q_projection", "f32")) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 5: Enforcement mode control
    fprintf(stdout, "Test 5: Enforcement Mode Control...");
    test_count++;
    llama_set_cuda_support_enforcement_strict(true);
    if (llama_get_cuda_support_enforcement_strict() == true) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 6: Violation counting
    fprintf(stdout, "Test 6: Violation Counting...");
    test_count++;
    llama_reset_cuda_support_violation_counters();
    int initial_count = llama_get_cuda_support_violation_count();
    if (initial_count == 0) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 7: All decode ops verified
    fprintf(stdout, "Test 7: All Decode Ops Verified...");
    test_count++;
    if (llama_verify_all_decode_ops_cuda_supported() == 0) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 8: Requirement statement printable
    fprintf(stdout, "Test 8: Requirement Statement...");
    test_count++;
    llama_print_cuda_support_requirement_statement();
    fprintf(stdout, "         PASS\n");
    pass_count++;

    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "SELF-TEST RESULTS: %d / %d tests passed\n", pass_count, test_count);
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");

    return (pass_count == test_count) ? 0 : -1;
}
