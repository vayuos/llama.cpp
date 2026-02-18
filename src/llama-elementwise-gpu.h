/**
 * SECTION 39: Eliminate CPU Elementwise Operations During Decode
 * Header
 *
 * This file implements GPU-exclusive enforcement for elementwise operations
 * (add, multiply, divide, scale). All elementwise operations must execute
 * exclusively on GPU during decode. CPU elementwise operations are forbidden.
 * Unfused or host-executed elementwise operations trigger hard failure.
 *
 * Rules:
 * - All elementwise add must execute on GPU during decode
 * - All elementwise multiply must execute on GPU during decode
 * - All elementwise divide must execute on GPU during decode
 * - All scalar operations must execute on GPU during decode
 * - No CPU elementwise kernels during decode
 * - No intermediate materialization for elementwise results
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ELEMENTWISE OPERATION PHASE ENUMERATION
// ============================================================================

enum llama_elementwise_phase {
    LLAMA_ELEMENTWISE_PHASE_NONE = 0,
    LLAMA_ELEMENTWISE_PHASE_PREFILL = 1,
    LLAMA_ELEMENTWISE_PHASE_DECODE = 2,
    LLAMA_ELEMENTWISE_PHASE_COMPLETE = 3,
};

// ============================================================================
// ELEMENTWISE STATE ENUMERATION
// ============================================================================

enum llama_gpu_elementwise_state {
    LLAMA_GPU_ELEMENTWISE_UNINITIALIZED = 0,
    LLAMA_GPU_ELEMENTWISE_INITIALIZED = 1,
    LLAMA_GPU_ELEMENTWISE_GPU_VERIFIED = 2,
    LLAMA_GPU_ELEMENTWISE_DECODE_ACTIVE = 3,
    LLAMA_GPU_ELEMENTWISE_COMPLETE = 4,
    LLAMA_GPU_ELEMENTWISE_ERROR = 5,
};

// ============================================================================
// ELEMENTWISE OPERATION TYPE ENUMERATION
// ============================================================================

enum llama_elementwise_operation_type {
    LLAMA_ELEMENTWISE_OP_NONE = 0,
    LLAMA_ELEMENTWISE_OP_ADD = 1,
    LLAMA_ELEMENTWISE_OP_MUL = 2,
    LLAMA_ELEMENTWISE_OP_DIV = 3,
    LLAMA_ELEMENTWISE_OP_MUL_SCALAR = 4,
    LLAMA_ELEMENTWISE_OP_ADD_SCALAR = 5,
    LLAMA_ELEMENTWISE_OP_DIV_SCALAR = 6,
};

// ============================================================================
// VIOLATION ENUMERATION
// ============================================================================

enum llama_elementwise_violation {
    LLAMA_ELEMENTWISE_VIOLATION_NONE = 0,
    LLAMA_ELEMENTWISE_VIOLATION_CPU_ADD_DECODE = 1,
    LLAMA_ELEMENTWISE_VIOLATION_CPU_MUL_DECODE = 2,
    LLAMA_ELEMENTWISE_VIOLATION_CPU_DIV_DECODE = 3,
    LLAMA_ELEMENTWISE_VIOLATION_CPU_SCALAR_DECODE = 4,
    LLAMA_ELEMENTWISE_VIOLATION_INTERMEDIATE_BUFFER = 5,
    LLAMA_ELEMENTWISE_VIOLATION_HOST_SEQUENCING = 6,
    LLAMA_ELEMENTWISE_VIOLATION_WRONG_BACKEND = 7,
};

// ============================================================================
// CONFIGURATION
// ============================================================================

struct llama_gpu_elementwise_config {
    bool forbid_cpu_elementwise;
    bool forbid_intermediate_buffer;
    bool cuda_backend_only;
    bool debug_tracking;
};

// ============================================================================
// OPERATION RECORD
// ============================================================================

struct llama_elementwise_operation_record {
    uint64_t operation_id;
    enum llama_elementwise_operation_type op_type;
    uint32_t layer_idx;
    uint64_t input_a_id;
    uint64_t input_b_id;
    uint64_t output_id;
    uint64_t element_count;
    bool was_gpu_executed;
    bool is_decode_phase;
};

// ============================================================================
// STATE RECORD
// ============================================================================

struct llama_gpu_elementwise_state_record {
    enum llama_gpu_elementwise_state state;
    enum llama_elementwise_phase current_phase;
    uint64_t total_operations_tracked;
    uint64_t gpu_operations_executed;
    uint64_t cpu_operations_detected;
    uint64_t decode_cpu_operations_blocked;
    int total_violations;
    enum llama_elementwise_violation last_violation;
};

// ============================================================================
// VALIDATION STATE
// ============================================================================

struct llama_gpu_elementwise_validation_state {
    struct llama_gpu_elementwise_config config;
    struct llama_gpu_elementwise_state_record state_record;
    void* operations_map;
    void* operation_history_vector;
    struct llama_elementwise_operation_record last_operation_record;
    int total_operations;
    int total_violations;
    bool enforcement_strict;
    bool decode_phase_active;
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

int llama_elementwise_gpu_init(void);
int llama_elementwise_gpu_configure(bool forbid_cpu, bool forbid_intermediate, bool cuda_only);
int llama_elementwise_gpu_set_phase(enum llama_elementwise_phase phase);
int llama_elementwise_gpu_begin_decode_phase(void);
int llama_elementwise_gpu_end_decode_phase(void);

// 10 enforcement points
int llama_elementwise_gpu_verify_gpu_backend_available(void);
int llama_elementwise_gpu_forbid_cpu_add_in_decode(void);
int llama_elementwise_gpu_forbid_cpu_mul_in_decode(void);
int llama_elementwise_gpu_forbid_cpu_div_in_decode(void);
int llama_elementwise_gpu_forbid_cpu_scalar_ops_in_decode(void);
int llama_elementwise_gpu_forbid_intermediate_materialization(void);
int llama_elementwise_gpu_forbid_host_sequencing(void);
int llama_elementwise_gpu_verify_all_gpu_executed(void);
int llama_elementwise_gpu_verify_no_cpu_operations(void);
int llama_elementwise_gpu_enforce_gpu_only_decode(void);

// Violation detection
int llama_elementwise_gpu_detect_cpu_add_decode(void);
int llama_elementwise_gpu_detect_cpu_mul_decode(void);
int llama_elementwise_gpu_detect_cpu_div_decode(void);
int llama_elementwise_gpu_detect_cpu_scalar_decode(void);
int llama_elementwise_gpu_detect_intermediate_buffer(uint64_t tensor_id);
int llama_elementwise_gpu_detect_host_sequencing(void);
int llama_elementwise_gpu_detect_wrong_backend(void);
int llama_elementwise_gpu_detect_unsupported_shape(void);

// Operation tracking
int llama_elementwise_gpu_record_operation(
    uint64_t op_id,
    enum llama_elementwise_operation_type op_type,
    uint32_t layer_idx,
    uint64_t in_a_id,
    uint64_t in_b_id,
    uint64_t out_id,
    uint64_t element_count
);
int llama_elementwise_gpu_mark_gpu_executed(uint64_t op_id);
int llama_elementwise_gpu_verify_operation_gpu_executed(uint64_t op_id);

// Verification
int llama_elementwise_gpu_verify_all_operations_tracked(void);
int llama_elementwise_gpu_verify_no_cpu_operations_decode(void);
int llama_elementwise_gpu_verify_no_intermediate_buffers(void);
int llama_elementwise_gpu_verify_backends_available(void);
int llama_elementwise_gpu_verify_decode_phase_locked(void);

// Query functions
struct llama_gpu_elementwise_state_record llama_elementwise_gpu_get_state_record(void);
enum llama_gpu_elementwise_state llama_elementwise_gpu_get_state(void);
uint64_t llama_elementwise_gpu_get_gpu_operations_count(void);
uint64_t llama_elementwise_gpu_get_cpu_operations_blocked_count(void);

// Diagnostics
void llama_elementwise_gpu_log_enforcement_enabled(void);
void llama_elementwise_gpu_log_decode_phase_active(void);
void llama_elementwise_gpu_print_state(void);
void llama_elementwise_gpu_print_operation_record(const struct llama_elementwise_operation_record* record);
void llama_elementwise_gpu_print_violation_summary(void);
void llama_elementwise_gpu_print_operation_summary(void);

// Violation reporting
void llama_elementwise_gpu_report_violation(
    enum llama_elementwise_violation violation_type,
    const char* location,
    const char* details
);

// Enforcement control
void llama_elementwise_gpu_set_enforcement_strict(bool strict);
bool llama_elementwise_gpu_get_enforcement_strict(void);
void llama_elementwise_gpu_set_debug_output(bool debug);

// Performance
int llama_elementwise_gpu_validate_performance_impact(void);
uint64_t llama_elementwise_gpu_get_kernel_count_reduction(void);

// Self-test
int llama_elementwise_gpu_selftest(void);

// Helpers
static inline const char* llama_elementwise_phase_name(enum llama_elementwise_phase phase) {
    switch (phase) {
        case LLAMA_ELEMENTWISE_PHASE_NONE: return "NONE";
        case LLAMA_ELEMENTWISE_PHASE_PREFILL: return "PREFILL";
        case LLAMA_ELEMENTWISE_PHASE_DECODE: return "DECODE";
        case LLAMA_ELEMENTWISE_PHASE_COMPLETE: return "COMPLETE";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_elementwise_operation_type_name(enum llama_elementwise_operation_type op) {
    switch (op) {
        case LLAMA_ELEMENTWISE_OP_NONE: return "NONE";
        case LLAMA_ELEMENTWISE_OP_ADD: return "ADD";
        case LLAMA_ELEMENTWISE_OP_MUL: return "MUL";
        case LLAMA_ELEMENTWISE_OP_DIV: return "DIV";
        case LLAMA_ELEMENTWISE_OP_MUL_SCALAR: return "MUL_SCALAR";
        case LLAMA_ELEMENTWISE_OP_ADD_SCALAR: return "ADD_SCALAR";
        case LLAMA_ELEMENTWISE_OP_DIV_SCALAR: return "DIV_SCALAR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_elementwise_violation_name(enum llama_elementwise_violation v) {
    switch (v) {
        case LLAMA_ELEMENTWISE_VIOLATION_NONE: return "NONE";
        case LLAMA_ELEMENTWISE_VIOLATION_CPU_ADD_DECODE: return "CPU_ADD_DECODE";
        case LLAMA_ELEMENTWISE_VIOLATION_CPU_MUL_DECODE: return "CPU_MUL_DECODE";
        case LLAMA_ELEMENTWISE_VIOLATION_CPU_DIV_DECODE: return "CPU_DIV_DECODE";
        case LLAMA_ELEMENTWISE_VIOLATION_CPU_SCALAR_DECODE: return "CPU_SCALAR_DECODE";
        case LLAMA_ELEMENTWISE_VIOLATION_INTERMEDIATE_BUFFER: return "INTERMEDIATE_BUFFER";
        case LLAMA_ELEMENTWISE_VIOLATION_HOST_SEQUENCING: return "HOST_SEQUENCING";
        case LLAMA_ELEMENTWISE_VIOLATION_WRONG_BACKEND: return "WRONG_BACKEND";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
