/**
 * SECTION 9: Convert Unsupported CUDA Ops Into Hard Decode Errors
 *
 * This file implements enforcement that any decode-critical operation lacking CUDA
 * support results in an immediate hard error. Unsupported CUDA ops on the decode
 * path are correctness failures, not performance issues. CPU fallback is forbidden.
 *
 * Core Principle:
 * "Decode-critical ops require mandatory CUDA support. Unsupported CUDA ops on the
 *  decode path are fatal errors. CUDA support is guaranteed before decode starts.
 *  CPU fallback for unsupported ops is impossible. Failures are early, explicit,
 *  and actionable. Decode throughput and invariants are protected by design."
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <map>
#include <vector>

// ============================================================================
// CUDA SUPPORT STATE DEFINITION
// ============================================================================

/**
 * Enum defining CUDA support status for an operation
 */
enum llama_cuda_support_status {
    LLAMA_CUDA_SUPPORT_UNKNOWN = 0,        // Support status not determined
    LLAMA_CUDA_SUPPORT_FULL = 1,           // Full CUDA support available
    LLAMA_CUDA_SUPPORT_PARTIAL = 2,        // Partial support (some data types)
    LLAMA_CUDA_SUPPORT_NONE = 3,           // No CUDA support
    LLAMA_CUDA_SUPPORT_UNSUPPORTED_DTYPE = 4,    // Unsupported data type
    LLAMA_CUDA_SUPPORT_UNSUPPORTED_SHAPE = 5,    // Unsupported tensor shape
    LLAMA_CUDA_SUPPORT_INVALID = 6,        // Invalid or error state
};

/**
 * Enum defining CUDA support requirement level for operations
 */
enum llama_cuda_requirement_level {
    LLAMA_CUDA_REQ_OPTIONAL = 0,           // CUDA not required (can use CPU)
    LLAMA_CUDA_REQ_DECODE_CRITICAL = 1,    // Required for decode-critical ops
    LLAMA_CUDA_REQ_MANDATORY = 2,          // Always required regardless of path
};

/**
 * Enum defining when CUDA support is validated
 */
enum llama_cuda_validation_point {
    LLAMA_CUDA_VALIDATE_UNKNOWN = 0,
    LLAMA_CUDA_VALIDATE_ADMISSION = 1,     // During decode admission
    LLAMA_CUDA_VALIDATE_GRAPH_BUILD = 2,   // During graph construction
    LLAMA_CUDA_VALIDATE_EXECUTION = 3,     // During execution (late discovery)
};

/**
 * Enum defining unsupported CUDA op violation types
 */
enum llama_cuda_violation_type {
    LLAMA_CUDA_VIOL_UNKNOWN = 0,
    LLAMA_CUDA_VIOL_UNSUPPORTED_OP = 1,               // Op has no CUDA support
    LLAMA_CUDA_VIOL_UNSUPPORTED_DTYPE = 2,            // Data type not supported
    LLAMA_CUDA_VIOL_UNSUPPORTED_SHAPE = 3,            // Tensor shape not supported
    LLAMA_CUDA_VIOL_MISSING_KERNEL = 4,               // CUDA kernel missing
    LLAMA_CUDA_VIOL_LATE_DISCOVERY = 5,               // Unsupported op found during execution
    LLAMA_CUDA_VIOL_PARTIAL_SUPPORT = 6,              // Only partial support for requirements
    LLAMA_CUDA_VIOL_UNSUPPORTED_FEATURE = 7,          // Required feature not supported
};

// ============================================================================
// CUDA SUPPORT STRUCTURES
// ============================================================================

/**
 * Structure defining CUDA support details for an operation
 */
struct llama_cuda_support_info {
    const char* operation_name;                    // Operation identifier
    enum llama_cuda_support_status status;         // Support status
    enum llama_cuda_requirement_level requirement; // Requirement level
    bool decode_critical_compatible;               // Compatible with decode-critical path?
    const char* supported_dtypes;                  // Supported data types (description)
    const char* supported_shapes;                  // Supported tensor shapes (description)
    const char* unsupported_reason;                // Why unsupported (if applicable)
};

/**
 * Structure tracking CUDA support validation state
 */
struct llama_cuda_support_validation_state {
    // Operation registry
    struct llama_cuda_support_info* operations;    // Array of operation support info
    int num_operations;                            // Number of registered operations
    int max_operations;                            // Capacity

    // Validation state
    bool admission_validation_complete;            // Admission validation done?
    bool all_ops_cuda_supported;                   // All ops have CUDA support?

    // Violation tracking
    int violation_count;                           // Total violations detected
    enum llama_cuda_violation_type last_violation_type;
    const char* last_violation_op;
    const char* last_violation_reason;

    // Late discovery tracking
    int late_discovery_count;                      // Unsupported ops found during execution
};

// ============================================================================
// CUDA SUPPORT ENUMERATION & REGISTRY
// ============================================================================

/**
 * Initialize CUDA support enforcement tracking
 */
int llama_cuda_support_enforce_init(void);

/**
 * Enumerate all decode-critical operations and their CUDA requirements
 * Must be called during system initialization.
 *
 * Returns: 0 = enumeration complete, -1 = error
 */
int llama_enumerate_cuda_requirements_for_decode(void);

/**
 * Register an operation's CUDA support details
 */
int llama_register_cuda_operation_support(
    const char* operation_name,
    enum llama_cuda_support_status support_status,
    enum llama_cuda_requirement_level requirement,
    bool decode_critical_compatible,
    const char* supported_dtypes,
    const char* supported_shapes,
    const char* unsupported_reason
);

/**
 * Get CUDA support information for an operation
 */
struct llama_cuda_support_info* llama_get_cuda_support_info(const char* operation_name);

/**
 * Check if operation has CUDA support for a specific data type
 */
bool llama_operation_has_cuda_support_for_dtype(
    const char* operation_name,
    const char* dtype
);

/**
 * Check if operation has CUDA support for a specific tensor shape
 */
bool llama_operation_has_cuda_support_for_shape(
    const char* operation_name,
    int num_dims,
    const int* shape
);

// ============================================================================
// CUDA SUPPORT VALIDATION AT DECODE ADMISSION
// ============================================================================

/**
 * ENFORCEMENT POINT 1: Validate CUDA support at decode admission
 * Verify every decode-critical op has valid CUDA backend implementation.
 * If any required CUDA op is unsupported, reject decode admission immediately.
 *
 * Returns: 0 = all ops supported, -1 = FATAL (unsupported op found)
 */
int llama_enforce_cuda_support_at_admission(
    const char** operation_names,
    bool* are_decode_critical,
    int num_operations
);

/**
 * ENFORCEMENT POINT 2: Validate CUDA support at graph construction
 * If decode graph includes op without CUDA support, abort graph construction.
 *
 * Returns: 0 = all ops supported, -1 = FATAL (abort graph construction)
 */
int llama_enforce_cuda_support_at_graph_build(
    const char* graph_name,
    const char** node_names,
    bool* are_decode_critical,
    int num_nodes
);

/**
 * ENFORCEMENT POINT 3: Fail fast on unsupported decode-critical ops
 * Hard failure when decode-critical op lacks CUDA support.
 * No fallback to CPU.
 *
 * Returns: 0 = supported, -1 = FATAL (unsupported op)
 */
int llama_enforce_no_unsupported_decode_critical_ops(
    const char* operation_name,
    bool is_decode_critical,
    enum llama_cuda_support_status support_status
);

/**
 * ENFORCEMENT POINT 4: Fail on unsupported data type for decode op
 * Decode-critical op with unsupported dtype is fatal error.
 *
 * Returns: 0 = dtype supported, -1 = FATAL (dtype not supported)
 */
int llama_enforce_cuda_dtype_support_for_decode(
    const char* operation_name,
    const char* dtype,
    bool is_decode_critical
);

/**
 * ENFORCEMENT POINT 5: Fail on unsupported tensor shape for decode op
 * Decode-critical op with unsupported shape is fatal error.
 *
 * Returns: 0 = shape supported, -1 = FATAL (shape not supported)
 */
int llama_enforce_cuda_shape_support_for_decode(
    const char* operation_name,
    int num_dims,
    const int* shape,
    bool is_decode_critical
);

/**
 * ENFORCEMENT POINT 6: Terminate on late discovery of unsupported op
 * If unsupported CUDA op discovered during decode execution, abort immediately.
 *
 * Returns: 0 = not unsupported, -1 = FATAL (unsupported op found)
 */
int llama_enforce_no_late_unsupported_cuda_discovery(
    const char* operation_name,
    enum llama_cuda_support_status discovered_status,
    bool is_decode_critical
);

// ============================================================================
// CUDA SUPPORT REQUIREMENT ENFORCEMENT
// ============================================================================

/**
 * Verify CUDA support for decode before execution begins
 * All decode-critical ops must have full CUDA support.
 *
 * Returns: 0 = all supported, -1 = FATAL (unsupported op found)
 */
int llama_verify_all_decode_ops_cuda_supported(void);

/**
 * Assert operation meets CUDA requirements for decode
 * Returns: 0 = meets requirements, -1 = FATAL (requirements not met)
 */
int llama_assert_operation_meets_cuda_requirements_for_decode(
    const char* operation_name,
    bool is_decode_critical
);

/**
 * Check if operation can fall back to CPU
 * Decode-critical ops cannot fall back to CPU even with partial support.
 *
 * Returns: 0 = CPU fallback allowed, -1 = CPU fallback forbidden
 */
int llama_check_cpu_fallback_allowed(
    const char* operation_name,
    bool is_decode_critical,
    enum llama_cuda_support_status support_status
);

// ============================================================================
// REMOVAL OF CPU FALLBACK LOGIC
// ============================================================================

/**
 * Assert that no CPU fallback logic exists for unsupported decode-critical ops
 * Unsupported ops on decode path must fail, not fall back.
 *
 * Returns: 0 = no fallback logic found, -1 = FATAL (fallback logic detected)
 */
int llama_assert_no_unsupported_op_cpu_fallback(
    const char* operation_name,
    bool is_decode_critical,
    bool fallback_logic_exists
);

/**
 * Remove CPU fallback for unsupported ops during decode
 * Replaces transparent CPU routing with hard failure.
 *
 * Returns: 0 = no fallback removed, -1 = fallback was removed (would have failed)
 */
int llama_prevent_unsupported_op_cpu_fallback(
    const char* operation_name,
    bool is_decode_critical,
    bool fallback_would_be_triggered
);

// ============================================================================
// EXPLICIT ERROR MESSAGES
// ============================================================================

/**
 * Report unsupported CUDA op violation with detailed error message
 */
void llama_report_unsupported_cuda_op_violation(
    const char* operation_name,
    bool is_decode_critical,
    enum llama_cuda_violation_type violation_type,
    const char* unsupported_reason
);

/**
 * Print comprehensive unsupported CUDA op diagnostics
 */
void llama_print_unsupported_cuda_op_diagnostics(
    const char* operation_name,
    bool is_decode_critical,
    enum llama_cuda_violation_type violation_type,
    enum llama_cuda_validation_point validation_point,
    const char* unsupported_reason
);

/**
 * Convert CUDA support status to human-readable string
 */
static inline const char* llama_cuda_support_status_name(
    enum llama_cuda_support_status status
) {
    switch (status) {
        case LLAMA_CUDA_SUPPORT_FULL:
            return "FULL";
        case LLAMA_CUDA_SUPPORT_PARTIAL:
            return "PARTIAL";
        case LLAMA_CUDA_SUPPORT_NONE:
            return "NONE";
        case LLAMA_CUDA_SUPPORT_UNSUPPORTED_DTYPE:
            return "UNSUPPORTED_DTYPE";
        case LLAMA_CUDA_SUPPORT_UNSUPPORTED_SHAPE:
            return "UNSUPPORTED_SHAPE";
        case LLAMA_CUDA_SUPPORT_INVALID:
            return "INVALID";
        default:
            return "UNKNOWN";
    }
}

/**
 * Convert violation type to human-readable string
 */
static inline const char* llama_cuda_violation_type_name(
    enum llama_cuda_violation_type violation_type
) {
    switch (violation_type) {
        case LLAMA_CUDA_VIOL_UNSUPPORTED_OP:
            return "UNSUPPORTED_OP";
        case LLAMA_CUDA_VIOL_UNSUPPORTED_DTYPE:
            return "UNSUPPORTED_DTYPE";
        case LLAMA_CUDA_VIOL_UNSUPPORTED_SHAPE:
            return "UNSUPPORTED_SHAPE";
        case LLAMA_CUDA_VIOL_MISSING_KERNEL:
            return "MISSING_KERNEL";
        case LLAMA_CUDA_VIOL_LATE_DISCOVERY:
            return "LATE_DISCOVERY";
        case LLAMA_CUDA_VIOL_PARTIAL_SUPPORT:
            return "PARTIAL_SUPPORT";
        case LLAMA_CUDA_VIOL_UNSUPPORTED_FEATURE:
            return "UNSUPPORTED_FEATURE";
        default:
            return "UNKNOWN";
    }
}

// ============================================================================
// REGRESSION PREVENTION
// ============================================================================

/**
 * Guard against regressions: newly added ops default to unsupported
 * Force developers to explicitly enable CUDA support for new operations.
 *
 * Returns: 0 = unsupported (safe default), needs explicit enable
 */
int llama_new_operation_default_unsupported_for_decode(
    const char* operation_name
);

/**
 * Assert that operation has explicit CUDA eligibility decision
 * Operations must be consciously marked as CUDA-supported.
 *
 * Returns: 0 = explicit decision made, -1 = no explicit decision
 */
int llama_assert_explicit_cuda_eligibility_decision(
    const char* operation_name
);

/**
 * Enable CUDA support for an operation (explicit decision required)
 * Must be called during operation registration.
 */
int llama_enable_cuda_support_for_operation(
    const char* operation_name,
    const char* supported_dtypes,
    const char* supported_shapes
);

// ============================================================================
// DECODE VS NON-DECODE DIFFERENTIATION
// ============================================================================

/**
 * Allow unsupported ops to run on CPU only when NON_CRITICAL
 * Decode-critical classification overrides all fallback logic.
 *
 * Returns: 0 = CPU fallback allowed, -1 = CPU fallback forbidden
 */
int llama_check_unsupported_op_cpu_allowed_by_criticality(
    bool is_decode_critical,
    bool fallback_to_cpu_requested
);

/**
 * Ensure decode-critical classification takes precedence
 * Unsupported ops cannot run on decode path regardless of fallback logic.
 *
 * Returns: 0 = classification enforced, -1 = enforcement failed
 */
int llama_enforce_decode_critical_precedence_over_fallback(
    const char* operation_name,
    bool is_decode_critical,
    enum llama_cuda_support_status support_status
);

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

/**
 * Enable/disable strict CUDA support enforcement
 * When enabled, unsupported ops cause immediate failure.
 */
void llama_set_cuda_support_enforcement_strict(bool enforce_strict);

/**
 * Get current enforcement mode
 */
bool llama_get_cuda_support_enforcement_strict(void);

/**
 * Get total unsupported CUDA op violations
 */
int llama_get_cuda_support_violation_count(void);

/**
 * Get late discovery violation count
 */
int llama_get_cuda_late_discovery_count(void);

/**
 * Reset CUDA support violation counters
 */
void llama_reset_cuda_support_violation_counters(void);

// ============================================================================
// EXPLICIT CUDA SUPPORT REQUIREMENT STATEMENT
// ============================================================================

/**
 * Print the CUDA support requirement principle
 */
void llama_print_cuda_support_requirement_statement(void);

// ============================================================================
// VALIDATION AND SELF-TEST
// ============================================================================

/**
 * Self-test: verify CUDA support enforcement mechanism works correctly
 */
int llama_cuda_support_enforce_selftest(void);

