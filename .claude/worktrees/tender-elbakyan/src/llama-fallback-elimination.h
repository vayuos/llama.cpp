/**
 * SECTION 8: Eliminate Silent CPU Backend Fallbacks
 *
 * This file implements elimination of silent CPU backend fallbacks on the decode path.
 * Any fallback from GPU to CPU is treated as a fatal correctness violation, not a
 * recovery mechanism. All fallback paths are audited, replaced with hard failures,
 * and validated at multiple enforcement points.
 *
 * Core Principle:
 * "Silent CPU fallback is prohibited on the decode path. Any fallback attempt is a
 *  fatal error. Decode-critical execution is GPU-only by construction. CPU fallback
 *  cannot occur silently. All violations are detected immediately."
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include "llama-backend-immutability-enforce.h"

// ============================================================================
// FALLBACK ELIMINATION STATE DEFINITION
// ============================================================================

/**
 * Enum defining fallback attempt locations
 */
enum llama_fallback_attempt_location {
    LLAMA_FALLBACK_UNKNOWN = 0,
    LLAMA_FALLBACK_BACKEND_DISPATCH = 1,       // During backend dispatch decision
    LLAMA_FALLBACK_GRAPH_EXECUTION = 2,        // During graph execution
    LLAMA_FALLBACK_KERNEL_DISPATCH = 3,        // Kernel selection/dispatch
    LLAMA_FALLBACK_SAMPLING = 4,               // Sampling operation
    LLAMA_FALLBACK_MEMORY_PLACEMENT = 5,       // Tensor placement mismatch
    LLAMA_FALLBACK_CAPABILITY_CHECK = 6,       // Runtime capability check
    LLAMA_FALLBACK_OOM_RECOVERY = 7,           // Out-of-memory recovery attempt
    LLAMA_FALLBACK_INVALID_STATE = 8,          // Invalid GPU state recovery
    LLAMA_FALLBACK_SHAPE_MISMATCH = 9,         // Tensor shape mismatch
};

/**
 * Enum defining reasons why GPU execution became unavailable
 */
enum llama_gpu_unavailability_reason {
    LLAMA_GPU_AVAILABLE = 0,                   // GPU available, no fallback needed
    LLAMA_GPU_UNAVAIL_NO_KERNEL = 1,           // Missing GPU kernel
    LLAMA_GPU_UNAVAIL_UNSUPPORTED_DTYPE = 2,   // Unsupported data type
    LLAMA_GPU_UNAVAIL_UNSUPPORTED_SHAPE = 3,   // Unsupported tensor shape
    LLAMA_GPU_UNAVAIL_MEMORY_PLACEMENT = 4,    // Tensor on CPU (memory mismatch)
    LLAMA_GPU_UNAVAIL_INVALID_CAPABILITY = 5,  // GPU capability missing
    LLAMA_GPU_UNAVAIL_OOM = 6,                 // GPU out of memory
    LLAMA_GPU_UNAVAIL_INVALID_STATE = 7,       // GPU in invalid state
    LLAMA_GPU_UNAVAIL_UNKNOWN = 8,             // Unknown/unspecified reason
};

/**
 * Enum defining fallback elimination violation types
 */
enum llama_fallback_violation_type {
    LLAMA_FALLBACK_VIOL_UNKNOWN = 0,
    LLAMA_FALLBACK_VIOL_SILENT_FALLBACK = 1,           // Silent CPU fallback detected
    LLAMA_FALLBACK_VIOL_DECODE_CRITICAL_CPU = 2,       // Decode-critical op on CPU
    LLAMA_FALLBACK_VIOL_MISSING_GPU_KERNEL = 3,        // Missing GPU kernel
    LLAMA_FALLBACK_VIOL_MEMORY_PLACEMENT = 4,          // Memory placement mismatch
    LLAMA_FALLBACK_VIOL_CAPABILITY_UNAVAILABLE = 5,    // Capability unavailable
    LLAMA_FALLBACK_VIOL_OOM_DURING_DECODE = 6,         // OOM during decode
    LLAMA_FALLBACK_VIOL_INVALID_GPU_STATE = 7,         // Invalid GPU state
    LLAMA_FALLBACK_VIOL_SHAPE_INCOMPATIBLE = 8,        // Incompatible tensor shape
};

// ============================================================================
// FALLBACK DETECTION STRUCTURES
// ============================================================================

/**
 * Structure recording a detected fallback attempt
 */
struct llama_fallback_attempt_record {
    const char* operation_name;                        // Operation attempting fallback
    enum llama_fallback_attempt_location location;    // Where fallback was detected
    enum llama_gpu_unavailability_reason reason;      // Why GPU unavailable
    bool is_decode_critical;                          // Is operation decode-critical?
    uint64_t detection_time_us;                       // When detected
    const char* diagnostic_message;                   // Diagnostic details
};

/**
 * Structure tracking fallback elimination state
 */
struct llama_fallback_elimination_state {
    // Fallback detection
    uint64_t total_attempts;                          // Total fallback attempts detected
    struct llama_fallback_attempt_record* attempts;   // Array of attempt records
    int max_attempts;                                 // Capacity

    // Violation tracking
    int violation_count;                              // Total violations
    enum llama_fallback_violation_type last_violation_type;
    const char* last_violation_message;

    // Audit state
    int audit_count;                                  // Number of audits performed
    int problematic_paths_found;                      // Paths that allow fallback

    // Decode-specific tracking
    bool decode_active;                               // Currently decoding?
    bool strict_enforcement_active;                   // Strict enforcement enabled?
};

// ============================================================================
// FALLBACK ELIMINATION CONTROL
// ============================================================================

/**
 * Initialize fallback elimination tracking
 */
int llama_fallback_elimination_init(void);

/**
 * Audit all fallback paths in the codebase
 * Identifies locations where CPU fallback can occur.
 * Returns: 0 = audit complete, count of problematic paths in *problematic_count
 */
int llama_audit_all_fallback_paths(int* problematic_count);

/**
 * Check if a fallback path exists for a given operation
 * Returns: 0 = no fallback, 1 = fallback exists, -1 = error
 */
int llama_fallback_path_exists(
    const char* operation_name,
    bool* fallback_path_exists
);

/**
 * Get diagnostic information about why fallback paths exist
 */
const char* llama_get_fallback_path_diagnostics(const char* operation_name);

// ============================================================================
// BACKEND DISPATCH HARDENING
// ============================================================================

/**
 * ENFORCEMENT POINT 1: Harden backend dispatch for decode-critical ops
 * Assert backend compatibility before execution.
 * If a decode-critical op reaches CPU dispatch → abort immediately.
 *
 * Returns: 0 = dispatch OK, -1 = FATAL (CPU dispatch for decode-critical)
 */
int llama_enforce_hardened_backend_dispatch(
    const char* operation_name,
    bool is_decode_critical,
    enum llama_backend_type assigned_backend
);

/**
 * ENFORCEMENT POINT 2: Fail on missing GPU kernel support
 * If a required GPU kernel is unavailable, reject decode admission.
 * Do not defer discovery until execution time.
 *
 * Returns: 0 = kernel available, -1 = FATAL (kernel missing)
 */
int llama_enforce_gpu_kernel_availability(
    const char* operation_name,
    bool kernel_available
);

/**
 * ENFORCEMENT POINT 3: Fail on memory-placement-induced fallback
 * If tensor placement would cause CPU execution, treat as invariant violation.
 * Do not copy tensors back to CPU.
 *
 * Returns: 0 = placement OK, -1 = FATAL (placement mismatch)
 */
int llama_enforce_gpu_memory_placement(
    const char* operation_name,
    bool tensor_on_gpu,
    bool operation_requires_gpu
);

/**
 * ENFORCEMENT POINT 4: Fail on runtime capability changes
 * If GPU execution becomes invalid mid-decode, terminate session.
 * Do not reroute work to CPU.
 *
 * Returns: 0 = capability valid, -1 = FATAL (capability lost)
 */
int llama_enforce_gpu_capability_stability(
    const char* operation_name,
    bool capability_available
);

/**
 * ENFORCEMENT POINT 5: Detect silent fallback attempts
 * Any attempt to fall back to CPU for decode-critical ops → abort.
 *
 * Returns: 0 = no fallback, -1 = FATAL (fallback attempted)
 */
int llama_detect_silent_fallback_attempt(
    const char* operation_name,
    bool is_decode_critical,
    bool fallback_attempted,
    enum llama_gpu_unavailability_reason reason
);

/**
 * ENFORCEMENT POINT 6: Validate decode-critical ops remain GPU-bound
 * Pre-execution verification that decode-critical ops haven't fallen back to CPU.
 *
 * Returns: 0 = GPU-bound, -1 = FATAL (CPU fallback detected)
 */
int llama_validate_decode_critical_gpu_binding(
    const char* operation_name,
    enum llama_backend_type executing_backend
);

// ============================================================================
// DECODE VS NON-DECODE DIFFERENTIATION
// ============================================================================

/**
 * Allow CPU fallback ONLY for non-critical tasks
 * For decode-critical tasks, CPU fallback is forbidden under all conditions.
 *
 * Returns: 0 = fallback allowed, -1 = FATAL (fallback not allowed)
 */
int llama_check_fallback_allowed_for_task_type(
    bool is_decode_critical,
    bool fallback_to_cpu_requested
);

/**
 * Assert that non-critical tasks CAN use CPU if needed
 * Returns: 0 = OK, -1 = configuration error
 */
int llama_assert_noncritical_can_use_cpu(bool is_decode_critical);

/**
 * Assert that decode-critical tasks CANNOT use CPU
 * Returns: 0 = OK, -1 = configuration error
 */
int llama_assert_decode_critical_gpu_only(bool is_decode_critical);

// ============================================================================
// GPU EXECUTION AVAILABILITY VERIFICATION
// ============================================================================

/**
 * Verify GPU kernel is available before decode begins
 * Called during admission control to catch kernel issues upfront.
 *
 * Returns: 0 = all kernels available, -1 = kernel missing
 */
int llama_verify_gpu_kernels_available_for_decode(void);

/**
 * Verify GPU memory placement is compatible
 * Called during admission control to check tensor placement.
 *
 * Returns: 0 = placement OK, -1 = placement issue found
 */
int llama_verify_gpu_memory_placement_for_decode(void);

/**
 * Verify GPU capability is sufficient for decode
 * Called during admission control to validate GPU features.
 *
 * Returns: 0 = capabilities sufficient, -1 = insufficient
 */
int llama_verify_gpu_capabilities_for_decode(void);

// ============================================================================
// FALLBACK MONITORING & DETECTION
// ============================================================================

/**
 * Monitor for fallback attempts at backend dispatch
 * Returns: 0 = no fallback, -1 = FATAL (fallback detected)
 */
int llama_monitor_backend_dispatch_fallback(
    const char* operation_name,
    bool is_decode_critical,
    bool fallback_triggered,
    enum llama_gpu_unavailability_reason reason
);

/**
 * Monitor for fallback attempts at graph execution
 * Returns: 0 = no fallback, -1 = FATAL (fallback detected)
 */
int llama_monitor_graph_execution_fallback(
    const char** operation_names,
    bool* are_decode_critical,
    bool* fallback_triggered,
    int num_operations,
    enum llama_gpu_unavailability_reason* reasons
);

/**
 * Monitor for fallback attempts at sampling
 * Returns: 0 = no fallback, -1 = FATAL (fallback detected)
 */
int llama_monitor_sampling_fallback(
    bool is_decode_critical,
    bool fallback_triggered,
    enum llama_gpu_unavailability_reason reason
);

/**
 * Check if GPU state became invalid (e.g., OOM, device error)
 * Returns: 0 = valid, -1 = FATAL (invalid state)
 */
int llama_check_gpu_state_validity_during_decode(void);

// ============================================================================
// EXPLICIT DIAGNOSTICS
// ============================================================================

/**
 * Report a fallback elimination violation with full diagnostics
 */
void llama_report_fallback_violation(
    const char* operation_name,
    enum llama_fallback_violation_type violation_type,
    enum llama_gpu_unavailability_reason reason,
    bool is_decode_critical
);

/**
 * Print comprehensive fallback violation diagnostics
 */
void llama_print_fallback_violation_diagnostics(
    const char* operation_name,
    enum llama_fallback_violation_type violation_type,
    enum llama_gpu_unavailability_reason reason,
    bool is_decode_critical
);

/**
 * Convert fallback location to human-readable string
 */
static inline const char* llama_fallback_location_name(
    enum llama_fallback_attempt_location location
) {
    switch (location) {
        case LLAMA_FALLBACK_BACKEND_DISPATCH:
            return "BACKEND_DISPATCH";
        case LLAMA_FALLBACK_GRAPH_EXECUTION:
            return "GRAPH_EXECUTION";
        case LLAMA_FALLBACK_KERNEL_DISPATCH:
            return "KERNEL_DISPATCH";
        case LLAMA_FALLBACK_SAMPLING:
            return "SAMPLING";
        case LLAMA_FALLBACK_MEMORY_PLACEMENT:
            return "MEMORY_PLACEMENT";
        case LLAMA_FALLBACK_CAPABILITY_CHECK:
            return "CAPABILITY_CHECK";
        case LLAMA_FALLBACK_OOM_RECOVERY:
            return "OOM_RECOVERY";
        case LLAMA_FALLBACK_INVALID_STATE:
            return "INVALID_STATE";
        case LLAMA_FALLBACK_SHAPE_MISMATCH:
            return "SHAPE_MISMATCH";
        default:
            return "UNKNOWN";
    }
}

/**
 * Convert GPU unavailability reason to human-readable string
 */
static inline const char* llama_gpu_unavailability_reason_name(
    enum llama_gpu_unavailability_reason reason
) {
    switch (reason) {
        case LLAMA_GPU_AVAILABLE:
            return "AVAILABLE";
        case LLAMA_GPU_UNAVAIL_NO_KERNEL:
            return "NO_KERNEL";
        case LLAMA_GPU_UNAVAIL_UNSUPPORTED_DTYPE:
            return "UNSUPPORTED_DTYPE";
        case LLAMA_GPU_UNAVAIL_UNSUPPORTED_SHAPE:
            return "UNSUPPORTED_SHAPE";
        case LLAMA_GPU_UNAVAIL_MEMORY_PLACEMENT:
            return "MEMORY_PLACEMENT";
        case LLAMA_GPU_UNAVAIL_INVALID_CAPABILITY:
            return "INVALID_CAPABILITY";
        case LLAMA_GPU_UNAVAIL_OOM:
            return "OOM";
        case LLAMA_GPU_UNAVAIL_INVALID_STATE:
            return "INVALID_STATE";
        case LLAMA_GPU_UNAVAIL_UNKNOWN:
            return "UNKNOWN";
        default:
            return "(invalid)";
    }
}

/**
 * Convert violation type to human-readable string
 */
static inline const char* llama_fallback_violation_type_name(
    enum llama_fallback_violation_type violation_type
) {
    switch (violation_type) {
        case LLAMA_FALLBACK_VIOL_SILENT_FALLBACK:
            return "SILENT_FALLBACK";
        case LLAMA_FALLBACK_VIOL_DECODE_CRITICAL_CPU:
            return "DECODE_CRITICAL_CPU";
        case LLAMA_FALLBACK_VIOL_MISSING_GPU_KERNEL:
            return "MISSING_GPU_KERNEL";
        case LLAMA_FALLBACK_VIOL_MEMORY_PLACEMENT:
            return "MEMORY_PLACEMENT";
        case LLAMA_FALLBACK_VIOL_CAPABILITY_UNAVAILABLE:
            return "CAPABILITY_UNAVAILABLE";
        case LLAMA_FALLBACK_VIOL_OOM_DURING_DECODE:
            return "OOM_DURING_DECODE";
        case LLAMA_FALLBACK_VIOL_INVALID_GPU_STATE:
            return "INVALID_GPU_STATE";
        case LLAMA_FALLBACK_VIOL_SHAPE_INCOMPATIBLE:
            return "SHAPE_INCOMPATIBLE";
        default:
            return "UNKNOWN";
    }
}

// ============================================================================
// VALIDATION HOOKS
// ============================================================================

/**
 * In debug builds, log any attempted fallback paths
 * Use this to catch regressions early.
 */
void llama_debug_log_fallback_attempt(
    const char* operation_name,
    enum llama_fallback_attempt_location location,
    enum llama_gpu_unavailability_reason reason
);

/**
 * In debug builds, assert fallback paths are unreachable for decode-critical ops
 */
int llama_debug_assert_no_decode_critical_fallback(
    const char* operation_name,
    bool is_decode_critical,
    bool fallback_attempted
);

/**
 * Enable/disable debug fallback logging
 */
void llama_set_debug_fallback_logging_enabled(bool enabled);

/**
 * Check if debug fallback logging is enabled
 */
bool llama_get_debug_fallback_logging_enabled(void);

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

/**
 * Enable/disable strict fallback elimination enforcement
 * When enabled, all fallback attempts cause immediate failure.
 */
void llama_set_fallback_elimination_enforcement_strict(bool enforce_strict);

/**
 * Get current enforcement mode
 */
bool llama_get_fallback_elimination_enforcement_strict(void);

/**
 * Get total fallback violations detected
 */
int llama_get_fallback_violation_count(void);

/**
 * Get decode-critical fallback violations
 */
int llama_get_decode_critical_fallback_violation_count(void);

/**
 * Reset fallback violation counters
 */
void llama_reset_fallback_violation_counters(void);

// ============================================================================
// EXPLICIT ZERO-FALLBACK POLICY STATEMENT
// ============================================================================

/**
 * Print the zero-fallback policy principle
 */
void llama_print_zero_fallback_policy_statement(void);

// ============================================================================
// VALIDATION AND SELF-TEST
// ============================================================================

/**
 * Self-test: verify fallback elimination mechanism works correctly
 */
int llama_fallback_elimination_selftest(void);

