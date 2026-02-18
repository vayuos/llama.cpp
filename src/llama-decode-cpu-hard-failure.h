/**
 * SECTION 4: Add Hard Failure on Decode-Critical CPU Execution
 *
 * This file implements hard failure checks at execution boundaries to ensure
 * that any attempt to execute decode-critical work on the CPU causes an
 * immediate fatal error.
 *
 * Core Principle:
 * "CPU execution on the decode-critical path is a fatal error, not a fallback option.
 *  Any violation is caught immediately. There is no recovery, rerouting, or degradation."
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>

// ============================================================================
// DECODE-CRITICAL CPU EXECUTION VIOLATION DETECTION
// ============================================================================

/**
 * Enum defining where decode-critical CPU execution violation was detected
 */
enum llama_decode_cpu_violation_location {
    LLAMA_CPU_VIOLATION_UNKNOWN = 0,
    LLAMA_CPU_VIOLATION_BACKEND_DISPATCH = 1,       // During backend dispatch
    LLAMA_CPU_VIOLATION_GRAPH_EXECUTION = 2,        // During graph execution
    LLAMA_CPU_VIOLATION_KERNEL_DISPATCH = 3,        // During kernel dispatch
    LLAMA_CPU_VIOLATION_SAMPLING = 4,               // During sampling operation
    LLAMA_CPU_VIOLATION_NODE_EXECUTION = 5,         // Node execution on CPU
    LLAMA_CPU_VIOLATION_MIXED_BACKEND_GRAPH = 6,    // Mixed CPU/GPU in decode graph
};

/**
 * Structure holding decode-critical CPU violation information
 */
struct llama_decode_cpu_violation {
    bool violation_detected;                        // True if violation occurred
    enum llama_decode_cpu_violation_location location;  // Where it was detected
    const char* operation_name;                     // Which decode-critical op failed
    const char* assigned_backend;                   // Backend it was assigned to
    const char* violation_message;                  // Detailed violation message
};

// ============================================================================
// HARD FAILURE ENFORCEMENT FUNCTIONS
// ============================================================================

/**
 * Assert that an operation is NOT decode-critical, or if it is, must be GPU-bound.
 * Called before every operation execution.
 *
 * Returns:
 *  0 = Operation is allowed to execute (not decode-critical OR is GPU-bound)
 * -1 = FATAL: Decode-critical op assigned to CPU (execution aborted)
 */
int llama_enforce_no_decode_critical_on_cpu(
    const char* operation_name,
    bool is_decode_critical,
    const char* assigned_backend
);

/**
 * Assert at backend dispatch: decode-critical ops cannot go to CPU
 *
 * Called during backend dispatch decision.
 * If a decode-critical op is about to be routed to CPU, abort immediately.
 *
 * Returns: 0 = Dispatch allowed, -1 = FATAL (CPU routing of decode-critical op)
 */
int llama_enforce_decode_critical_gpu_at_dispatch(
    const char* operation_name,
    bool is_decode_critical,
    const char* target_backend
);

/**
 * Assert during kernel dispatch: decode-critical kernels must be GPU-native
 *
 * Called when selecting kernel for execution.
 * If a decode-critical op has no GPU kernel, abort (don't fall back to CPU).
 *
 * Returns: 0 = Kernel available on GPU, -1 = FATAL (no GPU kernel)
 */
int llama_enforce_decode_critical_kernel_gpu_only(
    const char* operation_name,
    bool is_decode_critical,
    bool gpu_kernel_available,
    bool cpu_fallback_exists
);

/**
 * Assert during graph execution: detect mixed CPU/GPU decode graphs
 *
 * Called during graph execution to check for mixed-backend decode graphs.
 * If decode-critical nodes are split across CPU and GPU, abort.
 *
 * Returns: 0 = Graph is uniform (all decode-critical on GPU), -1 = FATAL (mixed backend)
 */
int llama_enforce_uniform_gpu_decode_graph(
    const char** decode_critical_ops,
    const char** op_backends,
    int num_ops
);

/**
 * Assert during sampling: sampling is decode-critical
 *
 * If sampling is detected on CPU backend, either:
 * - Fail immediately (if GPU sampling available), or
 * - Fail with "known limitation" marker (if GPU sampling not yet implemented)
 *
 * Returns: 0 = Sampling allowed on this backend, -1 = FATAL (CPU sampling forbidden)
 */
int llama_enforce_no_cpu_sampling(
    const char* sampling_backend,
    bool gpu_sampling_available
);

/**
 * Assert operation backend matches operation type
 *
 * For decode-critical ops: must be GPU
 * For non-critical ops: can be CPU or GPU
 *
 * Returns: 0 = Valid assignment, -1 = FATAL (decode-critical on CPU)
 */
int llama_assert_operation_backend_valid(
    const char* operation_name,
    bool is_decode_critical,
    const char* assigned_backend
);

// ============================================================================
// MIXED-BACKEND GRAPH DETECTION
// ============================================================================

/**
 * Detect mixed-backend decode graphs (decode-critical ops split across CPU and GPU)
 *
 * Returns:
 *  0 = Graph is uniform (no mixing)
 * -1 = FATAL: Graph has mixed backends for decode-critical ops
 */
int llama_detect_mixed_backend_decode_graph(
    const char** all_ops,
    bool* op_is_decode_critical,
    const char** op_backends,
    int num_ops,
    struct llama_decode_cpu_violation* violation_info
);

/**
 * Verify all decode-critical nodes in graph are GPU-bound
 * Called before graph execution begins.
 *
 * Returns: 0 = All decode-critical ops GPU-bound, -1 = FATAL (mixing detected)
 */
int llama_verify_decode_critical_nodes_gpu_only(
    const char** node_names,
    bool* node_is_decode_critical,
    const char** node_backends,
    int num_nodes
);

// ============================================================================
// VIOLATION DETECTION AND REPORTING
// ============================================================================

/**
 * Record a decode-critical CPU execution violation
 */
void llama_record_decode_cpu_violation(
    struct llama_decode_cpu_violation* violation,
    enum llama_decode_cpu_violation_location location,
    const char* operation_name,
    const char* assigned_backend,
    const char* violation_message
);

/**
 * Convert violation location to human-readable string
 */
static inline const char* llama_cpu_violation_location_name(
    enum llama_decode_cpu_violation_location location
) {
    switch (location) {
        case LLAMA_CPU_VIOLATION_UNKNOWN:
            return "UNKNOWN";
        case LLAMA_CPU_VIOLATION_BACKEND_DISPATCH:
            return "BACKEND_DISPATCH";
        case LLAMA_CPU_VIOLATION_GRAPH_EXECUTION:
            return "GRAPH_EXECUTION";
        case LLAMA_CPU_VIOLATION_KERNEL_DISPATCH:
            return "KERNEL_DISPATCH";
        case LLAMA_CPU_VIOLATION_SAMPLING:
            return "SAMPLING";
        case LLAMA_CPU_VIOLATION_NODE_EXECUTION:
            return "NODE_EXECUTION";
        case LLAMA_CPU_VIOLATION_MIXED_BACKEND_GRAPH:
            return "MIXED_BACKEND_GRAPH";
        default:
            return "(invalid)";
    }
}

/**
 * Print detailed violation diagnostics
 * Called when decode-critical CPU execution is detected.
 */
void llama_print_decode_cpu_violation_diagnostics(
    const struct llama_decode_cpu_violation* violation
);

// ============================================================================
// DEBUG ENFORCEMENT MODE
// ============================================================================

/**
 * Enable/disable strict decode-critical CPU execution checking
 * When enabled, violations cause immediate hard failure.
 * When disabled, violations are logged but may allow execution (testing only).
 */
void llama_set_decode_cpu_enforcement_strict(bool enforce_strict);

/**
 * Get current enforcement mode
 */
bool llama_get_decode_cpu_enforcement_strict(void);

/**
 * Get enforcement violation count
 * Tracks total violations detected (in any mode)
 */
int llama_get_decode_cpu_violation_count(void);

/**
 * Reset enforcement violation counter
 */
void llama_reset_decode_cpu_violation_counter(void);

// ============================================================================
// EXPLICIT CPU EXECUTION PROHIBITION STATEMENT
// ============================================================================

/**
 * Print the decode-critical CPU execution prohibition statement
 */
void llama_print_decode_critical_cpu_prohibition_statement(void);

// ============================================================================
// VALIDATION AND SELF-TEST
// ============================================================================

/**
 * Self-test: verify hard failure mechanism works correctly
 */
int llama_decode_cpu_hard_failure_selftest(void);

