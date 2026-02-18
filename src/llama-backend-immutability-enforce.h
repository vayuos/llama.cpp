/**
 * SECTION 6: Remove Runtime Backend Switching During Decode
 *
 * This file implements backend immutability enforcement to eliminate all runtime
 * backend switching during the decode phase. Once decode begins, backend selection
 * is frozen and immutable.
 *
 * Core Principle:
 * "Backend ownership is resolved once before decode and remains immutable for the
 *  entire decode lifetime. No per-token, per-layer, or per-operation backend
 *  re-evaluation or switching is permitted. Backend changes trigger immediate failure."
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
// BACKEND IMMUTABILITY STATE DEFINITION
// ============================================================================

/**
 * Enum defining the phase of backend resolution
 */
enum llama_backend_resolution_phase {
    LLAMA_BACKEND_PHASE_UNINITIALIZED = 0,     // No backend decision made yet
    LLAMA_BACKEND_PHASE_PREFILL = 1,           // During prefill (flexible, can change)
    LLAMA_BACKEND_PHASE_DECODE_FROZEN = 2,     // During decode (frozen, immutable)
    LLAMA_BACKEND_PHASE_TERMINATED = 3,        // Decode terminated or invalid
};

/**
 * Enum defining locations where backend immutability can be violated
 */
enum llama_backend_immutability_violation_location {
    LLAMA_BACKEND_VIOLATION_UNKNOWN = 0,
    LLAMA_BACKEND_VIOLATION_DECODE_LOOP_CHECK = 1,       // Backend checked in decode loop
    LLAMA_BACKEND_VIOLATION_PER_TOKEN_SWITCH = 2,        // Backend changed per-token
    LLAMA_BACKEND_VIOLATION_PER_LAYER_SWITCH = 3,        // Backend changed per-layer
    LLAMA_BACKEND_VIOLATION_PER_OP_SWITCH = 4,           // Backend changed per-operation
    LLAMA_BACKEND_VIOLATION_SHAPE_CHANGE_REEVAL = 5,     // Backend re-evaluated on shape change
    LLAMA_BACKEND_VIOLATION_CONTEXT_CHANGE_REEVAL = 6,   // Backend re-evaluated on context change
    LLAMA_BACKEND_VIOLATION_FALLBACK_PATH = 7,           // Fallback path attempted during decode
    LLAMA_BACKEND_VIOLATION_HEURISTIC_SELECTION = 8,     // Runtime heuristic selection during decode
    LLAMA_BACKEND_VIOLATION_CAPABILITY_CHECK = 9,        // Capability check during decode
    LLAMA_BACKEND_VIOLATION_INVALIDATION = 10,           // Backend became invalid during decode
};

/**
 * Enum defining backend types
 */
enum llama_backend_type {
    LLAMA_BACKEND_UNKNOWN = 0,
    LLAMA_BACKEND_CPU = 1,
    LLAMA_BACKEND_CUDA = 2,
    LLAMA_BACKEND_HIP = 3,
    LLAMA_BACKEND_METAL = 4,
    LLAMA_BACKEND_VULKAN = 5,
    LLAMA_BACKEND_ONEAPI = 6,
};

// ============================================================================
// BACKEND RESOLUTION RECORD
// ============================================================================

/**
 * Structure recording the resolved backend for a specific operation/layer/tensor
 */
struct llama_backend_resolution_record {
    const char* resource_identifier;           // Op name, layer name, or tensor name
    enum llama_backend_type resolved_backend;  // Resolved backend
    uint64_t resolution_time_us;               // When resolution occurred
    bool locked;                               // Is this resolution immutable?
    const char* resolution_reason;             // Why was this backend selected?
};

/**
 * Structure tracking backend immutability state for entire decode
 */
struct llama_backend_immutability_state {
    enum llama_backend_resolution_phase phase;              // Current resolution phase
    enum llama_backend_type decode_backend;                 // Frozen backend for decode phase

    // Resolution tracking
    uint64_t resolution_count;                              // Number of resolved operations
    struct llama_backend_resolution_record* resolutions;    // Array of resolution records
    int max_resolutions;                                    // Capacity

    // Invalidation tracking
    bool backend_invalid;                                   // Backend became invalid?
    const char* invalidation_reason;                        // Why invalid?
    uint64_t invalidation_time_us;                          // When invalidated

    // Immutability enforcement
    bool immutability_locked;                               // Backend frozen for decode?
    uint64_t freeze_time_us;                                // When backend was frozen

    // Violation tracking
    int violation_count;                                    // Total violations detected
    enum llama_backend_immutability_violation_location last_violation_location;
    const char* last_violation_message;
};

// ============================================================================
// BACKEND IMMUTABILITY CONTROL
// ============================================================================

/**
 * Initialize backend immutability tracking
 * Called once at context creation
 */
int llama_backend_immutability_init(void);

/**
 * Freeze backend resolution before first decode token
 * After this call, backend selection is immutable.
 *
 * Returns: 0 = Success, -1 = FATAL (cannot freeze backend)
 */
int llama_backend_immutability_freeze_for_decode(enum llama_backend_type backend);

/**
 * Check if backend is currently frozen
 */
bool llama_backend_immutability_is_frozen(void);

/**
 * Get the frozen backend type
 */
enum llama_backend_type llama_backend_immutability_get_frozen_backend(void);

/**
 * Verify backend has not changed since admission
 * Called during decode to ensure immutability invariant.
 *
 * Returns: 0 = Backend unchanged (valid), -1 = FATAL (backend changed)
 */
int llama_backend_immutability_verify_unchanged(enum llama_backend_type expected_backend);

/**
 * Record that backend became invalid during decode
 * Terminates decode phase.
 *
 * Returns: -1 (always fails)
 */
int llama_backend_immutability_record_invalidation(const char* reason);

// ============================================================================
// DECODE-LOOP BACKEND CHECK PROHIBITION
// ============================================================================

/**
 * Assert that backend resolution is NOT occurring in decode loop
 * Any backend check, heuristic selection, or re-evaluation during decode
 * triggers immediate failure.
 *
 * Returns: 0 = No backend check (valid), -1 = FATAL (backend check detected)
 */
int llama_assert_no_backend_check_in_decode_loop(
    const char* location_description,
    bool backend_check_attempted
);

/**
 * Assert no capability checks during decode
 * Capability checks imply potential fallback, which is forbidden.
 *
 * Returns: 0 = No capability check, -1 = FATAL (capability check during decode)
 */
int llama_assert_no_capability_check_during_decode(
    const char* operation_name,
    bool capability_check_performed
);

/**
 * Assert no heuristic backend selection during decode
 * Heuristic selection implies flexible/dynamic backend selection (forbidden).
 *
 * Returns: 0 = No heuristics, -1 = FATAL (heuristic selection during decode)
 */
int llama_assert_no_heuristic_backend_selection_during_decode(
    const char* selection_heuristic,
    bool heuristic_applied
);

// ============================================================================
// BACKEND IMMUTABILITY ENFORCEMENT POINTS
// ============================================================================

/**
 * ENFORCEMENT POINT 1: Backend immutability at graph execution start
 * Verify all ops in decode graph use frozen backend.
 *
 * Returns: 0 = All GPU-bound, -1 = FATAL (mixed backends or CPU found)
 */
int llama_enforce_backend_immutability_at_graph_execution(
    const char** operation_names,
    enum llama_backend_type* operation_backends,
    int num_operations
);

/**
 * ENFORCEMENT POINT 2: Backend immutability at operation dispatch
 * Verify operation is using frozen backend, not re-evaluating.
 *
 * Returns: 0 = Correct backend, -1 = FATAL (wrong backend or re-evaluation)
 */
int llama_enforce_backend_immutability_at_dispatch(
    const char* operation_name,
    enum llama_backend_type operation_backend,
    bool backend_was_reevaluated
);

/**
 * ENFORCEMENT POINT 3: Backend immutability across shape changes
 * Prevent backend re-evaluation when tensor shapes change.
 *
 * Returns: 0 = No re-evaluation, -1 = FATAL (backend re-evaluated on shape change)
 */
int llama_enforce_no_backend_reeval_on_shape_change(
    const char* operation_name,
    bool shape_changed,
    bool backend_reevaluated
);

/**
 * ENFORCEMENT POINT 4: Backend immutability across context changes
 * Prevent backend re-evaluation when context size changes.
 *
 * Returns: 0 = No re-evaluation, -1 = FATAL (backend re-evaluated on context change)
 */
int llama_enforce_no_backend_reeval_on_context_change(
    const char* operation_name,
    int old_context_size,
    int new_context_size,
    bool backend_reevaluated
);

/**
 * ENFORCEMENT POINT 5: Per-token backend immutability
 * Verify backend hasn't changed between tokens.
 *
 * Returns: 0 = Same backend, -1 = FATAL (backend changed per-token)
 */
int llama_enforce_backend_immutability_per_token(
    uint64_t token_id,
    const char* operation_name,
    enum llama_backend_type current_backend,
    enum llama_backend_type expected_backend
);

/**
 * ENFORCEMENT POINT 6: Per-layer backend immutability
 * Verify all layers use same backend.
 *
 * Returns: 0 = Uniform backend, -1 = FATAL (mixed backends across layers)
 */
int llama_enforce_backend_immutability_per_layer(
    int layer_id,
    const char* layer_name,
    enum llama_backend_type layer_backend,
    enum llama_backend_type expected_backend
);

/**
 * ENFORCEMENT POINT 7: Per-operation backend immutability
 * Verify operation is using same backend as previous execution.
 *
 * Returns: 0 = Same backend, -1 = FATAL (operation backend changed)
 */
int llama_enforce_backend_immutability_per_operation(
    const char* operation_name,
    enum llama_backend_type current_backend,
    enum llama_backend_type previous_backend,
    bool backend_changed
);

/**
 * ENFORCEMENT POINT 8: No fallback paths during decode
 * Verify no fallback mechanisms are being invoked.
 *
 * Returns: 0 = No fallback, -1 = FATAL (fallback path invoked during decode)
 */
int llama_enforce_no_fallback_paths_during_decode(
    const char* operation_name,
    bool fallback_attempted
);

/**
 * ENFORCEMENT POINT 9: Backend invalid during decode
 * Detect if backend became invalid while decode was in progress.
 *
 * Returns: 0 = Backend still valid, -1 = FATAL (backend invalidated)
 */
int llama_enforce_backend_validity_during_decode(
    const char* validity_check_location,
    bool backend_is_valid
);

/**
 * ENFORCEMENT POINT 10: Immutability pre-execution verification
 * Final verification before executing decode-critical operation.
 *
 * Returns: 0 = Immutability intact, -1 = FATAL (immutability violated)
 */
int llama_enforce_immutability_pre_execution(
    const char* operation_name,
    enum llama_backend_type operation_backend,
    bool immutability_intact
);

// ============================================================================
// PREFILL vs DECODE PHASE SEPARATION
// ============================================================================

/**
 * Enter prefill phase (flexible backend selection allowed)
 */
int llama_backend_phase_enter_prefill(void);

/**
 * Exit prefill phase and enter decode phase (freeze backend)
 * After this call, backend is immutable.
 *
 * Returns: 0 = Success, -1 = FATAL (cannot transition to decode)
 */
int llama_backend_phase_exit_prefill_enter_decode(void);

/**
 * Check if currently in decode phase
 */
bool llama_backend_phase_in_decode(void);

/**
 * Check if currently in prefill phase
 */
bool llama_backend_phase_in_prefill(void);

/**
 * Get current backend resolution phase
 */
enum llama_backend_resolution_phase llama_backend_phase_get_current(void);

// ============================================================================
// VIOLATION DETECTION AND REPORTING
// ============================================================================

/**
 * Record a backend immutability violation
 */
void llama_record_backend_immutability_violation(
    enum llama_backend_immutability_violation_location location,
    const char* operation_name,
    const char* violation_message
);

/**
 * Convert violation location to human-readable string
 */
static inline const char* llama_backend_violation_location_name(
    enum llama_backend_immutability_violation_location location
) {
    switch (location) {
        case LLAMA_BACKEND_VIOLATION_UNKNOWN:
            return "UNKNOWN";
        case LLAMA_BACKEND_VIOLATION_DECODE_LOOP_CHECK:
            return "DECODE_LOOP_CHECK";
        case LLAMA_BACKEND_VIOLATION_PER_TOKEN_SWITCH:
            return "PER_TOKEN_SWITCH";
        case LLAMA_BACKEND_VIOLATION_PER_LAYER_SWITCH:
            return "PER_LAYER_SWITCH";
        case LLAMA_BACKEND_VIOLATION_PER_OP_SWITCH:
            return "PER_OP_SWITCH";
        case LLAMA_BACKEND_VIOLATION_SHAPE_CHANGE_REEVAL:
            return "SHAPE_CHANGE_REEVAL";
        case LLAMA_BACKEND_VIOLATION_CONTEXT_CHANGE_REEVAL:
            return "CONTEXT_CHANGE_REEVAL";
        case LLAMA_BACKEND_VIOLATION_FALLBACK_PATH:
            return "FALLBACK_PATH";
        case LLAMA_BACKEND_VIOLATION_HEURISTIC_SELECTION:
            return "HEURISTIC_SELECTION";
        case LLAMA_BACKEND_VIOLATION_CAPABILITY_CHECK:
            return "CAPABILITY_CHECK";
        case LLAMA_BACKEND_VIOLATION_INVALIDATION:
            return "INVALIDATION";
        default:
            return "(invalid)";
    }
}

/**
 * Convert backend type to human-readable string
 */
static inline const char* llama_backend_type_name(enum llama_backend_type backend) {
    switch (backend) {
        case LLAMA_BACKEND_UNKNOWN:
            return "UNKNOWN";
        case LLAMA_BACKEND_CPU:
            return "CPU";
        case LLAMA_BACKEND_CUDA:
            return "CUDA";
        case LLAMA_BACKEND_HIP:
            return "HIP";
        case LLAMA_BACKEND_METAL:
            return "METAL";
        case LLAMA_BACKEND_VULKAN:
            return "VULKAN";
        case LLAMA_BACKEND_ONEAPI:
            return "ONEAPI";
        default:
            return "(invalid)";
    }
}

/**
 * Print detailed backend immutability violation diagnostics
 */
void llama_print_backend_immutability_violation_diagnostics(
    const struct llama_backend_immutability_state* state,
    enum llama_backend_immutability_violation_location violation_location,
    const char* violation_message
);

// ============================================================================
// BACKEND IMMUTABILITY AUDIT
// ============================================================================

/**
 * Audit backend resolution code for immutability violations
 * Identifies problematic code patterns that violate immutability.
 *
 * Returns: 0 = No violations found, >0 = Number of violations found
 */
int llama_audit_backend_resolution_code(void);

/**
 * Check if a code location is in decode loop
 */
bool llama_is_in_decode_loop(void);

/**
 * Check if a code location is attempting backend re-evaluation
 */
bool llama_is_attempting_backend_reeval(void);

/**
 * Check if a code location is attempting heuristic selection
 */
bool llama_is_attempting_heuristic_selection(void);

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

/**
 * Enable/disable strict backend immutability enforcement
 * When enabled, any violation causes immediate failure.
 * When disabled, violations are logged but may allow execution (testing only).
 */
void llama_set_backend_immutability_enforcement_strict(bool enforce_strict);

/**
 * Get current enforcement mode
 */
bool llama_get_backend_immutability_enforcement_strict(void);

/**
 * Get backend immutability violation count
 */
int llama_get_backend_immutability_violation_count(void);

/**
 * Reset backend immutability violation counter
 */
void llama_reset_backend_immutability_violation_counter(void);

// ============================================================================
// EXPLICIT BACKEND IMMUTABILITY STATEMENT
// ============================================================================

/**
 * Print the backend immutability principle and enforcement strategy
 */
void llama_print_backend_immutability_statement(void);

// ============================================================================
// VALIDATION AND SELF-TEST
// ============================================================================

/**
 * Self-test: verify backend immutability mechanism works correctly
 */
int llama_backend_immutability_selftest(void);

