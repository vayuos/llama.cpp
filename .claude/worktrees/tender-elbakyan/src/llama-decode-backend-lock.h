/**
 * SECTION 10: Add Decode-Time Backend Lock
 * Header
 *
 * This file implements a backend lock primitive that guarantees backend ownership
 * cannot change for the entire duration of decode. Once decode begins, backend
 * selection must be immutable until decode terminates.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "../ggml/include/ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BACKEND LOCK STATE ENUMERATION
// ============================================================================

/**
 * Backend lock state - tracks the lifecycle of backend ownership during decode
 */
enum llama_backend_lock_state {
    LLAMA_BACKEND_LOCK_UNLOCKED = 0,        // Backend not locked (prefill phase)
    LLAMA_BACKEND_LOCK_ACQUIRING = 1,       // Lock acquisition in progress
    LLAMA_BACKEND_LOCK_ACQUIRED = 2,        // Backend lock successfully acquired
    LLAMA_BACKEND_LOCK_RELEASING = 3,       // Lock release in progress
    LLAMA_BACKEND_LOCK_RELEASED = 4,        // Lock released (decode complete)
    LLAMA_BACKEND_LOCK_INVALID = 5,         // Lock in invalid state (error condition)
};

// ============================================================================
// BACKEND LOCK INVALIDATION REASON
// ============================================================================

/**
 * Reasons why a locked backend might become invalid during decode
 */
enum llama_backend_lock_invalidation_reason {
    LLAMA_BACKEND_LOCK_VALID = 0,           // Backend still valid
    LLAMA_BACKEND_LOCK_OOM = 1,             // Out of memory
    LLAMA_BACKEND_LOCK_DRIVER_ERROR = 2,    // GPU driver error
    LLAMA_BACKEND_LOCK_CAPABILITY_LOSS = 3, // Required capability lost
    LLAMA_BACKEND_LOCK_THERMAL_THROTTLE = 4,// Thermal throttling triggered
    LLAMA_BACKEND_LOCK_POWER_LIMIT = 5,     // Power limit triggered
    LLAMA_BACKEND_LOCK_RESET = 6,           // Backend reset occurred
    LLAMA_BACKEND_LOCK_UNKNOWN = 7,         // Unknown invalidation
};

// ============================================================================
// BACKEND LOCK VIOLATION TYPE
// ============================================================================

/**
 * Types of violations that can occur when the backend lock is active
 */
enum llama_backend_lock_violation_type {
    LLAMA_BACKEND_LOCK_VIOL_NONE = 0,                    // No violation
    LLAMA_BACKEND_LOCK_VIOL_BACKEND_CHANGE = 1,          // Attempted backend change
    LLAMA_BACKEND_LOCK_VIOL_RERESOLUTION = 2,            // Attempted re-resolution
    LLAMA_BACKEND_LOCK_VIOL_TENSOR_RELOCATION = 3,       // Attempted tensor relocation
    LLAMA_BACKEND_LOCK_VIOL_SHAPE_CHANGE = 4,            // Attempted shape change
    LLAMA_BACKEND_LOCK_VIOL_CAPABILITY_CHECK = 5,        // Attempted capability check
    LLAMA_BACKEND_LOCK_VIOL_MEMORY_PRESSURE = 6,         // Memory pressure response attempt
    LLAMA_BACKEND_LOCK_VIOL_FALLBACK_ATTEMPT = 7,        // Attempted fallback
    LLAMA_BACKEND_LOCK_VIOL_INVALIDATION = 8,            // Backend invalidation detected
};

// ============================================================================
// BACKEND LOCK VIOLATION LOCATION
// ============================================================================

/**
 * Locations where backend lock violations can be detected
 */
enum llama_backend_lock_violation_location {
    LLAMA_BACKEND_LOCK_LOC_UNKNOWN = 0,
    LLAMA_BACKEND_LOCK_LOC_ADMISSION = 1,
    LLAMA_BACKEND_LOCK_LOC_GRAPH_EXEC = 2,
    LLAMA_BACKEND_LOCK_LOC_TENSOR_OPS = 3,
    LLAMA_BACKEND_LOCK_LOC_MEMORY_MGT = 4,
    LLAMA_BACKEND_LOCK_LOC_BACKEND_SEL = 5,
    LLAMA_BACKEND_LOCK_LOC_CAPABILITY = 6,
    LLAMA_BACKEND_LOCK_LOC_FALLBACK = 7,
};

// ============================================================================
// BACKEND LOCK RECORD
// ============================================================================

/**
 * Struct to track backend lock state and history
 */
struct llama_backend_lock_record {
    enum llama_backend_lock_state state;
    enum ggml_backend_dev_type locked_backend;              // Which backend is locked
    uint64_t lock_acquire_time_ns;                      // Nanosecond timestamp of lock acquisition
    uint64_t lock_release_time_ns;                      // Nanosecond timestamp of lock release
    uint64_t decode_token_count;                        // Number of tokens decoded while locked
    bool lock_held;                                     // True = lock currently held
    bool backend_valid;                                 // True = locked backend is still valid
    enum llama_backend_lock_invalidation_reason invalidation_reason; // If invalid, why
    int violation_count;                                // Number of violations detected
    enum llama_backend_lock_violation_type last_violation; // Last violation type
    const char* last_violation_location;                // Location of last violation
};

// ============================================================================
// BACKEND LOCK VALIDATION STATE
// ============================================================================

/**
 * Global state for backend lock validation and enforcement
 */
struct llama_backend_lock_validation_state {
    struct llama_backend_lock_record lock_record;
    int total_violations;
    int total_invalidations;
    bool enforcement_strict;                            // True = abort on violation
    bool debug_verify_backend_identity;                 // True = verify backend per step
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
int llama_backend_lock_init(void);

// Lock lifecycle (3 enforcement points: 1-3)
int llama_backend_lock_acquire(enum ggml_backend_dev_type backend_to_lock);
int llama_backend_lock_release(void);
int llama_backend_lock_verify_held(void);

// Backend mutation prevention (3 enforcement points: 4-6)
int llama_backend_lock_prevent_backend_change(enum ggml_backend_dev_type new_backend);
int llama_backend_lock_prevent_reresolution(void);
int llama_backend_lock_prevent_tensor_relocation(void);

// Backend invalidation handling (2 enforcement points: 7-8)
int llama_backend_lock_check_validity(void);
int llama_backend_lock_terminate_on_invalidation(
    enum llama_backend_lock_invalidation_reason reason
);

// Query and diagnostic functions
bool llama_backend_lock_is_held(void);
enum ggml_backend_dev_type llama_backend_lock_get_locked_backend(void);
struct llama_backend_lock_record llama_backend_lock_get_record(void);
uint64_t llama_backend_lock_get_duration_ns(void);
int llama_backend_lock_get_violation_count(void);

// Scope management
int llama_backend_lock_assert_decode_phase_only(void);
int llama_backend_lock_assert_not_prefill(void);

// Verification functions
int llama_backend_lock_verify_all_operations_same_backend(
    const char** operation_names,
    int num_operations
);
int llama_backend_lock_assert_explicit_backend_decision(
    enum ggml_backend_dev_type backend
);

// Violation reporting
void llama_backend_lock_report_violation(
    enum llama_backend_lock_violation_type violation_type,
    enum llama_backend_lock_violation_location location,
    const char* details
);

// Diagnostics and logging
void llama_backend_lock_log_acquisition(void);
void llama_backend_lock_log_release(void);
void llama_backend_lock_print_status(void);
void llama_backend_lock_print_diagnostics(void);

// Enforcement mode control
void llama_backend_lock_set_enforcement_strict(bool strict);
bool llama_backend_lock_get_enforcement_strict(void);
void llama_backend_lock_set_debug_verify_backend_identity(bool verify);

// Validation
int llama_backend_lock_verify_immutability_invariant(void);
int llama_backend_lock_assert_backend_matches_locked(enum ggml_backend_dev_type actual_backend);

// Self-test suite
int llama_backend_lock_selftest(void);

// Helper/inline functions
static inline const char* llama_backend_lock_state_name(enum llama_backend_lock_state state) {
    switch (state) {
        case LLAMA_BACKEND_LOCK_UNLOCKED: return "UNLOCKED";
        case LLAMA_BACKEND_LOCK_ACQUIRING: return "ACQUIRING";
        case LLAMA_BACKEND_LOCK_ACQUIRED: return "ACQUIRED";
        case LLAMA_BACKEND_LOCK_RELEASING: return "RELEASING";
        case LLAMA_BACKEND_LOCK_RELEASED: return "RELEASED";
        case LLAMA_BACKEND_LOCK_INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_backend_lock_invalidation_reason_name(
    enum llama_backend_lock_invalidation_reason reason
) {
    switch (reason) {
        case LLAMA_BACKEND_LOCK_VALID: return "VALID";
        case LLAMA_BACKEND_LOCK_OOM: return "OUT_OF_MEMORY";
        case LLAMA_BACKEND_LOCK_DRIVER_ERROR: return "DRIVER_ERROR";
        case LLAMA_BACKEND_LOCK_CAPABILITY_LOSS: return "CAPABILITY_LOSS";
        case LLAMA_BACKEND_LOCK_THERMAL_THROTTLE: return "THERMAL_THROTTLE";
        case LLAMA_BACKEND_LOCK_POWER_LIMIT: return "POWER_LIMIT";
        case LLAMA_BACKEND_LOCK_RESET: return "BACKEND_RESET";
        case LLAMA_BACKEND_LOCK_UNKNOWN: return "UNKNOWN";
        default: return "INVALID";
    }
}

static inline const char* llama_backend_lock_violation_type_name(
    enum llama_backend_lock_violation_type violation_type
) {
    switch (violation_type) {
        case LLAMA_BACKEND_LOCK_VIOL_NONE: return "NONE";
        case LLAMA_BACKEND_LOCK_VIOL_BACKEND_CHANGE: return "BACKEND_CHANGE_ATTEMPTED";
        case LLAMA_BACKEND_LOCK_VIOL_RERESOLUTION: return "RERESOLUTION_ATTEMPTED";
        case LLAMA_BACKEND_LOCK_VIOL_TENSOR_RELOCATION: return "TENSOR_RELOCATION_ATTEMPTED";
        case LLAMA_BACKEND_LOCK_VIOL_SHAPE_CHANGE: return "SHAPE_CHANGE_ATTEMPTED";
        case LLAMA_BACKEND_LOCK_VIOL_CAPABILITY_CHECK: return "CAPABILITY_CHECK_ATTEMPTED";
        case LLAMA_BACKEND_LOCK_VIOL_MEMORY_PRESSURE: return "MEMORY_PRESSURE_RESPONSE";
        case LLAMA_BACKEND_LOCK_VIOL_FALLBACK_ATTEMPT: return "FALLBACK_ATTEMPT";
        case LLAMA_BACKEND_LOCK_VIOL_INVALIDATION: return "INVALIDATION";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
