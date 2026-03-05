/**
 * SECTION 17: Eliminate CPU token-step gating logic
 * Header
 *
 * This file implements enforcement that CPU no longer makes conditional decisions
 * about token progression. CPU cannot gate token advancement, check readiness,
 * or authorize next-token execution. Token-step progression is purely GPU-driven
 * as implicit consequence of GPU completion, not explicit CPU authorization.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CPU GATING DECISION TYPE ENUMERATION
// ============================================================================

/**
 * Types of CPU token-step gating decisions (all forbidden)
 */
enum llama_cpu_gating_decision {
    LLAMA_GATING_NONE = 0,
    LLAMA_GATING_SAMPLING_FINISHED = 1,         // "is sampling finished?"
    LLAMA_GATING_LOGITS_READY = 2,              // "are logits ready?"
    LLAMA_GATING_GPU_COMPLETE = 3,              // "did GPU complete?"
    LLAMA_GATING_READINESS_CHECK = 4,           // "is token ready to proceed?"
    LLAMA_GATING_CONTINUE_DECODE = 5,           // "should decode continue?"
    LLAMA_GATING_TOKEN_INDEX_ADVANCE = 6,       // "should token index advance?"
    LLAMA_GATING_STALL_CHECK = 7,               // "is decode stalled?"
    LLAMA_GATING_EXPLICIT_AUTHORIZATION = 8,    // explicit "go-ahead" signal
};

// ============================================================================
// CPU SYNCHRONIZATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of CPU synchronization that imply gating
 */
enum llama_cpu_sync_type {
    LLAMA_SYNC_NONE = 0,
    LLAMA_SYNC_POLLING_LOOP = 1,               // CPU polling loop
    LLAMA_SYNC_FLAG_POLLING = 2,               // CPU polling flags
    LLAMA_SYNC_SPIN_WAIT = 3,                  // CPU spinning on completion
    LLAMA_SYNC_EXPLICIT_WAIT = 4,              // CPU explicit wait call
    LLAMA_SYNC_BARRIER = 5,                    // CPU barrier before next token
    LLAMA_SYNC_READINESS_RECHECK = 6,          // CPU re-checking readiness
};

// ============================================================================
// TOKEN STEP BOUNDARY OWNERSHIP ENUMERATION
// ============================================================================

/**
 * Ownership of token-step boundaries
 */
enum llama_token_step_owner {
    LLAMA_STEP_OWNER_UNKNOWN = 0,
    LLAMA_STEP_OWNER_CPU = 1,                  // CPU owns boundary (forbidden)
    LLAMA_STEP_OWNER_GPU = 2,                  // GPU owns boundary (required)
    LLAMA_STEP_OWNER_SHARED = 3,               // Shared ownership (forbidden)
};

// ============================================================================
// CPU GATING VIOLATION DETECTION STATE ENUMERATION
// ============================================================================

/**
 * State of gating violation detection
 */
enum llama_gating_violation_state {
    LLAMA_GATING_STATE_CLEAN = 0,              // No violations
    LLAMA_GATING_STATE_VIOLATION_DETECTED = 1, // Violation found
    LLAMA_GATING_STATE_ENFORCING = 2,          // Enforcement active
    LLAMA_GATING_STATE_TERMINATED = 3,         // Terminated due to violation
};

// ============================================================================
// IMPLICIT COMPLETION SEMANTICS ENUMERATION
// ============================================================================

/**
 * Semantics of implicit GPU completion that drives progression
 */
enum llama_implicit_completion {
    LLAMA_COMPLETION_EXPLICIT = 0,             // Explicit CPU decision (forbidden)
    LLAMA_COMPLETION_IMPLICIT_GPU = 1,         // Implicit GPU internal completion
    LLAMA_COMPLETION_CONSEQUENCE = 2,          // Consequence of GPU state change
};

// ============================================================================
// CPU GATING ELIMINATION RECORD
// ============================================================================

/**
 * Record of CPU gating decisions and violations
 */
struct llama_cpu_gating_elimination_record {
    int total_gating_decisions_detected;        // CPU gating decisions found
    int total_synchronization_barriers;         // CPU sync barriers found
    int total_readiness_checks;                 // Readiness checks found
    enum llama_cpu_gating_decision last_decision; // Last decision detected
    enum llama_cpu_sync_type last_sync_type;    // Last sync type detected
    enum llama_token_step_owner step_owner;     // Who owns token steps
    bool cpu_gating_eliminated;                 // CPU gating fully eliminated
    bool gpu_implicit_completion;               // GPU implicit completion active
};

// ============================================================================
// TOKEN STEP GATING VALIDATION STATE
// ============================================================================

/**
 * Global validation state for token-step gating elimination
 */
struct llama_token_step_gating_validation_state {
    struct llama_cpu_gating_elimination_record gating_record;
    enum llama_gating_violation_state violation_state;
    int total_gating_violations;
    int total_sync_violations;
    int total_unauthorized_checks;
    bool enforcement_strict;                    // Abort on violation vs log only
    bool debug_detect_cpu_conditionals;         // Debug CPU conditionals
    bool debug_detect_cpu_barriers;             // Debug CPU barriers
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
int llama_token_step_gating_elimination_init(void);

// CPU gating elimination (5 enforcement points: 1-5)
int llama_token_step_gating_elimination_delete_can_proceed_checks(void);
int llama_token_step_gating_elimination_remove_cpu_barriers(void);
int llama_token_step_gating_elimination_prohibit_cpu_token_decisions(void);
int llama_token_step_gating_elimination_replace_with_implicit_semantics(void);
int llama_token_step_gating_elimination_remove_cpu_sync_loops(void);

// CPU control prevention (3 enforcement points: 6-8)
int llama_token_step_gating_elimination_forbid_cpu_waits_that_gate(void);
int llama_token_step_gating_elimination_move_boundaries_to_gpu(void);
int llama_token_step_gating_elimination_assert_gpu_token_step_owner(void);

// Invariant enforcement (2 enforcement points: 9-10)
int llama_token_step_gating_elimination_add_cpu_gating_invariants(void);
int llama_token_step_gating_elimination_audit_decode_call_sites(void);

// CPU gating violation detection
int llama_token_step_gating_elimination_detect_sampling_finished_check(void);
int llama_token_step_gating_elimination_detect_logits_ready_check(void);
int llama_token_step_gating_elimination_detect_gpu_complete_check(void);
int llama_token_step_gating_elimination_detect_readiness_check(void);
int llama_token_step_gating_elimination_detect_continue_decode_check(void);
int llama_token_step_gating_elimination_detect_token_index_advance_decision(void);
int llama_token_step_gating_elimination_detect_stall_check(void);
int llama_token_step_gating_elimination_detect_explicit_authorization(void);

// CPU synchronization violation detection
int llama_token_step_gating_elimination_detect_polling_loop(void);
int llama_token_step_gating_elimination_detect_flag_polling(void);
int llama_token_step_gating_elimination_detect_spin_wait(void);
int llama_token_step_gating_elimination_detect_explicit_wait(void);
int llama_token_step_gating_elimination_detect_barrier(void);

// GPU implicit completion verification
int llama_token_step_gating_elimination_enable_implicit_completion(void);
int llama_token_step_gating_elimination_verify_gpu_drives_progression(void);

// Query and verification functions
struct llama_cpu_gating_elimination_record llama_token_step_gating_elimination_get_record(void);
enum llama_gating_violation_state llama_token_step_gating_elimination_get_violation_state(void);
enum llama_token_step_owner llama_token_step_gating_elimination_get_step_owner(void);

// Verification functions
int llama_token_step_gating_elimination_verify_no_cpu_gating(void);
int llama_token_step_gating_elimination_verify_no_readiness_checks(void);
int llama_token_step_gating_elimination_verify_no_cpu_barriers(void);
int llama_token_step_gating_elimination_verify_gpu_owns_boundaries(void);
int llama_token_step_gating_elimination_verify_implicit_completion(void);

// Diagnostics and logging
void llama_token_step_gating_elimination_log_cpu_gating_eliminated(void);
void llama_token_step_gating_elimination_log_implicit_completion_active(void);
void llama_token_step_gating_elimination_print_gating_status(void);
void llama_token_step_gating_elimination_print_violation_summary(void);

// Violation reporting
void llama_token_step_gating_elimination_report_gating_decision(
    enum llama_cpu_gating_decision decision_type,
    const char* details
);
void llama_token_step_gating_elimination_report_sync_barrier(
    enum llama_cpu_sync_type sync_type
);

// Enforcement mode control
void llama_token_step_gating_elimination_set_enforcement_strict(bool strict);
bool llama_token_step_gating_elimination_get_enforcement_strict(void);
void llama_token_step_gating_elimination_set_debug_detect_cpu_conditionals(bool debug);
void llama_token_step_gating_elimination_set_debug_detect_cpu_barriers(bool debug);

// Self-test suite
int llama_token_step_gating_elimination_selftest(void);

// Helper/inline functions
static inline const char* llama_cpu_gating_decision_name(
    enum llama_cpu_gating_decision decision
) {
    switch (decision) {
        case LLAMA_GATING_NONE: return "NONE";
        case LLAMA_GATING_SAMPLING_FINISHED: return "SAMPLING_FINISHED";
        case LLAMA_GATING_LOGITS_READY: return "LOGITS_READY";
        case LLAMA_GATING_GPU_COMPLETE: return "GPU_COMPLETE";
        case LLAMA_GATING_READINESS_CHECK: return "READINESS_CHECK";
        case LLAMA_GATING_CONTINUE_DECODE: return "CONTINUE_DECODE";
        case LLAMA_GATING_TOKEN_INDEX_ADVANCE: return "TOKEN_INDEX_ADVANCE";
        case LLAMA_GATING_STALL_CHECK: return "STALL_CHECK";
        case LLAMA_GATING_EXPLICIT_AUTHORIZATION: return "EXPLICIT_AUTHORIZATION";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_cpu_sync_type_name(
    enum llama_cpu_sync_type sync_type
) {
    switch (sync_type) {
        case LLAMA_SYNC_NONE: return "NONE";
        case LLAMA_SYNC_POLLING_LOOP: return "POLLING_LOOP";
        case LLAMA_SYNC_FLAG_POLLING: return "FLAG_POLLING";
        case LLAMA_SYNC_SPIN_WAIT: return "SPIN_WAIT";
        case LLAMA_SYNC_EXPLICIT_WAIT: return "EXPLICIT_WAIT";
        case LLAMA_SYNC_BARRIER: return "BARRIER";
        case LLAMA_SYNC_READINESS_RECHECK: return "READINESS_RECHECK";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_token_step_owner_name(
    enum llama_token_step_owner owner
) {
    switch (owner) {
        case LLAMA_STEP_OWNER_UNKNOWN: return "UNKNOWN";
        case LLAMA_STEP_OWNER_CPU: return "CPU";
        case LLAMA_STEP_OWNER_GPU: return "GPU";
        case LLAMA_STEP_OWNER_SHARED: return "SHARED";
        default: return "INVALID";
    }
}

static inline const char* llama_gating_violation_state_name(
    enum llama_gating_violation_state state
) {
    switch (state) {
        case LLAMA_GATING_STATE_CLEAN: return "CLEAN";
        case LLAMA_GATING_STATE_VIOLATION_DETECTED: return "VIOLATION_DETECTED";
        case LLAMA_GATING_STATE_ENFORCING: return "ENFORCING";
        case LLAMA_GATING_STATE_TERMINATED: return "TERMINATED";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
