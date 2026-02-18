/**
 * SECTION 16: Remove CPU-driven decode loop progression
 * Header
 *
 * This file implements enforcement that CPU no longer owns decode-loop progression.
 * Decode progression is GPU-driven with CPU reduced to non-blocking initiator and
 * observer. CPU must not iterate per-token, advance position counters, or gate
 * token emission. Decode becomes a continuous GPU-controlled process.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// DECODE LOOP OWNERSHIP ENUMERATION
// ============================================================================

/**
 * Ownership of decode loop progression
 */
enum llama_decode_loop_owner {
    LLAMA_LOOP_OWNER_NONE = 0,          // No loop running
    LLAMA_LOOP_OWNER_CPU = 1,           // CPU owns loop (forbidden during decode)
    LLAMA_LOOP_OWNER_GPU = 2,           // GPU owns loop (required)
    LLAMA_LOOP_OWNER_TRANSITIONING = 3, // Ownership transitioning
    LLAMA_LOOP_OWNER_INVALID = 4,       // Invalid owner state
};

// ============================================================================
// CPU CONTROL VIOLATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of CPU control violations
 */
enum llama_cpu_control_violation {
    LLAMA_CPU_CTRL_NONE = 0,
    LLAMA_CPU_CTRL_OWNS_LOOP = 1,              // CPU owns decode loop
    LLAMA_CPU_CTRL_ADVANCES_TOKENS = 2,        // CPU advances token counters
    LLAMA_CPU_CTRL_GATE_CONDITION = 3,         // CPU checks gating condition
    LLAMA_CPU_CTRL_POLLS_COMPLETION = 4,       // CPU polls for kernel completion
    LLAMA_CPU_CTRL_WAITS_BETWEEN_TOKENS = 5,   // CPU waits/blocks between tokens
    LLAMA_CPU_CTRL_INVOKES_NEXT_TOKEN = 6,     // CPU invokes next-token explicitly
    LLAMA_CPU_CTRL_MUTATES_DECODE_STATE = 7,   // CPU mutates decode state
    LLAMA_CPU_CTRL_LOOP_ITERATION = 8,         // CPU performs loop iteration
};

// ============================================================================
// DECODE PROGRESSION STATE ENUMERATION
// ============================================================================

/**
 * State of decode progression (GPU-owned)
 */
enum llama_decode_progression_state {
    LLAMA_PROGRESSION_IDLE = 0,         // Not in decode
    LLAMA_PROGRESSION_STARTING = 1,     // Decode starting (CPU initiates)
    LLAMA_PROGRESSION_GPU_LOOP = 2,     // GPU loop running (autonomous)
    LLAMA_PROGRESSION_TOKEN_READY = 3,  // Token ready (GPU signals)
    LLAMA_PROGRESSION_COMPLETE = 4,     // Decode complete (GPU signals)
    LLAMA_PROGRESSION_ERROR = 5,        // Error state
};

// ============================================================================
// DECODE SIGNAL TYPE ENUMERATION
// ============================================================================

/**
 * Signals from GPU to CPU
 */
enum llama_decode_signal {
    LLAMA_SIGNAL_NONE = 0,
    LLAMA_SIGNAL_TOKEN_READY = 1,      // Token produced and ready
    LLAMA_SIGNAL_DECODE_COMPLETE = 2,  // Decode session complete
    LLAMA_SIGNAL_ERROR = 3,            // Error during decode
};

// ============================================================================
// LOOP STATE ENUMERATION
// ============================================================================

/**
 * CPU-side loop state (should be empty if GPU owns)
 */
enum llama_cpu_loop_state {
    LLAMA_LOOP_STATE_NONE = 0,         // No loop state
    LLAMA_LOOP_STATE_RUNNING = 1,      // Loop running (forbidden)
    LLAMA_LOOP_STATE_PAUSED = 2,       // Loop paused (forbidden)
    LLAMA_LOOP_STATE_WAITING = 3,      // Waiting (forbidden)
};

// ============================================================================
// DECODE LOOP OWNERSHIP RECORD
// ============================================================================

/**
 * Record of decode loop ownership
 */
struct llama_decode_loop_ownership_record {
    enum llama_decode_loop_owner current_owner;  // Current loop owner
    bool cpu_loop_eliminated;                    // CPU loop completely removed
    bool gpu_loop_active;                        // GPU loop running
    int cpu_control_violations;                  // Total CPU control violations
    enum llama_cpu_control_violation last_violation; // Last violation detected
    uint64_t gpu_loop_start_time_ns;             // When GPU loop started
    uint64_t total_tokens_produced_by_gpu;       // Tokens produced by GPU loop
};

// ============================================================================
// DECODE PROGRESSION RECORD
// ============================================================================

/**
 * GPU-owned decode progression state
 */
struct llama_decode_progression_record {
    enum llama_decode_progression_state current_state; // Current progression state
    uint64_t current_token_index;                      // Current token (GPU-managed)
    uint64_t tokens_produced;                          // Tokens produced so far
    enum llama_decode_signal last_signal;              // Last signal from GPU
    uint64_t last_signal_time_ns;                      // When last signal sent
    bool gpu_autonomous;                               // True = GPU autonomous
    bool cpu_polling_detected;                         // Detected CPU polling
    int cpu_wait_violations;                           // CPU waits detected
};

// ============================================================================
// CPU LOOP ELIMINATION VALIDATION STATE
// ============================================================================

/**
 * Global validation state for CPU loop elimination
 */
struct llama_decode_loop_elimination_validation_state {
    struct llama_decode_loop_ownership_record ownership_record;
    struct llama_decode_progression_record progression_record;
    int total_control_violations;
    int total_polling_detections;
    int total_wait_detections;
    bool enforcement_strict;                     // Abort on violation vs log only
    bool debug_detect_cpu_loop_attempts;         // Debug CPU loop attempts
    bool debug_detect_cpu_polling;               // Debug CPU polling
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
int llama_decode_loop_elimination_init(void);

// Loop ownership transfer (5 enforcement points: 1-5)
int llama_decode_loop_elimination_eliminate_cpu_loop(void);
int llama_decode_loop_elimination_transfer_ownership_to_gpu(void);
int llama_decode_loop_elimination_start_gpu_autonomous_loop(void);
int llama_decode_loop_elimination_forbid_cpu_token_iteration(void);
int llama_decode_loop_elimination_assert_gpu_loop_owns_progression(void);

// CPU control prevention (3 enforcement points: 6-8)
int llama_decode_loop_elimination_forbid_cpu_gating_conditions(void);
int llama_decode_loop_elimination_forbid_cpu_waits_between_tokens(void);
int llama_decode_loop_elimination_convert_cpu_to_signal_observer(void);

// GPU signal handling (2 enforcement points: 9-10)
int llama_decode_loop_elimination_enable_gpu_decode_signaling(void);
int llama_decode_loop_elimination_assert_gpu_drives_progression(void);

// CPU control violation detection
int llama_decode_loop_elimination_detect_cpu_owns_loop(void);
int llama_decode_loop_elimination_detect_cpu_advances_tokens(void);
int llama_decode_loop_elimination_detect_cpu_gate_condition(void);
int llama_decode_loop_elimination_detect_cpu_polls_completion(void);
int llama_decode_loop_elimination_detect_cpu_waits_between_tokens(void);
int llama_decode_loop_elimination_detect_cpu_invokes_next_token(void);
int llama_decode_loop_elimination_detect_cpu_mutates_decode_state(void);
int llama_decode_loop_elimination_detect_cpu_loop_iteration(void);

// GPU-driven progression tracking
int llama_decode_loop_elimination_signal_token_ready(uint64_t token_index);
int llama_decode_loop_elimination_signal_decode_complete(void);
int llama_decode_loop_elimination_signal_decode_error(const char* error_msg);
int llama_decode_loop_elimination_advance_gpu_token_index(void);

// Query and verification functions
struct llama_decode_loop_ownership_record llama_decode_loop_elimination_get_ownership_record(void);
struct llama_decode_progression_record llama_decode_loop_elimination_get_progression_record(void);
enum llama_decode_loop_owner llama_decode_loop_elimination_get_loop_owner(void);
enum llama_decode_progression_state llama_decode_loop_elimination_get_progression_state(void);

// Verification functions
int llama_decode_loop_elimination_verify_cpu_loop_eliminated(void);
int llama_decode_loop_elimination_verify_gpu_loop_active(void);
int llama_decode_loop_elimination_verify_no_cpu_gating(void);
int llama_decode_loop_elimination_verify_no_cpu_polling(void);
int llama_decode_loop_elimination_verify_gpu_autonomous(void);
int llama_decode_loop_elimination_verify_no_cpu_waits(void);

// Diagnostics and logging
void llama_decode_loop_elimination_log_cpu_loop_eliminated(void);
void llama_decode_loop_elimination_log_gpu_loop_started(void);
void llama_decode_loop_elimination_log_gpu_signal_sent(enum llama_decode_signal signal);
void llama_decode_loop_elimination_print_loop_ownership_status(void);
void llama_decode_loop_elimination_print_progression_status(void);
void llama_decode_loop_elimination_print_violation_summary(void);

// Violation reporting
void llama_decode_loop_elimination_report_control_violation(
    enum llama_cpu_control_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_decode_loop_elimination_set_enforcement_strict(bool strict);
bool llama_decode_loop_elimination_get_enforcement_strict(void);
void llama_decode_loop_elimination_set_debug_detect_cpu_loop_attempts(bool debug);
void llama_decode_loop_elimination_set_debug_detect_cpu_polling(bool debug);

// Self-test suite
int llama_decode_loop_elimination_selftest(void);

// Helper/inline functions
static inline const char* llama_decode_loop_owner_name(
    enum llama_decode_loop_owner owner
) {
    switch (owner) {
        case LLAMA_LOOP_OWNER_NONE: return "NONE";
        case LLAMA_LOOP_OWNER_CPU: return "CPU";
        case LLAMA_LOOP_OWNER_GPU: return "GPU";
        case LLAMA_LOOP_OWNER_TRANSITIONING: return "TRANSITIONING";
        case LLAMA_LOOP_OWNER_INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_cpu_control_violation_name(
    enum llama_cpu_control_violation violation
) {
    switch (violation) {
        case LLAMA_CPU_CTRL_NONE: return "NONE";
        case LLAMA_CPU_CTRL_OWNS_LOOP: return "OWNS_LOOP";
        case LLAMA_CPU_CTRL_ADVANCES_TOKENS: return "ADVANCES_TOKENS";
        case LLAMA_CPU_CTRL_GATE_CONDITION: return "GATE_CONDITION";
        case LLAMA_CPU_CTRL_POLLS_COMPLETION: return "POLLS_COMPLETION";
        case LLAMA_CPU_CTRL_WAITS_BETWEEN_TOKENS: return "WAITS_BETWEEN_TOKENS";
        case LLAMA_CPU_CTRL_INVOKES_NEXT_TOKEN: return "INVOKES_NEXT_TOKEN";
        case LLAMA_CPU_CTRL_MUTATES_DECODE_STATE: return "MUTATES_DECODE_STATE";
        case LLAMA_CPU_CTRL_LOOP_ITERATION: return "LOOP_ITERATION";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_decode_progression_state_name(
    enum llama_decode_progression_state state
) {
    switch (state) {
        case LLAMA_PROGRESSION_IDLE: return "IDLE";
        case LLAMA_PROGRESSION_STARTING: return "STARTING";
        case LLAMA_PROGRESSION_GPU_LOOP: return "GPU_LOOP";
        case LLAMA_PROGRESSION_TOKEN_READY: return "TOKEN_READY";
        case LLAMA_PROGRESSION_COMPLETE: return "COMPLETE";
        case LLAMA_PROGRESSION_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_decode_signal_name(
    enum llama_decode_signal signal
) {
    switch (signal) {
        case LLAMA_SIGNAL_NONE: return "NONE";
        case LLAMA_SIGNAL_TOKEN_READY: return "TOKEN_READY";
        case LLAMA_SIGNAL_DECODE_COMPLETE: return "DECODE_COMPLETE";
        case LLAMA_SIGNAL_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
