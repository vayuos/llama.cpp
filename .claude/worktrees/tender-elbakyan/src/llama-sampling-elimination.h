/**
 * SECTION 18: Remove CPU sampling from decode path
 * Header
 *
 * This file implements enforcement that CPU sampling is eliminated from the decode critical path.
 * All sampling operations (temperature, top-k, top-p, repetition penalty, etc.) must be GPU-resident.
 * CPU cannot invoke sampler, modify sampling parameters, or apply sampling logic during decode.
 * Sampling becomes a GPU-autonomous operation with CPU as observer only.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CPU SAMPLING OPERATION ENUMERATION
// ============================================================================

/**
 * Types of CPU sampling operations (all forbidden during decode)
 */
enum llama_cpu_sampling_operation {
    LLAMA_SAMPLING_NONE = 0,
    LLAMA_SAMPLING_CREATE_SAMPLER = 1,          // CPU creates sampler object
    LLAMA_SAMPLING_INVOKE = 2,                  // CPU calls sampler function
    LLAMA_SAMPLING_TEMPERATURE_SET = 3,         // CPU sets temperature parameter
    LLAMA_SAMPLING_TOP_K_SET = 4,               // CPU sets top-k parameter
    LLAMA_SAMPLING_TOP_P_SET = 5,               // CPU sets top-p parameter
    LLAMA_SAMPLING_REP_PENALTY_SET = 6,         // CPU sets repetition penalty
    LLAMA_SAMPLING_FREQ_PENALTY_SET = 7,        // CPU sets frequency penalty
    LLAMA_SAMPLING_PRESENCE_PENALTY_SET = 8,    // CPU sets presence penalty
    LLAMA_SAMPLING_MODIFY_LOGITS = 9,           // CPU modifies logits before sampling
    LLAMA_SAMPLING_APPLY_PENALTIES = 10,        // CPU applies penalties
    LLAMA_SAMPLING_SELECT_TOKEN = 11,           // CPU selects token from distribution
    LLAMA_SAMPLING_SEED_SET = 12,               // CPU sets random seed
    LLAMA_SAMPLING_GRAMMAR_SET = 13,            // CPU sets grammar constraint
};

// ============================================================================
// SAMPLING OPERATION CATEGORY ENUMERATION
// ============================================================================

/**
 * Categories of sampling operations by scope and impact
 */
enum llama_sampling_category {
    LLAMA_SAMPLING_CAT_NONE = 0,
    LLAMA_SAMPLING_CAT_OBJECT_LIFECYCLE = 1,    // Sampler creation/destruction
    LLAMA_SAMPLING_CAT_PARAMETER = 2,           // Parameter modification
    LLAMA_SAMPLING_CAT_LOGIT_TRANSFORM = 3,     // Logit modifications
    LLAMA_SAMPLING_CAT_EXECUTION = 4,           // Actual sampling execution
    LLAMA_SAMPLING_CAT_RANDOMNESS = 5,          // Random seed/state
    LLAMA_SAMPLING_CAT_CONSTRAINT = 6,          // Grammar/constraint logic
};

// ============================================================================
// GPU SAMPLING STATE ENUMERATION
// ============================================================================

/**
 * GPU-resident sampling state during decode
 */
enum llama_gpu_sampling_state {
    LLAMA_GPU_SAMPLING_UNINITIALIZED = 0,       // Not started
    LLAMA_GPU_SAMPLING_PREPARED = 1,            // GPU sampler prepared
    LLAMA_GPU_SAMPLING_AUTONOMOUS = 2,          // GPU executing sampling
    LLAMA_GPU_SAMPLING_TOKEN_READY = 3,         // Sampled token ready
    LLAMA_GPU_SAMPLING_ERROR = 4,               // Sampling error
};

// ============================================================================
// SAMPLING VIOLATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of CPU sampling violations
 */
enum llama_sampling_violation_type {
    LLAMA_SAMPLING_VIOLATION_NONE = 0,
    LLAMA_SAMPLING_VIOLATION_CPU_INVOKE = 1,               // CPU invoked sampler
    LLAMA_SAMPLING_VIOLATION_CPU_PARAMETER_CHANGE = 2,     // CPU changed parameter
    LLAMA_SAMPLING_VIOLATION_CPU_LOGIT_MODIFICATION = 3,   // CPU modified logits
    LLAMA_SAMPLING_VIOLATION_CPU_TOKEN_SELECTION = 4,      // CPU selected token
    LLAMA_SAMPLING_VIOLATION_SAMPLER_RECREATION = 5,       // Sampler recreated per-token
    LLAMA_SAMPLING_VIOLATION_PARAMETER_MISMATCH = 6,       // Parameter inconsistency
    LLAMA_SAMPLING_VIOLATION_SEED_CHANGE = 7,              // Random seed changed
    LLAMA_SAMPLING_VIOLATION_GRAMMAR_MODIFICATION = 8,     // Grammar changed
};

// ============================================================================
// SAMPLING OWNERSHIP MODEL ENUMERATION
// ============================================================================

/**
 * Owner of sampling operation
 */
enum llama_sampling_owner {
    LLAMA_SAMPLING_OWNER_UNKNOWN = 0,
    LLAMA_SAMPLING_OWNER_CPU = 1,      // CPU owns (forbidden during decode)
    LLAMA_SAMPLING_OWNER_GPU = 2,      // GPU owns (required during decode)
    LLAMA_SAMPLING_OWNER_SHARED = 3,   // Shared ownership (forbidden)
};

// ============================================================================
// PARAMETER MUTABILITY ENUMERATION
// ============================================================================

/**
 * Mutability state of sampling parameters
 */
enum llama_sampling_parameter_mutability {
    LLAMA_SAMPLING_PARAM_UNKNOWN = 0,
    LLAMA_SAMPLING_PARAM_MUTABLE = 1,          // Parameter mutable (forbidden)
    LLAMA_SAMPLING_PARAM_IMMUTABLE = 2,        // Parameter immutable (required)
    LLAMA_SAMPLING_PARAM_GPU_CONTROLLED = 3,   // GPU controls parameter
};

// ============================================================================
// SAMPLING OPERATION RECORD
// ============================================================================

/**
 * Record of a sampling operation attempt
 */
struct llama_sampling_operation_record {
    enum llama_cpu_sampling_operation operation;       // Operation type
    enum llama_sampling_category category;              // Operation category
    uint64_t timestamp_ns;                              // When it occurred
    uint32_t sequence_id;                               // Sequence this applies to
    const char * location;                              // Where it occurred (function/file)
    enum llama_sampling_violation_type violation;       // Violation type if any
    bool cpu_initiated;                                 // True if CPU initiated
    bool gpu_authorized;                                // True if GPU pre-authorized
};

// ============================================================================
// SAMPLING STATE RECORD
// ============================================================================

/**
 * Global state of sampling during decode
 */
struct llama_sampling_state_record {
    enum llama_sampling_owner current_owner;            // Current owner
    enum llama_gpu_sampling_state gpu_state;            // GPU state
    bool cpu_sampling_eliminated;                       // CPU sampling fully removed
    bool gpu_sampling_active;                           // GPU actively sampling
    bool parameters_gpu_controlled;                     // Parameters on GPU side
    int cpu_sampling_violations;                        // Total violations detected
    enum llama_sampling_violation_type last_violation;  // Last violation type
    uint64_t gpu_samples_produced;                      // Tokens sampled by GPU
    uint64_t gpu_sampling_start_time_ns;               // When GPU started sampling
};

// ============================================================================
// PARAMETER SNAPSHOT RECORD
// ============================================================================

/**
 * Snapshot of sampling parameters (for consistency checks)
 */
struct llama_sampling_parameter_snapshot {
    float temperature;                  // Temperature value
    int top_k;                          // Top-k value
    float top_p;                        // Top-p (nucleus) value
    float repeat_penalty;               // Repetition penalty
    float frequency_penalty;            // Frequency penalty
    float presence_penalty;             // Presence penalty
    uint64_t seed;                      // Random seed
    bool grammar_active;                // Grammar constraint active
    enum llama_sampling_parameter_mutability mutability;  // Can CPU change these?
    uint64_t snapshot_time_ns;          // When snapshot taken
};

// ============================================================================
// SAMPLING VALIDATION STATE
// ============================================================================

/**
 * Global validation state for sampling elimination
 */
struct llama_sampling_elimination_validation_state {
    struct llama_sampling_state_record state_record;
    struct llama_sampling_parameter_snapshot initial_params;
    struct llama_sampling_parameter_snapshot current_params;
    int total_operation_attempts;
    int total_violations;
    bool params_frozen;                 // Parameters immutable after initial set
    bool enforcement_strict;            // Abort on violation vs log only
    bool debug_detect_cpu_sampling;     // Debug CPU sampling attempts
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
int llama_sampling_elimination_init(void);

// Sampling ownership transfer (5 enforcement points: 1-5)
int llama_sampling_elimination_eliminate_cpu_sampler(void);
int llama_sampling_elimination_transfer_sampling_to_gpu(void);
int llama_sampling_elimination_freeze_sampling_parameters(void);
int llama_sampling_elimination_forbid_cpu_sampling_invoke(void);
int llama_sampling_elimination_assert_gpu_sampling_owns_execution(void);

// Parameter immutability (3 enforcement points: 6-8)
int llama_sampling_elimination_forbid_cpu_parameter_changes(void);
int llama_sampling_elimination_freeze_initial_parameters(void);
int llama_sampling_elimination_enable_gpu_parameter_control(void);

// Logit handling (2 enforcement points: 9-10)
int llama_sampling_elimination_forbid_cpu_logit_modification(void);
int llama_sampling_elimination_assert_gpu_controls_logits(void);

// CPU sampling violation detection
int llama_sampling_elimination_detect_cpu_invoke(void);
int llama_sampling_elimination_detect_cpu_parameter_change(void);
int llama_sampling_elimination_detect_cpu_logit_modification(void);
int llama_sampling_elimination_detect_cpu_token_selection(void);
int llama_sampling_elimination_detect_sampler_recreation(void);
int llama_sampling_elimination_detect_parameter_mismatch(void);
int llama_sampling_elimination_detect_seed_change(void);
int llama_sampling_elimination_detect_grammar_modification(void);

// GPU sampling state management
int llama_sampling_elimination_set_gpu_sampling_prepared(void);
int llama_sampling_elimination_set_gpu_sampling_autonomous(void);
int llama_sampling_elimination_signal_gpu_token_ready(int32_t token);

// GPU parameter control
int llama_sampling_elimination_snapshot_initial_parameters(void);
int llama_sampling_elimination_freeze_parameters(void);
int llama_sampling_elimination_transfer_parameters_to_gpu(void);

// Query and verification functions
struct llama_sampling_state_record llama_sampling_elimination_get_state_record(void);
struct llama_sampling_parameter_snapshot llama_sampling_elimination_get_current_parameters(void);
enum llama_sampling_owner llama_sampling_elimination_get_sampling_owner(void);
enum llama_gpu_sampling_state llama_sampling_elimination_get_gpu_sampling_state(void);

// Verification functions
int llama_sampling_elimination_verify_cpu_sampling_eliminated(void);
int llama_sampling_elimination_verify_gpu_sampling_active(void);
int llama_sampling_elimination_verify_parameters_immutable(void);
int llama_sampling_elimination_verify_no_cpu_parameter_changes(void);
int llama_sampling_elimination_verify_gpu_controls_sampling(void);
int llama_sampling_elimination_verify_no_cpu_logit_modifications(void);

// Diagnostics and logging
void llama_sampling_elimination_log_cpu_sampling_eliminated(void);
void llama_sampling_elimination_log_gpu_sampling_started(void);
void llama_sampling_elimination_log_token_sampled_by_gpu(int32_t token);
void llama_sampling_elimination_print_sampling_state(void);
void llama_sampling_elimination_print_parameter_state(void);
void llama_sampling_elimination_print_violation_summary(void);

// Violation reporting
void llama_sampling_elimination_report_sampling_violation(
    enum llama_sampling_violation_type violation_type,
    enum llama_cpu_sampling_operation operation,
    const char* details
);

// Enforcement mode control
void llama_sampling_elimination_set_enforcement_strict(bool strict);
bool llama_sampling_elimination_get_enforcement_strict(void);
void llama_sampling_elimination_set_debug_detect_cpu_sampling(bool debug);

// Self-test suite
int llama_sampling_elimination_selftest(void);

// Helper/inline functions
static inline const char* llama_cpu_sampling_operation_name(
    enum llama_cpu_sampling_operation op
) {
    switch (op) {
        case LLAMA_SAMPLING_NONE: return "NONE";
        case LLAMA_SAMPLING_CREATE_SAMPLER: return "CREATE_SAMPLER";
        case LLAMA_SAMPLING_INVOKE: return "INVOKE";
        case LLAMA_SAMPLING_TEMPERATURE_SET: return "TEMPERATURE_SET";
        case LLAMA_SAMPLING_TOP_K_SET: return "TOP_K_SET";
        case LLAMA_SAMPLING_TOP_P_SET: return "TOP_P_SET";
        case LLAMA_SAMPLING_REP_PENALTY_SET: return "REP_PENALTY_SET";
        case LLAMA_SAMPLING_FREQ_PENALTY_SET: return "FREQ_PENALTY_SET";
        case LLAMA_SAMPLING_PRESENCE_PENALTY_SET: return "PRESENCE_PENALTY_SET";
        case LLAMA_SAMPLING_MODIFY_LOGITS: return "MODIFY_LOGITS";
        case LLAMA_SAMPLING_APPLY_PENALTIES: return "APPLY_PENALTIES";
        case LLAMA_SAMPLING_SELECT_TOKEN: return "SELECT_TOKEN";
        case LLAMA_SAMPLING_SEED_SET: return "SEED_SET";
        case LLAMA_SAMPLING_GRAMMAR_SET: return "GRAMMAR_SET";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_sampling_violation_type_name(
    enum llama_sampling_violation_type violation
) {
    switch (violation) {
        case LLAMA_SAMPLING_VIOLATION_NONE: return "NONE";
        case LLAMA_SAMPLING_VIOLATION_CPU_INVOKE: return "CPU_INVOKE";
        case LLAMA_SAMPLING_VIOLATION_CPU_PARAMETER_CHANGE: return "CPU_PARAMETER_CHANGE";
        case LLAMA_SAMPLING_VIOLATION_CPU_LOGIT_MODIFICATION: return "CPU_LOGIT_MODIFICATION";
        case LLAMA_SAMPLING_VIOLATION_CPU_TOKEN_SELECTION: return "CPU_TOKEN_SELECTION";
        case LLAMA_SAMPLING_VIOLATION_SAMPLER_RECREATION: return "SAMPLER_RECREATION";
        case LLAMA_SAMPLING_VIOLATION_PARAMETER_MISMATCH: return "PARAMETER_MISMATCH";
        case LLAMA_SAMPLING_VIOLATION_SEED_CHANGE: return "SEED_CHANGE";
        case LLAMA_SAMPLING_VIOLATION_GRAMMAR_MODIFICATION: return "GRAMMAR_MODIFICATION";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_sampling_owner_name(
    enum llama_sampling_owner owner
) {
    switch (owner) {
        case LLAMA_SAMPLING_OWNER_UNKNOWN: return "UNKNOWN";
        case LLAMA_SAMPLING_OWNER_CPU: return "CPU";
        case LLAMA_SAMPLING_OWNER_GPU: return "GPU";
        case LLAMA_SAMPLING_OWNER_SHARED: return "SHARED";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_sampling_state_name(
    enum llama_gpu_sampling_state state
) {
    switch (state) {
        case LLAMA_GPU_SAMPLING_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_SAMPLING_PREPARED: return "PREPARED";
        case LLAMA_GPU_SAMPLING_AUTONOMOUS: return "AUTONOMOUS";
        case LLAMA_GPU_SAMPLING_TOKEN_READY: return "TOKEN_READY";
        case LLAMA_GPU_SAMPLING_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_sampling_category_name(
    enum llama_sampling_category category
) {
    switch (category) {
        case LLAMA_SAMPLING_CAT_NONE: return "NONE";
        case LLAMA_SAMPLING_CAT_OBJECT_LIFECYCLE: return "OBJECT_LIFECYCLE";
        case LLAMA_SAMPLING_CAT_PARAMETER: return "PARAMETER";
        case LLAMA_SAMPLING_CAT_LOGIT_TRANSFORM: return "LOGIT_TRANSFORM";
        case LLAMA_SAMPLING_CAT_EXECUTION: return "EXECUTION";
        case LLAMA_SAMPLING_CAT_RANDOMNESS: return "RANDOMNESS";
        case LLAMA_SAMPLING_CAT_CONSTRAINT: return "CONSTRAINT";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
