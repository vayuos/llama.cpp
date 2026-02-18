/**
 * SECTION 15: Enforce token-persistent graph execution model
 * Header
 *
 * This file implements enforcement that decode execution uses a token-persistent
 * graph model where a single decode graph instance is created once and executed
 * repeatedly for each token without CPU re-entry into graph control. The GPU owns
 * a long-lived execution context across token iterations. Graph lifetime exactly
 * matches decode lifetime.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// DECODE MODE STATE ENUMERATION
// ============================================================================

/**
 * Decode mode execution state
 */
enum llama_decode_mode_state {
    LLAMA_DECODE_MODE_INACTIVE = 0,      // Decode mode not active
    LLAMA_DECODE_MODE_INITIALIZING = 1,  // Decode mode entering
    LLAMA_DECODE_MODE_ACTIVE = 2,        // Decode mode active (graph persistent)
    LLAMA_DECODE_MODE_TERMINATING = 3,   // Decode mode exiting
    LLAMA_DECODE_MODE_TERMINATED = 4,    // Decode mode terminated
    LLAMA_DECODE_MODE_ERROR = 5,         // Decode mode in error state
};

// ============================================================================
// GRAPH LIFETIME BINDING STATE ENUMERATION
// ============================================================================

/**
 * Binding between graph lifetime and decode lifetime
 */
enum llama_graph_lifetime_binding {
    LLAMA_LIFETIME_UNBOUND = 0,          // Not bound to decode lifetime
    LLAMA_LIFETIME_BINDING_START = 1,    // Binding in progress at decode start
    LLAMA_LIFETIME_BOUND = 2,            // Bound to decode lifetime
    LLAMA_LIFETIME_LOCKED = 3,           // Binding locked (immutable)
    LLAMA_LIFETIME_INVALID = 4,          // Invalid binding state
};

// ============================================================================
// GRAPH PERSISTENCE STATE ENUMERATION
// ============================================================================

/**
 * Graph persistence across token iterations
 */
enum llama_graph_persistence_state {
    LLAMA_PERSISTENCE_UNINITIALIZED = 0, // Graph not initialized
    LLAMA_PERSISTENCE_FIRST_TOKEN = 1,   // Executing first token
    LLAMA_PERSISTENCE_PERSISTENT = 2,    // Graph persists across tokens
    LLAMA_PERSISTENCE_LOCKED = 3,        // Persistence locked (no changes)
    LLAMA_PERSISTENCE_INVALID = 4,       // Invalid persistence state
};

// ============================================================================
// CPU REENTRANCY VIOLATION ENUMERATION
// ============================================================================

/**
 * Types of CPU re-entry violations
 */
enum llama_cpu_reentrancy_violation {
    LLAMA_REENTRANCY_NONE = 0,
    LLAMA_REENTRANCY_GRAPH_REBUILD = 1,        // Graph rebuild attempted
    LLAMA_REENTRANCY_GRAPH_RESUBMIT = 2,       // Graph re-submitted per token
    LLAMA_REENTRANCY_TOKEN_COUNTER_UPDATE = 3, // Token counter updated by CPU
    LLAMA_REENTRANCY_GRAPH_INPUT_PATCH = 4,    // Graph inputs patched per token
    LLAMA_REENTRANCY_TENSOR_REBINDING = 5,     // Tensor rebinding per token
    LLAMA_REENTRANCY_BACKEND_REASSIGN = 6,     // Backend reassignment per token
    LLAMA_REENTRANCY_ORCHESTRATION = 7,        // CPU orchestration detected
    LLAMA_REENTRANCY_BUFFER_REBIND = 8,        // Buffer rebinding per token
};

// ============================================================================
// GRAPH STATE MUTATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of graph state mutations (forbidden during persistent execution)
 */
enum llama_graph_state_mutation {
    LLAMA_MUTATION_NONE = 0,
    LLAMA_MUTATION_LIFETIME_CHANGE = 1,        // Graph lifetime changed
    LLAMA_MUTATION_PERSISTENCE_BREAK = 2,      // Graph persistence broken
    LLAMA_MUTATION_BACKEND_CHANGE = 3,         // Backend changed
    LLAMA_MUTATION_EXECUTION_PLAN_CHANGE = 4,  // Execution plan changed
    LLAMA_MUTATION_INPUT_SHAPE_CHANGE = 5,     // Input tensor shape changed
    LLAMA_MUTATION_INPUT_LOCATION_CHANGE = 6,  // Input tensor location changed
    LLAMA_MUTATION_PARAMETER_INJECTION = 7,    // Parameters re-injected per token
};

// ============================================================================
// TOKEN PROGRESS TRACKING ENUMERATION
// ============================================================================

/**
 * Token progress through persistent execution
 */
enum llama_token_progress_state {
    LLAMA_TOKEN_PROGRESS_NONE = 0,
    LLAMA_TOKEN_PROGRESS_FETCHED = 1,          // Token fetched for processing
    LLAMA_TOKEN_PROGRESS_SUBMITTED = 2,        // Token submitted to GPU (via graph)
    LLAMA_TOKEN_PROGRESS_EXECUTING = 3,        // Token executing on GPU
    LLAMA_TOKEN_PROGRESS_COMPLETE = 4,         // Token processing complete on GPU
};

// ============================================================================
// DECODE MODE RECORD
// ============================================================================

/**
 * Record of decode mode state
 */
struct llama_decode_mode_record {
    enum llama_decode_mode_state mode;         // Current decode mode
    uint64_t graph_id;                         // Active graph ID (persistent)
    uint64_t decode_start_time_ns;             // When decode started
    uint64_t total_tokens_processed;           // Total tokens in decode session
    bool graph_lifetime_locked;                // Graph lifetime locked to decode
    bool graph_persistence_locked;             // Graph persistence locked
    bool cpu_reentrancy_forbidden;             // CPU re-entry forbidden
    enum llama_graph_lifetime_binding lifetime_binding; // Lifetime binding state
    enum llama_graph_persistence_state persistence; // Graph persistence state
};

// ============================================================================
// TOKEN PERSISTENT GRAPH STATE
// ============================================================================

/**
 * GPU-resident state for token-persistent graph execution
 */
struct llama_token_persistent_state {
    uint64_t graph_id;                         // Persistent graph ID
    uint64_t current_token_index;              // Current token being processed
    uint64_t kv_cache_offset;                  // GPU-resident KV cache offset
    uint64_t context_index;                    // GPU-resident context position
    enum llama_token_progress_state token_progress; // Token progress state
    int batch_size;                            // Batch size (fixed for persistence)
    bool state_gpu_resident;                   // True = state on GPU (persistent)
    uint64_t state_last_update_time_ns;        // Last update timestamp
};

// ============================================================================
// TOKEN PERSISTENT EXECUTION RECORD
// ============================================================================

/**
 * Record tracking token-persistent execution invariants
 */
struct llama_token_persistent_execution_record {
    uint64_t current_graph_id;                 // Current persistent graph
    enum llama_decode_mode_state decode_mode;  // Decode mode state
    int total_cpu_reentrancy_violations;       // Total CPU re-entry violations
    int total_graph_mutations;                 // Total graph state mutations
    enum llama_cpu_reentrancy_violation last_reentrancy; // Last violation
    enum llama_graph_state_mutation last_mutation; // Last mutation
    bool graph_input_stable;                   // Inputs fixed in shape/location
    bool graph_backend_immutable;              // Backend immutable
    bool execution_plan_immutable;             // Execution plan immutable
    bool token_progression_gpu_owned;          // Token progress owned by GPU
};

// ============================================================================
// TOKEN PERSISTENT EXECUTION VALIDATION STATE
// ============================================================================

/**
 * Global validation state for token-persistent execution
 */
struct llama_token_persistent_execution_validation_state {
    struct llama_decode_mode_record decode_mode_record;
    struct llama_token_persistent_execution_record execution_record;
    struct llama_token_persistent_state gpu_state;
    int total_reentrancy_violations;
    int total_mutation_violations;
    int total_graph_lifetime_mismatches;
    bool enforcement_strict;                   // Abort on violation vs log only
    bool debug_check_persistence_per_token;    // Check persistence at each token
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
int llama_token_persistent_init(void);

// Decode mode lifecycle (5 enforcement points: 1-5)
int llama_token_persistent_enter_decode_mode(uint64_t graph_id);
int llama_token_persistent_lock_graph_lifetime_to_decode(uint64_t graph_id);
int llama_token_persistent_bind_graph_to_decode_lifetime(uint64_t graph_id);
int llama_token_persistent_lock_graph_persistence_model(void);
int llama_token_persistent_assert_single_persistent_graph(void);

// CPU re-entry prevention (3 enforcement points: 6-8)
int llama_token_persistent_forbid_per_token_graph_resubmission(void);
int llama_token_persistent_forbid_per_token_rebinding(void);
int llama_token_persistent_forbid_cpu_orchestration(void);

// Graph state immutability (2 enforcement points: 9-10)
int llama_token_persistent_lock_graph_inputs_and_backend(void);
int llama_token_persistent_assert_execution_context_unchanged(void);

// GPU-owned token progression
int llama_token_persistent_enable_gpu_owned_token_progression(void);
int llama_token_persistent_update_gpu_token_state(
    uint64_t token_index,
    uint64_t kv_offset,
    uint64_t context_index
);

// Re-entry violation detection
int llama_token_persistent_detect_graph_rebuild_attempt(void);
int llama_token_persistent_detect_graph_resubmit_attempt(void);
int llama_token_persistent_detect_token_counter_cpu_update(void);
int llama_token_persistent_detect_graph_input_patch_attempt(void);
int llama_token_persistent_detect_tensor_rebinding_attempt(void);
int llama_token_persistent_detect_backend_reassignment_attempt(void);
int llama_token_persistent_detect_cpu_orchestration_attempt(void);

// Mutation detection
int llama_token_persistent_detect_graph_mutation(
    enum llama_graph_state_mutation mutation_type
);
int llama_token_persistent_detect_input_shape_change(void);
int llama_token_persistent_detect_input_location_change(void);

// Query and verification functions
struct llama_decode_mode_record llama_token_persistent_get_decode_mode_record(void);
struct llama_token_persistent_execution_record llama_token_persistent_get_execution_record(void);
struct llama_token_persistent_state llama_token_persistent_get_gpu_state(void);
enum llama_decode_mode_state llama_token_persistent_get_decode_mode(void);

// Persistence verification
int llama_token_persistent_verify_single_graph_throughout_decode(void);
int llama_token_persistent_verify_no_per_token_resubmission(void);
int llama_token_persistent_verify_graph_inputs_stable(void);
int llama_token_persistent_verify_gpu_owns_token_progression(void);
int llama_token_persistent_verify_no_cpu_orchestration(void);

// Lifetime binding verification
int llama_token_persistent_verify_graph_lifetime_matches_decode(void);
int llama_token_persistent_verify_graph_not_outliving_decode(void);
int llama_token_persistent_verify_decode_not_outliving_graph(void);

// Diagnostics and logging
void llama_token_persistent_log_decode_mode_entered(uint64_t graph_id);
void llama_token_persistent_log_graph_lifetime_bound(void);
void llama_token_persistent_log_persistence_locked(void);
void llama_token_persistent_print_decode_mode_status(void);
void llama_token_persistent_print_gpu_state_summary(void);
void llama_token_persistent_print_invariant_violations(void);

// Violation reporting
void llama_token_persistent_report_reentrancy_violation(
    enum llama_cpu_reentrancy_violation violation_type,
    const char* details
);
void llama_token_persistent_report_mutation_violation(
    enum llama_graph_state_mutation mutation_type
);

// Enforcement mode control
void llama_token_persistent_set_enforcement_strict(bool strict);
bool llama_token_persistent_get_enforcement_strict(void);
void llama_token_persistent_set_debug_check_persistence_per_token(bool debug);

// Exit decode mode
int llama_token_persistent_exit_decode_mode(void);
int llama_token_persistent_verify_decode_mode_exit_clean(void);

// Self-test suite
int llama_token_persistent_selftest(void);

// Helper/inline functions
static inline const char* llama_decode_mode_state_name(
    enum llama_decode_mode_state state
) {
    switch (state) {
        case LLAMA_DECODE_MODE_INACTIVE: return "INACTIVE";
        case LLAMA_DECODE_MODE_INITIALIZING: return "INITIALIZING";
        case LLAMA_DECODE_MODE_ACTIVE: return "ACTIVE";
        case LLAMA_DECODE_MODE_TERMINATING: return "TERMINATING";
        case LLAMA_DECODE_MODE_TERMINATED: return "TERMINATED";
        case LLAMA_DECODE_MODE_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_graph_lifetime_binding_name(
    enum llama_graph_lifetime_binding binding
) {
    switch (binding) {
        case LLAMA_LIFETIME_UNBOUND: return "UNBOUND";
        case LLAMA_LIFETIME_BINDING_START: return "BINDING_START";
        case LLAMA_LIFETIME_BOUND: return "BOUND";
        case LLAMA_LIFETIME_LOCKED: return "LOCKED";
        case LLAMA_LIFETIME_INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_graph_persistence_state_name(
    enum llama_graph_persistence_state persistence
) {
    switch (persistence) {
        case LLAMA_PERSISTENCE_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_PERSISTENCE_FIRST_TOKEN: return "FIRST_TOKEN";
        case LLAMA_PERSISTENCE_PERSISTENT: return "PERSISTENT";
        case LLAMA_PERSISTENCE_LOCKED: return "LOCKED";
        case LLAMA_PERSISTENCE_INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_cpu_reentrancy_violation_name(
    enum llama_cpu_reentrancy_violation violation
) {
    switch (violation) {
        case LLAMA_REENTRANCY_NONE: return "NONE";
        case LLAMA_REENTRANCY_GRAPH_REBUILD: return "GRAPH_REBUILD";
        case LLAMA_REENTRANCY_GRAPH_RESUBMIT: return "GRAPH_RESUBMIT";
        case LLAMA_REENTRANCY_TOKEN_COUNTER_UPDATE: return "TOKEN_COUNTER_UPDATE";
        case LLAMA_REENTRANCY_GRAPH_INPUT_PATCH: return "GRAPH_INPUT_PATCH";
        case LLAMA_REENTRANCY_TENSOR_REBINDING: return "TENSOR_REBINDING";
        case LLAMA_REENTRANCY_BACKEND_REASSIGN: return "BACKEND_REASSIGN";
        case LLAMA_REENTRANCY_ORCHESTRATION: return "ORCHESTRATION";
        case LLAMA_REENTRANCY_BUFFER_REBIND: return "BUFFER_REBIND";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_graph_state_mutation_name(
    enum llama_graph_state_mutation mutation
) {
    switch (mutation) {
        case LLAMA_MUTATION_NONE: return "NONE";
        case LLAMA_MUTATION_LIFETIME_CHANGE: return "LIFETIME_CHANGE";
        case LLAMA_MUTATION_PERSISTENCE_BREAK: return "PERSISTENCE_BREAK";
        case LLAMA_MUTATION_BACKEND_CHANGE: return "BACKEND_CHANGE";
        case LLAMA_MUTATION_EXECUTION_PLAN_CHANGE: return "EXECUTION_PLAN_CHANGE";
        case LLAMA_MUTATION_INPUT_SHAPE_CHANGE: return "INPUT_SHAPE_CHANGE";
        case LLAMA_MUTATION_INPUT_LOCATION_CHANGE: return "INPUT_LOCATION_CHANGE";
        case LLAMA_MUTATION_PARAMETER_INJECTION: return "PARAMETER_INJECTION";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_token_progress_state_name(
    enum llama_token_progress_state progress
) {
    switch (progress) {
        case LLAMA_TOKEN_PROGRESS_NONE: return "NONE";
        case LLAMA_TOKEN_PROGRESS_FETCHED: return "FETCHED";
        case LLAMA_TOKEN_PROGRESS_SUBMITTED: return "SUBMITTED";
        case LLAMA_TOKEN_PROGRESS_EXECUTING: return "EXECUTING";
        case LLAMA_TOKEN_PROGRESS_COMPLETE: return "COMPLETE";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
