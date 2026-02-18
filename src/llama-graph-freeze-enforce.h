/**
 * SECTION 11: Freeze Decode Graph Construction Pre-Decode
 * Header
 *
 * This file implements enforcement that the decode graph is fully constructed,
 * validated, and frozen before the first decode token is generated. No structural
 * graph changes are permitted during decode.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// GRAPH LIFECYCLE PHASE ENUMERATION
// ============================================================================

/**
 * Phases of graph lifecycle - strictly ordered and irreversible
 */
enum llama_graph_lifecycle_phase {
    LLAMA_GRAPH_PHASE_UNINITIALIZED = 0,    // Graph not yet created
    LLAMA_GRAPH_PHASE_PREFILL_BUILD = 1,    // Prefill graph construction allowed
    LLAMA_GRAPH_PHASE_PREFILL_EXEC = 2,     // Prefill execution phase
    LLAMA_GRAPH_PHASE_DECODE_BUILD = 3,     // Decode graph construction allowed
    LLAMA_GRAPH_PHASE_DECODE_FROZEN = 4,    // Decode graph frozen (no changes)
    LLAMA_GRAPH_PHASE_DECODE_EXEC = 5,      // Decode execution (immutable graph)
    LLAMA_GRAPH_PHASE_DECODE_COMPLETE = 6,  // Decode complete, graph released
};

// ============================================================================
// GRAPH FREEZE STATE ENUMERATION
// ============================================================================

/**
 * Freeze state of a decode graph
 */
enum llama_graph_freeze_state {
    LLAMA_GRAPH_FREEZE_UNFROZEN = 0,        // Graph is mutable
    LLAMA_GRAPH_FREEZE_FREEZING = 1,        // Freeze operation in progress
    LLAMA_GRAPH_FREEZE_FROZEN = 2,          // Graph is immutable
    LLAMA_GRAPH_FREEZE_INVALID = 3,         // Graph in invalid state
};

// ============================================================================
// GRAPH MUTATION VIOLATION TYPE
// ============================================================================

/**
 * Types of graph mutations that can be attempted
 */
enum llama_graph_mutation_type {
    LLAMA_GRAPH_MUT_NONE = 0,
    LLAMA_GRAPH_MUT_NODE_ADD = 1,           // Attempted node addition
    LLAMA_GRAPH_MUT_NODE_REMOVE = 2,        // Attempted node removal
    LLAMA_GRAPH_MUT_NODE_REORDER = 3,       // Attempted node reordering
    LLAMA_GRAPH_MUT_BACKEND_CHANGE = 4,     // Attempted backend reassignment
    LLAMA_GRAPH_MUT_SHAPE_CHANGE = 5,       // Attempted tensor shape change
    LLAMA_GRAPH_MUT_REBUILD = 6,            // Attempted graph rebuild
    LLAMA_GRAPH_MUT_REVALIDATE = 7,         // Attempted revalidation
    LLAMA_GRAPH_MUT_FALLBACK = 8,           // Attempted fallback execution
};

// ============================================================================
// GRAPH VALIDATION FAILURE TYPE
// ============================================================================

/**
 * Reasons why graph validation might fail at freeze time
 */
enum llama_graph_validation_failure_reason {
    LLAMA_GRAPH_VALID_OK = 0,
    LLAMA_GRAPH_VALID_CPU_NODE_DECODE_CRITICAL = 1,  // CPU node on decode path
    LLAMA_GRAPH_VALID_MIXED_BACKEND = 2,             // Mixed GPU/CPU backends
    LLAMA_GRAPH_VALID_BACKEND_LOCK_VIOLATED = 3,     // Backend lock not satisfied
    LLAMA_GRAPH_VALID_SHAPE_VARIABLE = 4,            // Variable tensor shape
    LLAMA_GRAPH_VALID_PLACEHOLDER_NODE = 5,          // Placeholder/unfilled node
    LLAMA_GRAPH_VALID_UNSAFE_ALLOCATION = 6,         // Unsafe memory allocation
    LLAMA_GRAPH_VALID_UNKNOWN = 7,
};

// ============================================================================
// GRAPH FREEZE RECORD
// ============================================================================

/**
 * Struct to track graph freeze state and validation
 */
struct llama_graph_freeze_record {
    enum llama_graph_lifecycle_phase current_phase;
    enum llama_graph_freeze_state freeze_state;
    uint64_t graph_id;                              // Unique graph identifier
    uint64_t graph_pointer;                         // Graph memory address
    bool graph_frozen;                              // True if frozen
    bool graph_valid;                               // True if valid for decode
    uint64_t nodes_count;                           // Number of nodes in graph
    uint64_t freeze_timestamp_ns;                   // When frozen (ns)
    int mutation_attempt_count;                     // Failed mutation attempts
    enum llama_graph_mutation_type last_mutation_attempt;
    enum llama_graph_validation_failure_reason validation_failure; // Why validation failed
};

// ============================================================================
// GRAPH FREEZE VALIDATION STATE
// ============================================================================

/**
 * Global state for graph freeze enforcement
 */
struct llama_graph_freeze_validation_state {
    struct llama_graph_freeze_record graph_record;
    int total_mutation_attempts;
    int total_validation_failures;
    bool enforcement_strict;
    bool debug_assert_frozen_per_step;              // Assert frozen before each exec step
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
int llama_graph_freeze_init(void);

// Phase management (5 enforcement points: 1-5)
int llama_graph_freeze_enter_prefill_build_phase(void);
int llama_graph_freeze_exit_prefill_phase(void);
int llama_graph_freeze_enter_decode_build_phase(void);
int llama_graph_freeze_enter_decode_exec_phase(void);
int llama_graph_freeze_exit_decode_phase(void);

// Graph construction control (3 enforcement points: 6-8)
int llama_graph_freeze_construct_decode_graph_once(
    uint64_t graph_id,
    uint64_t graph_pointer,
    uint64_t node_count
);
int llama_graph_freeze_validate_graph_before_freeze(void);
int llama_graph_freeze_freeze_graph(void);

// Mutation prevention (2 enforcement points: 9-10)
int llama_graph_freeze_prevent_mutation(enum llama_graph_mutation_type mutation_type);
int llama_graph_freeze_prevent_graph_rebuild(void);

// Shape change prevention
int llama_graph_freeze_prevent_shape_invalidation(void);

// Runtime assertions (1 enforcement point: 11)
int llama_graph_freeze_assert_frozen_at_decode_step(void);

// Query and diagnostic functions
bool llama_graph_freeze_is_graph_frozen(void);
enum llama_graph_lifecycle_phase llama_graph_freeze_get_current_phase(void);
struct llama_graph_freeze_record llama_graph_freeze_get_record(void);
int llama_graph_freeze_get_mutation_attempt_count(void);

// Validation functions
int llama_graph_freeze_verify_decode_critical_nodes_gpu_backed(void);
int llama_graph_freeze_verify_no_cpu_nodes_on_critical_path(void);
int llama_graph_freeze_verify_backend_lock_satisfied(void);
int llama_graph_freeze_verify_graph_structure_stable(void);

// Diagnostics and logging
void llama_graph_freeze_log_graph_frozen(void);
void llama_graph_freeze_log_phase_transition(
    enum llama_graph_lifecycle_phase from_phase,
    enum llama_graph_lifecycle_phase to_phase
);
void llama_graph_freeze_print_status(void);
void llama_graph_freeze_print_diagnostics(void);

// Violation reporting
void llama_graph_freeze_report_mutation_attempt(
    enum llama_graph_mutation_type mutation_type,
    const char* details
);

// Enforcement mode control
void llama_graph_freeze_set_enforcement_strict(bool strict);
bool llama_graph_freeze_get_enforcement_strict(void);
void llama_graph_freeze_set_debug_assert_frozen_per_step(bool assert_frozen);

// Self-test suite
int llama_graph_freeze_selftest(void);

// Helper/inline functions
static inline const char* llama_graph_lifecycle_phase_name(
    enum llama_graph_lifecycle_phase phase
) {
    switch (phase) {
        case LLAMA_GRAPH_PHASE_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GRAPH_PHASE_PREFILL_BUILD: return "PREFILL_BUILD";
        case LLAMA_GRAPH_PHASE_PREFILL_EXEC: return "PREFILL_EXEC";
        case LLAMA_GRAPH_PHASE_DECODE_BUILD: return "DECODE_BUILD";
        case LLAMA_GRAPH_PHASE_DECODE_FROZEN: return "DECODE_FROZEN";
        case LLAMA_GRAPH_PHASE_DECODE_EXEC: return "DECODE_EXEC";
        case LLAMA_GRAPH_PHASE_DECODE_COMPLETE: return "DECODE_COMPLETE";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_graph_freeze_state_name(
    enum llama_graph_freeze_state state
) {
    switch (state) {
        case LLAMA_GRAPH_FREEZE_UNFROZEN: return "UNFROZEN";
        case LLAMA_GRAPH_FREEZE_FREEZING: return "FREEZING";
        case LLAMA_GRAPH_FREEZE_FROZEN: return "FROZEN";
        case LLAMA_GRAPH_FREEZE_INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_graph_mutation_type_name(
    enum llama_graph_mutation_type mutation_type
) {
    switch (mutation_type) {
        case LLAMA_GRAPH_MUT_NONE: return "NONE";
        case LLAMA_GRAPH_MUT_NODE_ADD: return "NODE_ADD";
        case LLAMA_GRAPH_MUT_NODE_REMOVE: return "NODE_REMOVE";
        case LLAMA_GRAPH_MUT_NODE_REORDER: return "NODE_REORDER";
        case LLAMA_GRAPH_MUT_BACKEND_CHANGE: return "BACKEND_CHANGE";
        case LLAMA_GRAPH_MUT_SHAPE_CHANGE: return "SHAPE_CHANGE";
        case LLAMA_GRAPH_MUT_REBUILD: return "REBUILD";
        case LLAMA_GRAPH_MUT_REVALIDATE: return "REVALIDATE";
        case LLAMA_GRAPH_MUT_FALLBACK: return "FALLBACK";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_graph_validation_failure_reason_name(
    enum llama_graph_validation_failure_reason reason
) {
    switch (reason) {
        case LLAMA_GRAPH_VALID_OK: return "OK";
        case LLAMA_GRAPH_VALID_CPU_NODE_DECODE_CRITICAL: return "CPU_NODE_ON_CRITICAL_PATH";
        case LLAMA_GRAPH_VALID_MIXED_BACKEND: return "MIXED_BACKEND";
        case LLAMA_GRAPH_VALID_BACKEND_LOCK_VIOLATED: return "BACKEND_LOCK_VIOLATED";
        case LLAMA_GRAPH_VALID_SHAPE_VARIABLE: return "VARIABLE_SHAPE";
        case LLAMA_GRAPH_VALID_PLACEHOLDER_NODE: return "PLACEHOLDER_NODE";
        case LLAMA_GRAPH_VALID_UNSAFE_ALLOCATION: return "UNSAFE_ALLOCATION";
        case LLAMA_GRAPH_VALID_UNKNOWN: return "UNKNOWN";
        default: return "INVALID";
    }
}

#ifdef __cplusplus
}
#endif
