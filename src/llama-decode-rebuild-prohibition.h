/**
 * SECTION 12: Prohibit Graph Rebuilds During Decode
 * Header
 *
 * This file implements enforcement that graph rebuilds are completely forbidden
 * once decode has started. Any attempt to rebuild, invalidate, or regenerate
 * the graph during decode is treated as a fatal correctness error.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// DECODE PROGRESS STATE ENUMERATION
// ============================================================================

/**
 * State tracking whether decode is currently active
 */
enum llama_decode_progress_state {
    LLAMA_DECODE_PROGRESS_NOT_STARTED = 0,  // Decode has not begun
    LLAMA_DECODE_PROGRESS_STARTING = 1,     // Decode initialization in progress
    LLAMA_DECODE_PROGRESS_ACTIVE = 2,       // Decode is currently running
    LLAMA_DECODE_PROGRESS_PAUSED = 3,       // Decode paused (not used in strict mode)
    LLAMA_DECODE_PROGRESS_COMPLETED = 4,    // Decode session completed
};

// ============================================================================
// REBUILD TRIGGER TYPE ENUMERATION
// ============================================================================

/**
 * Types of conditions that might trigger a rebuild
 */
enum llama_rebuild_trigger_type {
    LLAMA_REBUILD_TRIGGER_NONE = 0,
    LLAMA_REBUILD_TRIGGER_SHAPE_MISMATCH = 1,      // Tensor shape changed
    LLAMA_REBUILD_TRIGGER_CONTEXT_GROWTH = 2,      // Context length increased
    LLAMA_REBUILD_TRIGGER_KV_CACHE_EXPANSION = 3,  // KV cache exceeded bounds
    LLAMA_REBUILD_TRIGGER_TOPOLOGY_CHANGE = 4,     // Graph topology changed
    LLAMA_REBUILD_TRIGGER_BACKEND_UNAVAILABLE = 5, // Backend became unavailable
    LLAMA_REBUILD_TRIGGER_MEMORY_REALLOCATION = 6, // Memory reallocation needed
    LLAMA_REBUILD_TRIGGER_VERSION_MISMATCH = 7,    // Graph version mismatch
    LLAMA_REBUILD_TRIGGER_AUTO_INVALIDATION = 8,   // Automatic invalidation detected
};

// ============================================================================
// REBUILD ATTEMPT LOCATION ENUMERATION
// ============================================================================

/**
 * Where a rebuild attempt was made from
 */
enum llama_rebuild_attempt_location {
    LLAMA_REBUILD_LOC_UNKNOWN = 0,
    LLAMA_REBUILD_LOC_GRAPH_REVALIDATE = 1,
    LLAMA_REBUILD_LOC_GRAPH_REGENERATE = 2,
    LLAMA_REBUILD_LOC_SHAPE_ADAPTATION = 3,
    LLAMA_REBUILD_LOC_KV_CACHE_EXTEND = 4,
    LLAMA_REBUILD_LOC_BACKEND_REASSIGN = 5,
    LLAMA_REBUILD_LOC_MEMORY_REALLOC = 6,
    LLAMA_REBUILD_LOC_AUTO_FALLBACK = 7,
};

// ============================================================================
// REBUILD VIOLATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of rebuild prohibition violations
 */
enum llama_rebuild_violation_type {
    LLAMA_REBUILD_VIOL_NONE = 0,
    LLAMA_REBUILD_VIOL_REBUILD_ATTEMPTED = 1,
    LLAMA_REBUILD_VIOL_REVALIDATE_ATTEMPTED = 2,
    LLAMA_REBUILD_VIOL_REGENERATE_ATTEMPTED = 3,
    LLAMA_REBUILD_VIOL_SHAPE_ADAPTATION = 4,
    LLAMA_REBUILD_VIOL_INVALIDATION = 5,
    LLAMA_REBUILD_VIOL_VERSION_MISMATCH = 6,
    LLAMA_REBUILD_VIOL_REBUILD_FLAG_SET = 7,
};

// ============================================================================
// REBUILD PROHIBITION RECORD
// ============================================================================

/**
 * Struct to track rebuild prohibition state and violations
 */
struct llama_rebuild_prohibition_record {
    enum llama_decode_progress_state decode_progress;
    bool decode_in_progress;                        // True if decode active
    uint64_t decode_start_timestamp_ns;             // When decode started
    uint64_t decode_step_count;                     // Number of steps completed
    uint64_t graph_id_at_decode_start;              // Graph ID when decode started
    uint32_t graph_version_at_decode_start;         // Graph version when decode started
    int rebuild_attempt_count;                      // Number of rebuild attempts
    enum llama_rebuild_trigger_type last_trigger;   // Last rebuild trigger
    enum llama_rebuild_attempt_location last_location; // Last attempt location
    enum llama_rebuild_violation_type last_violation; // Last violation type
};

// ============================================================================
// REBUILD PROHIBITION VALIDATION STATE
// ============================================================================

/**
 * Global state for rebuild prohibition enforcement
 */
struct llama_rebuild_prohibition_validation_state {
    struct llama_rebuild_prohibition_record prohibition_record;
    int total_rebuild_attempts;
    int total_rebuild_violations;
    bool enforcement_strict;
    bool debug_assert_graph_immutable_per_step;
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
int llama_rebuild_prohibition_init(void);

// Decode progress tracking (5 enforcement points: 1-5)
int llama_rebuild_prohibition_mark_decode_starting(
    uint64_t graph_id,
    uint32_t graph_version
);
int llama_rebuild_prohibition_mark_decode_active(void);
int llama_rebuild_prohibition_mark_decode_step_complete(void);
int llama_rebuild_prohibition_mark_decode_completed(void);
int llama_rebuild_prohibition_verify_decode_not_active(void);

// Rebuild entry point guards (6 enforcement points: 6-11)
int llama_rebuild_prohibition_guard_graph_revalidate(void);
int llama_rebuild_prohibition_guard_graph_regenerate(void);
int llama_rebuild_prohibition_guard_shape_adaptation(void);
int llama_rebuild_prohibition_guard_kv_cache_expansion(void);
int llama_rebuild_prohibition_guard_backend_reassignment(void);
int llama_rebuild_prohibition_guard_memory_reallocation(void);

// Rebuild flag checking
int llama_rebuild_prohibition_check_no_rebuild_flags_set(void);
int llama_rebuild_prohibition_check_graph_version_unchanged(
    uint64_t current_graph_id,
    uint32_t current_graph_version
);

// Late-discovered invalidation handling
int llama_rebuild_prohibition_handle_late_invalidation(
    enum llama_rebuild_trigger_type trigger_reason
);

// Query and diagnostic functions
bool llama_rebuild_prohibition_is_decode_active(void);
enum llama_decode_progress_state llama_rebuild_prohibition_get_decode_progress(void);
struct llama_rebuild_prohibition_record llama_rebuild_prohibition_get_record(void);
int llama_rebuild_prohibition_get_rebuild_attempt_count(void);
uint64_t llama_rebuild_prohibition_get_decode_step_count(void);

// Verification functions
int llama_rebuild_prohibition_verify_no_auto_rebuild_active(void);
int llama_rebuild_prohibition_verify_graph_stable_for_decode(void);
int llama_rebuild_prohibition_assert_not_in_rebuild_path(void);

// Violation reporting
void llama_rebuild_prohibition_report_rebuild_attempt(
    enum llama_rebuild_trigger_type trigger,
    enum llama_rebuild_attempt_location location,
    const char* reason
);

// Diagnostics and logging
void llama_rebuild_prohibition_log_decode_started(void);
void llama_rebuild_prohibition_log_decode_completed(void);
void llama_rebuild_prohibition_print_status(void);
void llama_rebuild_prohibition_print_diagnostics(void);

// Enforcement mode control
void llama_rebuild_prohibition_set_enforcement_strict(bool strict);
bool llama_rebuild_prohibition_get_enforcement_strict(void);
void llama_rebuild_prohibition_set_debug_assert_immutable_per_step(bool assert_immutable);

// Self-test suite
int llama_rebuild_prohibition_selftest(void);

// Helper/inline functions
static inline const char* llama_decode_progress_state_name(
    enum llama_decode_progress_state state
) {
    switch (state) {
        case LLAMA_DECODE_PROGRESS_NOT_STARTED: return "NOT_STARTED";
        case LLAMA_DECODE_PROGRESS_STARTING: return "STARTING";
        case LLAMA_DECODE_PROGRESS_ACTIVE: return "ACTIVE";
        case LLAMA_DECODE_PROGRESS_PAUSED: return "PAUSED";
        case LLAMA_DECODE_PROGRESS_COMPLETED: return "COMPLETED";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_rebuild_trigger_type_name(
    enum llama_rebuild_trigger_type trigger
) {
    switch (trigger) {
        case LLAMA_REBUILD_TRIGGER_NONE: return "NONE";
        case LLAMA_REBUILD_TRIGGER_SHAPE_MISMATCH: return "SHAPE_MISMATCH";
        case LLAMA_REBUILD_TRIGGER_CONTEXT_GROWTH: return "CONTEXT_GROWTH";
        case LLAMA_REBUILD_TRIGGER_KV_CACHE_EXPANSION: return "KV_CACHE_EXPANSION";
        case LLAMA_REBUILD_TRIGGER_TOPOLOGY_CHANGE: return "TOPOLOGY_CHANGE";
        case LLAMA_REBUILD_TRIGGER_BACKEND_UNAVAILABLE: return "BACKEND_UNAVAILABLE";
        case LLAMA_REBUILD_TRIGGER_MEMORY_REALLOCATION: return "MEMORY_REALLOCATION";
        case LLAMA_REBUILD_TRIGGER_VERSION_MISMATCH: return "VERSION_MISMATCH";
        case LLAMA_REBUILD_TRIGGER_AUTO_INVALIDATION: return "AUTO_INVALIDATION";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_rebuild_violation_type_name(
    enum llama_rebuild_violation_type violation
) {
    switch (violation) {
        case LLAMA_REBUILD_VIOL_NONE: return "NONE";
        case LLAMA_REBUILD_VIOL_REBUILD_ATTEMPTED: return "REBUILD_ATTEMPTED";
        case LLAMA_REBUILD_VIOL_REVALIDATE_ATTEMPTED: return "REVALIDATE_ATTEMPTED";
        case LLAMA_REBUILD_VIOL_REGENERATE_ATTEMPTED: return "REGENERATE_ATTEMPTED";
        case LLAMA_REBUILD_VIOL_SHAPE_ADAPTATION: return "SHAPE_ADAPTATION";
        case LLAMA_REBUILD_VIOL_INVALIDATION: return "INVALIDATION";
        case LLAMA_REBUILD_VIOL_VERSION_MISMATCH: return "VERSION_MISMATCH";
        case LLAMA_REBUILD_VIOL_REBUILD_FLAG_SET: return "REBUILD_FLAG_SET";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
