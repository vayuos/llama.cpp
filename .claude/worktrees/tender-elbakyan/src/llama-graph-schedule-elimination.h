/**
 * SECTION 14: Remove per-token graph scheduling logic
 * Header
 *
 * This file implements enforcement that all per-token graph scheduling and
 * traversal logic is eliminated from CPU execution during decode. The decode
 * graph executes as a predefined, fixed execution plan computed once at graph
 * build time. Dynamic traversal, node readiness checks, and topological sorts
 * are forbidden during decode.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// EXECUTION MODE ENUMERATION
// ============================================================================

/**
 * Execution mode for graph processing
 */
enum llama_graph_execution_mode {
    LLAMA_EXEC_MODE_DYNAMIC = 0,        // Dynamic traversal (forbidden during decode)
    LLAMA_EXEC_MODE_STATIC = 1,         // Static precomputed plan (required during decode)
    LLAMA_EXEC_MODE_SCHEDULING = 2,     // Scheduling in progress (transition state)
    LLAMA_EXEC_MODE_INVALID = 3,        // Invalid execution mode
};

// ============================================================================
// EXECUTION PLAN STATE ENUMERATION
// ============================================================================

/**
 * State of precomputed execution plan
 */
enum llama_execution_plan_state {
    LLAMA_PLAN_UNCOMPUTED = 0,         // Plan not yet computed
    LLAMA_PLAN_COMPUTING = 1,          // Computing plan in progress
    LLAMA_PLAN_COMPUTED = 2,           // Plan computed and valid
    LLAMA_PLAN_LOCKED = 3,             // Plan locked (immutable)
    LLAMA_PLAN_INVALID = 4,            // Plan invalidated
};

// ============================================================================
// SCHEDULING VIOLATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of scheduling violations detected
 */
enum llama_scheduling_violation_type {
    LLAMA_SCHED_VIOL_NONE = 0,
    LLAMA_SCHED_VIOL_DYNAMIC_TRAVERSAL = 1,     // Attempted dynamic graph traversal
    LLAMA_SCHED_VIOL_NODE_READINESS_CHECK = 2,  // Readiness check during decode
    LLAMA_SCHED_VIOL_TOPOLOGICAL_SORT = 3,      // Topological sort during decode
    LLAMA_SCHED_VIOL_DEPENDENCY_REEVAL = 4,     // Dependency re-evaluation
    LLAMA_SCHED_VIOL_EXECUTION_REORDER = 5,     // Execution order changed
    LLAMA_SCHED_VIOL_CONDITIONAL_SKIP = 6,      // Conditional node skip attempted
    LLAMA_SCHED_VIOL_SCHEDULER_CALLBACK = 7,    // Scheduler callback invoked
    LLAMA_SCHED_VIOL_SCHEDULING_ATTEMPT = 8,    // Scheduling API called during decode
};

// ============================================================================
// SCHEDULER CALLBACK TYPE ENUMERATION
// ============================================================================

/**
 * Types of scheduler callbacks that must be blocked
 */
enum llama_scheduler_callback_type {
    LLAMA_SCHED_CB_NONE = 0,
    LLAMA_SCHED_CB_PRE_EXECUTE = 1,             // Pre-execution callback
    LLAMA_SCHED_CB_POST_EXECUTE = 2,            // Post-execution callback
    LLAMA_SCHED_CB_NODE_SELECT = 3,             // Node selection callback
    LLAMA_SCHED_CB_DEPENDENCY_READY = 4,        // Dependency ready callback
    LLAMA_SCHED_CB_PRIORITY_UPDATE = 5,         // Priority update callback
    LLAMA_SCHED_CB_LOAD_BALANCE = 6,            // Load balancing callback
};

// ============================================================================
// EXECUTION PLAN SEGMENT
// ============================================================================

/**
 * Single segment in the precomputed execution plan
 */
struct llama_execution_plan_segment {
    uint64_t node_id;                           // Node ID to execute
    int segment_index;                          // Position in execution order
    int num_dependencies;                       // Number of dependencies
    uint64_t * dependency_node_ids;             // IDs of dependencies (allocated)
    bool can_parallelize;                       // Can this node parallelize with next
    uint32_t gpu_utilization_estimate;          // Estimated GPU utilization %
};

// ============================================================================
// EXECUTION PLAN RECORD
// ============================================================================

/**
 * Precomputed execution plan for a decode graph
 */
struct llama_execution_plan_record {
    uint64_t graph_id;                          // Graph ID this plan belongs to
    uint32_t graph_version;                     // Graph version when plan created
    int total_segments;                         // Total segments in plan
    struct llama_execution_plan_segment * segments; // Array of segments (allocated)
    enum llama_execution_plan_state plan_state; // Current plan state
    enum llama_graph_execution_mode exec_mode;  // Execution mode for this plan
    bool plan_immutable;                        // True = plan cannot change
    uint64_t plan_creation_time_ns;             // When plan was computed
    int num_execution_steps;                    // Expected steps to execute plan
};

// ============================================================================
// SCHEDULING ELIMINATION STATE
// ============================================================================

/**
 * Global state for scheduling elimination enforcement
 */
struct llama_graph_schedule_elimination_record {
    uint64_t current_graph_id;                  // Current active graph
    enum llama_graph_execution_mode current_mode; // Current execution mode
    enum llama_execution_plan_state plan_state; // Current plan state
    int total_plans_computed;                   // Plans computed so far
    int total_scheduling_attempts;              // Attempts to use dynamic scheduling
    int total_scheduling_violations;            // Violations detected
    enum llama_scheduling_violation_type last_violation; // Last violation type
    enum llama_scheduler_callback_type last_callback_blocked; // Last blocked callback
    bool static_execution_active;               // True = static execution mode active
    bool dynamic_traversal_forbidden;           // True = dynamic traversal forbidden
};

// ============================================================================
// SCHEDULE ELIMINATION VALIDATION STATE
// ============================================================================

/**
 * Global validation state for schedule elimination enforcement
 */
struct llama_graph_schedule_elimination_validation_state {
    struct llama_graph_schedule_elimination_record elimination_record;
    int total_mode_violations;
    int total_plan_mismatches;
    int total_callback_violations;
    bool enforcement_strict;                    // Abort on violation vs log only
    bool debug_assert_static_per_step;          // Assert static mode at each step
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
int llama_schedule_elimination_init(void);

// Execution plan computation (5 enforcement points: 1-5)
int llama_schedule_elimination_precompute_execution_order(
    uint64_t graph_id,
    uint32_t graph_version
);
int llama_schedule_elimination_store_execution_plan(
    uint64_t graph_id,
    struct llama_execution_plan_record * plan
);
int llama_schedule_elimination_linearize_graph_traversal(uint64_t graph_id);
int llama_schedule_elimination_bind_plan_to_graph(uint64_t graph_id);
int llama_schedule_elimination_lock_execution_plan(uint64_t graph_id);

// Dynamic scheduling prevention (3 enforcement points: 6-8)
int llama_schedule_elimination_forbid_dynamic_traversal(void);
int llama_schedule_elimination_forbid_scheduler_callbacks(void);
int llama_schedule_elimination_fail_on_scheduling_api(void);

// Static execution mode enforcement (2 enforcement points: 9-10)
int llama_schedule_elimination_enable_static_execution_mode(uint64_t graph_id);
int llama_schedule_elimination_assert_static_mode_active(void);

// Scheduling violation detection
int llama_schedule_elimination_detect_dynamic_traversal_attempt(void);
int llama_schedule_elimination_detect_readiness_check_attempt(void);
int llama_schedule_elimination_detect_topological_sort_attempt(void);
int llama_schedule_elimination_detect_dependency_reevaluation(void);
int llama_schedule_elimination_detect_execution_reorder_attempt(void);
int llama_schedule_elimination_detect_conditional_skip_attempt(void);

// Scheduler callback blocking
int llama_schedule_elimination_block_scheduler_callback(
    enum llama_scheduler_callback_type callback_type
);
int llama_schedule_elimination_verify_no_callbacks_registered(void);

// Query and diagnostic functions
struct llama_execution_plan_record llama_schedule_elimination_get_plan(uint64_t graph_id);
struct llama_graph_schedule_elimination_record llama_schedule_elimination_get_record(void);
int llama_schedule_elimination_get_plan_segment_count(uint64_t graph_id);
enum llama_graph_execution_mode llama_schedule_elimination_get_execution_mode(void);

// Verification functions
int llama_schedule_elimination_verify_plan_complete(uint64_t graph_id);
int llama_schedule_elimination_verify_plan_immutable(uint64_t graph_id);
int llama_schedule_elimination_verify_no_dynamic_scheduling(void);
int llama_schedule_elimination_verify_static_mode_throughout(void);

// Diagnostics and logging
void llama_schedule_elimination_log_plan_computed(uint64_t graph_id, int num_segments);
void llama_schedule_elimination_log_static_mode_enabled(void);
void llama_schedule_elimination_print_execution_plan(uint64_t graph_id);
void llama_schedule_elimination_print_status(void);

// Violation reporting
void llama_schedule_elimination_report_scheduling_attempt(
    enum llama_scheduling_violation_type violation_type,
    const char* details
);
void llama_schedule_elimination_report_callback_invocation(
    enum llama_scheduler_callback_type callback_type
);

// Enforcement mode control
void llama_schedule_elimination_set_enforcement_strict(bool strict);
bool llama_schedule_elimination_get_enforcement_strict(void);
void llama_schedule_elimination_set_debug_assert_static_per_step(bool debug);

// Static execution interface
int llama_schedule_elimination_execute_static_plan_step(uint64_t graph_id);
int llama_schedule_elimination_reset_plan_execution_state(uint64_t graph_id);

// Self-test suite
int llama_schedule_elimination_selftest(void);

// Helper/inline functions
static inline const char* llama_graph_execution_mode_name(
    enum llama_graph_execution_mode mode
) {
    switch (mode) {
        case LLAMA_EXEC_MODE_DYNAMIC: return "DYNAMIC";
        case LLAMA_EXEC_MODE_STATIC: return "STATIC";
        case LLAMA_EXEC_MODE_SCHEDULING: return "SCHEDULING";
        case LLAMA_EXEC_MODE_INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_execution_plan_state_name(
    enum llama_execution_plan_state state
) {
    switch (state) {
        case LLAMA_PLAN_UNCOMPUTED: return "UNCOMPUTED";
        case LLAMA_PLAN_COMPUTING: return "COMPUTING";
        case LLAMA_PLAN_COMPUTED: return "COMPUTED";
        case LLAMA_PLAN_LOCKED: return "LOCKED";
        case LLAMA_PLAN_INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_scheduling_violation_type_name(
    enum llama_scheduling_violation_type violation
) {
    switch (violation) {
        case LLAMA_SCHED_VIOL_NONE: return "NONE";
        case LLAMA_SCHED_VIOL_DYNAMIC_TRAVERSAL: return "DYNAMIC_TRAVERSAL";
        case LLAMA_SCHED_VIOL_NODE_READINESS_CHECK: return "NODE_READINESS_CHECK";
        case LLAMA_SCHED_VIOL_TOPOLOGICAL_SORT: return "TOPOLOGICAL_SORT";
        case LLAMA_SCHED_VIOL_DEPENDENCY_REEVAL: return "DEPENDENCY_REEVAL";
        case LLAMA_SCHED_VIOL_EXECUTION_REORDER: return "EXECUTION_REORDER";
        case LLAMA_SCHED_VIOL_CONDITIONAL_SKIP: return "CONDITIONAL_SKIP";
        case LLAMA_SCHED_VIOL_SCHEDULER_CALLBACK: return "SCHEDULER_CALLBACK";
        case LLAMA_SCHED_VIOL_SCHEDULING_ATTEMPT: return "SCHEDULING_ATTEMPT";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_scheduler_callback_type_name(
    enum llama_scheduler_callback_type callback
) {
    switch (callback) {
        case LLAMA_SCHED_CB_NONE: return "NONE";
        case LLAMA_SCHED_CB_PRE_EXECUTE: return "PRE_EXECUTE";
        case LLAMA_SCHED_CB_POST_EXECUTE: return "POST_EXECUTE";
        case LLAMA_SCHED_CB_NODE_SELECT: return "NODE_SELECT";
        case LLAMA_SCHED_CB_DEPENDENCY_READY: return "DEPENDENCY_READY";
        case LLAMA_SCHED_CB_PRIORITY_UPDATE: return "PRIORITY_UPDATE";
        case LLAMA_SCHED_CB_LOAD_BALANCE: return "LOAD_BALANCE";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
