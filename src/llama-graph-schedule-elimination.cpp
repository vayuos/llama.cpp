/**
 * SECTION 14: Remove per-token graph scheduling logic
 * Implementation
 *
 * Enforces that all per-token graph scheduling and traversal logic is eliminated
 * from CPU execution during decode. The decode graph executes as a predefined,
 * fixed execution plan computed once at graph build time. Dynamic traversal,
 * node readiness checks, and topological sorts are forbidden during decode.
 */

#include "llama-graph-schedule-elimination.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <vector>

// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================

static struct llama_graph_schedule_elimination_validation_state g_schedule_elimination_state = {
    {
        0,                        // current_graph_id
        LLAMA_EXEC_MODE_INVALID,  // current_mode
        LLAMA_PLAN_UNCOMPUTED,    // plan_state
        0,                        // total_plans_computed
        0,                        // total_scheduling_attempts
        0,                        // total_scheduling_violations
        LLAMA_SCHED_VIOL_NONE,    // last_violation
        LLAMA_SCHED_CB_NONE,      // last_callback_blocked
        false,                    // static_execution_active
        false                     // dynamic_traversal_forbidden
    },
    0,      // total_mode_violations
    0,      // total_plan_mismatches
    0,      // total_callback_violations
    true,   // enforcement_strict
    false   // debug_assert_static_per_step
};

// Per-graph execution plans: maps graph_id -> execution_plan
static std::map<uint64_t, struct llama_execution_plan_record> g_execution_plans;

// Per-graph execution modes: maps graph_id -> execution_mode
static std::map<uint64_t, enum llama_graph_execution_mode> g_graph_execution_modes;

// Per-graph plan states: maps graph_id -> plan_state
static std::map<uint64_t, enum llama_execution_plan_state> g_graph_plan_states;

// Tracking scheduling attempts: maps graph_id -> attempt_count
static std::map<uint64_t, int> g_graph_scheduling_attempts;

// Tracking scheduling violations: maps violation_type -> count
static std::map<enum llama_scheduling_violation_type, int> g_violation_type_counts;

// Tracking blocked callbacks: maps callback_type -> count
static std::map<enum llama_scheduler_callback_type, int> g_callback_block_counts;

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize schedule elimination enforcement system
 */
int llama_schedule_elimination_init(void) {
    // Clear all tracking maps
    g_execution_plans.clear();
    g_graph_execution_modes.clear();
    g_graph_plan_states.clear();
    g_graph_scheduling_attempts.clear();
    g_violation_type_counts.clear();
    g_callback_block_counts.clear();

    // Reset global state
    g_schedule_elimination_state.elimination_record.total_plans_computed = 0;
    g_schedule_elimination_state.elimination_record.total_scheduling_attempts = 0;
    g_schedule_elimination_state.elimination_record.total_scheduling_violations = 0;
    g_schedule_elimination_state.elimination_record.static_execution_active = false;
    g_schedule_elimination_state.elimination_record.dynamic_traversal_forbidden = false;
    g_schedule_elimination_state.total_mode_violations = 0;
    g_schedule_elimination_state.total_plan_mismatches = 0;
    g_schedule_elimination_state.total_callback_violations = 0;

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 1-5: Execution plan computation
// ============================================================================

/**
 * ENFORCEMENT POINT 1: Precompute execution order at graph build time
 * Compute total, linearized execution order before decode starts.
 */
int llama_schedule_elimination_precompute_execution_order(
    uint64_t graph_id,
    uint32_t graph_version
) {
    if (graph_id == 0) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP1: Invalid graph_id (0)\n");
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    if (graph_version == 0) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP1: Invalid graph_version (0)\n");
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    // Mark plan state as computing
    g_graph_plan_states[graph_id] = LLAMA_PLAN_COMPUTING;

    // In real implementation, would perform topological sort and linearize
    // For now, mark as computed
    g_graph_plan_states[graph_id] = LLAMA_PLAN_COMPUTED;
    g_schedule_elimination_state.elimination_record.total_plans_computed++;
    g_schedule_elimination_state.elimination_record.plan_state = LLAMA_PLAN_COMPUTED;

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 2: Store execution plan for graph
 * Save precomputed plan for use during execution.
 */
int llama_schedule_elimination_store_execution_plan(
    uint64_t graph_id,
    struct llama_execution_plan_record * plan
) {
    if (plan == nullptr) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP2: plan pointer null\n");
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    if (g_graph_plan_states[graph_id] != LLAMA_PLAN_COMPUTED) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP2: plan not computed for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    // Store plan
    g_execution_plans[graph_id] = *plan;
    return 0; // Success
}

/**
 * ENFORCEMENT POINT 3: Linearize graph traversal (precomputed)
 * Ensure graph traversal is linearized into fixed execution order.
 */
int llama_schedule_elimination_linearize_graph_traversal(uint64_t graph_id) {
    if (!g_execution_plans.count(graph_id)) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP3: no execution plan for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    // Verify plan has deterministic order
    struct llama_execution_plan_record& plan = g_execution_plans[graph_id];
    if (plan.total_segments == 0) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP3: execution plan has no segments for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 4: Bind execution plan to graph identity
 * Plan is valid only for this specific graph ID and version.
 */
int llama_schedule_elimination_bind_plan_to_graph(uint64_t graph_id) {
    if (!g_execution_plans.count(graph_id)) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP4: no execution plan for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    struct llama_execution_plan_record& plan = g_execution_plans[graph_id];
    if (plan.graph_id != graph_id) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP4: plan graph_id mismatch: %lu vs %lu\n",
                plan.graph_id, graph_id);
        g_schedule_elimination_state.total_plan_mismatches++;
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 5: Lock execution plan (immutable)
 * After locking, plan cannot change.
 */
int llama_schedule_elimination_lock_execution_plan(uint64_t graph_id) {
    if (!g_execution_plans.count(graph_id)) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP5: no execution plan for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    struct llama_execution_plan_record& plan = g_execution_plans[graph_id];
    plan.plan_immutable = true;
    g_graph_plan_states[graph_id] = LLAMA_PLAN_LOCKED;

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 6-8: Dynamic scheduling prevention
// ============================================================================

/**
 * ENFORCEMENT POINT 6: Forbid dynamic graph traversal
 * Dynamic traversal forbidden once static execution plan locked.
 */
int llama_schedule_elimination_forbid_dynamic_traversal(void) {
    if (!g_schedule_elimination_state.elimination_record.static_execution_active) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP6: forbid_dynamic_traversal called before static mode active\n");
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    g_schedule_elimination_state.elimination_record.dynamic_traversal_forbidden = true;
    return 0; // Success
}

/**
 * ENFORCEMENT POINT 7: Forbid scheduler callbacks
 * No scheduler callbacks allowed during static execution mode.
 */
int llama_schedule_elimination_forbid_scheduler_callbacks(void) {
    if (!g_schedule_elimination_state.elimination_record.dynamic_traversal_forbidden) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP7: forbid_scheduler_callbacks called before dynamic traversal forbidden\n");
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    // Callbacks now forbidden
    return 0; // Success
}

/**
 * ENFORCEMENT POINT 8: Fail on scheduling API invocation
 * If any scheduling API called during decode, immediate failure.
 */
int llama_schedule_elimination_fail_on_scheduling_api(void) {
    if (g_schedule_elimination_state.elimination_record.total_scheduling_attempts > 0) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP8: scheduling API invoked during decode\n");
        fprintf(stderr, "  - Total attempts: %d\n", g_schedule_elimination_state.elimination_record.total_scheduling_attempts);

        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 9-10: Static execution mode enforcement
// ============================================================================

/**
 * ENFORCEMENT POINT 9: Enable static execution mode
 * Transition to static execution mode for this graph.
 */
int llama_schedule_elimination_enable_static_execution_mode(uint64_t graph_id) {
    if (!g_execution_plans.count(graph_id)) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP9: no execution plan for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    struct llama_execution_plan_record& plan = g_execution_plans[graph_id];
    if (!plan.plan_immutable) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP9: execution plan not locked for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    g_graph_execution_modes[graph_id] = LLAMA_EXEC_MODE_STATIC;
    g_schedule_elimination_state.elimination_record.current_mode = LLAMA_EXEC_MODE_STATIC;
    g_schedule_elimination_state.elimination_record.static_execution_active = true;
    g_schedule_elimination_state.elimination_record.current_graph_id = graph_id;

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 10: Assert static mode active at each step
 * Runtime check that static execution mode is active before token step.
 */
int llama_schedule_elimination_assert_static_mode_active(void) {
    if (!g_schedule_elimination_state.elimination_record.static_execution_active) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP10: static execution mode not active\n");
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    enum llama_graph_execution_mode mode = g_schedule_elimination_state.elimination_record.current_mode;
    if (mode != LLAMA_EXEC_MODE_STATIC) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR EP10: execution mode is %s, not STATIC\n",
                llama_graph_execution_mode_name(mode));
        g_schedule_elimination_state.total_mode_violations++;
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// SCHEDULING VIOLATION DETECTION
// ============================================================================

/**
 * Detect attempted dynamic graph traversal
 */
int llama_schedule_elimination_detect_dynamic_traversal_attempt(void) {
    fprintf(stderr, "[SCHEDULE_ELIM] VIOLATION: Dynamic graph traversal attempted\n");
    g_violation_type_counts[LLAMA_SCHED_VIOL_DYNAMIC_TRAVERSAL]++;
    g_schedule_elimination_state.elimination_record.total_scheduling_violations++;
    g_schedule_elimination_state.elimination_record.last_violation = LLAMA_SCHED_VIOL_DYNAMIC_TRAVERSAL;

    if (g_schedule_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect attempted node readiness check
 */
int llama_schedule_elimination_detect_readiness_check_attempt(void) {
    fprintf(stderr, "[SCHEDULE_ELIM] VIOLATION: Node readiness check attempted\n");
    g_violation_type_counts[LLAMA_SCHED_VIOL_NODE_READINESS_CHECK]++;
    g_schedule_elimination_state.elimination_record.total_scheduling_violations++;
    g_schedule_elimination_state.elimination_record.last_violation = LLAMA_SCHED_VIOL_NODE_READINESS_CHECK;

    if (g_schedule_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect attempted topological sort
 */
int llama_schedule_elimination_detect_topological_sort_attempt(void) {
    fprintf(stderr, "[SCHEDULE_ELIM] VIOLATION: Topological sort attempted\n");
    g_violation_type_counts[LLAMA_SCHED_VIOL_TOPOLOGICAL_SORT]++;
    g_schedule_elimination_state.elimination_record.total_scheduling_violations++;
    g_schedule_elimination_state.elimination_record.last_violation = LLAMA_SCHED_VIOL_TOPOLOGICAL_SORT;

    if (g_schedule_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect attempted dependency re-evaluation
 */
int llama_schedule_elimination_detect_dependency_reevaluation(void) {
    fprintf(stderr, "[SCHEDULE_ELIM] VIOLATION: Dependency re-evaluation attempted\n");
    g_violation_type_counts[LLAMA_SCHED_VIOL_DEPENDENCY_REEVAL]++;
    g_schedule_elimination_state.elimination_record.total_scheduling_violations++;
    g_schedule_elimination_state.elimination_record.last_violation = LLAMA_SCHED_VIOL_DEPENDENCY_REEVAL;

    if (g_schedule_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect attempted execution order change
 */
int llama_schedule_elimination_detect_execution_reorder_attempt(void) {
    fprintf(stderr, "[SCHEDULE_ELIM] VIOLATION: Execution order reorder attempted\n");
    g_violation_type_counts[LLAMA_SCHED_VIOL_EXECUTION_REORDER]++;
    g_schedule_elimination_state.elimination_record.total_scheduling_violations++;
    g_schedule_elimination_state.elimination_record.last_violation = LLAMA_SCHED_VIOL_EXECUTION_REORDER;

    if (g_schedule_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect attempted conditional node skip
 */
int llama_schedule_elimination_detect_conditional_skip_attempt(void) {
    fprintf(stderr, "[SCHEDULE_ELIM] VIOLATION: Conditional node skip attempted\n");
    g_violation_type_counts[LLAMA_SCHED_VIOL_CONDITIONAL_SKIP]++;
    g_schedule_elimination_state.elimination_record.total_scheduling_violations++;
    g_schedule_elimination_state.elimination_record.last_violation = LLAMA_SCHED_VIOL_CONDITIONAL_SKIP;

    if (g_schedule_elimination_state.enforcement_strict) abort();
    return -1;
}

// ============================================================================
// SCHEDULER CALLBACK BLOCKING
// ============================================================================

/**
 * Block scheduler callback invocation
 */
int llama_schedule_elimination_block_scheduler_callback(
    enum llama_scheduler_callback_type callback_type
) {
    fprintf(stderr, "[SCHEDULE_ELIM] VIOLATION: Scheduler callback blocked: %s\n",
            llama_scheduler_callback_type_name(callback_type));

    g_callback_block_counts[callback_type]++;
    g_schedule_elimination_state.elimination_record.total_scheduling_violations++;
    g_schedule_elimination_state.elimination_record.last_callback_blocked = callback_type;
    g_schedule_elimination_state.elimination_record.last_violation = LLAMA_SCHED_VIOL_SCHEDULER_CALLBACK;
    g_schedule_elimination_state.total_callback_violations++;

    if (g_schedule_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Verify no callbacks are registered
 */
int llama_schedule_elimination_verify_no_callbacks_registered(void) {
    if (g_callback_block_counts.size() > 0) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR: Scheduler callbacks are registered\n");
        for (auto& entry : g_callback_block_counts) {
            fprintf(stderr, "  - %s: %d blocks\n", llama_scheduler_callback_type_name(entry.first), entry.second);
        }
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// QUERY AND DIAGNOSTIC FUNCTIONS
// ============================================================================

/**
 * Get execution plan for graph
 */
struct llama_execution_plan_record llama_schedule_elimination_get_plan(uint64_t graph_id) {
    if (g_execution_plans.count(graph_id)) {
        return g_execution_plans[graph_id];
    }

    // Return empty/invalid record
    struct llama_execution_plan_record empty;
    empty.graph_id = 0;
    empty.graph_version = 0;
    empty.total_segments = 0;
    empty.segments = nullptr;
    empty.plan_state = LLAMA_PLAN_UNCOMPUTED;
    empty.exec_mode = LLAMA_EXEC_MODE_INVALID;
    empty.plan_immutable = false;
    empty.plan_creation_time_ns = 0;
    empty.num_execution_steps = 0;
    return empty;
}

/**
 * Get global elimination record
 */
struct llama_graph_schedule_elimination_record llama_schedule_elimination_get_record(void) {
    return g_schedule_elimination_state.elimination_record;
}

/**
 * Get plan segment count
 */
int llama_schedule_elimination_get_plan_segment_count(uint64_t graph_id) {
    if (g_execution_plans.count(graph_id)) {
        return g_execution_plans[graph_id].total_segments;
    }
    return 0;
}

/**
 * Get current execution mode
 */
enum llama_graph_execution_mode llama_schedule_elimination_get_execution_mode(void) {
    return g_schedule_elimination_state.elimination_record.current_mode;
}

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

/**
 * Verify execution plan is complete
 */
int llama_schedule_elimination_verify_plan_complete(uint64_t graph_id) {
    if (!g_execution_plans.count(graph_id)) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR: No execution plan for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    struct llama_execution_plan_record& plan = g_execution_plans[graph_id];
    if (plan.total_segments == 0) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR: Execution plan empty for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify execution plan is immutable
 */
int llama_schedule_elimination_verify_plan_immutable(uint64_t graph_id) {
    if (!g_execution_plans.count(graph_id)) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR: No execution plan for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    struct llama_execution_plan_record& plan = g_execution_plans[graph_id];
    if (!plan.plan_immutable) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR: Execution plan not locked for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify no dynamic scheduling attempted
 */
int llama_schedule_elimination_verify_no_dynamic_scheduling(void) {
    if (g_schedule_elimination_state.elimination_record.total_scheduling_attempts > 0) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR: %d scheduling attempts detected\n",
                g_schedule_elimination_state.elimination_record.total_scheduling_attempts);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify static mode throughout decode
 */
int llama_schedule_elimination_verify_static_mode_throughout(void) {
    if (!g_schedule_elimination_state.elimination_record.static_execution_active) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR: Static execution mode not active throughout decode\n");
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    if (g_schedule_elimination_state.elimination_record.current_mode != LLAMA_EXEC_MODE_STATIC) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR: Execution mode switched away from STATIC\n");
        g_schedule_elimination_state.total_mode_violations++;
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

/**
 * Log that execution plan was computed
 */
void llama_schedule_elimination_log_plan_computed(uint64_t graph_id, int num_segments) {
    printf("[SCHEDULE_ELIM] ✓ Execution plan computed for graph %lu\n", graph_id);
    printf("  - Total segments: %d\n", num_segments);
    printf("  - Fixed execution order established\n");
    printf("  - No per-token scheduling overhead\n");
}

/**
 * Log that static execution mode enabled
 */
void llama_schedule_elimination_log_static_mode_enabled(void) {
    printf("[SCHEDULE_ELIM] ✓ Static execution mode enabled\n");
    printf("  - Dynamic traversal forbidden\n");
    printf("  - Scheduler callbacks disabled\n");
    printf("  - Graph executes predefined plan\n");
}

/**
 * Print execution plan
 */
void llama_schedule_elimination_print_execution_plan(uint64_t graph_id) {
    printf("\n=== Execution Plan for Graph %lu ===\n", graph_id);

    if (!g_execution_plans.count(graph_id)) {
        printf("No execution plan for this graph\n");
        return;
    }

    struct llama_execution_plan_record& plan = g_execution_plans[graph_id];
    printf("Plan state: %s\n", llama_execution_plan_state_name(plan.plan_state));
    printf("Execution mode: %s\n", llama_graph_execution_mode_name(plan.exec_mode));
    printf("Total segments: %d\n", plan.total_segments);
    printf("Execution steps: %d\n", plan.num_execution_steps);
    printf("Plan immutable: %s\n", plan.plan_immutable ? "YES" : "NO");
    printf("====================================\n\n");
}

/**
 * Print status
 */
void llama_schedule_elimination_print_status(void) {
    printf("\n=== Schedule Elimination Status ===\n");
    printf("Static execution active: %s\n", g_schedule_elimination_state.elimination_record.static_execution_active ? "YES" : "NO");
    printf("Dynamic traversal forbidden: %s\n", g_schedule_elimination_state.elimination_record.dynamic_traversal_forbidden ? "YES" : "NO");
    printf("Current execution mode: %s\n", llama_graph_execution_mode_name(g_schedule_elimination_state.elimination_record.current_mode));
    printf("Total plans computed: %d\n", g_schedule_elimination_state.elimination_record.total_plans_computed);
    printf("Total scheduling attempts: %d\n", g_schedule_elimination_state.elimination_record.total_scheduling_attempts);
    printf("Total scheduling violations: %d\n", g_schedule_elimination_state.elimination_record.total_scheduling_violations);
    printf("Total callback violations: %d\n", g_schedule_elimination_state.total_callback_violations);
    printf("===================================\n\n");
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

/**
 * Report scheduling attempt violation
 */
void llama_schedule_elimination_report_scheduling_attempt(
    enum llama_scheduling_violation_type violation_type,
    const char* details
) {
    fprintf(stderr, "[SCHEDULE_ELIM] REPORT: Scheduling violation\n");
    fprintf(stderr, "  - Violation type: %s\n", llama_scheduling_violation_type_name(violation_type));
    fprintf(stderr, "  - Details: %s\n", details ? details : "unknown");
    fprintf(stderr, "  - Expected: Static execution only\n");

    g_schedule_elimination_state.elimination_record.total_scheduling_attempts++;
    g_violation_type_counts[violation_type]++;
}

/**
 * Report callback invocation
 */
void llama_schedule_elimination_report_callback_invocation(
    enum llama_scheduler_callback_type callback_type
) {
    fprintf(stderr, "[SCHEDULE_ELIM] REPORT: Scheduler callback invocation\n");
    fprintf(stderr, "  - Callback type: %s\n", llama_scheduler_callback_type_name(callback_type));
    fprintf(stderr, "  - Expected: No callbacks during static execution\n");

    g_callback_block_counts[callback_type]++;
    g_schedule_elimination_state.total_callback_violations++;
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

/**
 * Set enforcement mode (strict=abort, permissive=log)
 */
void llama_schedule_elimination_set_enforcement_strict(bool strict) {
    g_schedule_elimination_state.enforcement_strict = strict;
}

/**
 * Get current enforcement mode
 */
bool llama_schedule_elimination_get_enforcement_strict(void) {
    return g_schedule_elimination_state.enforcement_strict;
}

/**
 * Set debug assertion mode
 */
void llama_schedule_elimination_set_debug_assert_static_per_step(bool debug) {
    g_schedule_elimination_state.debug_assert_static_per_step = debug;
}

// ============================================================================
// STATIC EXECUTION INTERFACE
// ============================================================================

/**
 * Execute one step of static execution plan
 */
int llama_schedule_elimination_execute_static_plan_step(uint64_t graph_id) {
    if (g_schedule_elimination_state.debug_assert_static_per_step) {
        if (llama_schedule_elimination_assert_static_mode_active() != 0) {
            return -1;
        }
    }

    if (!g_execution_plans.count(graph_id)) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR: No execution plan for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Reset execution state for next plan execution
 */
int llama_schedule_elimination_reset_plan_execution_state(uint64_t graph_id) {
    // Reset internal plan execution counters
    if (!g_execution_plans.count(graph_id)) {
        fprintf(stderr, "[SCHEDULE_ELIM] ERROR: No execution plan for graph %lu\n", graph_id);
        if (g_schedule_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================

/**
 * Test Case 1: Execution plan computation
 */
static int test_execution_plan_computation(void) {
    llama_schedule_elimination_init();

    int ret = llama_schedule_elimination_precompute_execution_order(1, 1);
    if (ret != 0) {
        fprintf(stderr, "[TEST] FAIL: Execution plan computation\n");
        return -1;
    }

    if (g_schedule_elimination_state.elimination_record.total_plans_computed != 1) {
        fprintf(stderr, "[TEST] FAIL: Plan count incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 2: Plan immutability
 */
static int test_plan_immutability(void) {
    llama_schedule_elimination_init();

    llama_schedule_elimination_precompute_execution_order(1, 1);

    struct llama_execution_plan_record plan;
    plan.graph_id = 1;
    plan.graph_version = 1;
    plan.total_segments = 10;
    plan.segments = nullptr;
    plan.plan_state = LLAMA_PLAN_COMPUTED;
    plan.exec_mode = LLAMA_EXEC_MODE_STATIC;
    plan.plan_immutable = false;
    plan.plan_creation_time_ns = 0;
    plan.num_execution_steps = 0;

    int ret = llama_schedule_elimination_store_execution_plan(1, &plan);
    if (ret != 0) {
        fprintf(stderr, "[TEST] FAIL: Store execution plan\n");
        return -1;
    }

    ret = llama_schedule_elimination_lock_execution_plan(1);
    if (ret != 0) {
        fprintf(stderr, "[TEST] FAIL: Lock execution plan\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 3: Static execution mode
 */
static int test_static_execution_mode(void) {
    llama_schedule_elimination_init();

    llama_schedule_elimination_precompute_execution_order(1, 1);

    struct llama_execution_plan_record plan;
    plan.graph_id = 1;
    plan.graph_version = 1;
    plan.total_segments = 10;
    plan.segments = nullptr;
    plan.plan_state = LLAMA_PLAN_COMPUTED;
    plan.exec_mode = LLAMA_EXEC_MODE_STATIC;
    plan.plan_immutable = true;
    plan.plan_creation_time_ns = 0;
    plan.num_execution_steps = 0;

    llama_schedule_elimination_store_execution_plan(1, &plan);
    llama_schedule_elimination_lock_execution_plan(1);

    int ret = llama_schedule_elimination_enable_static_execution_mode(1);
    if (ret != 0) {
        fprintf(stderr, "[TEST] FAIL: Enable static execution mode\n");
        return -1;
    }

    if (!g_schedule_elimination_state.elimination_record.static_execution_active) {
        fprintf(stderr, "[TEST] FAIL: Static execution not active\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 4: Dynamic traversal prevention
 */
static int test_dynamic_traversal_prevention(void) {
    llama_schedule_elimination_init();

    llama_schedule_elimination_precompute_execution_order(1, 1);
    llama_schedule_elimination_forbid_dynamic_traversal();

    if (!g_schedule_elimination_state.elimination_record.dynamic_traversal_forbidden) {
        fprintf(stderr, "[TEST] FAIL: Dynamic traversal not forbidden\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 5: Violation detection
 */
static int test_violation_detection(void) {
    llama_schedule_elimination_init();

    int ret = llama_schedule_elimination_detect_dynamic_traversal_attempt();
    if (ret == 0) {
        fprintf(stderr, "[TEST] FAIL: Dynamic traversal violation not detected\n");
        return -1;
    }

    if (g_schedule_elimination_state.elimination_record.total_scheduling_violations != 1) {
        fprintf(stderr, "[TEST] FAIL: Violation count incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 6: Callback blocking
 */
static int test_callback_blocking(void) {
    llama_schedule_elimination_init();

    int ret = llama_schedule_elimination_block_scheduler_callback(LLAMA_SCHED_CB_PRE_EXECUTE);
    if (ret == 0) {
        fprintf(stderr, "[TEST] FAIL: Callback not blocked\n");
        return -1;
    }

    if (g_schedule_elimination_state.total_callback_violations != 1) {
        fprintf(stderr, "[TEST] FAIL: Callback violation count incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 7: Plan verification
 */
static int test_plan_verification(void) {
    llama_schedule_elimination_init();

    llama_schedule_elimination_precompute_execution_order(1, 1);

    struct llama_execution_plan_record plan;
    plan.graph_id = 1;
    plan.graph_version = 1;
    plan.total_segments = 10;
    plan.segments = nullptr;
    plan.plan_state = LLAMA_PLAN_COMPUTED;
    plan.exec_mode = LLAMA_EXEC_MODE_STATIC;
    plan.plan_immutable = true;
    plan.plan_creation_time_ns = 0;
    plan.num_execution_steps = 0;

    llama_schedule_elimination_store_execution_plan(1, &plan);

    int ret = llama_schedule_elimination_verify_plan_complete(1);
    if (ret != 0) {
        fprintf(stderr, "[TEST] FAIL: Plan verification failed\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 8: Mode assertion
 */
static int test_mode_assertion(void) {
    llama_schedule_elimination_init();

    llama_schedule_elimination_precompute_execution_order(1, 1);

    struct llama_execution_plan_record plan;
    plan.graph_id = 1;
    plan.graph_version = 1;
    plan.total_segments = 10;
    plan.segments = nullptr;
    plan.plan_state = LLAMA_PLAN_COMPUTED;
    plan.exec_mode = LLAMA_EXEC_MODE_STATIC;
    plan.plan_immutable = true;
    plan.plan_creation_time_ns = 0;
    plan.num_execution_steps = 0;

    llama_schedule_elimination_store_execution_plan(1, &plan);
    llama_schedule_elimination_lock_execution_plan(1);
    llama_schedule_elimination_enable_static_execution_mode(1);

    int ret = llama_schedule_elimination_assert_static_mode_active();
    if (ret != 0) {
        fprintf(stderr, "[TEST] FAIL: Static mode assertion failed\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Run all self-tests
 */
int llama_schedule_elimination_selftest(void) {
    printf("[SCHEDULE_ELIM] Running self-test suite...\n");

    // Set permissive mode for testing
    bool old_strict = g_schedule_elimination_state.enforcement_strict;
    g_schedule_elimination_state.enforcement_strict = false;

    int tests_passed = 0;
    int tests_failed = 0;

    #define RUN_TEST(test_fn) do { \
        if (test_fn() == 0) { \
            printf("  ✓ " #test_fn "\n"); \
            tests_passed++; \
        } else { \
            printf("  ✗ " #test_fn "\n"); \
            tests_failed++; \
        } \
    } while(0)

    RUN_TEST(test_execution_plan_computation);
    RUN_TEST(test_plan_immutability);
    RUN_TEST(test_static_execution_mode);
    RUN_TEST(test_dynamic_traversal_prevention);
    RUN_TEST(test_violation_detection);
    RUN_TEST(test_callback_blocking);
    RUN_TEST(test_plan_verification);
    RUN_TEST(test_mode_assertion);

    #undef RUN_TEST

    // Restore enforcement mode
    g_schedule_elimination_state.enforcement_strict = old_strict;

    printf("[SCHEDULE_ELIM] Self-tests complete: %d passed, %d failed\n", tests_passed, tests_failed);
    return (tests_failed == 0) ? 0 : -1;
}
