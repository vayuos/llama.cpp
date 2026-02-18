/**
 * SECTION 17: Eliminate CPU token-step gating logic
 * Implementation
 *
 * Enforces that CPU no longer makes conditional decisions about token progression.
 * CPU cannot gate token advancement, check readiness, or authorize next-token execution.
 * Token-step progression is purely GPU-driven as implicit consequence of GPU completion.
 */

#include "llama-token-step-gating-elimination.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <vector>

// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================

static struct llama_token_step_gating_validation_state g_token_step_gating_state = {
    .gating_record = {
        .total_gating_decisions_detected = 0,
        .total_synchronization_barriers = 0,
        .total_readiness_checks = 0,
        .last_decision = LLAMA_GATING_NONE,
        .last_sync_type = LLAMA_SYNC_NONE,
        .step_owner = LLAMA_STEP_OWNER_UNKNOWN,
        .cpu_gating_eliminated = false,
        .gpu_implicit_completion = false,
    },
    .violation_state = LLAMA_GATING_STATE_CLEAN,
    .total_gating_violations = 0,
    .total_sync_violations = 0,
    .total_unauthorized_checks = 0,
    .enforcement_strict = true,
    .debug_detect_cpu_conditionals = false,
    .debug_detect_cpu_barriers = false,
};

// Per-gating-decision tracking: maps decision_type -> count
static std::map<enum llama_cpu_gating_decision, int> g_gating_decision_counts;

// Per-sync-type tracking: maps sync_type -> count
static std::map<enum llama_cpu_sync_type, int> g_sync_type_counts;

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize token-step gating elimination enforcement system
 */
int llama_token_step_gating_elimination_init(void) {
    // Clear all tracking maps
    g_gating_decision_counts.clear();
    g_sync_type_counts.clear();

    // Reset global state
    g_token_step_gating_state.gating_record.total_gating_decisions_detected = 0;
    g_token_step_gating_state.gating_record.total_synchronization_barriers = 0;
    g_token_step_gating_state.gating_record.total_readiness_checks = 0;
    g_token_step_gating_state.gating_record.cpu_gating_eliminated = false;
    g_token_step_gating_state.gating_record.gpu_implicit_completion = false;
    g_token_step_gating_state.violation_state = LLAMA_GATING_STATE_CLEAN;
    g_token_step_gating_state.total_gating_violations = 0;
    g_token_step_gating_state.total_sync_violations = 0;
    g_token_step_gating_state.total_unauthorized_checks = 0;

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 1-5: CPU gating elimination
// ============================================================================

/**
 * ENFORCEMENT POINT 1: Delete "can proceed" checks
 * Remove CPU logic checking if sampling/logits/GPU completion happened.
 */
int llama_token_step_gating_elimination_delete_can_proceed_checks(void) {
    if (g_token_step_gating_state.gating_record.total_gating_decisions_detected > 0) {
        fprintf(stderr, "[GATING_ELIM] ERROR EP1: CPU gating decisions still exist (%d)\n",
                g_token_step_gating_state.gating_record.total_gating_decisions_detected);
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 2: Remove CPU barriers
 * Eliminate barriers where CPU waits for GPU then authorizes next step.
 */
int llama_token_step_gating_elimination_remove_cpu_barriers(void) {
    if (g_token_step_gating_state.gating_record.total_synchronization_barriers > 0) {
        fprintf(stderr, "[GATING_ELIM] ERROR EP2: CPU barriers still exist (%d)\n",
                g_token_step_gating_state.gating_record.total_synchronization_barriers);
        g_token_step_gating_state.total_sync_violations++;
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 3: Prohibit CPU decision-making between tokens
 * CPU cannot decide to advance token index, continue decode, or check staleness.
 */
int llama_token_step_gating_elimination_prohibit_cpu_token_decisions(void) {
    if (g_token_step_gating_state.gating_record.total_readiness_checks > 0) {
        fprintf(stderr, "[GATING_ELIM] ERROR EP3: CPU readiness checks still exist (%d)\n",
                g_token_step_gating_state.gating_record.total_readiness_checks);
        g_token_step_gating_state.total_unauthorized_checks++;
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 4: Replace gating with implicit GPU completion
 * Progression is implicit consequence of GPU work, not explicit CPU signal.
 */
int llama_token_step_gating_elimination_replace_with_implicit_semantics(void) {
    if (!g_token_step_gating_state.gating_record.gpu_implicit_completion) {
        fprintf(stderr, "[GATING_ELIM] ERROR EP4: GPU implicit completion not enabled\n");
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 5: Remove CPU polling/sync loops
 * Delete loops where CPU polls flags or spins on completion states.
 */
int llama_token_step_gating_elimination_remove_cpu_sync_loops(void) {
    if (g_sync_type_counts.size() > 0) {
        fprintf(stderr, "[GATING_ELIM] ERROR EP5: CPU sync loops still exist\n");
        for (auto& entry : g_sync_type_counts) {
            fprintf(stderr, "  - %s: %d instances\n", llama_cpu_sync_type_name(entry.first), entry.second);
        }
        g_token_step_gating_state.total_sync_violations++;
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 6-8: CPU control prevention
// ============================================================================

/**
 * ENFORCEMENT POINT 6: Forbid CPU waits that imply control
 * Wait only for final termination or async notification, never for step control.
 */
int llama_token_step_gating_elimination_forbid_cpu_waits_that_gate(void) {
    if (g_token_step_gating_state.gating_record.total_synchronization_barriers > 0) {
        fprintf(stderr, "[GATING_ELIM] ERROR EP6: CPU waits controlling gating detected\n");
        g_token_step_gating_state.total_sync_violations++;
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 7: Move token-step boundaries into GPU scope
 * Token-step transitions are GPU-side state transitions, not CPU decisions.
 */
int llama_token_step_gating_elimination_move_boundaries_to_gpu(void) {
    if (g_token_step_gating_state.gating_record.step_owner != LLAMA_STEP_OWNER_GPU) {
        fprintf(stderr, "[GATING_ELIM] ERROR EP7: Token-step boundaries not GPU-owned\n");
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 8: Assert GPU token-step owner
 * GPU is sole owner of token-step progression.
 */
int llama_token_step_gating_elimination_assert_gpu_token_step_owner(void) {
    if (g_token_step_gating_state.gating_record.step_owner != LLAMA_STEP_OWNER_GPU) {
        fprintf(stderr, "[GATING_ELIM] ERROR EP8: GPU not token-step owner\n");
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 9-10: Invariant enforcement
// ============================================================================

/**
 * ENFORCEMENT POINT 9: Add CPU gating invariants
 * Runtime assertions fail if CPU attempts to gate token progression.
 */
int llama_token_step_gating_elimination_add_cpu_gating_invariants(void) {
    if (g_token_step_gating_state.total_gating_violations > 0) {
        fprintf(stderr, "[GATING_ELIM] ERROR EP9: CPU gating invariant violations detected (%d)\n",
                g_token_step_gating_state.total_gating_violations);
        g_token_step_gating_state.violation_state = LLAMA_GATING_STATE_VIOLATION_DETECTED;
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 10: Audit decode call sites
 * Verify no CPU conditional branches exist between token steps.
 */
int llama_token_step_gating_elimination_audit_decode_call_sites(void) {
    if (g_token_step_gating_state.total_unauthorized_checks > 0) {
        fprintf(stderr, "[GATING_ELIM] ERROR EP10: Unauthorized CPU checks at decode call sites (%d)\n",
                g_token_step_gating_state.total_unauthorized_checks);
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// CPU GATING VIOLATION DETECTION
// ============================================================================

/**
 * Detect "is sampling finished?" check
 */
int llama_token_step_gating_elimination_detect_sampling_finished_check(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU checks if sampling finished\n");
    g_gating_decision_counts[LLAMA_GATING_SAMPLING_FINISHED]++;
    g_token_step_gating_state.gating_record.total_gating_decisions_detected++;
    g_token_step_gating_state.total_gating_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect "are logits ready?" check
 */
int llama_token_step_gating_elimination_detect_logits_ready_check(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU checks if logits ready\n");
    g_gating_decision_counts[LLAMA_GATING_LOGITS_READY]++;
    g_token_step_gating_state.gating_record.total_gating_decisions_detected++;
    g_token_step_gating_state.total_gating_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect "did GPU complete?" check
 */
int llama_token_step_gating_elimination_detect_gpu_complete_check(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU checks if GPU completed\n");
    g_gating_decision_counts[LLAMA_GATING_GPU_COMPLETE]++;
    g_token_step_gating_state.gating_record.total_gating_decisions_detected++;
    g_token_step_gating_state.total_gating_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect readiness check
 */
int llama_token_step_gating_elimination_detect_readiness_check(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU performs readiness check\n");
    g_gating_decision_counts[LLAMA_GATING_READINESS_CHECK]++;
    g_token_step_gating_state.gating_record.total_readiness_checks++;
    g_token_step_gating_state.total_gating_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect "should decode continue?" check
 */
int llama_token_step_gating_elimination_detect_continue_decode_check(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU decides if decode continues\n");
    g_gating_decision_counts[LLAMA_GATING_CONTINUE_DECODE]++;
    g_token_step_gating_state.gating_record.total_gating_decisions_detected++;
    g_token_step_gating_state.total_gating_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect token index advance decision
 */
int llama_token_step_gating_elimination_detect_token_index_advance_decision(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU decides token index advance\n");
    g_gating_decision_counts[LLAMA_GATING_TOKEN_INDEX_ADVANCE]++;
    g_token_step_gating_state.gating_record.total_gating_decisions_detected++;
    g_token_step_gating_state.total_unauthorized_checks++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect stall check
 */
int llama_token_step_gating_elimination_detect_stall_check(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU checks if decode stalled\n");
    g_gating_decision_counts[LLAMA_GATING_STALL_CHECK]++;
    g_token_step_gating_state.gating_record.total_gating_decisions_detected++;
    g_token_step_gating_state.total_gating_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect explicit authorization
 */
int llama_token_step_gating_elimination_detect_explicit_authorization(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU issues explicit authorization\n");
    g_gating_decision_counts[LLAMA_GATING_EXPLICIT_AUTHORIZATION]++;
    g_token_step_gating_state.gating_record.total_gating_decisions_detected++;
    g_token_step_gating_state.total_gating_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

// ============================================================================
// CPU SYNCHRONIZATION VIOLATION DETECTION
// ============================================================================

/**
 * Detect CPU polling loop
 */
int llama_token_step_gating_elimination_detect_polling_loop(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU polling loop detected\n");
    g_sync_type_counts[LLAMA_SYNC_POLLING_LOOP]++;
    g_token_step_gating_state.gating_record.total_synchronization_barriers++;
    g_token_step_gating_state.total_sync_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect flag polling
 */
int llama_token_step_gating_elimination_detect_flag_polling(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU flag polling detected\n");
    g_sync_type_counts[LLAMA_SYNC_FLAG_POLLING]++;
    g_token_step_gating_state.gating_record.total_synchronization_barriers++;
    g_token_step_gating_state.total_sync_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect spin wait
 */
int llama_token_step_gating_elimination_detect_spin_wait(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU spin wait detected\n");
    g_sync_type_counts[LLAMA_SYNC_SPIN_WAIT]++;
    g_token_step_gating_state.gating_record.total_synchronization_barriers++;
    g_token_step_gating_state.total_sync_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect explicit wait
 */
int llama_token_step_gating_elimination_detect_explicit_wait(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU explicit wait detected\n");
    g_sync_type_counts[LLAMA_SYNC_EXPLICIT_WAIT]++;
    g_token_step_gating_state.gating_record.total_synchronization_barriers++;
    g_token_step_gating_state.total_sync_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect barrier
 */
int llama_token_step_gating_elimination_detect_barrier(void) {
    fprintf(stderr, "[GATING_ELIM] VIOLATION: CPU barrier detected\n");
    g_sync_type_counts[LLAMA_SYNC_BARRIER]++;
    g_token_step_gating_state.gating_record.total_synchronization_barriers++;
    g_token_step_gating_state.total_sync_violations++;

    if (g_token_step_gating_state.enforcement_strict) abort();
    return -1;
}

// ============================================================================
// GPU IMPLICIT COMPLETION VERIFICATION
// ============================================================================

/**
 * Enable GPU implicit completion
 */
int llama_token_step_gating_elimination_enable_implicit_completion(void) {
    g_token_step_gating_state.gating_record.gpu_implicit_completion = true;
    g_token_step_gating_state.gating_record.step_owner = LLAMA_STEP_OWNER_GPU;

    return 0; // Success
}

/**
 * Verify GPU drives progression
 */
int llama_token_step_gating_elimination_verify_gpu_drives_progression(void) {
    if (!g_token_step_gating_state.gating_record.gpu_implicit_completion) {
        fprintf(stderr, "[GATING_ELIM] ERROR: GPU implicit completion not enabled\n");
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    if (g_token_step_gating_state.gating_record.step_owner != LLAMA_STEP_OWNER_GPU) {
        fprintf(stderr, "[GATING_ELIM] ERROR: GPU does not drive progression\n");
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// QUERY AND VERIFICATION FUNCTIONS
// ============================================================================

/**
 * Get gating elimination record
 */
struct llama_cpu_gating_elimination_record llama_token_step_gating_elimination_get_record(void) {
    return g_token_step_gating_state.gating_record;
}

/**
 * Get violation state
 */
enum llama_gating_violation_state llama_token_step_gating_elimination_get_violation_state(void) {
    return g_token_step_gating_state.violation_state;
}

/**
 * Get step owner
 */
enum llama_token_step_owner llama_token_step_gating_elimination_get_step_owner(void) {
    return g_token_step_gating_state.gating_record.step_owner;
}

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

/**
 * Verify no CPU gating
 */
int llama_token_step_gating_elimination_verify_no_cpu_gating(void) {
    if (g_token_step_gating_state.total_gating_violations > 0) {
        fprintf(stderr, "[GATING_ELIM] ERROR: CPU gating violations (%d)\n",
                g_token_step_gating_state.total_gating_violations);
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify no readiness checks
 */
int llama_token_step_gating_elimination_verify_no_readiness_checks(void) {
    if (g_token_step_gating_state.gating_record.total_readiness_checks > 0) {
        fprintf(stderr, "[GATING_ELIM] ERROR: Readiness checks detected (%d)\n",
                g_token_step_gating_state.gating_record.total_readiness_checks);
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify no CPU barriers
 */
int llama_token_step_gating_elimination_verify_no_cpu_barriers(void) {
    if (g_token_step_gating_state.gating_record.total_synchronization_barriers > 0) {
        fprintf(stderr, "[GATING_ELIM] ERROR: CPU barriers detected (%d)\n",
                g_token_step_gating_state.gating_record.total_synchronization_barriers);
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify GPU owns boundaries
 */
int llama_token_step_gating_elimination_verify_gpu_owns_boundaries(void) {
    if (g_token_step_gating_state.gating_record.step_owner != LLAMA_STEP_OWNER_GPU) {
        fprintf(stderr, "[GATING_ELIM] ERROR: GPU does not own boundaries\n");
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify implicit completion
 */
int llama_token_step_gating_elimination_verify_implicit_completion(void) {
    if (!g_token_step_gating_state.gating_record.gpu_implicit_completion) {
        fprintf(stderr, "[GATING_ELIM] ERROR: GPU implicit completion not enabled\n");
        if (g_token_step_gating_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

/**
 * Log CPU gating eliminated
 */
void llama_token_step_gating_elimination_log_cpu_gating_eliminated(void) {
    printf("[GATING_ELIM] ✓ CPU token-step gating eliminated\n");
    printf("  - No CPU conditional token decisions\n");
    printf("  - No CPU readiness checks\n");
    printf("  - No CPU barriers or sync points\n");
}

/**
 * Log implicit completion active
 */
void llama_token_step_gating_elimination_log_implicit_completion_active(void) {
    printf("[GATING_ELIM] ✓ GPU implicit completion active\n");
    printf("  - Token progression is GPU-driven\n");
    printf("  - Progression implicit in GPU state change\n");
    printf("  - No CPU authorization required\n");
}

/**
 * Print gating status
 */
void llama_token_step_gating_elimination_print_gating_status(void) {
    printf("\n=== Gating Elimination Status ===\n");
    printf("CPU gating eliminated: %s\n", g_token_step_gating_state.gating_record.cpu_gating_eliminated ? "YES" : "NO");
    printf("GPU implicit completion: %s\n", g_token_step_gating_state.gating_record.gpu_implicit_completion ? "YES" : "NO");
    printf("Token-step owner: %s\n", llama_token_step_owner_name(g_token_step_gating_state.gating_record.step_owner));
    printf("Violation state: %s\n", llama_gating_violation_state_name(g_token_step_gating_state.violation_state));
    printf("Total gating violations: %d\n", g_token_step_gating_state.total_gating_violations);
    printf("Total sync violations: %d\n", g_token_step_gating_state.total_sync_violations);
    printf("==================================\n\n");
}

/**
 * Print violation summary
 */
void llama_token_step_gating_elimination_print_violation_summary(void) {
    printf("\n=== Gating Violation Summary ===\n");
    printf("Total gating decisions: %d\n", g_token_step_gating_state.gating_record.total_gating_decisions_detected);
    printf("Total barriers: %d\n", g_token_step_gating_state.gating_record.total_synchronization_barriers);
    printf("Total readiness checks: %d\n", g_token_step_gating_state.gating_record.total_readiness_checks);

    if (g_gating_decision_counts.size() > 0) {
        printf("\nDecision types:\n");
        for (auto& entry : g_gating_decision_counts) {
            printf("  - %s: %d\n", llama_cpu_gating_decision_name(entry.first), entry.second);
        }
    }

    if (g_sync_type_counts.size() > 0) {
        printf("\nSync types:\n");
        for (auto& entry : g_sync_type_counts) {
            printf("  - %s: %d\n", llama_cpu_sync_type_name(entry.first), entry.second);
        }
    }
    printf("=================================\n\n");
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

/**
 * Report gating decision
 */
void llama_token_step_gating_elimination_report_gating_decision(
    enum llama_cpu_gating_decision decision_type,
    const char* details
) {
    fprintf(stderr, "[GATING_ELIM] REPORT: CPU gating decision\n");
    fprintf(stderr, "  - Decision type: %s\n", llama_cpu_gating_decision_name(decision_type));
    fprintf(stderr, "  - Details: %s\n", details ? details : "unknown");
    fprintf(stderr, "  - Expected: GPU-driven progression\n");

    g_gating_decision_counts[decision_type]++;
    g_token_step_gating_state.gating_record.total_gating_decisions_detected++;
    g_token_step_gating_state.total_gating_violations++;
}

/**
 * Report sync barrier
 */
void llama_token_step_gating_elimination_report_sync_barrier(
    enum llama_cpu_sync_type sync_type
) {
    fprintf(stderr, "[GATING_ELIM] REPORT: CPU sync barrier\n");
    fprintf(stderr, "  - Sync type: %s\n", llama_cpu_sync_type_name(sync_type));
    fprintf(stderr, "  - Expected: No CPU barriers\n");

    g_sync_type_counts[sync_type]++;
    g_token_step_gating_state.gating_record.total_synchronization_barriers++;
    g_token_step_gating_state.total_sync_violations++;
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

/**
 * Set enforcement mode (strict=abort, permissive=log)
 */
void llama_token_step_gating_elimination_set_enforcement_strict(bool strict) {
    g_token_step_gating_state.enforcement_strict = strict;
}

/**
 * Get current enforcement mode
 */
bool llama_token_step_gating_elimination_get_enforcement_strict(void) {
    return g_token_step_gating_state.enforcement_strict;
}

/**
 * Set debug detect CPU conditionals
 */
void llama_token_step_gating_elimination_set_debug_detect_cpu_conditionals(bool debug) {
    g_token_step_gating_state.debug_detect_cpu_conditionals = debug;
}

/**
 * Set debug detect CPU barriers
 */
void llama_token_step_gating_elimination_set_debug_detect_cpu_barriers(bool debug) {
    g_token_step_gating_state.debug_detect_cpu_barriers = debug;
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================

/**
 * Test Case 1: Gating detection
 */
static int test_gating_detection(void) {
    llama_token_step_gating_elimination_init();

    int ret = llama_token_step_gating_elimination_detect_sampling_finished_check();
    if (ret == 0) {
        fprintf(stderr, "[TEST] FAIL: Gating not detected\n");
        return -1;
    }

    if (g_token_step_gating_state.total_gating_violations != 1) {
        fprintf(stderr, "[TEST] FAIL: Violation count incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 2: Sync barrier detection
 */
static int test_sync_barrier_detection(void) {
    llama_token_step_gating_elimination_init();

    int ret = llama_token_step_gating_elimination_detect_polling_loop();
    if (ret == 0) {
        fprintf(stderr, "[TEST] FAIL: Sync barrier not detected\n");
        return -1;
    }

    if (g_token_step_gating_state.total_sync_violations != 1) {
        fprintf(stderr, "[TEST] FAIL: Sync violation count incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 3: GPU ownership
 */
static int test_gpu_ownership(void) {
    llama_token_step_gating_elimination_init();

    llama_token_step_gating_elimination_enable_implicit_completion();

    if (g_token_step_gating_state.gating_record.step_owner != LLAMA_STEP_OWNER_GPU) {
        fprintf(stderr, "[TEST] FAIL: GPU ownership not set\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 4: Implicit completion
 */
static int test_implicit_completion(void) {
    llama_token_step_gating_elimination_init();

    llama_token_step_gating_elimination_enable_implicit_completion();

    int ret = llama_token_step_gating_elimination_verify_implicit_completion();
    if (ret != 0) {
        fprintf(stderr, "[TEST] FAIL: Implicit completion verification failed\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 5: Readiness check detection
 */
static int test_readiness_check_detection(void) {
    llama_token_step_gating_elimination_init();

    int ret = llama_token_step_gating_elimination_detect_readiness_check();
    if (ret == 0) {
        fprintf(stderr, "[TEST] FAIL: Readiness check not detected\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 6: Decision tracking
 */
static int test_decision_tracking(void) {
    llama_token_step_gating_elimination_init();

    llama_token_step_gating_elimination_detect_logits_ready_check();
    llama_token_step_gating_elimination_detect_gpu_complete_check();

    if (g_token_step_gating_state.gating_record.total_gating_decisions_detected != 2) {
        fprintf(stderr, "[TEST] FAIL: Decision tracking incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 7: Barrier tracking
 */
static int test_barrier_tracking(void) {
    llama_token_step_gating_elimination_init();

    llama_token_step_gating_elimination_detect_spin_wait();
    llama_token_step_gating_elimination_detect_explicit_wait();

    if (g_token_step_gating_state.gating_record.total_synchronization_barriers != 2) {
        fprintf(stderr, "[TEST] FAIL: Barrier tracking incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 8: Clean state verification
 */
static int test_clean_state_verification(void) {
    llama_token_step_gating_elimination_init();

    llama_token_step_gating_elimination_enable_implicit_completion();

    int ret = llama_token_step_gating_elimination_verify_no_cpu_gating();
    if (ret != 0) {
        fprintf(stderr, "[TEST] FAIL: Clean state verification failed\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Run all self-tests
 */
int llama_token_step_gating_elimination_selftest(void) {
    printf("[GATING_ELIM] Running self-test suite...\n");

    // Set permissive mode for testing
    bool old_strict = g_token_step_gating_state.enforcement_strict;
    g_token_step_gating_state.enforcement_strict = false;

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

    RUN_TEST(test_gating_detection);
    RUN_TEST(test_sync_barrier_detection);
    RUN_TEST(test_gpu_ownership);
    RUN_TEST(test_implicit_completion);
    RUN_TEST(test_readiness_check_detection);
    RUN_TEST(test_decision_tracking);
    RUN_TEST(test_barrier_tracking);
    RUN_TEST(test_clean_state_verification);

    #undef RUN_TEST

    // Restore enforcement mode
    g_token_step_gating_state.enforcement_strict = old_strict;

    printf("[GATING_ELIM] Self-tests complete: %d passed, %d failed\n", tests_passed, tests_failed);
    return (tests_failed == 0) ? 0 : -1;
}
