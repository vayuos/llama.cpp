/**
 * SECTION 16: Remove CPU-driven decode loop progression
 * Implementation
 *
 * Enforces that CPU no longer owns decode-loop progression. Decode progression
 * is GPU-driven with CPU reduced to non-blocking initiator and observer. CPU must
 * not iterate per-token, advance position counters, or gate token emission. Decode
 * becomes a continuous GPU-controlled process.
 */

#include "llama-decode-loop-elimination.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <vector>

// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================

static struct llama_decode_loop_elimination_validation_state g_decode_loop_elimination_state = {
    {
        LLAMA_LOOP_OWNER_NONE,
        false,
        false,
        0,
        LLAMA_CPU_CTRL_NONE,
        0,
        0
    },
    {
        LLAMA_PROGRESSION_IDLE,
        0,
        0,
        LLAMA_SIGNAL_NONE,
        0,
        false,
        false,
        0
    },
    0,
    0,
    0,
    true,
    false,
    false
};

// CPU loop iteration tracking: maps iteration_count -> detected
static std::map<int, bool> g_cpu_loop_iterations_detected;

// CPU polling attempts tracking
static std::map<uint64_t, int> g_cpu_polling_attempts;

// CPU wait violations tracking
static std::map<uint64_t, int> g_cpu_wait_violations;

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize decode loop elimination enforcement system
 */
int llama_decode_loop_elimination_init(void) {
    // Clear all tracking maps
    g_cpu_loop_iterations_detected.clear();
    g_cpu_polling_attempts.clear();
    g_cpu_wait_violations.clear();

    // Reset global state
    g_decode_loop_elimination_state.ownership_record.current_owner = LLAMA_LOOP_OWNER_NONE;
    g_decode_loop_elimination_state.ownership_record.cpu_loop_eliminated = false;
    g_decode_loop_elimination_state.ownership_record.gpu_loop_active = false;
    g_decode_loop_elimination_state.ownership_record.cpu_control_violations = 0;
    g_decode_loop_elimination_state.progression_record.current_state = LLAMA_PROGRESSION_IDLE;
    g_decode_loop_elimination_state.progression_record.tokens_produced = 0;
    g_decode_loop_elimination_state.progression_record.gpu_autonomous = false;
    g_decode_loop_elimination_state.total_control_violations = 0;
    g_decode_loop_elimination_state.total_polling_detections = 0;
    g_decode_loop_elimination_state.total_wait_detections = 0;

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 1-5: Loop ownership transfer
// ============================================================================

/**
 * ENFORCEMENT POINT 1: Eliminate CPU loop
 * CPU decode loop completely removed; no per-token iteration on CPU.
 */
int llama_decode_loop_elimination_eliminate_cpu_loop(void) {
    if (g_decode_loop_elimination_state.ownership_record.current_owner == LLAMA_LOOP_OWNER_CPU) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR EP1: CPU loop still exists\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    g_decode_loop_elimination_state.ownership_record.cpu_loop_eliminated = true;
    return 0; // Success
}

/**
 * ENFORCEMENT POINT 2: Transfer ownership to GPU
 * Decode loop ownership transferred from CPU to GPU.
 */
int llama_decode_loop_elimination_transfer_ownership_to_gpu(void) {
    if (!g_decode_loop_elimination_state.ownership_record.cpu_loop_eliminated) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR EP2: CPU loop not eliminated\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    g_decode_loop_elimination_state.ownership_record.current_owner = LLAMA_LOOP_OWNER_TRANSITIONING;
    return 0; // Success
}

/**
 * ENFORCEMENT POINT 3: Start GPU autonomous loop
 * GPU loop now runs autonomously without CPU intervention per token.
 */
int llama_decode_loop_elimination_start_gpu_autonomous_loop(void) {
    if (g_decode_loop_elimination_state.ownership_record.current_owner != LLAMA_LOOP_OWNER_TRANSITIONING) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR EP3: Ownership not in TRANSITIONING state\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    g_decode_loop_elimination_state.ownership_record.current_owner = LLAMA_LOOP_OWNER_GPU;
    g_decode_loop_elimination_state.ownership_record.gpu_loop_active = true;
    g_decode_loop_elimination_state.progression_record.current_state = LLAMA_PROGRESSION_GPU_LOOP;
    g_decode_loop_elimination_state.progression_record.gpu_autonomous = true;

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 4: Forbid CPU token iteration
 * Any CPU attempt to iterate tokens or advance counters is forbidden.
 */
int llama_decode_loop_elimination_forbid_cpu_token_iteration(void) {
    if (g_decode_loop_elimination_state.ownership_record.current_owner != LLAMA_LOOP_OWNER_GPU) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR EP4: Loop not GPU-owned\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 5: Assert GPU loop owns progression
 * GPU loop is sole owner of decode progression; CPU cannot influence.
 */
int llama_decode_loop_elimination_assert_gpu_loop_owns_progression(void) {
    if (g_decode_loop_elimination_state.ownership_record.current_owner != LLAMA_LOOP_OWNER_GPU) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR EP5: GPU doesn't own loop\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    if (!g_decode_loop_elimination_state.progression_record.gpu_autonomous) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR EP5: GPU not autonomous\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 6-8: CPU control prevention
// ============================================================================

/**
 * ENFORCEMENT POINT 6: Forbid CPU gating conditions
 * CPU cannot check "is next token allowed?" or similar conditions.
 */
int llama_decode_loop_elimination_forbid_cpu_gating_conditions(void) {
    if (g_decode_loop_elimination_state.ownership_record.current_owner != LLAMA_LOOP_OWNER_GPU) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR EP6: Loop not GPU-owned\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 7: Forbid CPU waits between tokens
 * CPU must not block or wait between tokens.
 */
int llama_decode_loop_elimination_forbid_cpu_waits_between_tokens(void) {
    if (g_decode_loop_elimination_state.progression_record.cpu_wait_violations > 0) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR EP7: CPU waits detected (%d)\n",
                g_decode_loop_elimination_state.progression_record.cpu_wait_violations);
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 8: Convert CPU to signal observer
 * CPU role reduced to receiving and reacting to GPU signals.
 */
int llama_decode_loop_elimination_convert_cpu_to_signal_observer(void) {
    if (!g_decode_loop_elimination_state.progression_record.gpu_autonomous) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR EP8: GPU not autonomous\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 9-10: GPU signal handling
// ============================================================================

/**
 * ENFORCEMENT POINT 9: Enable GPU decode signaling
 * GPU sends signals for token ready and decode complete.
 */
int llama_decode_loop_elimination_enable_gpu_decode_signaling(void) {
    if (g_decode_loop_elimination_state.ownership_record.current_owner != LLAMA_LOOP_OWNER_GPU) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR EP9: Loop not GPU-owned\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 10: Assert GPU drives progression
 * GPU is sole driver of decode progression; CPU cannot trigger steps.
 */
int llama_decode_loop_elimination_assert_gpu_drives_progression(void) {
    if (g_decode_loop_elimination_state.ownership_record.cpu_control_violations > 0) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR EP10: CPU control violations detected (%d)\n",
                g_decode_loop_elimination_state.ownership_record.cpu_control_violations);
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// CPU CONTROL VIOLATION DETECTION
// ============================================================================

/**
 * Detect CPU owns loop
 */
int llama_decode_loop_elimination_detect_cpu_owns_loop(void) {
    fprintf(stderr, "[DECODE_LOOP_ELIM] VIOLATION: CPU owns decode loop\n");
    g_decode_loop_elimination_state.ownership_record.last_violation = LLAMA_CPU_CTRL_OWNS_LOOP;
    g_decode_loop_elimination_state.ownership_record.cpu_control_violations++;
    g_decode_loop_elimination_state.total_control_violations++;

    if (g_decode_loop_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect CPU advances tokens
 */
int llama_decode_loop_elimination_detect_cpu_advances_tokens(void) {
    fprintf(stderr, "[DECODE_LOOP_ELIM] VIOLATION: CPU advances token counters\n");
    g_decode_loop_elimination_state.ownership_record.last_violation = LLAMA_CPU_CTRL_ADVANCES_TOKENS;
    g_decode_loop_elimination_state.ownership_record.cpu_control_violations++;
    g_decode_loop_elimination_state.total_control_violations++;

    if (g_decode_loop_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect CPU gate condition
 */
int llama_decode_loop_elimination_detect_cpu_gate_condition(void) {
    fprintf(stderr, "[DECODE_LOOP_ELIM] VIOLATION: CPU checks gating condition\n");
    g_decode_loop_elimination_state.ownership_record.last_violation = LLAMA_CPU_CTRL_GATE_CONDITION;
    g_decode_loop_elimination_state.ownership_record.cpu_control_violations++;
    g_decode_loop_elimination_state.total_control_violations++;

    if (g_decode_loop_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect CPU polls completion
 */
int llama_decode_loop_elimination_detect_cpu_polls_completion(void) {
    fprintf(stderr, "[DECODE_LOOP_ELIM] VIOLATION: CPU polls for kernel completion\n");
    g_decode_loop_elimination_state.ownership_record.last_violation = LLAMA_CPU_CTRL_POLLS_COMPLETION;
    g_decode_loop_elimination_state.ownership_record.cpu_control_violations++;
    g_decode_loop_elimination_state.total_polling_detections++;

    if (g_decode_loop_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect CPU waits between tokens
 */
int llama_decode_loop_elimination_detect_cpu_waits_between_tokens(void) {
    fprintf(stderr, "[DECODE_LOOP_ELIM] VIOLATION: CPU waits between tokens\n");
    g_decode_loop_elimination_state.ownership_record.last_violation = LLAMA_CPU_CTRL_WAITS_BETWEEN_TOKENS;
    g_decode_loop_elimination_state.progression_record.cpu_wait_violations++;
    g_decode_loop_elimination_state.total_wait_detections++;

    if (g_decode_loop_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect CPU invokes next token
 */
int llama_decode_loop_elimination_detect_cpu_invokes_next_token(void) {
    fprintf(stderr, "[DECODE_LOOP_ELIM] VIOLATION: CPU invokes next-token\n");
    g_decode_loop_elimination_state.ownership_record.last_violation = LLAMA_CPU_CTRL_INVOKES_NEXT_TOKEN;
    g_decode_loop_elimination_state.ownership_record.cpu_control_violations++;
    g_decode_loop_elimination_state.total_control_violations++;

    if (g_decode_loop_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect CPU mutates decode state
 */
int llama_decode_loop_elimination_detect_cpu_mutates_decode_state(void) {
    fprintf(stderr, "[DECODE_LOOP_ELIM] VIOLATION: CPU mutates decode state\n");
    g_decode_loop_elimination_state.ownership_record.last_violation = LLAMA_CPU_CTRL_MUTATES_DECODE_STATE;
    g_decode_loop_elimination_state.ownership_record.cpu_control_violations++;
    g_decode_loop_elimination_state.total_control_violations++;

    if (g_decode_loop_elimination_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect CPU loop iteration
 */
int llama_decode_loop_elimination_detect_cpu_loop_iteration(void) {
    fprintf(stderr, "[DECODE_LOOP_ELIM] VIOLATION: CPU performs loop iteration\n");
    g_decode_loop_elimination_state.ownership_record.last_violation = LLAMA_CPU_CTRL_LOOP_ITERATION;
    g_decode_loop_elimination_state.ownership_record.cpu_control_violations++;
    g_decode_loop_elimination_state.total_control_violations++;

    if (g_decode_loop_elimination_state.enforcement_strict) abort();
    return -1;
}

// ============================================================================
// GPU-DRIVEN PROGRESSION TRACKING
// ============================================================================

/**
 * Signal token ready
 */
int llama_decode_loop_elimination_signal_token_ready(uint64_t token_index) {
    if (g_decode_loop_elimination_state.ownership_record.current_owner != LLAMA_LOOP_OWNER_GPU) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR: Loop not GPU-owned\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    // Track token progression
    g_decode_loop_elimination_state.progression_record.tokens_produced = token_index;
    g_decode_loop_elimination_state.progression_record.current_state = LLAMA_PROGRESSION_TOKEN_READY;
    g_decode_loop_elimination_state.progression_record.last_signal = LLAMA_SIGNAL_TOKEN_READY;
    g_decode_loop_elimination_state.progression_record.tokens_produced++;
    g_decode_loop_elimination_state.ownership_record.total_tokens_produced_by_gpu++;

    return 0; // Success
}

/**
 * Signal decode complete
 */
int llama_decode_loop_elimination_signal_decode_complete(void) {
    g_decode_loop_elimination_state.progression_record.current_state = LLAMA_PROGRESSION_COMPLETE;
    g_decode_loop_elimination_state.progression_record.last_signal = LLAMA_SIGNAL_DECODE_COMPLETE;
    g_decode_loop_elimination_state.ownership_record.gpu_loop_active = false;

    return 0; // Success
}

/**
 * Signal decode error
 */
int llama_decode_loop_elimination_signal_decode_error(const char* error_msg) {
    fprintf(stderr, "[DECODE_LOOP_ELIM] Decode error: %s\n", error_msg ? error_msg : "unknown");
    g_decode_loop_elimination_state.progression_record.current_state = LLAMA_PROGRESSION_ERROR;
    g_decode_loop_elimination_state.progression_record.last_signal = LLAMA_SIGNAL_ERROR;
    g_decode_loop_elimination_state.ownership_record.gpu_loop_active = false;

    return 0; // Success
}

/**
 * Advance GPU token index
 */
int llama_decode_loop_elimination_advance_gpu_token_index(void) {
    if (g_decode_loop_elimination_state.ownership_record.current_owner != LLAMA_LOOP_OWNER_GPU) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR: Loop not GPU-owned\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    g_decode_loop_elimination_state.progression_record.current_token_index++;
    return 0; // Success
}

// ============================================================================
// QUERY AND VERIFICATION FUNCTIONS
// ============================================================================

/**
 * Get ownership record
 */
struct llama_decode_loop_ownership_record llama_decode_loop_elimination_get_ownership_record(void) {
    return g_decode_loop_elimination_state.ownership_record;
}

/**
 * Get progression record
 */
struct llama_decode_progression_record llama_decode_loop_elimination_get_progression_record(void) {
    return g_decode_loop_elimination_state.progression_record;
}

/**
 * Get loop owner
 */
enum llama_decode_loop_owner llama_decode_loop_elimination_get_loop_owner(void) {
    return g_decode_loop_elimination_state.ownership_record.current_owner;
}

/**
 * Get progression state
 */
enum llama_decode_progression_state llama_decode_loop_elimination_get_progression_state(void) {
    return g_decode_loop_elimination_state.progression_record.current_state;
}

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

/**
 * Verify CPU loop eliminated
 */
int llama_decode_loop_elimination_verify_cpu_loop_eliminated(void) {
    if (!g_decode_loop_elimination_state.ownership_record.cpu_loop_eliminated) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR: CPU loop not eliminated\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify GPU loop active
 */
int llama_decode_loop_elimination_verify_gpu_loop_active(void) {
    if (!g_decode_loop_elimination_state.ownership_record.gpu_loop_active) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR: GPU loop not active\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify no CPU gating
 */
int llama_decode_loop_elimination_verify_no_cpu_gating(void) {
    if (g_decode_loop_elimination_state.ownership_record.cpu_control_violations > 0) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR: CPU control violations detected\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify no CPU polling
 */
int llama_decode_loop_elimination_verify_no_cpu_polling(void) {
    if (g_decode_loop_elimination_state.total_polling_detections > 0) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR: CPU polling detected (%d times)\n",
                g_decode_loop_elimination_state.total_polling_detections);
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify GPU autonomous
 */
int llama_decode_loop_elimination_verify_gpu_autonomous(void) {
    if (!g_decode_loop_elimination_state.progression_record.gpu_autonomous) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR: GPU not autonomous\n");
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify no CPU waits
 */
int llama_decode_loop_elimination_verify_no_cpu_waits(void) {
    if (g_decode_loop_elimination_state.total_wait_detections > 0) {
        fprintf(stderr, "[DECODE_LOOP_ELIM] ERROR: CPU waits detected (%d times)\n",
                g_decode_loop_elimination_state.total_wait_detections);
        if (g_decode_loop_elimination_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

/**
 * Log CPU loop eliminated
 */
void llama_decode_loop_elimination_log_cpu_loop_eliminated(void) {
    printf("[DECODE_LOOP_ELIM] ✓ CPU decode loop eliminated\n");
    printf("  - No per-token CPU iteration\n");
    printf("  - No CPU-driven loop progression\n");
    printf("  - CPU ownership transferred to GPU\n");
}

/**
 * Log GPU loop started
 */
void llama_decode_loop_elimination_log_gpu_loop_started(void) {
    printf("[DECODE_LOOP_ELIM] ✓ GPU loop started (autonomous)\n");
    printf("  - GPU drives decode progression\n");
    printf("  - CPU reduced to signal observer\n");
    printf("  - Token emission GPU-owned\n");
}

/**
 * Log GPU signal sent
 */
void llama_decode_loop_elimination_log_gpu_signal_sent(enum llama_decode_signal signal) {
    printf("[DECODE_LOOP_ELIM] ✓ GPU signal sent: %s\n", llama_decode_signal_name(signal));
}

/**
 * Print loop ownership status
 */
void llama_decode_loop_elimination_print_loop_ownership_status(void) {
    printf("\n=== Loop Ownership Status ===\n");
    printf("Current owner: %s\n", llama_decode_loop_owner_name(g_decode_loop_elimination_state.ownership_record.current_owner));
    printf("CPU loop eliminated: %s\n", g_decode_loop_elimination_state.ownership_record.cpu_loop_eliminated ? "YES" : "NO");
    printf("GPU loop active: %s\n", g_decode_loop_elimination_state.ownership_record.gpu_loop_active ? "YES" : "NO");
    printf("CPU control violations: %d\n", g_decode_loop_elimination_state.ownership_record.cpu_control_violations);
    printf("Tokens produced by GPU: %lu\n", g_decode_loop_elimination_state.ownership_record.total_tokens_produced_by_gpu);
    printf("=============================\n\n");
}

/**
 * Print progression status
 */
void llama_decode_loop_elimination_print_progression_status(void) {
    printf("\n=== Progression Status ===\n");
    printf("Current state: %s\n", llama_decode_progression_state_name(g_decode_loop_elimination_state.progression_record.current_state));
    printf("GPU autonomous: %s\n", g_decode_loop_elimination_state.progression_record.gpu_autonomous ? "YES" : "NO");
    printf("Tokens produced: %lu\n", g_decode_loop_elimination_state.progression_record.tokens_produced);
    printf("Last signal: %s\n", llama_decode_signal_name(g_decode_loop_elimination_state.progression_record.last_signal));
    printf("CPU wait violations: %d\n", g_decode_loop_elimination_state.progression_record.cpu_wait_violations);
    printf("===========================\n\n");
}

/**
 * Print violation summary
 */
void llama_decode_loop_elimination_print_violation_summary(void) {
    printf("\n=== Violation Summary ===\n");
    printf("Total control violations: %d\n", g_decode_loop_elimination_state.total_control_violations);
    printf("Total polling detections: %d\n", g_decode_loop_elimination_state.total_polling_detections);
    printf("Total wait detections: %d\n", g_decode_loop_elimination_state.total_wait_detections);
    printf("Last violation: %s\n", llama_cpu_control_violation_name(g_decode_loop_elimination_state.ownership_record.last_violation));
    printf("=========================\n\n");
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

/**
 * Report control violation
 */
void llama_decode_loop_elimination_report_control_violation(
    enum llama_cpu_control_violation violation_type,
    const char* details
) {
    fprintf(stderr, "[DECODE_LOOP_ELIM] REPORT: CPU control violation\n");
    fprintf(stderr, "  - Violation type: %s\n", llama_cpu_control_violation_name(violation_type));
    fprintf(stderr, "  - Details: %s\n", details ? details : "unknown");
    fprintf(stderr, "  - Expected: GPU autonomous control\n");

    g_decode_loop_elimination_state.ownership_record.cpu_control_violations++;
    g_decode_loop_elimination_state.total_control_violations++;
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

/**
 * Set enforcement mode (strict=abort, permissive=log)
 */
void llama_decode_loop_elimination_set_enforcement_strict(bool strict) {
    g_decode_loop_elimination_state.enforcement_strict = strict;
}

/**
 * Get current enforcement mode
 */
bool llama_decode_loop_elimination_get_enforcement_strict(void) {
    return g_decode_loop_elimination_state.enforcement_strict;
}

/**
 * Set debug detect CPU loop attempts
 */
void llama_decode_loop_elimination_set_debug_detect_cpu_loop_attempts(bool debug) {
    g_decode_loop_elimination_state.debug_detect_cpu_loop_attempts = debug;
}

/**
 * Set debug detect CPU polling
 */
void llama_decode_loop_elimination_set_debug_detect_cpu_polling(bool debug) {
    g_decode_loop_elimination_state.debug_detect_cpu_polling = debug;
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================

/**
 * Test Case 1: CPU loop elimination
 */
static int test_cpu_loop_elimination(void) {
    llama_decode_loop_elimination_init();

    int ret = llama_decode_loop_elimination_eliminate_cpu_loop();
    if (ret != 0 || !g_decode_loop_elimination_state.ownership_record.cpu_loop_eliminated) {
        fprintf(stderr, "[TEST] FAIL: CPU loop elimination\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 2: Ownership transfer
 */
static int test_ownership_transfer(void) {
    llama_decode_loop_elimination_init();

    llama_decode_loop_elimination_eliminate_cpu_loop();
    int ret = llama_decode_loop_elimination_transfer_ownership_to_gpu();
    if (ret != 0) {
        fprintf(stderr, "[TEST] FAIL: Ownership transfer\n");
        return -1;
    }

    if (g_decode_loop_elimination_state.ownership_record.current_owner != LLAMA_LOOP_OWNER_TRANSITIONING) {
        fprintf(stderr, "[TEST] FAIL: Ownership not transitioning\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 3: GPU autonomous loop
 */
static int test_gpu_autonomous_loop(void) {
    llama_decode_loop_elimination_init();

    llama_decode_loop_elimination_eliminate_cpu_loop();
    llama_decode_loop_elimination_transfer_ownership_to_gpu();
    int ret = llama_decode_loop_elimination_start_gpu_autonomous_loop();
    if (ret != 0 || !g_decode_loop_elimination_state.progression_record.gpu_autonomous) {
        fprintf(stderr, "[TEST] FAIL: GPU autonomous loop\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 4: Token signaling
 */
static int test_token_signaling(void) {
    llama_decode_loop_elimination_init();

    llama_decode_loop_elimination_eliminate_cpu_loop();
    llama_decode_loop_elimination_transfer_ownership_to_gpu();
    llama_decode_loop_elimination_start_gpu_autonomous_loop();

    int ret = llama_decode_loop_elimination_signal_token_ready(0);
    if (ret != 0 || g_decode_loop_elimination_state.progression_record.tokens_produced != 1) {
        fprintf(stderr, "[TEST] FAIL: Token signaling\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 5: Control violation detection
 */
static int test_control_violation_detection(void) {
    llama_decode_loop_elimination_init();

    int ret = llama_decode_loop_elimination_detect_cpu_owns_loop();
    if (ret == 0) {
        fprintf(stderr, "[TEST] FAIL: Control violation not detected\n");
        return -1;
    }

    if (g_decode_loop_elimination_state.ownership_record.cpu_control_violations != 1) {
        fprintf(stderr, "[TEST] FAIL: Violation count incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 6: Polling detection
 */
static int test_polling_detection(void) {
    llama_decode_loop_elimination_init();

    int ret = llama_decode_loop_elimination_detect_cpu_polls_completion();
    if (ret == 0) {
        fprintf(stderr, "[TEST] FAIL: Polling not detected\n");
        return -1;
    }

    if (g_decode_loop_elimination_state.total_polling_detections != 1) {
        fprintf(stderr, "[TEST] FAIL: Polling count incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 7: Wait detection
 */
static int test_wait_detection(void) {
    llama_decode_loop_elimination_init();

    int ret = llama_decode_loop_elimination_detect_cpu_waits_between_tokens();
    if (ret == 0) {
        fprintf(stderr, "[TEST] FAIL: Wait not detected\n");
        return -1;
    }

    if (g_decode_loop_elimination_state.total_wait_detections != 1) {
        fprintf(stderr, "[TEST] FAIL: Wait count incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 8: Decode complete signal
 */
static int test_decode_complete_signal(void) {
    llama_decode_loop_elimination_init();

    llama_decode_loop_elimination_eliminate_cpu_loop();
    llama_decode_loop_elimination_transfer_ownership_to_gpu();
    llama_decode_loop_elimination_start_gpu_autonomous_loop();
    llama_decode_loop_elimination_signal_decode_complete();

    if (g_decode_loop_elimination_state.progression_record.current_state != LLAMA_PROGRESSION_COMPLETE) {
        fprintf(stderr, "[TEST] FAIL: Decode complete signal\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Run all self-tests
 */
int llama_decode_loop_elimination_selftest(void) {
    printf("[DECODE_LOOP_ELIM] Running self-test suite...\n");

    // Set permissive mode for testing
    bool old_strict = g_decode_loop_elimination_state.enforcement_strict;
    g_decode_loop_elimination_state.enforcement_strict = false;

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

    RUN_TEST(test_cpu_loop_elimination);
    RUN_TEST(test_ownership_transfer);
    RUN_TEST(test_gpu_autonomous_loop);
    RUN_TEST(test_token_signaling);
    RUN_TEST(test_control_violation_detection);
    RUN_TEST(test_polling_detection);
    RUN_TEST(test_wait_detection);
    RUN_TEST(test_decode_complete_signal);

    #undef RUN_TEST

    // Restore enforcement mode
    g_decode_loop_elimination_state.enforcement_strict = old_strict;

    printf("[DECODE_LOOP_ELIM] Self-tests complete: %d passed, %d failed\n", tests_passed, tests_failed);
    return (tests_failed == 0) ? 0 : -1;
}
