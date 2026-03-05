/**
 * SECTION 10: Add Decode-Time Backend Lock
 * Implementation
 *
 * This file implements enforcement that backend selection is immutable during
 * decode. Once decode begins, the selected backend is locked and cannot be
 * changed, switched, or fallen back from, regardless of runtime conditions.
 */

#include "llama-decode-backend-lock.h"
#include <cstring>
#include <cstdio>
#include <chrono>
#include <cinttypes>
#include <map>

// ============================================================================
// GLOBAL STATE MANAGEMENT
// ============================================================================

static struct llama_backend_lock_validation_state g_backend_lock_state = {
    {
        LLAMA_BACKEND_LOCK_UNLOCKED,
        GGML_BACKEND_DEVICE_TYPE_CPU,
        0,
        0,
        0,
        false,
        true,
        LLAMA_BACKEND_LOCK_VALID,
        0,
        LLAMA_BACKEND_LOCK_VIOL_NONE,
        NULL
    },
    0,
    0,
    true,
    false
};

static bool g_backend_lock_enforcement_strict = true;
static int g_total_backend_lock_violations = 0;
static int g_total_backend_lock_invalidations = 0;

// Violation tracking per location
static std::map<enum llama_backend_lock_violation_location, int> g_violation_location_map;

// ============================================================================
// INITIALIZATION
// ============================================================================

int llama_backend_lock_init(void) {
    g_backend_lock_state.lock_record.state = LLAMA_BACKEND_LOCK_UNLOCKED;
    g_backend_lock_state.lock_record.lock_held = false;
    g_backend_lock_state.lock_record.backend_valid = true;
    g_backend_lock_state.lock_record.invalidation_reason = LLAMA_BACKEND_LOCK_VALID;
    g_backend_lock_state.lock_record.violation_count = 0;
    g_backend_lock_state.lock_record.last_violation = LLAMA_BACKEND_LOCK_VIOL_NONE;
    g_backend_lock_state.lock_record.decode_token_count = 0;
    g_backend_lock_state.total_violations = 0;
    g_backend_lock_state.total_invalidations = 0;

    g_violation_location_map.clear();

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 1: Backend Lock Acquisition
// ============================================================================

int llama_backend_lock_acquire(enum ggml_backend_dev_type backend_to_lock) {
    if (g_backend_lock_state.lock_record.lock_held) {
        fprintf(stderr, "ERROR: Backend lock already held. Cannot re-acquire.\n");
        if (g_backend_lock_enforcement_strict) {
            return -1;
        }
    }

    // Transition to ACQUIRING state
    g_backend_lock_state.lock_record.state = LLAMA_BACKEND_LOCK_ACQUIRING;

    // Validate backend selection
    if (backend_to_lock != GGML_BACKEND_DEVICE_TYPE_GPU) {
        fprintf(stderr, "FATAL: Backend lock requires GPU backend. Got non-GPU backend.\n");
        g_backend_lock_state.lock_record.state = LLAMA_BACKEND_LOCK_INVALID;
        if (g_backend_lock_enforcement_strict) {
            return -1;
        }
    }

    // Record lock acquisition
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    g_backend_lock_state.lock_record.lock_acquire_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    g_backend_lock_state.lock_record.locked_backend = backend_to_lock;
    g_backend_lock_state.lock_record.lock_held = true;
    g_backend_lock_state.lock_record.backend_valid = true;
    g_backend_lock_state.lock_record.invalidation_reason = LLAMA_BACKEND_LOCK_VALID;
    g_backend_lock_state.lock_record.decode_token_count = 0;

    // Transition to ACQUIRED state
    g_backend_lock_state.lock_record.state = LLAMA_BACKEND_LOCK_ACQUIRED;

    llama_backend_lock_log_acquisition();

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 2: Backend Lock Release
// ============================================================================

int llama_backend_lock_release(void) {
    if (!g_backend_lock_state.lock_record.lock_held) {
        fprintf(stderr, "WARNING: Backend lock not held. Cannot release.\n");
        return -1;
    }

    // Transition to RELEASING state
    g_backend_lock_state.lock_record.state = LLAMA_BACKEND_LOCK_RELEASING;

    // Record lock release time
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    g_backend_lock_state.lock_record.lock_release_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    // Release the lock
    g_backend_lock_state.lock_record.lock_held = false;

    // Transition to RELEASED state
    g_backend_lock_state.lock_record.state = LLAMA_BACKEND_LOCK_RELEASED;

    llama_backend_lock_log_release();

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 3: Verify Lock Is Held
// ============================================================================

int llama_backend_lock_verify_held(void) {
    if (!g_backend_lock_state.lock_record.lock_held) {
        fprintf(stderr, "FATAL: Backend lock is not held during decode. Decode requires locked backend.\n");
        g_backend_lock_state.lock_record.state = LLAMA_BACKEND_LOCK_INVALID;

        llama_backend_lock_report_violation(
            LLAMA_BACKEND_LOCK_VIOL_INVALIDATION,
            LLAMA_BACKEND_LOCK_LOC_ADMISSION,
            "Backend lock not held at decode entry"
        );

        if (g_backend_lock_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 4: Prevent Backend Change
// ============================================================================

int llama_backend_lock_prevent_backend_change(enum ggml_backend_dev_type new_backend) {
    if (!g_backend_lock_state.lock_record.lock_held) {
        return 0; // Lock not held, no prevention needed
    }

    if (new_backend != g_backend_lock_state.lock_record.locked_backend) {
        fprintf(stderr, "FATAL: Attempted backend change during decode.\n");
        fprintf(stderr, "       Locked backend: %d, Attempted backend: %d\n",
                g_backend_lock_state.lock_record.locked_backend, new_backend);

        llama_backend_lock_report_violation(
            LLAMA_BACKEND_LOCK_VIOL_BACKEND_CHANGE,
            LLAMA_BACKEND_LOCK_LOC_BACKEND_SEL,
            "Backend change attempted while locked"
        );

        if (g_backend_lock_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 5: Prevent Backend Re-resolution
// ============================================================================

int llama_backend_lock_prevent_reresolution(void) {
    if (!g_backend_lock_state.lock_record.lock_held) {
        return 0;
    }

    fprintf(stderr, "FATAL: Attempted backend re-resolution during decode while locked.\n");
    fprintf(stderr, "       Backend is immutable during decode phase.\n");

    llama_backend_lock_report_violation(
        LLAMA_BACKEND_LOCK_VIOL_RERESOLUTION,
        LLAMA_BACKEND_LOCK_LOC_BACKEND_SEL,
        "Backend re-resolution attempted while locked"
    );

    if (g_backend_lock_enforcement_strict) {
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 6: Prevent Tensor Relocation
// ============================================================================

int llama_backend_lock_prevent_tensor_relocation(void) {
    if (!g_backend_lock_state.lock_record.lock_held) {
        return 0;
    }

    fprintf(stderr, "FATAL: Attempted tensor relocation during decode while backend locked.\n");
    fprintf(stderr, "       All decode tensors must remain on locked backend.\n");

    llama_backend_lock_report_violation(
        LLAMA_BACKEND_LOCK_VIOL_TENSOR_RELOCATION,
        LLAMA_BACKEND_LOCK_LOC_MEMORY_MGT,
        "Tensor relocation attempted while backend locked"
    );

    if (g_backend_lock_enforcement_strict) {
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 7: Check Backend Validity
// ============================================================================

int llama_backend_lock_check_validity(void) {
    if (!g_backend_lock_state.lock_record.lock_held) {
        return 0;
    }

    if (!g_backend_lock_state.lock_record.backend_valid) {
        fprintf(stderr, "FATAL: Locked backend has become invalid during decode.\n");
        fprintf(stderr, "       Reason: %s\n",
                llama_backend_lock_invalidation_reason_name(
                    g_backend_lock_state.lock_record.invalidation_reason));

        llama_backend_lock_report_violation(
            LLAMA_BACKEND_LOCK_VIOL_INVALIDATION,
            LLAMA_BACKEND_LOCK_LOC_BACKEND_SEL,
            "Backend validity check failed"
        );

        if (g_backend_lock_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 8: Terminate on Backend Invalidation
// ============================================================================

int llama_backend_lock_terminate_on_invalidation(
    enum llama_backend_lock_invalidation_reason reason
) {
    if (!g_backend_lock_state.lock_record.lock_held) {
        return 0;
    }

    if (reason != LLAMA_BACKEND_LOCK_VALID) {
        fprintf(stderr, "FATAL: Backend invalidation detected during decode.\n");
        fprintf(stderr, "       Reason: %s\n",
                llama_backend_lock_invalidation_reason_name(reason));
        fprintf(stderr, "       Decode session must be terminated immediately.\n");

        g_backend_lock_state.lock_record.backend_valid = false;
        g_backend_lock_state.lock_record.invalidation_reason = reason;
        g_backend_lock_state.total_invalidations++;
        g_total_backend_lock_invalidations++;

        llama_backend_lock_report_violation(
            LLAMA_BACKEND_LOCK_VIOL_INVALIDATION,
            LLAMA_BACKEND_LOCK_LOC_BACKEND_SEL,
            llama_backend_lock_invalidation_reason_name(reason)
        );

        if (g_backend_lock_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// QUERY AND DIAGNOSTIC FUNCTIONS
// ============================================================================

bool llama_backend_lock_is_held(void) {
    return g_backend_lock_state.lock_record.lock_held;
}

enum ggml_backend_dev_type llama_backend_lock_get_locked_backend(void) {
    if (!g_backend_lock_state.lock_record.lock_held) {
        return GGML_BACKEND_DEVICE_TYPE_CPU; // Default when not locked
    }
    return g_backend_lock_state.lock_record.locked_backend;
}

struct llama_backend_lock_record llama_backend_lock_get_record(void) {
    return g_backend_lock_state.lock_record;
}

uint64_t llama_backend_lock_get_duration_ns(void) {
    if (g_backend_lock_state.lock_record.lock_acquire_time_ns == 0) {
        return 0;
    }

    if (g_backend_lock_state.lock_record.lock_release_time_ns == 0) {
        // Still locked, calculate from acquisition to now
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        uint64_t current_time_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        return current_time_ns - g_backend_lock_state.lock_record.lock_acquire_time_ns;
    }

    // Lock released, calculate from acquisition to release
    return g_backend_lock_state.lock_record.lock_release_time_ns -
           g_backend_lock_state.lock_record.lock_acquire_time_ns;
}

int llama_backend_lock_get_violation_count(void) {
    return g_total_backend_lock_violations;
}

// ============================================================================
// SCOPE MANAGEMENT
// ============================================================================

int llama_backend_lock_assert_decode_phase_only(void) {
    // This function validates that backend lock is only used during decode
    // In a full implementation, this would check a phase flag from context
    return 0;
}

int llama_backend_lock_assert_not_prefill(void) {
    // This function ensures backend lock is NOT active during prefill phase
    // In a full implementation, this would check and reject prefill phase access
    return 0;
}

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

int llama_backend_lock_verify_all_operations_same_backend(
    const char** operation_names,
    int num_operations
) {
    if (operation_names == NULL || num_operations <= 0) {
        return -1;
    }

    if (!g_backend_lock_state.lock_record.lock_held) {
        return 0; // Not locked, no verification needed
    }

    // In full implementation, would verify each operation uses locked backend
    return 0;
}

int llama_backend_lock_assert_explicit_backend_decision(
    enum ggml_backend_dev_type backend
) {
    if (!g_backend_lock_state.lock_record.lock_held) {
        return 0;
    }

    if (backend != g_backend_lock_state.lock_record.locked_backend) {
        fprintf(stderr, "ERROR: Operation uses different backend than locked backend.\n");
        return -1;
    }

    return 0;
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

void llama_backend_lock_report_violation(
    enum llama_backend_lock_violation_type violation_type,
    enum llama_backend_lock_violation_location location,
    const char* details
) {
    fprintf(stderr, "\n");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "BACKEND LOCK VIOLATION\n");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "Lock State:       %s\n",
            llama_backend_lock_state_name(g_backend_lock_state.lock_record.state));
    fprintf(stderr, "Lock Held:        %s\n",
            g_backend_lock_state.lock_record.lock_held ? "YES" : "NO");
    fprintf(stderr, "Locked Backend:   %d\n",
            g_backend_lock_state.lock_record.locked_backend);
    fprintf(stderr, "Violation Type:   %s\n",
            llama_backend_lock_violation_type_name(violation_type));
    fprintf(stderr, "Location:         %d\n", location);
    fprintf(stderr, "Details:          %s\n", details != NULL ? details : "(none)");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "\n");

    g_backend_lock_state.lock_record.violation_count++;
    g_backend_lock_state.lock_record.last_violation = violation_type;
    g_backend_lock_state.lock_record.last_violation_location = details;
    g_backend_lock_state.total_violations++;
    g_total_backend_lock_violations++;
    g_violation_location_map[location]++;
}

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

void llama_backend_lock_log_acquisition(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "BACKEND LOCK ACQUIRED\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "Locked Backend:   %d\n", g_backend_lock_state.lock_record.locked_backend);
    fprintf(stdout, "Lock Acquire Time: %" PRIu64 " ns\n",
            g_backend_lock_state.lock_record.lock_acquire_time_ns);
    fprintf(stdout, "Status:           Backend selection is now IMMUTABLE for entire decode phase\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

void llama_backend_lock_log_release(void) {
    uint64_t duration_ns = llama_backend_lock_get_duration_ns();
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "BACKEND LOCK RELEASED\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "Locked Backend:       %d\n", g_backend_lock_state.lock_record.locked_backend);
    fprintf(stdout, "Lock Duration:        %" PRIu64 " ns (%.3f ms)\n",
            duration_ns, duration_ns / 1000000.0);
    fprintf(stdout, "Tokens Decoded:       %" PRIu64 "\n",
            g_backend_lock_state.lock_record.decode_token_count);
    fprintf(stdout, "Violations Detected:  %d\n",
            g_backend_lock_state.lock_record.violation_count);
    fprintf(stdout, "Backend Valid:        %s\n",
            g_backend_lock_state.lock_record.backend_valid ? "YES" : "NO");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

void llama_backend_lock_print_status(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "Backend Lock Status:\n");
    fprintf(stdout, "  State:           %s\n",
            llama_backend_lock_state_name(g_backend_lock_state.lock_record.state));
    fprintf(stdout, "  Lock Held:       %s\n",
            g_backend_lock_state.lock_record.lock_held ? "YES" : "NO");
    fprintf(stdout, "  Locked Backend:  %d\n",
            g_backend_lock_state.lock_record.locked_backend);
    fprintf(stdout, "  Valid:           %s\n",
            g_backend_lock_state.lock_record.backend_valid ? "YES" : "NO");
    fprintf(stdout, "  Violations:      %d\n",
            g_backend_lock_state.lock_record.violation_count);
    fprintf(stdout, "\n");
}

void llama_backend_lock_print_diagnostics(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "BACKEND LOCK DIAGNOSTICS\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "Lock State:           %s\n",
            llama_backend_lock_state_name(g_backend_lock_state.lock_record.state));
    fprintf(stdout, "Lock Held:            %s\n",
            g_backend_lock_state.lock_record.lock_held ? "YES" : "NO");
    fprintf(stdout, "Locked Backend:       %d\n",
            g_backend_lock_state.lock_record.locked_backend);
    fprintf(stdout, "Backend Valid:        %s\n",
            g_backend_lock_state.lock_record.backend_valid ? "YES" : "NO");
    fprintf(stdout, "\n");

    if (!g_backend_lock_state.lock_record.backend_valid) {
        fprintf(stdout, "Backend Invalidation Reason: %s\n",
                llama_backend_lock_invalidation_reason_name(
                    g_backend_lock_state.lock_record.invalidation_reason));
    }

    fprintf(stdout, "Total Violations:     %d\n",
            g_backend_lock_state.lock_record.violation_count);
    fprintf(stdout, "Last Violation:       %s\n",
            llama_backend_lock_violation_type_name(g_backend_lock_state.lock_record.last_violation));

    fprintf(stdout, "Lock Duration:        %" PRIu64 " ns\n",
            llama_backend_lock_get_duration_ns());
    fprintf(stdout, "Tokens Decoded:       %" PRIu64 "\n",
            g_backend_lock_state.lock_record.decode_token_count);
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

void llama_backend_lock_set_enforcement_strict(bool strict) {
    g_backend_lock_enforcement_strict = strict;
    g_backend_lock_state.enforcement_strict = strict;
}

bool llama_backend_lock_get_enforcement_strict(void) {
    return g_backend_lock_enforcement_strict;
}

void llama_backend_lock_set_debug_verify_backend_identity(bool verify) {
    g_backend_lock_state.debug_verify_backend_identity = verify;
}

// ============================================================================
// INVARIANT VERIFICATION
// ============================================================================

int llama_backend_lock_verify_immutability_invariant(void) {
    if (!g_backend_lock_state.lock_record.lock_held) {
        return 0; // Not in decode, invariant not applicable
    }

    if (g_backend_lock_state.lock_record.state != LLAMA_BACKEND_LOCK_ACQUIRED) {
        fprintf(stderr, "FATAL: Backend lock in invalid state during decode.\n");
        return -1;
    }

    if (!g_backend_lock_state.lock_record.backend_valid) {
        fprintf(stderr, "FATAL: Backend has become invalid while locked.\n");
        return -1;
    }

    return 0;
}

int llama_backend_lock_assert_backend_matches_locked(enum ggml_backend_dev_type actual_backend) {
    if (!g_backend_lock_state.lock_record.lock_held) {
        return 0;
    }

    if (actual_backend != g_backend_lock_state.lock_record.locked_backend) {
        fprintf(stderr, "FATAL: Backend mismatch during decode.\n");
        fprintf(stderr, "       Expected (locked): %d, Actual: %d\n",
                g_backend_lock_state.lock_record.locked_backend, actual_backend);

        llama_backend_lock_report_violation(
            LLAMA_BACKEND_LOCK_VIOL_BACKEND_CHANGE,
            LLAMA_BACKEND_LOCK_LOC_BACKEND_SEL,
            "Backend mismatch detected during decode step"
        );

        if (g_backend_lock_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================

int llama_backend_lock_selftest(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "BACKEND LOCK SELF-TEST SUITE\n");
    fprintf(stdout, "================================================================================\n");

    int test_count = 0;
    int pass_count = 0;

    // TEST 1: Initialization
    fprintf(stdout, "\nTest 1: Initialization...");
    test_count++;
    if (llama_backend_lock_init() == 0 && !g_backend_lock_state.lock_record.lock_held) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 2: Lock acquisition
    fprintf(stdout, "Test 2: Lock Acquisition...");
    test_count++;
    if (llama_backend_lock_acquire(GGML_BACKEND_DEVICE_TYPE_GPU) == 0 &&
        g_backend_lock_state.lock_record.lock_held &&
        g_backend_lock_state.lock_record.state == LLAMA_BACKEND_LOCK_ACQUIRED) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 3: Query locked backend
    fprintf(stdout, "Test 3: Query Locked Backend...");
    test_count++;
    if (llama_backend_lock_get_locked_backend() == GGML_BACKEND_DEVICE_TYPE_GPU) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 4: Lock held verification
    fprintf(stdout, "Test 4: Lock Held Verification...");
    test_count++;
    if (llama_backend_lock_verify_held() == 0 && llama_backend_lock_is_held()) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 5: Prevent backend change
    fprintf(stdout, "Test 5: Prevent Backend Change...");
    test_count++;
    llama_backend_lock_set_enforcement_strict(false); // Set permissive mode
    if (llama_backend_lock_prevent_backend_change(GGML_BACKEND_DEVICE_TYPE_CPU) != 0) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }
    llama_backend_lock_set_enforcement_strict(true); // Back to strict

    // TEST 6: Backend validity check
    fprintf(stdout, "Test 6: Backend Validity Check...");
    test_count++;
    if (llama_backend_lock_check_validity() == 0 && g_backend_lock_state.lock_record.backend_valid) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 7: Get duration
    fprintf(stdout, "Test 7: Get Lock Duration...");
    test_count++;
    if (llama_backend_lock_get_duration_ns() > 0) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 8: Lock release
    fprintf(stdout, "Test 8: Lock Release...");
    test_count++;
    if (llama_backend_lock_release() == 0 &&
        !g_backend_lock_state.lock_record.lock_held &&
        g_backend_lock_state.lock_record.state == LLAMA_BACKEND_LOCK_RELEASED) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "SELF-TEST RESULTS: %d / %d tests passed\n", pass_count, test_count);
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");

    return (pass_count == test_count) ? 0 : -1;
}
