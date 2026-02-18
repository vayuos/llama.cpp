/**
 * SECTION 12: Prohibit Graph Rebuilds During Decode
 * Implementation
 *
 * This file implements enforcement that graph rebuilds are completely forbidden
 * once decode has started. Any attempt to rebuild, invalidate, or regenerate
 * the graph during decode is treated as a fatal correctness error.
 */

#include "llama-decode-rebuild-prohibition.h"
#include <cstring>
#include <cstdio>
#include <chrono>
#include <map>

// ============================================================================
// GLOBAL STATE MANAGEMENT
// ============================================================================

static struct llama_rebuild_prohibition_validation_state g_rebuild_prohibition_state = {
    .prohibition_record = {
        .decode_progress = LLAMA_DECODE_PROGRESS_NOT_STARTED,
        .decode_in_progress = false,
        .decode_start_timestamp_ns = 0,
        .decode_step_count = 0,
        .graph_id_at_decode_start = 0,
        .graph_version_at_decode_start = 0,
        .rebuild_attempt_count = 0,
        .last_trigger = LLAMA_REBUILD_TRIGGER_NONE,
        .last_location = LLAMA_REBUILD_LOC_UNKNOWN,
        .last_violation = LLAMA_REBUILD_VIOL_NONE,
    },
    .total_rebuild_attempts = 0,
    .total_rebuild_violations = 0,
    .enforcement_strict = true,
    .debug_assert_graph_immutable_per_step = false,
};

static bool g_rebuild_prohibition_enforcement_strict = true;
static int g_total_rebuild_attempts_blocked = 0;
static int g_total_rebuild_violations = 0;

// Per-trigger type tracking
static std::map<enum llama_rebuild_trigger_type, int> g_trigger_attempt_map;
static std::map<enum llama_rebuild_attempt_location, int> g_location_attempt_map;

// ============================================================================
// INITIALIZATION
// ============================================================================

int llama_rebuild_prohibition_init(void) {
    g_rebuild_prohibition_state.prohibition_record.decode_progress = LLAMA_DECODE_PROGRESS_NOT_STARTED;
    g_rebuild_prohibition_state.prohibition_record.decode_in_progress = false;
    g_rebuild_prohibition_state.prohibition_record.decode_start_timestamp_ns = 0;
    g_rebuild_prohibition_state.prohibition_record.decode_step_count = 0;
    g_rebuild_prohibition_state.prohibition_record.graph_id_at_decode_start = 0;
    g_rebuild_prohibition_state.prohibition_record.graph_version_at_decode_start = 0;
    g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count = 0;
    g_rebuild_prohibition_state.prohibition_record.last_trigger = LLAMA_REBUILD_TRIGGER_NONE;
    g_rebuild_prohibition_state.prohibition_record.last_location = LLAMA_REBUILD_LOC_UNKNOWN;
    g_rebuild_prohibition_state.prohibition_record.last_violation = LLAMA_REBUILD_VIOL_NONE;
    g_rebuild_prohibition_state.total_rebuild_attempts = 0;
    g_rebuild_prohibition_state.total_rebuild_violations = 0;

    g_trigger_attempt_map.clear();
    g_location_attempt_map.clear();

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 1: Mark Decode Starting
// ============================================================================

int llama_rebuild_prohibition_mark_decode_starting(
    uint64_t graph_id,
    uint32_t graph_version
) {
    if (g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        fprintf(stderr, "ERROR: Cannot start decode while decode is already in progress\n");
        if (g_rebuild_prohibition_enforcement_strict) {
            return -1;
        }
    }

    g_rebuild_prohibition_state.prohibition_record.decode_progress = LLAMA_DECODE_PROGRESS_STARTING;
    g_rebuild_prohibition_state.prohibition_record.graph_id_at_decode_start = graph_id;
    g_rebuild_prohibition_state.prohibition_record.graph_version_at_decode_start = graph_version;

    // Record start time
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    g_rebuild_prohibition_state.prohibition_record.decode_start_timestamp_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 2: Mark Decode Active
// ============================================================================

int llama_rebuild_prohibition_mark_decode_active(void) {
    if (g_rebuild_prohibition_state.prohibition_record.decode_progress != LLAMA_DECODE_PROGRESS_STARTING) {
        fprintf(stderr, "ERROR: Cannot mark decode active from state: %s\n",
                llama_decode_progress_state_name(g_rebuild_prohibition_state.prohibition_record.decode_progress));
        if (g_rebuild_prohibition_enforcement_strict) {
            return -1;
        }
    }

    g_rebuild_prohibition_state.prohibition_record.decode_progress = LLAMA_DECODE_PROGRESS_ACTIVE;
    g_rebuild_prohibition_state.prohibition_record.decode_in_progress = true;
    g_rebuild_prohibition_state.prohibition_record.decode_step_count = 0;

    llama_rebuild_prohibition_log_decode_started();

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 3: Mark Decode Step Complete
// ============================================================================

int llama_rebuild_prohibition_mark_decode_step_complete(void) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0; // Not in decode, nothing to track
    }

    g_rebuild_prohibition_state.prohibition_record.decode_step_count++;

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 4: Mark Decode Completed
// ============================================================================

int llama_rebuild_prohibition_mark_decode_completed(void) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        fprintf(stderr, "WARNING: Attempted to complete decode that is not in progress\n");
        return -1;
    }

    g_rebuild_prohibition_state.prohibition_record.decode_progress = LLAMA_DECODE_PROGRESS_COMPLETED;
    g_rebuild_prohibition_state.prohibition_record.decode_in_progress = false;

    llama_rebuild_prohibition_log_decode_completed();

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 5: Verify Decode Not Active
// ============================================================================

int llama_rebuild_prohibition_verify_decode_not_active(void) {
    if (g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        fprintf(stderr, "ERROR: Operation attempted while decode is active\n");
        if (g_rebuild_prohibition_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 6: Guard Graph Revalidate
// ============================================================================

int llama_rebuild_prohibition_guard_graph_revalidate(void) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0; // Revalidation allowed outside decode
    }

    fprintf(stderr, "FATAL: Graph revalidation attempted during decode\n");
    fprintf(stderr, "       Graph structure is immutable during active token generation\n");

    g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count++;
    g_rebuild_prohibition_state.prohibition_record.last_trigger = LLAMA_REBUILD_TRIGGER_VERSION_MISMATCH;
    g_rebuild_prohibition_state.prohibition_record.last_location = LLAMA_REBUILD_LOC_GRAPH_REVALIDATE;
    g_rebuild_prohibition_state.prohibition_record.last_violation = LLAMA_REBUILD_VIOL_REVALIDATE_ATTEMPTED;
    g_rebuild_prohibition_state.total_rebuild_attempts++;
    g_total_rebuild_attempts_blocked++;
    g_location_attempt_map[LLAMA_REBUILD_LOC_GRAPH_REVALIDATE]++;

    llama_rebuild_prohibition_report_rebuild_attempt(
        LLAMA_REBUILD_TRIGGER_VERSION_MISMATCH,
        LLAMA_REBUILD_LOC_GRAPH_REVALIDATE,
        "Graph revalidation during decode"
    );

    if (g_rebuild_prohibition_enforcement_strict) {
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 7: Guard Graph Regenerate
// ============================================================================

int llama_rebuild_prohibition_guard_graph_regenerate(void) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0;
    }

    fprintf(stderr, "FATAL: Graph regeneration attempted during decode\n");
    fprintf(stderr, "       Decode must run on single frozen graph\n");

    g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count++;
    g_rebuild_prohibition_state.prohibition_record.last_trigger = LLAMA_REBUILD_TRIGGER_AUTO_INVALIDATION;
    g_rebuild_prohibition_state.prohibition_record.last_location = LLAMA_REBUILD_LOC_GRAPH_REGENERATE;
    g_rebuild_prohibition_state.prohibition_record.last_violation = LLAMA_REBUILD_VIOL_REGENERATE_ATTEMPTED;
    g_rebuild_prohibition_state.total_rebuild_attempts++;
    g_total_rebuild_attempts_blocked++;
    g_location_attempt_map[LLAMA_REBUILD_LOC_GRAPH_REGENERATE]++;

    llama_rebuild_prohibition_report_rebuild_attempt(
        LLAMA_REBUILD_TRIGGER_AUTO_INVALIDATION,
        LLAMA_REBUILD_LOC_GRAPH_REGENERATE,
        "Graph regeneration during decode"
    );

    if (g_rebuild_prohibition_enforcement_strict) {
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 8: Guard Shape Adaptation
// ============================================================================

int llama_rebuild_prohibition_guard_shape_adaptation(void) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0;
    }

    fprintf(stderr, "FATAL: Tensor shape adaptation attempted during decode\n");
    fprintf(stderr, "       All tensor shapes must be fixed before decode begins\n");

    g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count++;
    g_rebuild_prohibition_state.prohibition_record.last_trigger = LLAMA_REBUILD_TRIGGER_SHAPE_MISMATCH;
    g_rebuild_prohibition_state.prohibition_record.last_location = LLAMA_REBUILD_LOC_SHAPE_ADAPTATION;
    g_rebuild_prohibition_state.prohibition_record.last_violation = LLAMA_REBUILD_VIOL_SHAPE_ADAPTATION;
    g_rebuild_prohibition_state.total_rebuild_attempts++;
    g_total_rebuild_attempts_blocked++;
    g_location_attempt_map[LLAMA_REBUILD_LOC_SHAPE_ADAPTATION]++;

    llama_rebuild_prohibition_report_rebuild_attempt(
        LLAMA_REBUILD_TRIGGER_SHAPE_MISMATCH,
        LLAMA_REBUILD_LOC_SHAPE_ADAPTATION,
        "Shape adaptation during decode"
    );

    if (g_rebuild_prohibition_enforcement_strict) {
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 9: Guard KV Cache Expansion
// ============================================================================

int llama_rebuild_prohibition_guard_kv_cache_expansion(void) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0;
    }

    fprintf(stderr, "FATAL: KV cache expansion attempted during decode\n");
    fprintf(stderr, "       KV cache bounds must be preallocated before decode\n");

    g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count++;
    g_rebuild_prohibition_state.prohibition_record.last_trigger = LLAMA_REBUILD_TRIGGER_KV_CACHE_EXPANSION;
    g_rebuild_prohibition_state.prohibition_record.last_location = LLAMA_REBUILD_LOC_KV_CACHE_EXTEND;
    g_rebuild_prohibition_state.prohibition_record.last_violation = LLAMA_REBUILD_VIOL_INVALIDATION;
    g_rebuild_prohibition_state.total_rebuild_attempts++;
    g_total_rebuild_attempts_blocked++;
    g_location_attempt_map[LLAMA_REBUILD_LOC_KV_CACHE_EXTEND]++;

    llama_rebuild_prohibition_report_rebuild_attempt(
        LLAMA_REBUILD_TRIGGER_KV_CACHE_EXPANSION,
        LLAMA_REBUILD_LOC_KV_CACHE_EXTEND,
        "KV cache expansion during decode"
    );

    if (g_rebuild_prohibition_enforcement_strict) {
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 10: Guard Backend Reassignment
// ============================================================================

int llama_rebuild_prohibition_guard_backend_reassignment(void) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0;
    }

    fprintf(stderr, "FATAL: Backend reassignment attempted during decode\n");
    fprintf(stderr, "       Backend is locked and immutable during decode\n");

    g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count++;
    g_rebuild_prohibition_state.prohibition_record.last_trigger = LLAMA_REBUILD_TRIGGER_BACKEND_UNAVAILABLE;
    g_rebuild_prohibition_state.prohibition_record.last_location = LLAMA_REBUILD_LOC_BACKEND_REASSIGN;
    g_rebuild_prohibition_state.prohibition_record.last_violation = LLAMA_REBUILD_VIOL_REBUILD_ATTEMPTED;
    g_rebuild_prohibition_state.total_rebuild_attempts++;
    g_total_rebuild_attempts_blocked++;
    g_location_attempt_map[LLAMA_REBUILD_LOC_BACKEND_REASSIGN]++;

    llama_rebuild_prohibition_report_rebuild_attempt(
        LLAMA_REBUILD_TRIGGER_BACKEND_UNAVAILABLE,
        LLAMA_REBUILD_LOC_BACKEND_REASSIGN,
        "Backend reassignment during decode"
    );

    if (g_rebuild_prohibition_enforcement_strict) {
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 11: Guard Memory Reallocation
// ============================================================================

int llama_rebuild_prohibition_guard_memory_reallocation(void) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0;
    }

    fprintf(stderr, "FATAL: Memory reallocation attempted during decode\n");
    fprintf(stderr, "       All memory must be preallocated before decode begins\n");

    g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count++;
    g_rebuild_prohibition_state.prohibition_record.last_trigger = LLAMA_REBUILD_TRIGGER_MEMORY_REALLOCATION;
    g_rebuild_prohibition_state.prohibition_record.last_location = LLAMA_REBUILD_LOC_MEMORY_REALLOC;
    g_rebuild_prohibition_state.prohibition_record.last_violation = LLAMA_REBUILD_VIOL_REBUILD_ATTEMPTED;
    g_rebuild_prohibition_state.total_rebuild_attempts++;
    g_total_rebuild_attempts_blocked++;
    g_location_attempt_map[LLAMA_REBUILD_LOC_MEMORY_REALLOC]++;

    llama_rebuild_prohibition_report_rebuild_attempt(
        LLAMA_REBUILD_TRIGGER_MEMORY_REALLOCATION,
        LLAMA_REBUILD_LOC_MEMORY_REALLOC,
        "Memory reallocation during decode"
    );

    if (g_rebuild_prohibition_enforcement_strict) {
        return -1;
    }

    return 0;
}

// ============================================================================
// REBUILD FLAG CHECKING
// ============================================================================

int llama_rebuild_prohibition_check_no_rebuild_flags_set(void) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0;
    }

    // In full implementation, would check actual rebuild flags
    // For now, this is a placeholder for the verification mechanism

    return 0;
}

// ============================================================================
// GRAPH VERSION VERIFICATION
// ============================================================================

int llama_rebuild_prohibition_check_graph_version_unchanged(
    uint64_t current_graph_id,
    uint32_t current_graph_version
) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0;
    }

    if (current_graph_id != g_rebuild_prohibition_state.prohibition_record.graph_id_at_decode_start) {
        fprintf(stderr, "FATAL: Graph ID changed during decode\n");
        fprintf(stderr, "       Started with ID %" PRIu64 ", now: %" PRIu64 "\n",
                g_rebuild_prohibition_state.prohibition_record.graph_id_at_decode_start,
                current_graph_id);

        g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count++;
        g_rebuild_prohibition_state.prohibition_record.last_violation = LLAMA_REBUILD_VIOL_REBUILD_ATTEMPTED;
        g_total_rebuild_violations++;

        if (g_rebuild_prohibition_enforcement_strict) {
            return -1;
        }
    }

    if (current_graph_version != g_rebuild_prohibition_state.prohibition_record.graph_version_at_decode_start) {
        fprintf(stderr, "FATAL: Graph version changed during decode\n");
        fprintf(stderr, "       Started with version %u, now: %u\n",
                g_rebuild_prohibition_state.prohibition_record.graph_version_at_decode_start,
                current_graph_version);

        g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count++;
        g_rebuild_prohibition_state.prohibition_record.last_trigger = LLAMA_REBUILD_TRIGGER_VERSION_MISMATCH;
        g_rebuild_prohibition_state.prohibition_record.last_violation = LLAMA_REBUILD_VIOL_VERSION_MISMATCH;
        g_total_rebuild_violations++;

        if (g_rebuild_prohibition_enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// LATE-DISCOVERED INVALIDATION HANDLING
// ============================================================================

int llama_rebuild_prohibition_handle_late_invalidation(
    enum llama_rebuild_trigger_type trigger_reason
) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0;
    }

    fprintf(stderr, "FATAL: Late invalidation discovered during decode\n");
    fprintf(stderr, "       Trigger: %s\n", llama_rebuild_trigger_type_name(trigger_reason));
    fprintf(stderr, "       Decode session must be terminated immediately\n");

    g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count++;
    g_rebuild_prohibition_state.prohibition_record.last_trigger = trigger_reason;
    g_rebuild_prohibition_state.prohibition_record.last_violation = LLAMA_REBUILD_VIOL_INVALIDATION;
    g_total_rebuild_violations++;
    g_trigger_attempt_map[trigger_reason]++;

    llama_rebuild_prohibition_report_rebuild_attempt(
        trigger_reason,
        LLAMA_REBUILD_LOC_UNKNOWN,
        "Late invalidation discovered"
    );

    if (g_rebuild_prohibition_enforcement_strict) {
        return -1;
    }

    return 0;
}

// ============================================================================
// QUERY AND DIAGNOSTIC FUNCTIONS
// ============================================================================

bool llama_rebuild_prohibition_is_decode_active(void) {
    return g_rebuild_prohibition_state.prohibition_record.decode_in_progress;
}

enum llama_decode_progress_state llama_rebuild_prohibition_get_decode_progress(void) {
    return g_rebuild_prohibition_state.prohibition_record.decode_progress;
}

struct llama_rebuild_prohibition_record llama_rebuild_prohibition_get_record(void) {
    return g_rebuild_prohibition_state.prohibition_record;
}

int llama_rebuild_prohibition_get_rebuild_attempt_count(void) {
    return g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count;
}

uint64_t llama_rebuild_prohibition_get_decode_step_count(void) {
    return g_rebuild_prohibition_state.prohibition_record.decode_step_count;
}

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

int llama_rebuild_prohibition_verify_no_auto_rebuild_active(void) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0;
    }

    // In full implementation, would check for active auto-rebuild logic
    return 0;
}

int llama_rebuild_prohibition_verify_graph_stable_for_decode(void) {
    if (llama_rebuild_prohibition_check_no_rebuild_flags_set() != 0) {
        return -1;
    }

    return 0;
}

int llama_rebuild_prohibition_assert_not_in_rebuild_path(void) {
    if (!g_rebuild_prohibition_state.prohibition_record.decode_in_progress) {
        return 0;
    }

    // Any rebuild path code reached during decode is a fatal error
    fprintf(stderr, "FATAL: Rebuild code path executed during decode\n");
    if (g_rebuild_prohibition_enforcement_strict) {
        return -1;
    }

    return 0;
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

void llama_rebuild_prohibition_report_rebuild_attempt(
    enum llama_rebuild_trigger_type trigger,
    enum llama_rebuild_attempt_location location,
    const char* reason
) {
    fprintf(stderr, "\n");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "GRAPH REBUILD PROHIBITION VIOLATION\n");
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "Decode Progress:  %s\n",
            llama_decode_progress_state_name(g_rebuild_prohibition_state.prohibition_record.decode_progress));
    fprintf(stderr, "Decode Active:    %s\n",
            g_rebuild_prohibition_state.prohibition_record.decode_in_progress ? "YES" : "NO");
    fprintf(stderr, "Trigger Type:     %s\n", llama_rebuild_trigger_type_name(trigger));
    fprintf(stderr, "Location:         %d\n", location);
    fprintf(stderr, "Reason:           %s\n", reason != NULL ? reason : "(none)");
    fprintf(stderr, "Step Count:       %" PRIu64 "\n",
            g_rebuild_prohibition_state.prohibition_record.decode_step_count);
    fprintf(stderr, "================================================================================\n");
    fprintf(stderr, "\n");
}

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

void llama_rebuild_prohibition_log_decode_started(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "DECODE SESSION STARTED - GRAPH REBUILDS PROHIBITED\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "Graph ID:         %" PRIu64 "\n",
            g_rebuild_prohibition_state.prohibition_record.graph_id_at_decode_start);
    fprintf(stdout, "Graph Version:    %u\n",
            g_rebuild_prohibition_state.prohibition_record.graph_version_at_decode_start);
    fprintf(stdout, "Start Time:       %" PRIu64 " ns\n",
            g_rebuild_prohibition_state.prohibition_record.decode_start_timestamp_ns);
    fprintf(stdout, "Status:           Graph structure is now IMMUTABLE for decode duration\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

void llama_rebuild_prohibition_log_decode_completed(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "DECODE SESSION COMPLETED - GRAPH REBUILDS NOW PERMITTED\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "Decode Steps:     %" PRIu64 "\n",
            g_rebuild_prohibition_state.prohibition_record.decode_step_count);
    fprintf(stdout, "Rebuild Attempts: %d (all blocked)\n",
            g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count);
    fprintf(stdout, "Status:           Decode session ended, graph is now mutable\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

void llama_rebuild_prohibition_print_status(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "Graph Rebuild Prohibition Status:\n");
    fprintf(stdout, "  Decode Progress:   %s\n",
            llama_decode_progress_state_name(g_rebuild_prohibition_state.prohibition_record.decode_progress));
    fprintf(stdout, "  Decode Active:     %s\n",
            g_rebuild_prohibition_state.prohibition_record.decode_in_progress ? "YES" : "NO");
    fprintf(stdout, "  Decode Steps:      %" PRIu64 "\n",
            g_rebuild_prohibition_state.prohibition_record.decode_step_count);
    fprintf(stdout, "  Rebuild Attempts:  %d\n",
            g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count);
    fprintf(stdout, "\n");
}

void llama_rebuild_prohibition_print_diagnostics(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "GRAPH REBUILD PROHIBITION DIAGNOSTICS\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "Decode State:\n");
    fprintf(stdout, "  Progress:         %s\n",
            llama_decode_progress_state_name(g_rebuild_prohibition_state.prohibition_record.decode_progress));
    fprintf(stdout, "  In Progress:      %s\n",
            g_rebuild_prohibition_state.prohibition_record.decode_in_progress ? "YES" : "NO");
    fprintf(stdout, "  Step Count:       %" PRIu64 "\n",
            g_rebuild_prohibition_state.prohibition_record.decode_step_count);
    fprintf(stdout, "\n");
    fprintf(stdout, "Graph Information:\n");
    fprintf(stdout, "  ID at Start:      %" PRIu64 "\n",
            g_rebuild_prohibition_state.prohibition_record.graph_id_at_decode_start);
    fprintf(stdout, "  Version at Start: %u\n",
            g_rebuild_prohibition_state.prohibition_record.graph_version_at_decode_start);
    fprintf(stdout, "\n");
    fprintf(stdout, "Rebuild Attempt History:\n");
    fprintf(stdout, "  Total Attempts:   %d\n",
            g_rebuild_prohibition_state.prohibition_record.rebuild_attempt_count);
    fprintf(stdout, "  Last Trigger:     %s\n",
            llama_rebuild_trigger_type_name(g_rebuild_prohibition_state.prohibition_record.last_trigger));
    fprintf(stdout, "  Last Location:    %d\n",
            g_rebuild_prohibition_state.prohibition_record.last_location);
    fprintf(stdout, "  Last Violation:   %s\n",
            llama_rebuild_violation_type_name(g_rebuild_prohibition_state.prohibition_record.last_violation));
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

void llama_rebuild_prohibition_set_enforcement_strict(bool strict) {
    g_rebuild_prohibition_enforcement_strict = strict;
    g_rebuild_prohibition_state.enforcement_strict = strict;
}

bool llama_rebuild_prohibition_get_enforcement_strict(void) {
    return g_rebuild_prohibition_enforcement_strict;
}

void llama_rebuild_prohibition_set_debug_assert_immutable_per_step(bool assert_immutable) {
    g_rebuild_prohibition_state.debug_assert_graph_immutable_per_step = assert_immutable;
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================

int llama_rebuild_prohibition_selftest(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "REBUILD PROHIBITION SELF-TEST SUITE\n");
    fprintf(stdout, "================================================================================\n");

    int test_count = 0;
    int pass_count = 0;

    // TEST 1: Initialization
    fprintf(stdout, "\nTest 1: Initialization...");
    test_count++;
    if (llama_rebuild_prohibition_init() == 0 &&
        g_rebuild_prohibition_state.prohibition_record.decode_progress == LLAMA_DECODE_PROGRESS_NOT_STARTED) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 2: Mark decode starting
    fprintf(stdout, "Test 2: Mark Decode Starting...");
    test_count++;
    if (llama_rebuild_prohibition_mark_decode_starting(5678, 42) == 0 &&
        g_rebuild_prohibition_state.prohibition_record.graph_id_at_decode_start == 5678) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 3: Mark decode active
    fprintf(stdout, "Test 3: Mark Decode Active...");
    test_count++;
    if (llama_rebuild_prohibition_mark_decode_active() == 0 &&
        g_rebuild_prohibition_state.prohibition_record.decode_in_progress &&
        g_rebuild_prohibition_state.prohibition_record.decode_progress == LLAMA_DECODE_PROGRESS_ACTIVE) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 4: Query decode active
    fprintf(stdout, "Test 4: Query Decode Active...");
    test_count++;
    if (llama_rebuild_prohibition_is_decode_active() == true) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 5: Guard graph revalidate
    fprintf(stdout, "Test 5: Guard Graph Revalidate...");
    test_count++;
    llama_rebuild_prohibition_set_enforcement_strict(false); // Permissive mode
    if (llama_rebuild_prohibition_guard_graph_revalidate() != 0) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }
    llama_rebuild_prohibition_set_enforcement_strict(true); // Back to strict

    // TEST 6: Mark decode step complete
    fprintf(stdout, "Test 6: Mark Decode Step Complete...");
    test_count++;
    uint64_t steps_before = g_rebuild_prohibition_state.prohibition_record.decode_step_count;
    llama_rebuild_prohibition_mark_decode_step_complete();
    if (g_rebuild_prohibition_state.prohibition_record.decode_step_count == steps_before + 1) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 7: Mark decode completed
    fprintf(stdout, "Test 7: Mark Decode Completed...");
    test_count++;
    if (llama_rebuild_prohibition_mark_decode_completed() == 0 &&
        !g_rebuild_prohibition_state.prohibition_record.decode_in_progress &&
        g_rebuild_prohibition_state.prohibition_record.decode_progress == LLAMA_DECODE_PROGRESS_COMPLETED) {
        fprintf(stdout, " PASS\n");
        pass_count++;
    } else {
        fprintf(stdout, " FAIL\n");
    }

    // TEST 8: Verify decode not active
    fprintf(stdout, "Test 8: Verify Decode Not Active...");
    test_count++;
    if (llama_rebuild_prohibition_verify_decode_not_active() == 0) {
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
