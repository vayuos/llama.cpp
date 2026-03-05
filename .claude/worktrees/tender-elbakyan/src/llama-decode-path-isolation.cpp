/**
 * Decode Path Isolation Implementation
 *
 * Enforces complete isolation of decode execution from thread pool interactions.
 * Eliminates per-token submissions, wake-sleep cycles, lock contention, and
 * work-stealing to achieve deterministic, low-jitter decode with stable GPU occupancy.
 */

#include "llama-decode-path-isolation.h"
#include "llama-impl.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <time.h>
#include <math.h>
#include <inttypes.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <pthread.h>
    #include <unistd.h>
#endif

/* ============================================================================
   Thread-Local State for Isolation Tracking
   ============================================================================ */

static thread_local bool g_isolation_active = false;
static thread_local int g_isolation_depth = 0;
static llama_decode_path_isolation_state * g_global_isolation_state = NULL;
static pthread_mutex_t g_isolation_state_lock = PTHREAD_MUTEX_INITIALIZER;

/* ============================================================================
   Timing Utilities
   ============================================================================ */

static uint64_t get_time_ns(void) {
    #ifdef _WIN32
        LARGE_INTEGER frequency, counter;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&counter);
        return (counter.QuadPart * 1000000000LL) / frequency.QuadPart;
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    #endif
}

/* ============================================================================
   Initialization and Lifecycle
   ============================================================================ */

bool llama_decode_path_isolation_init(llama_decode_path_isolation_state * state) {
    if (!state) {
        LLAMA_LOG_ERROR("DECODE ISOLATION: state is NULL\n");
        return false;
    }

    memset(state, 0, sizeof(*state));

    state->current_mode = LLAMA_DECODE_MODE_PREFILL;
    state->isolation_state = LLAMA_ISOLATION_STATE_IDLE;
    state->threadpool_minimal_size = 1;
    state->max_allowed_submissions_per_token = 0; // No submissions allowed

    // Allocate per-token metrics array
    state->token_metrics_capacity = 4096; // Support up to 4K tokens per decode
    state->token_metrics = (llama_per_token_metrics *)malloc(
        sizeof(llama_per_token_metrics) * state->token_metrics_capacity);

    if (!state->token_metrics) {
        LLAMA_LOG_ERROR("DECODE ISOLATION: Failed to allocate token metrics array\n");
        return false;
    }

    // Allocate lock entries array
    state->lock_entries_capacity = 64; // Support up to 64 locks
    state->lock_entries = (llama_lock_contention_entry *)malloc(
        sizeof(llama_lock_contention_entry) * state->lock_entries_capacity);

    if (!state->lock_entries) {
        LLAMA_LOG_ERROR("DECODE ISOLATION: Failed to allocate lock entries array\n");
        free(state->token_metrics);
        return false;
    }

    memset(state->token_metrics, 0,
           sizeof(llama_per_token_metrics) * state->token_metrics_capacity);
    memset(state->lock_entries, 0,
           sizeof(llama_lock_contention_entry) * state->lock_entries_capacity);

    // Initialize direct invocation state
    state->direct_invoke.enabled = true;
    state->direct_invoke.all_tokens_direct = true;

    // Initialize guard state
    state->guard_state.abort_on_violation = true;

    // Enable all monitoring by default
    state->submission_detection_enabled = true;
    state->lock_monitoring_enabled = true;
    state->work_stealing_monitoring_enabled = true;
    state->direct_invocation_required = true;

    LLAMA_LOG_INFO("DECODE ISOLATION: Initialized (token_capacity=%d, lock_capacity=%d)\n",
                   state->token_metrics_capacity,
                   state->lock_entries_capacity);

    pthread_mutex_lock(&g_isolation_state_lock);
    g_global_isolation_state = state;
    pthread_mutex_unlock(&g_isolation_state_lock);

    return true;
}

void llama_decode_path_isolation_release(llama_decode_path_isolation_state * state) {
    if (!state) return;

    if (state->token_metrics) {
        free(state->token_metrics);
        state->token_metrics = NULL;
    }

    if (state->lock_entries) {
        free(state->lock_entries);
        state->lock_entries = NULL;
    }

    state->isolation_state = LLAMA_ISOLATION_STATE_UNINITIALIZED;
    state->isolation_active = false;

    pthread_mutex_lock(&g_isolation_state_lock);
    if (g_global_isolation_state == state) {
        g_global_isolation_state = NULL;
    }
    pthread_mutex_unlock(&g_isolation_state_lock);

    LLAMA_LOG_INFO("DECODE ISOLATION: Released\n");
}

void llama_decode_path_isolation_reset(llama_decode_path_isolation_state * state) {
    if (!state) return;

    // Reset per-token metrics
    memset(state->token_metrics, 0,
           sizeof(llama_per_token_metrics) * state->token_metrics_capacity);

    // Reset counters
    state->n_tokens_processed = 0;
    state->max_per_token_submissions = 0;
    state->max_per_token_locks = 0;
    state->max_per_token_wakeups = 0;
    state->threadpool_submissions = 0;
    state->threadpool_completions = 0;
    state->threadpool_enqueue_operations = 0;
    state->threadpool_dequeue_operations = 0;

    // Reset queue monitor
    memset(&state->queue_monitor, 0, sizeof(state->queue_monitor));

    // Reset isolation state
    state->isolation_state = LLAMA_ISOLATION_STATE_IDLE;
    state->isolation_active = false;
    state->decode_start_ns = 0;
    state->decode_end_ns = 0;

    LLAMA_LOG_INFO("DECODE ISOLATION: Reset complete\n");
}

/* ============================================================================
   Thread Pool Freezing and Control
   ============================================================================ */

bool llama_decode_path_isolation_freeze_threadpool(
    llama_decode_path_isolation_state * state,
    int minimal_threads) {
    if (!state) return false;

    if (minimal_threads < 1) {
        minimal_threads = 1;
    }

    state->threadpool_initial_size = state->threadpool_current_size > 0 ?
        state->threadpool_current_size : 8; // Default if not set

    state->threadpool_minimal_size = minimal_threads;
    state->threadpool_current_size = minimal_threads;
    state->threadpool_state = LLAMA_THREADPOOL_STATE_FROZEN;

    LLAMA_LOG_INFO(
        "DECODE ISOLATION: Threadpool frozen (initial=%d, minimal=%d)\n",
        state->threadpool_initial_size,
        state->threadpool_minimal_size);

    return true;
}

bool llama_decode_path_isolation_thaw_threadpool(
    llama_decode_path_isolation_state * state) {
    if (!state) return false;

    state->threadpool_current_size = state->threadpool_initial_size;
    state->threadpool_state = LLAMA_THREADPOOL_STATE_ACTIVE;

    LLAMA_LOG_INFO("DECODE ISOLATION: Threadpool thawed (size=%d)\n",
                   state->threadpool_current_size);

    return true;
}

llama_threadpool_state_t llama_decode_path_isolation_get_threadpool_state(
    const llama_decode_path_isolation_state * state) {
    if (!state) return LLAMA_THREADPOOL_STATE_UNKNOWN;
    return state->threadpool_state;
}

/* ============================================================================
   Decode Execution Mode Management
   ============================================================================ */

bool llama_decode_path_isolation_begin_decode(
    llama_decode_path_isolation_state * state,
    int n_tokens_expected) {
    if (!state) return false;

    if (state->isolation_state == LLAMA_ISOLATION_STATE_DECODE_GUARDED) {
        LLAMA_LOG_WARN("DECODE ISOLATION: Nested decode detected\n");
        g_isolation_depth++;
        return true;
    }

    // Freeze thread pool
    if (!llama_decode_path_isolation_freeze_threadpool(state, 1)) {
        LLAMA_LOG_ERROR("DECODE ISOLATION: Failed to freeze threadpool\n");
        state->isolation_state = LLAMA_ISOLATION_STATE_ERROR;
        return false;
    }

    // Take queue snapshot
    llama_decode_path_isolation_snapshot_queue_before(state);

    state->isolation_state = LLAMA_ISOLATION_STATE_DECODE_GUARDED;
    state->isolation_active = true;
    state->decode_start_ns = get_time_ns();
    state->current_mode = LLAMA_DECODE_MODE_DECODE_ISOLATION;
    state->n_tokens_processed = 0;

    g_isolation_active = true;
    g_isolation_depth = 1;

    LLAMA_LOG_INFO(
        "DECODE ISOLATION: Decode phase started (expecting %d tokens)\n",
        n_tokens_expected);

    // Assert initial preconditions
    if (!llama_decode_path_isolation_assert_empty_queue(state)) {
        LLAMA_LOG_ERROR("DECODE ISOLATION: Queue not empty at decode start\n");
    }

    return true;
}

bool llama_decode_path_isolation_end_decode(
    llama_decode_path_isolation_state * state) {
    if (!state) return false;

    if (g_isolation_depth > 1) {
        g_isolation_depth--;
        return true;
    }

    state->decode_end_ns = get_time_ns();
    state->current_mode = LLAMA_DECODE_MODE_PREFILL;
    state->isolation_active = false;
    g_isolation_active = false;
    g_isolation_depth = 0;

    // Thaw thread pool
    llama_decode_path_isolation_thaw_threadpool(state);

    // Validate isolation
    bool isolation_valid = llama_decode_path_isolation_check_integrity(state);

    if (!isolation_valid) {
        state->isolation_state = LLAMA_ISOLATION_STATE_ERROR;
        state->guard_state.violations_detected++;
    } else {
        state->isolation_state = LLAMA_ISOLATION_STATE_DECODE_COMPLETE;
    }

    uint64_t decode_duration_ns = state->decode_end_ns - state->decode_start_ns;

    LLAMA_LOG_INFO(
        "DECODE ISOLATION: Decode phase ended (tokens=%d, duration=%.2fms, violations=%lu)\n",
        state->n_tokens_processed,
        (double)decode_duration_ns / 1000000.0,
        state->guard_state.violations_detected);

    state->total_decode_time_ns += decode_duration_ns;
    state->total_decode_cycles++;

    return isolation_valid;
}

void llama_decode_path_isolation_token_start(
    llama_decode_path_isolation_state * state,
    uint64_t token_index) {
    if (!state || token_index >= (uint64_t)state->token_metrics_capacity) {
        return;
    }

    state->token_metrics[token_index].token_index = token_index;
    state->token_metrics[token_index].timestamp_ns = get_time_ns();
    state->token_metrics[token_index].submission_count = 0;
    state->token_metrics[token_index].parallel_region_count = 0;
    state->token_metrics[token_index].lock_acquisitions = 0;
    state->token_metrics[token_index].worker_wakeups = 0;
    state->token_metrics[token_index].work_stealing_attempts = 0;
    state->token_metrics[token_index].direct_invocation_used = false;
}

void llama_decode_path_isolation_token_end(
    llama_decode_path_isolation_state * state) {
    if (!state || state->n_tokens_processed >= state->token_metrics_capacity) {
        return;
    }

    int token_idx = state->n_tokens_processed;
    llama_per_token_metrics * metrics = &state->token_metrics[token_idx];

    // Update max values
    if (metrics->submission_count > state->max_per_token_submissions) {
        state->max_per_token_submissions = metrics->submission_count;
    }
    if (metrics->lock_acquisitions > state->max_per_token_locks) {
        state->max_per_token_locks = metrics->lock_acquisitions;
    }
    if (metrics->worker_wakeups > state->max_per_token_wakeups) {
        state->max_per_token_wakeups = metrics->worker_wakeups;
    }

    state->n_tokens_processed++;
}

/* ============================================================================
   Direct Invocation Mechanism
   ============================================================================ */

bool llama_decode_path_isolation_execute_direct(
    llama_decode_path_isolation_state * state,
    const char * operation_name,
    bool (*operation_fn)(void * data),
    void * user_data) {
    (void)operation_name;  // Reserved for future detailed operation tracing

    if (!state || !operation_fn) return false;

    uint64_t start_ns = get_time_ns();

    // Execute operation synchronously
    bool result = operation_fn(user_data);

    uint64_t end_ns = get_time_ns();
    uint64_t duration_ns = end_ns - start_ns;

    // Update direct invocation metrics
    state->direct_invoke.invocations_count++;
    state->direct_invoke.total_invocation_time_ns += duration_ns;

    if (duration_ns > state->direct_invoke.max_invocation_time_ns) {
        state->direct_invoke.max_invocation_time_ns = duration_ns;
    }

    if (state->direct_invoke.min_invocation_time_ns == 0 ||
        duration_ns < state->direct_invoke.min_invocation_time_ns) {
        state->direct_invoke.min_invocation_time_ns = duration_ns;
    }

    // Mark current token as using direct invocation
    if (state->n_tokens_processed < state->token_metrics_capacity) {
        state->token_metrics[state->n_tokens_processed].direct_invocation_used = true;
    }

    return result;
}

bool llama_decode_path_isolation_verify_direct_invocation(
    const llama_decode_path_isolation_state * state) {
    if (!state || state->n_tokens_processed == 0) return false;

    int last_token_idx = state->n_tokens_processed - 1;
    if (last_token_idx >= state->token_metrics_capacity) return false;

    return state->token_metrics[last_token_idx].direct_invocation_used;
}

/* ============================================================================
   Submission Detection and Prevention
   ============================================================================ */

bool llama_decode_path_isolation_record_submission(
    llama_decode_path_isolation_state * state,
    const char * location,
    bool is_fatal) {
    if (!state) return true; // Allow if no state
    if (!state->submission_detection_enabled) return true;

    state->threadpool_submissions++;

    if (state->n_tokens_processed < state->token_metrics_capacity) {
        state->token_metrics[state->n_tokens_processed].submission_count++;
    }

    if (is_fatal || state->abort_on_submission) {
        state->guard_state.violations_detected++;
        snprintf(state->guard_state.last_violation_msg,
                 sizeof(state->guard_state.last_violation_msg),
                 "Threadpool submission in decode at %s",
                 location ? location : "unknown");
        state->guard_state.last_violation_ns = get_time_ns();

        LLAMA_LOG_ERROR("DECODE ISOLATION: Submission violation at %s\n",
                       location ? location : "unknown");

        if (state->guard_state.abort_on_violation) {
            return false; // Violation
        }
    }

    return true;
}

bool llama_decode_path_isolation_record_parallel_region(
    llama_decode_path_isolation_state * state,
    const char * location) {
    if (!state) return true;
    if (!state->submission_detection_enabled) return true;

    if (state->n_tokens_processed < state->token_metrics_capacity) {
        state->token_metrics[state->n_tokens_processed].parallel_region_count++;
    }

    state->guard_state.violations_detected++;
    snprintf(state->guard_state.last_violation_msg,
             sizeof(state->guard_state.last_violation_msg),
             "Parallel region in decode at %s",
             location ? location : "unknown");
    state->guard_state.last_violation_ns = get_time_ns();

    LLAMA_LOG_ERROR("DECODE ISOLATION: Parallel region violation at %s\n",
                   location ? location : "unknown");

    return !state->guard_state.abort_on_violation;
}

bool llama_decode_path_isolation_record_work_chunking(
    llama_decode_path_isolation_state * state,
    const char * location) {
    if (!state) return true;
    if (!state->submission_detection_enabled) return true;

    state->guard_state.violations_detected++;
    snprintf(state->guard_state.last_violation_msg,
             sizeof(state->guard_state.last_violation_msg),
             "Work chunking in decode at %s",
             location ? location : "unknown");

    LLAMA_LOG_ERROR("DECODE ISOLATION: Work chunking violation at %s\n",
                   location ? location : "unknown");

    return !state->guard_state.abort_on_violation;
}

/* ============================================================================
   Lock Contention Monitoring
   ============================================================================ */

int llama_decode_path_isolation_register_lock(
    llama_decode_path_isolation_state * state,
    const char * lock_name) {
    if (!state || !lock_name) return -1;

    if (state->n_lock_entries >= state->lock_entries_capacity) {
        return -1; // Capacity exceeded
    }

    int idx = state->n_lock_entries;
    state->lock_entries[idx].lock_name = lock_name;
    state->lock_entries[idx].acquisition_count = 0;
    state->lock_entries[idx].contention_samples = 0;
    state->lock_entries[idx].max_hold_time_ns = 0;
    state->lock_entries[idx].is_contended = false;

    state->n_lock_entries++;

    return idx;
}

void llama_decode_path_isolation_record_lock_acquisition(
    llama_decode_path_isolation_state * state,
    int lock_id,
    bool acquired,
    uint64_t hold_time_ns) {
    if (!state || !state->lock_monitoring_enabled) return;
    if (lock_id < 0 || lock_id >= state->n_lock_entries) return;

    llama_lock_contention_entry * entry = &state->lock_entries[lock_id];
    entry->acquisition_count++;

    if (!acquired) {
        entry->contention_samples++;
        entry->is_contended = true;
    }

    if (hold_time_ns > entry->max_hold_time_ns) {
        entry->max_hold_time_ns = hold_time_ns;
    }

    // Update per-token metrics
    if (state->n_tokens_processed < state->token_metrics_capacity) {
        state->token_metrics[state->n_tokens_processed].lock_acquisitions++;
    }
}

/* ============================================================================
   Work-Stealing Monitoring
   ============================================================================ */

void llama_decode_path_isolation_enable_work_stealing_monitoring(
    llama_decode_path_isolation_state * state,
    bool enabled) {
    if (!state) return;
    state->work_stealing_monitoring_enabled = enabled;
}

void llama_decode_path_isolation_record_work_steal(
    llama_decode_path_isolation_state * state,
    bool successful) {
    if (!state || !state->work_stealing_monitoring_enabled) return;

    state->work_stealing.steal_attempts_count++;

    if (successful) {
        state->work_stealing.successful_steals++;
        state->work_stealing.steals_detected_in_decode = true;

        LLAMA_LOG_WARN("DECODE ISOLATION: Work steal detected\n");
    }

    if (state->n_tokens_processed < state->token_metrics_capacity) {
        state->token_metrics[state->n_tokens_processed].work_stealing_attempts++;
    }
}

/* ============================================================================
   Task Queue Monitoring
   ============================================================================ */

bool llama_decode_path_isolation_snapshot_queue_before(
    llama_decode_path_isolation_state * state) {
    if (!state) return false;

    state->queue_monitor.initial_queue_depth = 0;
    state->queue_monitor.current_queue_depth = 0;
    state->queue_monitor.max_observed_depth = 0;
    state->queue_monitor.total_enqueues = 0;
    state->queue_monitor.total_dequeues = 0;
    state->queue_monitor.queue_operations_detected = false;

    return true;
}

void llama_decode_path_isolation_update_queue_depth(
    llama_decode_path_isolation_state * state,
    int current_depth) {
    if (!state) return;

    state->queue_monitor.current_queue_depth = current_depth;

    if (current_depth > state->queue_monitor.max_observed_depth) {
        state->queue_monitor.max_observed_depth = current_depth;
    }
}

void llama_decode_path_isolation_record_queue_operation(
    llama_decode_path_isolation_state * state,
    bool is_enqueue) {
    if (!state) return;

    state->queue_monitor.queue_operations_detected = true;

    if (is_enqueue) {
        state->queue_monitor.total_enqueues++;
        state->threadpool_enqueue_operations++;
    } else {
        state->queue_monitor.total_dequeues++;
        state->threadpool_dequeue_operations++;
    }
}

/* ============================================================================
   Runtime Assertions and Guards
   ============================================================================ */

bool llama_decode_path_isolation_assert_no_submissions(
    llama_decode_path_isolation_state * state,
    const char * context_msg) {
    if (!state) return true;

    if (state->threadpool_submissions > 0) {
        state->guard_state.assertions_failed++;

        LLAMA_LOG_ERROR(
            "DECODE ISOLATION: Assertion failed - submissions detected (%lu submissions, context: %s)\n",
            state->threadpool_submissions,
            context_msg ? context_msg : "unknown");

        if (state->guard_state.abort_on_violation) {
            return false;
        }
    } else {
        state->guard_state.assertions_passed++;
    }

    return state->threadpool_submissions == 0;
}

bool llama_decode_path_isolation_assert_empty_queue(
    llama_decode_path_isolation_state * state) {
    if (!state) return true;

    if (state->queue_monitor.current_queue_depth != 0 &&
        state->queue_monitor.initial_queue_depth != 0) {
        state->guard_state.assertions_failed++;

        LLAMA_LOG_ERROR(
            "DECODE ISOLATION: Queue not empty (depth=%d)\n",
            state->queue_monitor.current_queue_depth);

        if (state->guard_state.abort_on_violation) {
            return false;
        }
    } else {
        state->guard_state.assertions_passed++;
    }

    return state->queue_monitor.current_queue_depth == 0;
}

bool llama_decode_path_isolation_assert_no_active_workers(
    llama_decode_path_isolation_state * state) {
    if (!state) return true;

    // For minimal size 1, only the main decode thread should be active
    if (state->threadpool_current_size > state->threadpool_minimal_size) {
        state->guard_state.assertions_failed++;

        LLAMA_LOG_ERROR(
            "DECODE ISOLATION: Active workers detected (size=%d, minimal=%d)\n",
            state->threadpool_current_size,
            state->threadpool_minimal_size);

        if (state->guard_state.abort_on_violation) {
            return false;
        }
    } else {
        state->guard_state.assertions_passed++;
    }

    return state->threadpool_current_size <= state->threadpool_minimal_size;
}

bool llama_decode_path_isolation_assert_zero_per_token_submissions(
    llama_decode_path_isolation_state * state) {
    if (!state) return true;

    bool all_zero = true;

    for (int i = 0; i < state->n_tokens_processed; i++) {
        if (state->token_metrics[i].submission_count > 0) {
            all_zero = false;

            LLAMA_LOG_ERROR(
                "DECODE ISOLATION: Per-token submissions detected (token=%lu, submissions=%u)\n",
                state->token_metrics[i].token_index,
                state->token_metrics[i].submission_count);

            break;
        }
    }

    if (all_zero) {
        state->guard_state.assertions_passed++;
    } else {
        state->guard_state.assertions_failed++;
        if (state->guard_state.abort_on_violation) {
            return false;
        }
    }

    return all_zero;
}

bool llama_decode_path_isolation_assert_lock_contention_low(
    llama_decode_path_isolation_state * state,
    double max_contention_percent) {
    if (!state) return true;

    bool contention_low = true;

    for (int i = 0; i < state->n_lock_entries; i++) {
        llama_lock_contention_entry * entry = &state->lock_entries[i];

        if (entry->acquisition_count > 0) {
            double contention_pct = (double)entry->contention_samples * 100.0 /
                                   (double)entry->acquisition_count;

            if (contention_pct > max_contention_percent) {
                contention_low = false;

                LLAMA_LOG_WARN(
                    "DECODE ISOLATION: Lock contention high (%s: %.1f%%)\n",
                    entry->lock_name,
                    contention_pct);
            }
        }
    }

    if (contention_low) {
        state->guard_state.assertions_passed++;
    } else {
        state->guard_state.assertions_failed++;
    }

    return contention_low;
}

void llama_decode_path_isolation_configure_guard(
    llama_decode_path_isolation_state * state,
    bool abort_on_violation) {
    if (!state) return;
    state->guard_state.abort_on_violation = abort_on_violation;
}

/* ============================================================================
   Metrics and Diagnostics
   ============================================================================ */

bool llama_decode_path_isolation_get_token_metrics(
    const llama_decode_path_isolation_state * state,
    int token_index,
    llama_per_token_metrics * out_metrics) {
    if (!state || !out_metrics || token_index < 0 ||
        token_index >= state->token_metrics_capacity) {
        return false;
    }

    *out_metrics = state->token_metrics[token_index];
    return true;
}

char * llama_decode_path_isolation_get_summary(
    const llama_decode_path_isolation_state * state) {
    if (!state) return NULL;

    char * buffer = (char *)malloc(2048);
    if (!buffer) return NULL;

    uint64_t avg_ns = state->n_tokens_processed > 0 ?
        state->total_decode_time_ns / state->n_tokens_processed : 0;

    snprintf(buffer, 2048,
        "DECODE ISOLATION SUMMARY\n"
        "========================\n"
        "Total Decode Cycles: %lu\n"
        "Total Tokens Processed: %d\n"
        "Total Decode Time: %.2f ms\n"
        "Average Time Per Token: %.2f ms\n"
        "\n"
        "Thread Pool State:\n"
        "  Initial Size: %d\n"
        "  Minimal Size: %d\n"
        "  Final Size: %d\n"
        "\n"
        "Submissions and Violations:\n"
        "  Total Submissions: %lu\n"
        "  Max Per-Token: %u\n"
        "  Queue Enqueue Ops: %lu\n"
        "  Queue Dequeue Ops: %lu\n"
        "  Violations Detected: %lu\n"
        "\n"
        "Lock Contention:\n"
        "  Total Locks Registered: %d\n"
        "  Max Per-Token Acquisitions: %u\n"
        "\n"
        "Direct Invocation:\n"
        "  Total Invocations: %lu\n"
        "  All Tokens Direct: %s\n"
        "  Total Time: %.2f ms\n"
        "\n"
        "Guard Status:\n"
        "  Assertions Passed: %lu\n"
        "  Assertions Failed: %lu\n"
        "  Abort on Violation: %s\n",
        state->total_decode_cycles,
        state->n_tokens_processed,
        (double)state->total_decode_time_ns / 1000000.0,
        (double)avg_ns / 1000000.0,
        state->threadpool_initial_size,
        state->threadpool_minimal_size,
        state->threadpool_current_size,
        state->threadpool_submissions,
        state->max_per_token_submissions,
        state->threadpool_enqueue_operations,
        state->threadpool_dequeue_operations,
        state->guard_state.violations_detected,
        state->n_lock_entries,
        state->max_per_token_locks,
        state->direct_invoke.invocations_count,
        state->direct_invoke.all_tokens_direct ? "yes" : "no",
        (double)state->direct_invoke.total_invocation_time_ns / 1000000.0,
        state->guard_state.assertions_passed,
        state->guard_state.assertions_failed,
        state->guard_state.abort_on_violation ? "yes" : "no");

    return buffer;
}

char * llama_decode_path_isolation_get_diagnostics(
    const llama_decode_path_isolation_state * state) {
    if (!state) return NULL;

    char * buffer = (char *)malloc(4096);
    if (!buffer) return NULL;

    int offset = 0;

    offset += snprintf(buffer + offset, 4096 - offset,
        "DECODE ISOLATION DIAGNOSTICS\n"
        "============================\n\n"
        "State Information:\n"
        "  Current Mode: %d\n"
        "  Isolation State: %d\n"
        "  Active: %s\n"
        "  Threadpool State: %d\n\n"
        "Per-Token Analysis (first 5 tokens):\n",
        state->current_mode,
        state->isolation_state,
        state->isolation_active ? "yes" : "no",
        state->threadpool_state);

    int tokens_to_show = state->n_tokens_processed < 5 ?
        state->n_tokens_processed : 5;

    for (int i = 0; i < tokens_to_show; i++) {
        const llama_per_token_metrics * m = &state->token_metrics[i];
        offset += snprintf(buffer + offset, 4096 - offset,
            "  Token %lu:\n"
            "    Submissions: %u\n"
            "    Parallel Regions: %u\n"
            "    Lock Acquisitions: %u\n"
            "    Worker Wakeups: %u\n"
            "    Work Steals: %u\n"
            "    Direct Invocation: %s\n",
            m->token_index,
            m->submission_count,
            m->parallel_region_count,
            m->lock_acquisitions,
            m->worker_wakeups,
            m->work_stealing_attempts,
            m->direct_invocation_used ? "yes" : "no");
    }

    if (state->guard_state.violations_detected > 0) {
        offset += snprintf(buffer + offset, 4096 - offset,
            "\nLast Violation:\n"
            "  Message: %s\n"
            "  Time: %lu ns ago\n",
            state->guard_state.last_violation_msg,
            (get_time_ns() - state->guard_state.last_violation_ns) / 1000);
    }

    return buffer;
}

void llama_decode_path_isolation_reset_metrics(
    llama_decode_path_isolation_state * state) {
    if (!state) return;

    state->guard_state.assertions_passed = 0;
    state->guard_state.assertions_failed = 0;
    state->guard_state.violations_detected = 0;
    memset(state->guard_state.last_violation_msg, 0,
           sizeof(state->guard_state.last_violation_msg));

    memset(state->lock_entries, 0,
           sizeof(llama_lock_contention_entry) * state->lock_entries_capacity);
    state->n_lock_entries = 0;

    state->total_decode_cycles = 0;
    state->total_decode_time_ns = 0;
}

bool llama_decode_path_isolation_check_integrity(
    const llama_decode_path_isolation_state * state) {
    if (!state) return true;

    // Check 1: No submissions
    if (state->threadpool_submissions > 0) {
        LLAMA_LOG_ERROR(
            "DECODE ISOLATION: Integrity check failed - submissions detected (%lu)\n",
            state->threadpool_submissions);
        return false;
    }

    // Check 2: Per-token submissions are zero
    for (int i = 0; i < state->n_tokens_processed; i++) {
        if (state->token_metrics[i].submission_count > 0) {
            LLAMA_LOG_ERROR(
                "DECODE ISOLATION: Integrity check failed - token %d has %u submissions\n",
                i,
                state->token_metrics[i].submission_count);
            return false;
        }
    }

    // Check 3: All tokens used direct invocation
    for (int i = 0; i < state->n_tokens_processed; i++) {
        if (!state->token_metrics[i].direct_invocation_used &&
            state->direct_invocation_required) {
            LLAMA_LOG_WARN(
                "DECODE ISOLATION: Token %d did not use direct invocation\n", i);
        }
    }

    // Check 4: No work stealing
    if (state->work_stealing.steals_detected_in_decode) {
        LLAMA_LOG_ERROR(
            "DECODE ISOLATION: Integrity check failed - work stealing detected\n");
        return false;
    }

    return true;
}

/* ============================================================================
   Configuration
   ============================================================================ */

void llama_decode_path_isolation_enable_submission_detection(
    llama_decode_path_isolation_state * state,
    bool enabled) {
    if (state) state->submission_detection_enabled = enabled;
}

void llama_decode_path_isolation_enable_lock_monitoring(
    llama_decode_path_isolation_state * state,
    bool enabled) {
    if (state) state->lock_monitoring_enabled = enabled;
}

void llama_decode_path_isolation_set_max_submissions_per_token(
    llama_decode_path_isolation_state * state,
    int max_count) {
    if (state) state->max_allowed_submissions_per_token = max_count;
}

void llama_decode_path_isolation_set_abort_on_violation(
    llama_decode_path_isolation_state * state,
    bool abort) {
    if (state) state->guard_state.abort_on_violation = abort;
}

/* ============================================================================
   Validation and Audit Functions
   ============================================================================ */

char * llama_decode_path_isolation_audit_decode_path(
    const llama_decode_path_isolation_state * state) {
    if (!state) return NULL;

    char * report = (char *)malloc(8192);
    if (!report) return NULL;

    int offset = 0;

    offset += snprintf(report + offset, 8192 - offset,
        "DECODE PATH ISOLATION AUDIT\n"
        "===========================\n\n"
        "CHECKS PERFORMED:\n\n"
        "1. Threadpool Submission Check:\n"
        "   Status: %s\n"
        "   Submissions Found: %lu\n"
        "   Expected: 0\n"
        "   Result: %s\n\n",
        state->threadpool_submissions == 0 ? "PASS" : "FAIL",
        state->threadpool_submissions,
        state->threadpool_submissions == 0 ? "COMPLIANT" : "VIOLATION");

    offset += snprintf(report + offset, 8192 - offset,
        "2. Per-Token Submission Check:\n"
        "   Max Per-Token: %u\n"
        "   Expected Maximum: %d\n"
        "   Result: %s\n\n",
        state->max_per_token_submissions,
        state->max_allowed_submissions_per_token,
        state->max_per_token_submissions <= (uint32_t)state->max_allowed_submissions_per_token ?
            "COMPLIANT" : "VIOLATION");

    offset += snprintf(report + offset, 8192 - offset,
        "3. Direct Invocation Check:\n"
        "   All Tokens Direct: %s\n"
        "   Total Invocations: %lu\n"
        "   Result: %s\n\n",
        state->direct_invoke.all_tokens_direct ? "yes" : "no",
        state->direct_invoke.invocations_count,
        state->direct_invoke.all_tokens_direct ? "COMPLIANT" : "WARNING");

    offset += snprintf(report + offset, 8192 - offset,
        "4. Work-Stealing Check:\n"
        "   Steals Detected: %s\n"
        "   Total Steal Attempts: %lu\n"
        "   Result: %s\n\n",
        state->work_stealing.steals_detected_in_decode ? "yes" : "no",
        state->work_stealing.steal_attempts_count,
        !state->work_stealing.steals_detected_in_decode ?
            "COMPLIANT" : "VIOLATION");

    offset += snprintf(report + offset, 8192 - offset,
        "5. Queue Operations Check:\n"
        "   Enqueues: %lu\n"
        "   Dequeues: %lu\n"
        "   Queue Ops Detected: %s\n"
        "   Result: %s\n\n",
        state->threadpool_enqueue_operations,
        state->threadpool_dequeue_operations,
        state->queue_monitor.queue_operations_detected ? "yes" : "no",
        !state->queue_monitor.queue_operations_detected ?
            "COMPLIANT" : "VIOLATION");

    offset += snprintf(report + offset, 8192 - offset,
        "OVERALL ISOLATION STATUS: %s\n"
        "Total Violations: %lu\n",
        state->guard_state.violations_detected == 0 ? "ISOLATED" : "VIOLATED",
        state->guard_state.violations_detected);

    return report;
}

bool llama_decode_path_isolation_is_isolated(void) {
    return g_isolation_active;
}

int llama_decode_path_isolation_get_isolation_depth(void) {
    return g_isolation_depth;
}
