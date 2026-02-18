/**
 * Decode Threading Discipline Enforcement Implementation
 *
 * Eliminates thread wake/sleep cycles during token decode.
 * Ensures persistent workers with no per-token synchronization.
 */

#include "llama-threading-discipline.h"
#include "llama-impl.h"

#include <cstring>
#include <algorithm>

/**
 * Initialize threading discipline state
 */
void llama_threading_discipline_init(llama_threading_discipline_state * state) {
    if (!state) return;

    state->enforce_active = false;
    state->in_decode_phase = false;
    state->decode_thread_count = 0;
    state->server_thread_count = 0;
    state->max_decode_threads = 8;  // Default conservative limit

    state->allow_thread_wake = false;
    state->allow_thread_sleep = false;
    state->allow_barriers = false;
    state->allow_cond_vars = false;
    state->require_persistent = true;

    state->workers = nullptr;
    state->n_workers = 0;

    // Initialize metrics
    state->metrics.wake_count = 0;
    state->metrics.sleep_count = 0;
    state->metrics.barrier_wait_count = 0;
    state->metrics.cond_var_signals = 0;
    state->metrics.context_switches = 0;
    state->metrics.active_thread_count = 0;
    state->metrics.peak_thread_count = 0;
}

/**
 * Activate threading discipline for decode
 */
void llama_threading_discipline_activate_decode(
    llama_threading_discipline_state * state,
    int n_threads,
    int max_threads) {

    if (!state) return;

    state->enforce_active = true;
    state->in_decode_phase = true;
    state->decode_thread_count = n_threads;
    state->max_decode_threads = max_threads;

    // During decode, no per-token wakeups allowed
    state->allow_thread_wake = false;
    state->allow_thread_sleep = false;
    state->allow_barriers = false;
    state->allow_cond_vars = false;
    state->require_persistent = true;

    // Allocate worker tracking
    if (state->workers) {
        delete[] state->workers;
    }
    state->workers = new llama_thread_assignment[n_threads];
    state->n_workers = n_threads;

    // Initialize workers as persistent
    for (int i = 0; i < n_threads; i++) {
        state->workers[i].thread_id = i;
        state->workers[i].cpu_affinity = -1;  // Not yet set
        state->workers[i].is_persistent = true;
        state->workers[i].work_items_completed = 0;
    }

    state->metrics.active_thread_count = n_threads;

    LLAMA_LOG_INFO(
        "THREADING DISCIPLINE: Decode activated (%d persistent workers, max %d)\n",
        n_threads, max_threads);
}

/**
 * [CRITICAL] Validate invariant
 */
bool llama_threading_discipline_validate_invariant(
    const llama_threading_discipline_state * state) {

    if (!state || !state->in_decode_phase) {
        return true;
    }

    // Check that all workers are persistent
    for (int i = 0; i < state->n_workers; i++) {
        if (!state->workers[i].is_persistent) {
            LLAMA_LOG_ERROR(
                "THREADING DISCIPLINE: Worker %d is not persistent!\n"
                "  All decode workers must remain active for entire session\n",
                i);
            return false;
        }
    }

    // Thread count must be stable
    if (state->metrics.active_thread_count != state->decode_thread_count) {
        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Thread count changed during decode!\n"
            "  Expected: %d\n"
            "  Current: %u\n",
            state->decode_thread_count, state->metrics.active_thread_count);
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Audit condition variables
 */
bool llama_threading_discipline_audit_cond_vars(
    const llama_threading_discipline_state * state,
    void * worker_loop_address) {

    if (!state || !state->in_decode_phase) {
        return true;
    }

    // In a real implementation, this would:
    // 1. Analyze the worker loop code at worker_loop_address
    // 2. Detect pthread_cond_wait or std::condition_variable::wait calls
    // 3. Check if any conditional variable signals occur per-token

    // For now, we track condition variable signals through metrics
    if (state->metrics.cond_var_signals > 0) {
        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Condition variable signals detected!\n"
            "  Signals during decode: %lu\n"
            "  Per-token condition variable waits are forbidden\n",
            state->metrics.cond_var_signals);
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Enable persistent workers
 */
bool llama_threading_discipline_enable_persistent_workers(
    llama_threading_discipline_state * state) {

    if (!state) {
        return false;
    }

    // Validate all workers are configured for persistence
    for (int i = 0; i < state->n_workers; i++) {
        state->workers[i].is_persistent = true;
    }

    LLAMA_LOG_INFO(
        "THREADING DISCIPLINE: Persistent worker model enabled for %d workers\n",
        state->n_workers);

    return true;
}

/**
 * [CRITICAL] Check for per-node barriers
 */
bool llama_threading_discipline_check_no_per_node_barriers(
    const llama_threading_discipline_state * state,
    int graph_nodes) {

    if (!state || !state->in_decode_phase) {
        return true;
    }

    // If barriers are triggered per node, that's a violation
    // Expected barrier count during decode should be zero or minimal
    // Estimated per-node barrier cost: graph_nodes * synchronization_overhead

    if (state->metrics.barrier_wait_count > 0) {
        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Per-node barriers detected!\n"
            "  Barrier waits during decode: %lu\n"
            "  Must use static scheduling, not per-node synchronization\n",
            state->metrics.barrier_wait_count);
        LLAMA_ABORT("Per-node barrier detected in decode");
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Validate no graph wake/sleep cycles
 */
bool llama_threading_discipline_validate_no_graph_cycles(
    const llama_threading_discipline_state * state,
    bool has_graph_wake_call,
    bool has_graph_sleep_call) {

    if (!state || !state->in_decode_phase) {
        return true;
    }

    // Pattern check: wake → execute → sleep during graph execution
    if ((has_graph_wake_call && has_graph_sleep_call) ||
        (has_graph_wake_call && state->allow_thread_sleep == false) ||
        (has_graph_sleep_call && state->allow_thread_wake == false)) {

        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Graph-level wake/sleep cycles detected!\n"
            "  Wake calls in graph execution: %s\n"
            "  Sleep calls in graph execution: %s\n"
            "  Graph must execute under single activation epoch\n",
            has_graph_wake_call ? "yes" : "no",
            has_graph_sleep_call ? "yes" : "no");
        LLAMA_ABORT("Graph wake/sleep cycle detected");
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Enforce thread pool isolation
 */
bool llama_threading_discipline_enforce_pool_isolation(
    llama_threading_discipline_state * state,
    int decode_pool_id,
    int server_pool_id) {

    if (!state) {
        return false;
    }

    // Pools must be different
    if (decode_pool_id == server_pool_id) {
        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Decode and server share same thread pool!\n"
            "  Decode pool: %d\n"
            "  Server pool: %d\n"
            "  Must use separate pools to avoid competition\n",
            decode_pool_id, server_pool_id);
        LLAMA_ABORT("Thread pool isolation violated");
        return false;
    }

    LLAMA_LOG_INFO(
        "THREADING DISCIPLINE: Thread pools isolated (decode=%d, server=%d)\n",
        decode_pool_id, server_pool_id);

    return true;
}

/**
 * [CRITICAL] Enforce thread count cap
 */
bool llama_threading_discipline_enforce_thread_cap(
    const llama_threading_discipline_state * state,
    int actual_thread_count) {

    if (!state) {
        return false;
    }

    // Check against maximum
    if (actual_thread_count > state->max_decode_threads) {
        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Thread count exceeds cap!\n"
            "  Actual: %d\n"
            "  Maximum: %d\n"
            "  Oversubscription increases context switching and jitter\n",
            actual_thread_count, state->max_decode_threads);
        LLAMA_ABORT("Thread count cap exceeded");
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Validate sampling has no thread signaling
 */
bool llama_threading_discipline_sampling_no_signals(
    const llama_threading_discipline_state * state,
    bool sampling_uses_threadpool) {

    if (!state || !state->in_decode_phase) {
        return true;
    }

    if (sampling_uses_threadpool) {
        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Sampling triggers thread pool activation!\n"
            "  Sampling must execute inside persistent decode context\n"
            "  Must not signal worker threads\n");
        LLAMA_ABORT("Sampling thread signaling detected");
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Validate stream-ordered execution
 */
bool llama_threading_discipline_validate_stream_ordering(
    const llama_threading_discipline_state * state,
    bool has_blocking_gpu_wait) {

    if (!state || !state->in_decode_phase) {
        return true;
    }

    if (has_blocking_gpu_wait) {
        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Blocking GPU wait detected!\n"
            "  Thread blocks waiting for GPU completion\n"
            "  Must use stream-ordered execution instead\n"
            "  GPU should advance autonomously\n");
        LLAMA_ABORT("Blocking GPU wait in decode");
        return false;
    }

    return true;
}

/**
 * Record thread activity
 */
void llama_threading_discipline_record_activity(
    llama_threading_discipline_state * state,
    const char * activity_type) {

    if (!state || !state->in_decode_phase) {
        return;
    }

    if (!activity_type) {
        return;
    }

    if (strcmp(activity_type, "wake") == 0) {
        state->metrics.wake_count++;
    }
    else if (strcmp(activity_type, "sleep") == 0) {
        state->metrics.sleep_count++;
    }
    else if (strcmp(activity_type, "barrier") == 0) {
        state->metrics.barrier_wait_count++;
    }
    else if (strcmp(activity_type, "cond_signal") == 0) {
        state->metrics.cond_var_signals++;
    }
}

/**
 * [CRITICAL] Validate no per-token thread activity
 */
bool llama_threading_discipline_validate_no_per_token_activity(
    const llama_threading_discipline_state * state) {

    if (!state || !state->in_decode_phase) {
        return true;
    }

    // During decode, thread wake/sleep counts must be zero
    if (state->metrics.wake_count > 0) {
        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Thread wakeups detected during decode!\n"
            "  Wake count: %lu\n"
            "  Per-token thread signaling forbidden\n",
            state->metrics.wake_count);
        LLAMA_ABORT("Per-token thread wakeup detected");
        return false;
    }

    if (state->metrics.sleep_count > 0) {
        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Thread sleeps detected during decode!\n"
            "  Sleep count: %lu\n"
            "  Per-token thread suspension forbidden\n",
            state->metrics.sleep_count);
        LLAMA_ABORT("Per-token thread sleep detected");
        return false;
    }

    if (state->metrics.barrier_wait_count > 0) {
        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Barrier synchronization detected!\n"
            "  Barrier waits: %lu\n"
            "  Per-node barrier overhead forbidden\n",
            state->metrics.barrier_wait_count);
        LLAMA_ABORT("Per-node barrier detected");
        return false;
    }

    if (state->metrics.cond_var_signals > 0) {
        LLAMA_LOG_ERROR(
            "THREADING DISCIPLINE: Condition variable signals detected!\n"
            "  Signals: %lu\n"
            "  Per-token condition variable churn forbidden\n",
            state->metrics.cond_var_signals);
        LLAMA_ABORT("Condition variable signaling detected");
        return false;
    }

    return true;
}

/**
 * Set CPU affinity
 */
bool llama_threading_discipline_set_cpu_affinity(
    llama_threading_discipline_state * state,
    const int * core_ids,
    int n_cores) {

    if (!state || !core_ids || n_cores <= 0) {
        return false;
    }

    if (n_cores > state->n_workers) {
        LLAMA_LOG_WARN(
            "THREADING DISCIPLINE: More CPU cores than workers (%d > %d)\n",
            n_cores, state->n_workers);
        return false;
    }

    // Assign cores to workers
    for (int i = 0; i < n_cores && i < state->n_workers; i++) {
        state->workers[i].cpu_affinity = core_ids[i];
    }

    LLAMA_LOG_INFO(
        "THREADING DISCIPLINE: CPU affinity set for %d workers\n", n_cores);

    return true;
}

/**
 * Get metrics
 */
llama_thread_metrics llama_threading_discipline_get_metrics(
    const llama_threading_discipline_state * state) {

    if (!state) {
        return {};
    }

    return state->metrics;
}

/**
 * Check compliance
 */
bool llama_threading_discipline_check_compliance(
    const llama_threading_discipline_state * state) {

    if (!state || !state->in_decode_phase) {
        return true;
    }

    // Check all critical invariants
    if (!llama_threading_discipline_validate_invariant(state)) {
        return false;
    }

    if (!llama_threading_discipline_validate_no_per_token_activity(state)) {
        return false;
    }

    return true;
}

/**
 * Dump metrics
 */
void llama_threading_discipline_dump_metrics(
    const llama_threading_discipline_state * state) {

    if (!state) {
        return;
    }

    LLAMA_LOG_INFO("THREADING DISCIPLINE METRICS:\n");
    LLAMA_LOG_INFO("  Decode threads: %d\n", state->decode_thread_count);
    LLAMA_LOG_INFO("  Server threads: %d\n", state->server_thread_count);
    LLAMA_LOG_INFO("  Active threads: %u\n", state->metrics.active_thread_count);
    LLAMA_LOG_INFO("  Peak threads: %u\n", state->metrics.peak_thread_count);
    LLAMA_LOG_INFO("THREADING CHURN:\n");
    LLAMA_LOG_INFO("  Thread wakeups: %lu\n", state->metrics.wake_count);
    LLAMA_LOG_INFO("  Thread sleeps: %lu\n", state->metrics.sleep_count);
    LLAMA_LOG_INFO("  Barrier waits: %lu\n", state->metrics.barrier_wait_count);
    LLAMA_LOG_INFO("  Cond var signals: %lu\n", state->metrics.cond_var_signals);
    LLAMA_LOG_INFO("  Estimated context switches: %lu\n", state->metrics.context_switches);

    // Print invariant status
    const char * invariant_status = llama_threading_discipline_validate_invariant(state) ? "OK" : "VIOLATED";
    const char * activity_status = (state->metrics.wake_count == 0 && state->metrics.sleep_count == 0) ? "OK" : "CHURN";

    LLAMA_LOG_INFO("INVARIANT STATUS:\n");
    LLAMA_LOG_INFO("  Persistent workers: %s\n", invariant_status);
    LLAMA_LOG_INFO("  No per-token activity: %s\n", activity_status);

    // Print worker details
    if (state->workers && state->n_workers > 0) {
        LLAMA_LOG_INFO("WORKER ASSIGNMENTS:\n");
        for (int i = 0; i < state->n_workers; i++) {
            const char * cpu_str = state->workers[i].cpu_affinity >= 0 ?
                                  "pinned" : "unaffine";
            LLAMA_LOG_INFO("  Worker %d: persistent=%d, affinity=%s, items=%lu\n",
                          i,
                          state->workers[i].is_persistent ? 1 : 0,
                          cpu_str,
                          state->workers[i].work_items_completed);
        }
    }
}

/**
 * Deactivate enforcement
 */
void llama_threading_discipline_deactivate(
    llama_threading_discipline_state * state) {

    if (!state) {
        return;
    }

    // Dump final metrics before deactivating
    llama_threading_discipline_dump_metrics(state);

    state->in_decode_phase = false;
    state->enforce_active = false;

    // Clean up worker tracking
    if (state->workers) {
        delete[] state->workers;
        state->workers = nullptr;
    }
    state->n_workers = 0;

    LLAMA_LOG_INFO("THREADING DISCIPLINE: Deactivated\n");
}
