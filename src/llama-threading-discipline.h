#pragma once

/**
 * Decode Threading Discipline Enforcement
 *
 * Eliminates thread wake/sleep cycles during token-by-token decode.
 * Ensures persistent workers with no per-token signaling or synchronization.
 */

#include <cstdint>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Thread activity metrics
 */
typedef struct {
    uint64_t wake_count;           ///< Number of worker thread wakeups
    uint64_t sleep_count;          ///< Number of worker thread sleeps
    uint64_t barrier_wait_count;   ///< Number of barrier synchronizations
    uint64_t cond_var_signals;     ///< Condition variable signal count
    uint64_t context_switches;     ///< Context switch count (estimated)
    uint32_t active_thread_count;  ///< Currently active worker threads
    uint32_t peak_thread_count;    ///< Peak thread count during decode
} llama_thread_metrics;

/**
 * Thread work assignment
 */
typedef struct {
    int thread_id;                 ///< Worker thread ID
    int cpu_affinity;              ///< CPU core affinity (-1 = not set)
    bool is_persistent;            ///< Whether thread stays active
    uint64_t work_items_completed; ///< Work items processed
} llama_thread_assignment;

/**
 * Threading discipline enforcement state
 */
typedef struct {
    bool             enforce_active;           ///< Enforcement enabled
    bool             in_decode_phase;          ///< Currently in decode
    int              decode_thread_count;      ///< Threads allocated for decode
    int              server_thread_count;      ///< Threads for server/HTTP
    int              max_decode_threads;       ///< Maximum allowed decode threads

    // Invariant tracking
    bool             allow_thread_wake;        ///< Whether thread wakes permitted
    bool             allow_thread_sleep;       ///< Whether thread sleeps permitted
    bool             allow_barriers;           ///< Whether barriers permitted
    bool             allow_cond_vars;          ///< Whether condition vars permitted
    bool             require_persistent;       ///< Require persistent workers

    // Worker state
    llama_thread_assignment * workers;         ///< Worker thread assignments
    int              n_workers;                ///< Number of workers

    // Metrics
    llama_thread_metrics metrics;
} llama_threading_discipline_state;

/**
 * Initialize threading discipline enforcement
 */
void llama_threading_discipline_init(llama_threading_discipline_state * state);

/**
 * Activate threading discipline for decode
 *
 * Enables enforcement and configures for decode phase.
 * Sets up persistent worker threads.
 *
 * @param state Discipline enforcement state
 * @param n_threads Number of decode worker threads
 * @param max_threads Maximum allowed threads
 */
void llama_threading_discipline_activate_decode(
    llama_threading_discipline_state * state,
    int n_threads,
    int max_threads);

/**
 * [CRITICAL] Establish the invariant
 *
 * No worker thread may be put to sleep/woken per token.
 * Validates that thread set is stable and persistent.
 *
 * @param state Discipline state
 * @return true if invariant maintained, false if violations detected
 */
bool llama_threading_discipline_validate_invariant(
    const llama_threading_discipline_state * state);

/**
 * [CRITICAL] Audit thread pool condition variable usage
 *
 * Detects per-token signaling on condition variables.
 * Scans worker loop for pthread_cond_wait/std::condition_variable::wait calls.
 * Aborts if per-token wakeups detected.
 *
 * @param state Discipline state
 * @param worker_loop_address Address of worker loop code
 * @return true if no per-token signaling, false if violations
 */
bool llama_threading_discipline_audit_cond_vars(
    const llama_threading_discipline_state * state,
    void * worker_loop_address);

/**
 * [CRITICAL] Convert workers to persistent spin model
 *
 * Configures decode workers for persistent operation:
 * - Launched once at decode start
 * - Kept active for entire session
 * - Use bounded spin-wait or cooperative loop
 * - No per-token thread suspension
 *
 * @param state Discipline state
 * @return true if persistent model configured, false otherwise
 */
bool llama_threading_discipline_enable_persistent_workers(
    llama_threading_discipline_state * state);

/**
 * [CRITICAL] Check for per-node barriers
 *
 * Detects barriers triggered after each graph node or kernel dispatch.
 * These must be eliminated and replaced with static scheduling.
 *
 * @param state Discipline state
 * @param graph_nodes Number of nodes that would trigger barriers
 * @return true if no per-node barriers, false if detected
 */
bool llama_threading_discipline_check_no_per_node_barriers(
    const llama_threading_discipline_state * state,
    int graph_nodes);

/**
 * [CRITICAL] Validate graph-level wake cycles eliminated
 *
 * Checks that graph execution does not follow:
 * Wake workers → execute task → sleep workers pattern
 *
 * Validates execution operates under single activation epoch.
 *
 * @param state Discipline state
 * @param has_graph_wake_call Whether graph execution calls thread_wake
 * @param has_graph_sleep_call Whether graph execution calls thread_sleep
 * @return true if no per-graph wake/sleep cycles
 */
bool llama_threading_discipline_validate_no_graph_cycles(
    const llama_threading_discipline_state * state,
    bool has_graph_wake_call,
    bool has_graph_sleep_call);

/**
 * [CRITICAL] Enforce thread pool isolation
 *
 * Separates decode thread pool from server/HTTP thread pool.
 * Prevents decode workers from competing on same condition variables.
 *
 * @param state Discipline state
 * @param decode_pool_id Thread pool ID for decode
 * @param server_pool_id Thread pool ID for server/HTTP
 * @return true if pools properly isolated, false if shared
 */
bool llama_threading_discipline_enforce_pool_isolation(
    llama_threading_discipline_state * state,
    int decode_pool_id,
    int server_pool_id);

/**
 * [CRITICAL] Enforce decode thread count cap
 *
 * Limits decode threads to minimum necessary.
 * Prevents oversubscription and unnecessary context switching.
 * Validates: n_threads <= (CPU_cores / 2) or configured_max
 *
 * @param state Discipline state
 * @param actual_thread_count Actual threads configured
 * @return true if within cap, false if oversubscribed
 */
bool llama_threading_discipline_enforce_thread_cap(
    const llama_threading_discipline_state * state,
    int actual_thread_count);

/**
 * [CRITICAL] Remove sampling thread pool activation
 *
 * Ensures sampling does not trigger thread pool signaling.
 * Sampling must execute inside persistent decode context.
 *
 * @param state Discipline state
 * @param sampling_uses_threadpool Whether sampling calls thread_wake
 * @return true if sampling is thread-pool free, false otherwise
 */
bool llama_threading_discipline_sampling_no_signals(
    const llama_threading_discipline_state * state,
    bool sampling_uses_threadpool);

/**
 * [CRITICAL] Validate stream-ordered execution
 *
 * For GPU stream operations, ensures blocking waits are replaced
 * with stream-ordered execution model.
 * Thread should not block; GPU advances autonomously.
 *
 * @param state Discipline state
 * @param has_blocking_gpu_wait Whether thread blocks on GPU
 * @return true if stream-ordered, false if blocking waits exist
 */
bool llama_threading_discipline_validate_stream_ordering(
    const llama_threading_discipline_state * state,
    bool has_blocking_gpu_wait);

/**
 * [CRITICAL] Record thread activity
 *
 * Called whenever a thread wake, sleep, or barrier occurs.
 * Used to detect per-token signaling during decode.
 *
 * @param state Discipline state
 * @param activity_type "wake", "sleep", "barrier", "cond_signal"
 */
void llama_threading_discipline_record_activity(
    llama_threading_discipline_state * state,
    const char * activity_type);

/**
 * [CRITICAL] Validate thread activity during decode
 *
 * Asserts that during decode, thread activities are zero.
 * If decode_thread_wake_count or decode_thread_sleep_count > 0: abort.
 *
 * @param state Discipline state
 * @return true if no per-token thread activity, false otherwise
 */
bool llama_threading_discipline_validate_no_per_token_activity(
    const llama_threading_discipline_state * state);

/**
 * Set CPU affinity for decode workers
 *
 * Pins decode workers to specific CPU cores.
 * Reduces context switching and improves cache locality.
 *
 * @param state Discipline state
 * @param core_ids Array of CPU core IDs
 * @param n_cores Number of cores to use
 * @return true if affinity set, false otherwise
 */
bool llama_threading_discipline_set_cpu_affinity(
    llama_threading_discipline_state * state,
    const int * core_ids,
    int n_cores);

/**
 * Get thread activity metrics
 *
 * @param state Discipline state
 * @return Thread activity metrics
 */
llama_thread_metrics llama_threading_discipline_get_metrics(
    const llama_threading_discipline_state * state);

/**
 * Check invariant compliance
 *
 * Returns true if all threading discipline invariants are maintained.
 *
 * @param state Discipline state
 * @return true if compliant, false if violations
 */
bool llama_threading_discipline_check_compliance(
    const llama_threading_discipline_state * state);

/**
 * Dump threading metrics and status
 *
 * Logs detailed breakdown of thread activity.
 *
 * @param state Discipline state
 */
void llama_threading_discipline_dump_metrics(
    const llama_threading_discipline_state * state);

/**
 * Deactivate threading discipline enforcement
 *
 * @param state Discipline state
 */
void llama_threading_discipline_deactivate(
    llama_threading_discipline_state * state);

#ifdef __cplusplus
}
#endif
