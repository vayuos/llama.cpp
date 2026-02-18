#pragma once

/**
 * Oversubscription Control for LLAMA Decode Optimization
 *
 * REQUIREMENT #45: Strict CPU thread concurrency control during decode phase.
 * Prevents thread oversubscription that degrades decode performance through
 * excessive context switching, cache eviction, and scheduling contention.
 *
 * Core Invariant During Decode:
 *   active_runnable_threads <= required_decode_orchestration_threads
 *   Where required_decode_orchestration_threads = 1-2 (control + optional auxiliary)
 *
 * Enforcement Strategy:
 *   1. Override user thread count for decode phase (prefill_threads = N, decode_threads = 1-2)
 *   2. Disable CPU backend workers during decode
 *   3. Force OpenMP thread count = 1
 *   4. Suspend all background helper threads
 *   5. Prevent dynamic thread creation mid-decode
 *   6. Eliminate per-token parallel micro-scheduling
 *   7. Runtime detection and validation of violations
 *
 * Expected Outcomes:
 *   - Minimal context switching (< 10 switches per token for 128-token sequence)
 *   - Reduced CPU contention (single active CPU thread context)
 *   - Stable GPU kernel dispatch timing (predictable latency)
 *   - Reduced latency jitter (std dev of per-token decode time < 5%)
 *   - Improved sustained tokens/sec (10-15% improvement over baseline)
 *   - CPU usage bounded and predictable (100% of 1 CPU core during decode)
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Oversubscription state machine states
 */
typedef enum {
    LLAMA_OVERSUBSCRIPTION_UNINITIALIZED = 0,
    LLAMA_OVERSUBSCRIPTION_CONFIGURED = 1,     // Thread counts determined, not yet enforced
    LLAMA_OVERSUBSCRIPTION_PREFILL_ACTIVE = 2, // Prefill phase: user thread count active
    LLAMA_OVERSUBSCRIPTION_DECODE_ACTIVE = 3,  // Decode phase: strict minimal threads
    LLAMA_OVERSUBSCRIPTION_RELEASED = 4        // Session ended, threads released
} llama_oversubscription_state_t;

/**
 * OpenMP enforcement mode during decode
 */
typedef enum {
    LLAMA_OPENMP_NORMAL = 0,       // No OpenMP override (default)
    LLAMA_OPENMP_OVERRIDE_TO_ONE = 1, // Force num_threads(1) during decode
    LLAMA_OPENMP_DISABLED = 2      // Completely disable OpenMP parallelization
} llama_openmp_mode_t;

/**
 * Per-phase thread count configuration
 */
typedef struct {
    // User-specified thread count (respected during prefill)
    int user_thread_count;

    // Minimal thread count for decode phase
    int decode_thread_count;        // Typically 1-2

    // Decode orchestration breakdown
    int decode_control_threads;     // 1 control/dispatch thread (required)
    int decode_auxiliary_threads;   // 0-1 optional auxiliary orchestration thread

    // Maximum allowed runnable threads during decode
    int max_runnable_threads;       // Hard limit for oversubscription detection

    // Background thread suppression
    int background_cache_threads;   // Cache maintenance threads to suspend
    int background_async_threads;   // Async memory cleaners to suspend
    int background_logging_threads; // Logging workers to suspend

} llama_thread_count_config;

/**
 * Per-thread execution context tracking
 */
typedef struct {
    uint32_t thread_id;             // OS thread ID
    bool is_decode_worker;          // true if assigned to decode phase
    bool is_prefill_worker;         // true if assigned to prefill phase
    bool is_background_worker;      // true if background helper (to be suspended)
    bool is_omp_worker;             // true if created by OpenMP parallelization

    // Thread state during decode
    bool active_in_decode;          // true if runnable during decode
    uint64_t decode_activation_time; // ns when thread became active in decode
    int context_switches;            // Context switch count during this decode session
    int wake_attempts;               // Number of times this thread was woken per token

    // Violations
    bool oversubscription_violation; // Detected during decode
    uint64_t wake_count_per_token;  // How many times woken in most recent token
} llama_thread_context;

/**
 * Background helper thread registration
 */
typedef struct {
    uint32_t thread_id;             // OS thread ID
    const char * thread_name;       // Name for diagnostics
    bool is_suspended;              // true if currently suspended
    uint64_t suspension_count;      // Count of suspension events

    // Thread attributes
    bool is_cache_maintenance;      // Cache flush/cleanup thread
    bool is_async_memory_cleaner;   // Async memory management
    bool is_logging_worker;         // Async logging thread
    bool is_monitoring_thread;      // Performance monitoring
} llama_background_thread;

/**
 * Oversubscription enforcement state
 */
typedef struct {
    // Configuration
    llama_thread_count_config thread_config;
    llama_openmp_mode_t openmp_mode;

    // State machine
    llama_oversubscription_state_t state;
    uint64_t state_transition_time; // ns of last state transition

    // User thread configuration (from --threads parameter)
    int user_thread_count;          // N from user (used for prefill)
    int actual_prefill_threads;     // Threads actually used for prefill

    // Decode phase configuration
    int decode_thread_count;        // 1-2 threads for decode
    bool gpu_exclusive_decode;      // true if GPU handles all decode
    bool cpu_backend_disabled;      // true if CPU workers are disabled during decode

    // OpenMP state tracking
    int omp_prev_num_threads;       // OpenMP thread count before override
    bool omp_override_active;       // true if currently forcing num_threads=1

    // Thread tracking
    llama_thread_context * threads; // Array of tracked threads
    int n_threads_tracked;          // Number of tracked threads
    int max_threads_capacity;       // Capacity of threads array

    // Background thread tracking
    llama_background_thread * background_threads; // Array of background threads
    int n_background_threads;       // Number of background threads registered
    int max_background_capacity;    // Capacity of background_threads array

    // Decode phase metrics
    uint64_t tokens_generated;      // Total tokens in current decode
    uint64_t context_switches_total; // Total context switches during decode
    uint64_t wake_events_total;     // Total thread wake events during decode
    uint64_t per_token_max_runnable; // Peak runnable thread count per token

    // Violation tracking
    uint64_t oversubscription_violations; // Count of violations detected
    uint64_t thread_creation_violations;  // Attempt to create threads mid-decode
    uint64_t background_thread_violations; // Background threads woken during decode
    uint64_t omp_parallel_violations;     // Parallel regions created during decode
    uint64_t cpu_backend_violations;      // CPU ops scheduled during GPU-exclusive decode

    // Metrics for diagnostics
    uint64_t last_oversubscription_check; // ns of last validation check
    uint64_t prefill_start_time;    // ns when prefill started
    uint64_t decode_start_time;     // ns when decode started
    uint64_t decode_end_time;       // ns when decode ended (0 if active)

    // Enforcement policy
    bool abort_on_oversubscription; // If true, abort when violations detected
    bool abort_on_omp_expansion;    // If true, abort when OpenMP creates extra threads
    bool abort_on_background_wake;  // If true, abort when background threads wake

} llama_oversubscription_control;

/**
 * Initialize oversubscription control state
 *
 * Detects available CPU threads and initializes tracking structures.
 * Does not enforce any restrictions yet - just sets up state.
 *
 * @param control Oversubscription control state to initialize
 * @return true if initialization successful
 */
bool llama_oversubscription_init(llama_oversubscription_control * control);

/**
 * Release oversubscription control state and cleanup
 *
 * @param control Oversubscription control state
 */
void llama_oversubscription_release(llama_oversubscription_control * control);

/**
 * [CRITICAL] Configure thread counts for prefill and decode phases
 *
 * Sets up the thread count policy:
 * - Prefill uses user-specified thread count (N from --threads)
 * - Decode uses minimal thread count (1-2)
 *
 * Example on 8-core system with --threads 8:
 *   Prefill: 8 threads active (user configuration)
 *   Decode: 1 control thread + 0 auxiliary = 1 thread total
 *
 * This configuration is applied when transitioning to decode phase.
 *
 * @param control Oversubscription control state
 * @param user_thread_count Thread count from --threads parameter
 * @param gpu_exclusive_decode true if GPU handles all decode (CPU minimal)
 * @return true if configuration successful
 */
bool llama_oversubscription_configure_thread_counts(
    llama_oversubscription_control * control,
    int user_thread_count,
    bool gpu_exclusive_decode);

/**
 * [CRITICAL] Begin prefill phase with user thread count
 *
 * Transitions to PREFILL_ACTIVE state. Allows all user threads to execute.
 * This is called before the prefill phase starts (before processing input prompt).
 *
 * @param control Oversubscription control state
 * @return true if prefill phase started successfully
 */
bool llama_oversubscription_begin_prefill(llama_oversubscription_control * control);

/**
 * [CRITICAL] Transition from prefill to decode phase
 *
 * Enters DECODE_ACTIVE state and applies all oversubscription controls:
 * - Reduce thread count to 1-2
 * - Force OpenMP num_threads = 1
 * - Disable CPU backend workers
 * - Suspend background threads
 * - Prevent new thread creation
 *
 * This is the critical barrier between prefill and autonomous decode.
 *
 * @param control Oversubscription control state
 * @return true if decode phase activated successfully
 */
bool llama_oversubscription_begin_decode(llama_oversubscription_control * control);

/**
 * [CRITICAL] Override user thread count during decode
 *
 * Internally called by begin_decode. Reduces active thread count from
 * user_thread_count to decode_thread_count (typically 1-2).
 *
 * Mechanism:
 * - Park excess threads
 * - Force worker pool to minimal size
 * - Update scheduler to only wake decode threads
 *
 * @param control Oversubscription control state
 * @param target_thread_count New thread count (typically 1-2)
 * @return true if thread override successful
 */
bool llama_oversubscription_override_thread_count(
    llama_oversubscription_control * control,
    int target_thread_count);

/**
 * [CRITICAL] Disable CPU backend workers during decode
 *
 * Marks all CPU worker threads as inactive. If GPU-exclusive decode is enabled,
 * CPU workers should never execute during decode phase.
 *
 * For GPU-exclusive decode:
 * - CPU workers remain parked (never woken)
 * - All compute on GPU backend
 * - CPU workers only execute during prefill or host-side tasks
 *
 * @param control Oversubscription control state
 * @return true if CPU backend disabled successfully
 */
bool llama_oversubscription_disable_cpu_backend(llama_oversubscription_control * control);

/**
 * [CRITICAL] Force OpenMP thread count to 1 during decode
 *
 * Internally calls omp_set_num_threads(1) and tracks override.
 * Prevents OpenMP from spawning parallel regions with multiple threads.
 *
 * @param control Oversubscription control state
 * @param force_disable If true, disable OpenMP entirely; if false, just set num_threads=1
 * @return true if OpenMP override successful
 */
bool llama_oversubscription_override_openmp(
    llama_oversubscription_control * control,
    bool force_disable);

/**
 * [CRITICAL] Suspend all background helper threads
 *
 * Parks cache maintenance threads, async memory cleaners, and logging workers.
 * These threads should not execute during decode to avoid contention.
 *
 * Background threads include:
 * - KV cache flush/compaction threads
 * - Async memory management threads
 * - Logging worker threads
 * - Performance monitoring threads
 *
 * @param control Oversubscription control state
 * @return true if background threads suspended successfully
 */
bool llama_oversubscription_suspend_background_threads(llama_oversubscription_control * control);

/**
 * [CRITICAL] Register a background helper thread
 *
 * Called during initialization to register threads that should be suspended
 * during decode phase.
 *
 * @param control Oversubscription control state
 * @param thread_id OS thread ID
 * @param thread_name Human-readable name (e.g., "cache_maintenance", "logging_worker")
 * @param is_cache_maintenance If true, marks as cache thread
 * @param is_async_memory If true, marks as async memory thread
 * @param is_logging If true, marks as logging thread
 * @return true if registration successful
 */
bool llama_oversubscription_register_background_thread(
    llama_oversubscription_control * control,
    uint32_t thread_id,
    const char * thread_name,
    bool is_cache_maintenance,
    bool is_async_memory,
    bool is_logging);

/**
 * [CRITICAL] Track a decode worker thread
 *
 * Registers a thread as a decode worker. Used to track runnable thread count
 * and detect violations.
 *
 * @param control Oversubscription control state
 * @param thread_id OS thread ID
 * @param is_prefill_worker If true, can be used for prefill phase
 * @return true if registration successful
 */
bool llama_oversubscription_register_decode_worker(
    llama_oversubscription_control * control,
    uint32_t thread_id,
    bool is_prefill_worker);

/**
 * [CRITICAL] Count currently active runnable threads during decode
 *
 * Returns the number of threads that are currently runnable (not parked,
 * not blocked) during decode phase.
 *
 * Used for oversubscription detection:
 *   if (active_runnable_threads > max_allowed_threads) { VIOLATION! }
 *
 * @param control Oversubscription control state
 * @return Count of active runnable threads, or -1 on error
 */
int llama_oversubscription_count_active_runnable_threads(
    const llama_oversubscription_control * control);

/**
 * [CRITICAL] Detect oversubscription at runtime
 *
 * Checks if active runnable thread count exceeds allowed decode thread count.
 * Called periodically during decode to catch violations.
 *
 * Algorithm:
 *   1. Count active runnable threads
 *   2. Compare against max_allowed_threads
 *   3. If exceeded: log violation, increment counter
 *   4. If abort_on_oversubscription: abort execution
 *
 * @param control Oversubscription control state
 * @return true if no oversubscription detected, false if violated
 */
bool llama_oversubscription_detect_oversubscription(llama_oversubscription_control * control);

/**
 * [CRITICAL] Prevent per-token parallel scheduling
 *
 * Called at the start of each token generation to validate that no
 * multiple threads are being woken for control/orchestration work.
 *
 * Per-token scheduling should be:
 *   - Single control thread handles token setup
 *   - GPU kernel launched from single thread
 *   - Completion signaled back to control thread
 *   - NO micro-task parallel scheduling
 *
 * @param control Oversubscription control state
 * @return true if per-token scheduling valid, false if violation
 */
bool llama_oversubscription_validate_per_token_scheduling(llama_oversubscription_control * control);

/**
 * [CRITICAL] Prevent dynamic thread creation mid-decode
 *
 * Rejects attempts to:
 * - Create new threads during decode
 * - Lazy-initialize thread pool threads
 * - Auto-scale thread pools
 * - Activate work-stealing from other domains
 *
 * Thread pool must be static during decode - no new threads allowed.
 *
 * @param control Oversubscription control state
 * @param new_thread_id OS thread ID of thread being created
 * @return true if thread creation allowed, false/abort if decode active and denied
 */
bool llama_oversubscription_check_thread_creation_allowed(
    const llama_oversubscription_control * control,
    uint32_t new_thread_id);

/**
 * [CRITICAL] Prevent work-stealing and auto-scaling
 *
 * During decode, thread pool work distribution must be static:
 * - No dynamic work-stealing from other thread pools
 * - No auto-scaling based on queue depth
 * - No lazy thread initialization
 * - Thread assignments immutable
 *
 * @param control Oversubscription control state
 * @return true if current work distribution valid, false if auto-scaling attempted
 */
bool llama_oversubscription_validate_static_thread_pool(llama_oversubscription_control * control);

/**
 * [CRITICAL] Prevent background thread wakeup during decode
 *
 * Validates that background helper threads remain parked during decode.
 * Detects if any background thread has been woken for:
 * - Cache maintenance
 * - Async memory cleanup
 * - Logging operations
 * - Monitoring tasks
 *
 * @param control Oversubscription control state
 * @return true if all background threads remain suspended, false if any woken
 */
bool llama_oversubscription_validate_background_threads_parked(llama_oversubscription_control * control);

/**
 * [CRITICAL] Detect per-token oversubscription violations
 *
 * Called after each token is generated to check:
 * - How many threads were woken for this token
 * - Did they exceed the allowed count
 * - Any background threads wake during token
 *
 * Logs detailed metrics about thread activity.
 *
 * @param control Oversubscription control state
 * @return true if token generated with proper thread count, false if violations
 */
bool llama_oversubscription_check_per_token_threads(llama_oversubscription_control * control);

/**
 * [CRITICAL] End decode phase and restore original thread count
 *
 * Transitions back to normal state. Restores:
 * - Original thread count (from user --threads parameter)
 * - OpenMP to previous thread count
 * - CPU backend workers to active
 * - Background threads to active
 *
 * @param control Oversubscription control state
 * @return true if decode phase ended successfully
 */
bool llama_oversubscription_end_decode(llama_oversubscription_control * control);

/**
 * [CRITICAL] Assert oversubscription control state intact
 *
 * Validates that:
 * - No illegal thread creations occurred
 * - Background threads remain parked
 * - Thread count matches configured
 * - No oversubscription violations
 *
 * Aborts if inconsistencies detected.
 *
 * @param control Oversubscription control state
 * @return true if state valid
 */
bool llama_oversubscription_assert_control_intact(const llama_oversubscription_control * control);

/**
 * Get current oversubscription control state
 *
 * @param control Oversubscription control state
 * @return Current state enum
 */
llama_oversubscription_state_t llama_oversubscription_get_state(
    const llama_oversubscription_control * control);

/**
 * Check if decode phase is currently active
 *
 * @param control Oversubscription control state
 * @return true if in DECODE_ACTIVE state
 */
bool llama_oversubscription_is_decode_active(const llama_oversubscription_control * control);

/**
 * [DEBUG] Dump oversubscription control configuration
 *
 * Writes human-readable configuration and metrics to log.
 *
 * Example output:
 *
 * OVERSUBSCRIPTION CONTROL CONFIG:
 *   User thread count: 8 (from --threads)
 *   Decode thread count: 1 (control) + 0 (auxiliary) = 1 total
 *   GPU-exclusive decode: yes
 *   Current state: DECODE_ACTIVE
 *   Tokens generated: 42
 *   Total context switches: 8 (0.19 per token)
 *   Total thread wake events: 47 (1.12 per token)
 *   Oversubscription violations: 0
 *   Background threads suspended: 3
 *     cache_maintenance_worker [suspended 1 time]
 *     async_memory_cleaner [suspended 1 time]
 *     logging_worker [suspended 1 time]
 *
 * @param control Oversubscription control state
 */
void llama_oversubscription_dump_config(const llama_oversubscription_control * control);

/**
 * [DEBUG] Get violation statistics
 *
 * Returns counts of detected oversubscription violations.
 *
 * @param control Oversubscription control state
 * @param out_oversubscription Count of oversubscription violations
 * @param out_thread_creation Count of illegal thread creation attempts
 * @param out_background_wake Count of background thread wake violations
 * @param out_omp_expansion Count of OpenMP expansion violations
 */
void llama_oversubscription_get_violations(
    const llama_oversubscription_control * control,
    uint64_t * out_oversubscription,
    uint64_t * out_thread_creation,
    uint64_t * out_background_wake,
    uint64_t * out_omp_expansion);

/**
 * [DEBUG] Get decode phase metrics
 *
 * Returns performance metrics collected during decode phase.
 *
 * @param control Oversubscription control state
 * @param out_tokens_generated Total tokens generated in current/last decode
 * @param out_context_switches Total context switches during decode
 * @param out_wake_events Total thread wake events
 * @param out_peak_runnable_threads Peak runnable thread count across all tokens
 * @param out_avg_runnable_threads Average runnable threads per token
 */
void llama_oversubscription_get_metrics(
    const llama_oversubscription_control * control,
    uint64_t * out_tokens_generated,
    uint64_t * out_context_switches,
    uint64_t * out_wake_events,
    int * out_peak_runnable_threads,
    double * out_avg_runnable_threads);

/**
 * [DEBUG] Get thread-specific metrics
 *
 * Returns metrics for a specific thread during decode.
 *
 * @param control Oversubscription control state
 * @param thread_id OS thread ID
 * @param out_active true if thread active in decode
 * @param out_context_switches Context switches for this thread
 * @param out_wake_count Wake events for this thread
 * @return true if thread found, false otherwise
 */
bool llama_oversubscription_get_thread_metrics(
    const llama_oversubscription_control * control,
    uint32_t thread_id,
    bool * out_active,
    int * out_context_switches,
    int * out_wake_count);

#ifdef __cplusplus
}
#endif
