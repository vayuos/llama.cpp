#pragma once

/**
 * Decode Path Isolation Enforcement for LLAMA
 *
 * Complete isolation of decode execution from thread pool interactions.
 * Eliminates all per-token thread pool submissions, wake-sleep cycles,
 * lock contention, and work-stealing to achieve deterministic, low-jitter
 * decode execution with stable GPU occupancy.
 *
 * Key Properties:
 * - No threadpool_submit() calls inside decode loop
 * - No per-token parallel region creation
 * - No per-token task wake/sleep cycles
 * - No work-stealing attempts during decode
 * - Direct synchronous invocation of decode operations
 * - Thread pool frozen at minimal size during decode
 * - Runtime guards and assertion system
 *
 * Expected Outcome:
 * - Zero thread pool task submissions per token
 * - Zero lock acquisitions per token
 * - Zero worker thread wake-ups per token
 * - Stable CPU dispatch timing
 * - Higher GPU occupancy
 * - Improved tokens/sec stability
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Decode execution mode - controls thread pool interaction during decode
 */
typedef enum {
    LLAMA_DECODE_MODE_UNINITIALIZED = 0,
    LLAMA_DECODE_MODE_PREFILL = 1,          // Prefill phase - threadpool active
    LLAMA_DECODE_MODE_DECODE_ISOLATION = 2, // Decode phase - threadpool frozen, isolated
    LLAMA_DECODE_MODE_SHUTDOWN = 3          // Shutdown phase - thread pool released
} llama_decode_mode_t;

/**
 * Thread pool state snapshot - captured at decode start
 */
typedef enum {
    LLAMA_THREADPOOL_STATE_UNKNOWN = 0,
    LLAMA_THREADPOOL_STATE_ACTIVE = 1,      // Pool actively processing work
    LLAMA_THREADPOOL_STATE_IDLE = 2,        // Pool idle, workers parked
    LLAMA_THREADPOOL_STATE_FROZEN = 3,      // Pool frozen at minimal size
    LLAMA_THREADPOOL_STATE_DISABLED = 4     // Pool completely disabled for decode
} llama_threadpool_state_t;

/**
 * Lock contention tracking entry
 */
typedef struct {
    const char * lock_name;                 // Name/location of lock
    uint64_t acquisition_count;             // Total acquisitions during decode
    uint64_t contention_samples;            // Times lock was already held
    uint64_t max_hold_time_ns;              // Maximum lock hold duration
    bool is_contended;                      // true if contention detected
} llama_lock_contention_entry;

/**
 * Per-token submission tracking
 */
typedef struct {
    uint64_t token_index;                   // Token number in decode sequence
    uint64_t timestamp_ns;                  // When token processing started
    uint32_t submission_count;              // Number of threadpool submissions
    uint32_t parallel_region_count;         // Number of parallel regions created
    uint32_t lock_acquisitions;             // Total lock acquisitions
    uint32_t worker_wakeups;                // Worker thread wake-up calls
    uint32_t work_stealing_attempts;        // Work-stealing attempts
    bool direct_invocation_used;            // true if direct invocation used
} llama_per_token_metrics;

/**
 * Thread pool queue monitoring state
 */
typedef struct {
    int initial_queue_depth;                // Task queue depth before decode
    int current_queue_depth;                // Current task queue depth
    int max_observed_depth;                 // Max depth during decode
    uint64_t total_enqueues;                // Total enqueue calls
    uint64_t total_dequeues;                // Total dequeue calls
    bool queue_operations_detected;         // true if queue ops in decode loop
} llama_threadpool_queue_monitor;

/**
 * Direct invocation state and metrics
 */
typedef struct {
    bool enabled;                           // Direct invocation enabled
    uint64_t invocations_count;             // Total direct invocations
    uint64_t total_invocation_time_ns;      // Total time spent in direct invocation
    uint64_t max_invocation_time_ns;        // Maximum single invocation time
    uint64_t min_invocation_time_ns;        // Minimum single invocation time
    bool all_tokens_direct;                 // true if all tokens used direct invocation
} llama_direct_invocation_state;

/**
 * Work-stealing monitoring
 */
typedef struct {
    bool work_stealing_enabled;             // true if pool supports work-stealing
    uint64_t steal_attempts_count;          // Total steal attempts during decode
    uint64_t successful_steals;             // Successful steals
    bool steals_detected_in_decode;         // true if steals occurred during decode
} llama_work_stealing_monitor;

/**
 * Decode path isolation state machine
 */
typedef enum {
    LLAMA_ISOLATION_STATE_UNINITIALIZED = 0,
    LLAMA_ISOLATION_STATE_IDLE = 1,         // Not in decode
    LLAMA_ISOLATION_STATE_PREFILL_ACTIVE = 2, // Prefill in progress
    LLAMA_ISOLATION_STATE_DECODE_GUARDED = 3, // Decode with isolation active
    LLAMA_ISOLATION_STATE_DECODE_COMPLETE = 4, // Decode completed
    LLAMA_ISOLATION_STATE_ERROR = 5         // Isolation violation detected
} llama_decode_isolation_state_t;

/**
 * Runtime guard and assertion tracking
 */
typedef struct {
    uint64_t assertions_passed;             // Total passed assertions
    uint64_t assertions_failed;             // Total failed assertions
    uint64_t violations_detected;           // Total isolation violations
    char last_violation_msg[512];           // Last violation message
    bool abort_on_violation;                // true = abort on violation, false = log only
    uint64_t last_violation_ns;             // Timestamp of last violation
} llama_isolation_guard_state;

/**
 * Complete decode path isolation state
 */
typedef struct {
    // Execution mode and state
    llama_decode_mode_t current_mode;       // Current decode mode
    llama_decode_isolation_state_t isolation_state; // Isolation state machine
    bool isolation_active;                  // true when isolation is enforced
    uint64_t decode_start_ns;               // Timestamp when decode started
    uint64_t decode_end_ns;                 // Timestamp when decode completed

    // Thread pool monitoring
    llama_threadpool_state_t threadpool_state; // Captured thread pool state
    int threadpool_initial_size;            // Thread pool size before freeze
    int threadpool_minimal_size;            // Minimal size during decode (typically 1)
    int threadpool_current_size;            // Current effective size
    uint64_t threadpool_submissions;        // Total submissions during decode
    uint64_t threadpool_completions;        // Total completions
    uint64_t threadpool_enqueue_operations; // Enqueue operations detected
    uint64_t threadpool_dequeue_operations; // Dequeue operations detected

    // Per-token metrics
    llama_per_token_metrics * token_metrics; // Per-token tracking
    int n_tokens_processed;                 // Tokens processed in current decode
    int token_metrics_capacity;             // Capacity of token_metrics array
    uint32_t max_per_token_submissions;     // Max submissions in any token
    uint32_t max_per_token_locks;           // Max lock acquisitions in any token
    uint32_t max_per_token_wakeups;         // Max wakeups in any token

    // Queue monitoring
    llama_threadpool_queue_monitor queue_monitor; // Queue state tracking

    // Direct invocation state
    llama_direct_invocation_state direct_invoke; // Direct invocation metrics

    // Work-stealing monitoring
    llama_work_stealing_monitor work_stealing; // Work-stealing tracking

    // Lock contention tracking
    llama_lock_contention_entry * lock_entries; // Per-lock contention data
    int n_lock_entries;                     // Number of tracked locks
    int lock_entries_capacity;              // Capacity of lock_entries array

    // Guard and assertions
    llama_isolation_guard_state guard_state; // Runtime guard state

    // Metrics summary
    uint64_t total_decode_cycles;           // Total decode calls
    uint64_t total_decode_time_ns;          // Total decode execution time
    double avg_jitter_ns;                   // Average jitter between tokens
    double std_dev_jitter_ns;               // Standard deviation of jitter
    uint64_t gpu_occupancy_samples;         // GPU occupancy measurements
    double avg_gpu_occupancy_percent;       // Average GPU occupancy %

    // Configuration
    bool submission_detection_enabled;      // true = detect threadpool submissions
    bool lock_monitoring_enabled;           // true = monitor lock contention
    bool work_stealing_monitoring_enabled;  // true = track work-stealing
    bool direct_invocation_required;        // true = enforce direct invocation
    bool abort_on_submission;               // true = abort if submission detected
    int max_allowed_submissions_per_token;  // Max before violation (0=none allowed)
} llama_decode_path_isolation_state;

/* ============================================================================
   Initialization and Lifecycle Management
   ============================================================================ */

/**
 * Initialize decode path isolation state
 * Must be called once before any decode operations
 *
 * @param state Output state structure
 * @return true on success, false on failure
 */
bool llama_decode_path_isolation_init(llama_decode_path_isolation_state * state);

/**
 * Release all resources associated with isolation state
 *
 * @param state State to release
 */
void llama_decode_path_isolation_release(llama_decode_path_isolation_state * state);

/**
 * Reset isolation state for a new decode sequence
 * Call before starting a new token generation loop
 *
 * @param state State to reset
 */
void llama_decode_path_isolation_reset(llama_decode_path_isolation_state * state);

/* ============================================================================
   Thread Pool Freezing and Control
   ============================================================================ */

/**
 * Freeze thread pool before decode starts
 * Resizes pool to minimal size and parks unused workers
 *
 * @param state Isolation state
 * @param minimal_threads Minimum threads to keep active (typically 1)
 * @return true on success
 */
bool llama_decode_path_isolation_freeze_threadpool(
    llama_decode_path_isolation_state * state,
    int minimal_threads);

/**
 * Thaw thread pool after decode completes
 * Restores thread pool to original size
 *
 * @param state Isolation state
 * @return true on success
 */
bool llama_decode_path_isolation_thaw_threadpool(
    llama_decode_path_isolation_state * state);

/**
 * Get current thread pool state
 *
 * @param state Isolation state
 * @return Current threadpool state
 */
llama_threadpool_state_t llama_decode_path_isolation_get_threadpool_state(
    const llama_decode_path_isolation_state * state);

/* ============================================================================
   Decode Execution Mode Management
   ============================================================================ */

/**
 * Begin decode phase with full isolation
 * Activates all enforcement guards and freezes thread pool
 *
 * @param state Isolation state
 * @param n_tokens_expected Expected number of tokens to process
 * @return true on success
 */
bool llama_decode_path_isolation_begin_decode(
    llama_decode_path_isolation_state * state,
    int n_tokens_expected);

/**
 * End decode phase
 * Validates isolation, collects metrics, thaws thread pool
 *
 * @param state Isolation state
 * @return true if isolation was maintained throughout decode
 */
bool llama_decode_path_isolation_end_decode(
    llama_decode_path_isolation_state * state);

/**
 * Signal start of per-token processing
 * Used for per-token metric collection
 *
 * @param state Isolation state
 * @param token_index Index of token being processed
 */
void llama_decode_path_isolation_token_start(
    llama_decode_path_isolation_state * state,
    uint64_t token_index);

/**
 * Signal end of per-token processing
 *
 * @param state Isolation state
 */
void llama_decode_path_isolation_token_end(
    llama_decode_path_isolation_state * state);

/* ============================================================================
   Direct Invocation Mechanism
   ============================================================================ */

/**
 * Execute operation directly without thread pool submission
 * Synchronous execution in current thread
 *
 * @param state Isolation state
 * @param operation_name Name of operation for tracking
 * @param operation_fn Function to execute directly
 * @param user_data Context data for operation
 * @return true on success
 */
bool llama_decode_path_isolation_execute_direct(
    llama_decode_path_isolation_state * state,
    const char * operation_name,
    bool (*operation_fn)(void * data),
    void * user_data);

/**
 * Verify that direct invocation was used for last token
 *
 * @param state Isolation state
 * @return true if direct invocation was used
 */
bool llama_decode_path_isolation_verify_direct_invocation(
    const llama_decode_path_isolation_state * state);

/* ============================================================================
   Submission Detection and Prevention
   ============================================================================ */

/**
 * Record detection of threadpool submission in decode path
 * Used by instrumented threadpool code to report violations
 *
 * @param state Isolation state
 * @param location Code location/function name
 * @param is_fatal true if this is a fatal violation
 * @return true if submission allowed, false if violation
 */
bool llama_decode_path_isolation_record_submission(
    llama_decode_path_isolation_state * state,
    const char * location,
    bool is_fatal);

/**
 * Record detection of parallel region creation
 *
 * @param state Isolation state
 * @param location Code location
 * @return true if allowed, false if violation
 */
bool llama_decode_path_isolation_record_parallel_region(
    llama_decode_path_isolation_state * state,
    const char * location);

/**
 * Record detection of per-token work chunking
 *
 * @param state Isolation state
 * @param location Code location
 * @return true if allowed, false if violation
 */
bool llama_decode_path_isolation_record_work_chunking(
    llama_decode_path_isolation_state * state,
    const char * location);

/* ============================================================================
   Lock Contention Monitoring
   ============================================================================ */

/**
 * Register a lock for contention monitoring
 *
 * @param state Isolation state
 * @param lock_name Name of lock
 * @return Lock entry index, or negative on error
 */
int llama_decode_path_isolation_register_lock(
    llama_decode_path_isolation_state * state,
    const char * lock_name);

/**
 * Record lock acquisition attempt
 *
 * @param state Isolation state
 * @param lock_id Lock ID from registration
 * @param acquired true if acquired immediately, false if contended
 * @param hold_time_ns How long lock was held
 */
void llama_decode_path_isolation_record_lock_acquisition(
    llama_decode_path_isolation_state * state,
    int lock_id,
    bool acquired,
    uint64_t hold_time_ns);

/* ============================================================================
   Work-Stealing Monitoring
   ============================================================================ */

/**
 * Enable work-stealing monitoring
 *
 * @param state Isolation state
 * @param enabled true to enable monitoring
 */
void llama_decode_path_isolation_enable_work_stealing_monitoring(
    llama_decode_path_isolation_state * state,
    bool enabled);

/**
 * Record work-stealing attempt
 *
 * @param state Isolation state
 * @param successful true if steal succeeded
 */
void llama_decode_path_isolation_record_work_steal(
    llama_decode_path_isolation_state * state,
    bool successful);

/* ============================================================================
   Task Queue Monitoring
   ============================================================================ */

/**
 * Take snapshot of thread pool queue before decode
 *
 * @param state Isolation state
 * @return true on success
 */
bool llama_decode_path_isolation_snapshot_queue_before(
    llama_decode_path_isolation_state * state);

/**
 * Update queue monitoring with current depth
 *
 * @param state Isolation state
 * @param current_depth Current queue depth
 */
void llama_decode_path_isolation_update_queue_depth(
    llama_decode_path_isolation_state * state,
    int current_depth);

/**
 * Record a queue operation (enqueue or dequeue)
 *
 * @param state Isolation state
 * @param is_enqueue true for enqueue, false for dequeue
 */
void llama_decode_path_isolation_record_queue_operation(
    llama_decode_path_isolation_state * state,
    bool is_enqueue);

/* ============================================================================
   Runtime Assertions and Guards
   ============================================================================ */

/**
 * Assert that no threadpool submissions occurred in current decode
 * Aborts or logs violation based on configuration
 *
 * @param state Isolation state
 * @param context_msg Message describing assertion context
 * @return true if assertion passed
 */
bool llama_decode_path_isolation_assert_no_submissions(
    llama_decode_path_isolation_state * state,
    const char * context_msg);

/**
 * Assert that thread pool queue is empty
 *
 * @param state Isolation state
 * @return true if assertion passed
 */
bool llama_decode_path_isolation_assert_empty_queue(
    llama_decode_path_isolation_state * state);

/**
 * Assert that no worker threads are active for decode
 *
 * @param state Isolation state
 * @return true if assertion passed
 */
bool llama_decode_path_isolation_assert_no_active_workers(
    llama_decode_path_isolation_state * state);

/**
 * Assert that per-token submissions are zero
 *
 * @param state Isolation state
 * @return true if assertion passed
 */
bool llama_decode_path_isolation_assert_zero_per_token_submissions(
    llama_decode_path_isolation_state * state);

/**
 * Assert that lock contention is below threshold
 *
 * @param state Isolation state
 * @param max_contention_percent Maximum allowed contention percentage
 * @return true if assertion passed
 */
bool llama_decode_path_isolation_assert_lock_contention_low(
    llama_decode_path_isolation_state * state,
    double max_contention_percent);

/**
 * Configure guard behavior
 *
 * @param state Isolation state
 * @param abort_on_violation true to abort, false to log only
 */
void llama_decode_path_isolation_configure_guard(
    llama_decode_path_isolation_state * state,
    bool abort_on_violation);

/* ============================================================================
   Metrics and Diagnostics
   ============================================================================ */

/**
 * Get per-token metrics for a specific token
 *
 * @param state Isolation state
 * @param token_index Index of token
 * @param out_metrics Output metrics structure
 * @return true if metrics available
 */
bool llama_decode_path_isolation_get_token_metrics(
    const llama_decode_path_isolation_state * state,
    int token_index,
    llama_per_token_metrics * out_metrics);

/**
 * Get summary metrics for entire decode sequence
 *
 * @param state Isolation state
 * @return Newly allocated summary string (caller must free)
 */
char * llama_decode_path_isolation_get_summary(
    const llama_decode_path_isolation_state * state);

/**
 * Get detailed diagnostics report
 *
 * @param state Isolation state
 * @return Newly allocated diagnostics report (caller must free)
 */
char * llama_decode_path_isolation_get_diagnostics(
    const llama_decode_path_isolation_state * state);

/**
 * Reset all metrics counters
 *
 * @param state Isolation state
 */
void llama_decode_path_isolation_reset_metrics(
    llama_decode_path_isolation_state * state);

/**
 * Check if isolation was maintained throughout decode
 *
 * @param state Isolation state
 * @return true if isolation maintained (no violations)
 */
bool llama_decode_path_isolation_check_integrity(
    const llama_decode_path_isolation_state * state);

/* ============================================================================
   Configuration
   ============================================================================ */

/**
 * Enable/disable submission detection
 *
 * @param state Isolation state
 * @param enabled true to enable
 */
void llama_decode_path_isolation_enable_submission_detection(
    llama_decode_path_isolation_state * state,
    bool enabled);

/**
 * Enable/disable lock monitoring
 *
 * @param state Isolation state
 * @param enabled true to enable
 */
void llama_decode_path_isolation_enable_lock_monitoring(
    llama_decode_path_isolation_state * state,
    bool enabled);

/**
 * Set maximum allowed submissions per token (0 = none allowed)
 *
 * @param state Isolation state
 * @param max_count Maximum allowed
 */
void llama_decode_path_isolation_set_max_submissions_per_token(
    llama_decode_path_isolation_state * state,
    int max_count);

/**
 * Enable/disable abort on violation
 *
 * @param state Isolation state
 * @param abort true to abort, false to log only
 */
void llama_decode_path_isolation_set_abort_on_violation(
    llama_decode_path_isolation_state * state,
    bool abort);

/* ============================================================================
   Validation and Audit Functions
   ============================================================================ */

/**
 * Validate decode path for thread pool isolation compliance
 * Performs comprehensive audit of decode critical path
 *
 * @param state Isolation state
 * @return Newly allocated audit report (caller must free)
 */
char * llama_decode_path_isolation_audit_decode_path(
    const llama_decode_path_isolation_state * state);

/**
 * Check if current thread is in isolation mode
 *
 * @return true if isolation is active
 */
bool llama_decode_path_isolation_is_isolated(void);

/**
 * Get current isolation depth (for nested isolation support)
 *
 * @return Current isolation depth
 */
int llama_decode_path_isolation_get_isolation_depth(void);

#ifdef __cplusplus
}
#endif

