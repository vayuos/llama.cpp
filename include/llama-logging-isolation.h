#pragma once

/**
 * llama-logging-isolation.h
 *
 * Complete elimination of logging overhead during decode-critical window.
 * Implements comprehensive logging blackout enforcement with runtime guards,
 * atomic metrics collection, and deferred logging after decode completion.
 *
 * Requirements Enforced (10 Rules):
 * 1. Define Decode Logging Blackout Window
 *    - Window: decode_start → first_token_forward → ... → last_token_selected → decode_end
 *    - Zero logging overhead during this window
 *    - Complete elimination of:
 *      * Info logs (LOG_INF)
 *      * Debug logs (LOG_DBG)
 *      * Timing logs
 *      * JSON event logs
 *      * Per-token progress logs
 *
 * 2. Identify All Logging Calls on Decode Path
 *    - Audit server.cpp, server-task.cpp, server-context.cpp
 *    - Audit llama_decode, llama_sampler functions
 *    - Find: LOG(), fprintf, std::cout, std::cerr, spdlog
 *    - Result: Remove or gate with decode_active flag
 *
 * 3. Introduce Decode-Mode Logging Guard
 *    - Runtime flag: decode_active = true during decode
 *    - Compile-time: #ifdef LLAMA_LOGGING_ISOLATION guards
 *    - Production: compile out logging entirely
 *
 * 4. Eliminate Per-Token Log Emission
 *    - Zero "token generated" logs
 *    - Zero streaming progress logs
 *    - Zero per-step timing logs
 *    - Reduces mutex acquisition, string formatting, heap allocation, syscalls
 *
 * 5. Disable Structured Logging During Decode
 *    - Buffer counters atomically only
 *    - Flush after decode completes
 *    - Never serialize during decode
 *
 * 6. Disable Error Logging Inside Hot Path
 *    - Convert to error code propagation
 *    - Defer logging to outer control layer
 *    - Avoid string formatting inside decode
 *
 * 7. Disable Verbose Server Modes During Decode
 *    - Force-disable --verbose flag
 *    - Disable per-request debug
 *    - Disable slot transition prints
 *    - Disable performance tracing
 *
 * 8. Remove Logging Locks from Decode
 *    - Logging uses global mutex
 *    - Ensure decode thread never touches logging structures
 *    - Verify zero lock contention
 *
 * 9. Validate Logging Isolation
 *    - Measure: tokens/sec with logging enabled
 *    - Measure: tokens/sec with logging disabled
 *    - Verify: logging doesn't reduce throughput
 *
 * 10. Expected Outcome
 *     - No log formatting during decode
 *     - No mutex contention from logging
 *     - No I/O syscalls per token
 *     - Reduced CPU jitter
 *     - Stable GPU dispatch cadence
 *     - Decode path becomes silent and computation-only
 *
 * Key Metrics Tracked:
 * - Per-token logging operations (target: 0)
 * - Tokens/sec with logging vs without
 * - Logging mutex acquisitions per token (target: 0)
 * - Deferred log buffer depth
 * - I/O syscalls per token (target: 0)
 * - CPU jitter reduction from logging elimination
 */

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <mutex>
#include <array>
#include <memory>
#include <functional>
#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

// Maximum number of deferred log entries to buffer
#define LLAMA_LOGGING_ISOLATION_MAX_DEFERRED_LOGS 10000

// Maximum length of a single deferred log message
#define LLAMA_LOGGING_ISOLATION_MAX_LOG_MESSAGE_LEN 512

// Logging blackout state machine states
#define LLAMA_LOGGING_STATE_UNINITIALIZED 0
#define LLAMA_LOGGING_STATE_IDLE 1
#define LLAMA_LOGGING_STATE_PREFILL_ACTIVE 2
#define LLAMA_LOGGING_STATE_DECODE_BLACKOUT 3
#define LLAMA_LOGGING_STATE_DECODE_COMPLETE 4
#define LLAMA_LOGGING_STATE_ERROR 5

// Log level definitions (must match common/log.h)
#define LLAMA_LOG_LEVEL_ERROR 1
#define LLAMA_LOG_LEVEL_WARN 2
#define LLAMA_LOG_LEVEL_INFO 3
#define LLAMA_LOG_LEVEL_DEBUG 4

// Compile-time configuration
#ifndef LLAMA_LOGGING_ISOLATION_ENABLED
#define LLAMA_LOGGING_ISOLATION_ENABLED 1
#endif

#ifndef LLAMA_LOGGING_COLLECT_METRICS
#define LLAMA_LOGGING_COLLECT_METRICS 1
#endif

#ifndef LLAMA_LOGGING_DEFER_FLUSH
#define LLAMA_LOGGING_DEFER_FLUSH 1
#endif

// ============================================================================
// ATOMIC METRICS STRUCTURES
// ============================================================================

/**
 * Per-token logging metrics (lock-free)
 */
typedef struct {
    std::atomic<uint64_t> log_calls_blocked;        // Total LOG_*() calls blocked
    std::atomic<uint64_t> fprintf_calls_blocked;    // Total fprintf() calls blocked
    std::atomic<uint64_t> string_formats_prevented; // String formatting operations prevented
    std::atomic<uint64_t> mutex_acquisitions;       // Logging mutex acquisition attempts
    std::atomic<uint64_t> io_syscalls_prevented;    // I/O syscalls prevented
    std::atomic<uint64_t> heap_allocs_prevented;    // Heap allocations prevented
    std::atomic<uint64_t> cache_line_invalidations_prevented; // Cache pollution prevented
} llama_per_token_logging_metrics;

/**
 * Decode-window logging isolation state
 */
typedef struct {
    // State machine
    std::atomic<int32_t> isolation_state;           // Current isolation state
    std::atomic<bool> decode_active;                // true during decode blackout
    std::atomic<bool> logging_disabled;             // true when logging is suppressed

    // Timing
    std::atomic<uint64_t> decode_start_ns;          // When decode started
    std::atomic<uint64_t> decode_end_ns;            // When decode ended
    std::atomic<uint64_t> first_token_ns;           // First token forward time
    std::atomic<uint64_t> last_token_ns;            // Last token selected time

    // Metrics
    std::atomic<uint64_t> decode_tokens_processed;  // Tokens processed during decode
    std::atomic<uint64_t> decode_duration_ns;       // Total decode duration

    // Lock contention detection
    std::atomic<uint64_t> logging_lock_waits;       // Lock acquisition wait count
    std::atomic<uint64_t> logging_lock_max_wait_ns; // Maximum lock wait time

    // Error tracking
    std::atomic<bool> isolation_violation;          // true if logging during decode detected
    std::atomic<uint32_t> violation_count;          // Number of violations

    // Per-token metrics (accumulated)
    std::atomic<uint64_t> tokens_with_logging;      // Tokens that attempted logging
    std::atomic<uint64_t> tokens_zero_logging;      // Tokens with zero logging operations
} llama_logging_isolation_state;

/**
 * Deferred log entry (buffered for post-decode flush)
 */
typedef struct {
    uint64_t timestamp_ns;                          // When log was deferred
    int32_t log_level;                              // LLAMA_LOG_LEVEL_*
    char message[LLAMA_LOGGING_ISOLATION_MAX_LOG_MESSAGE_LEN];
} llama_deferred_log_entry;

/**
 * Deferred logging buffer (ring buffer, atomic operations only)
 */
typedef struct {
    std::atomic<uint32_t> write_index;              // Write position (modulo capacity)
    std::atomic<uint32_t> read_index;               // Read position (modulo capacity)
    std::atomic<uint32_t> entry_count;              // Number of entries in buffer
    std::atomic<bool> overflow;                     // true if buffer overflowed
    std::atomic<uint64_t> total_deferred;           // Total entries deferred

    // Ring buffer of deferred logs
    llama_deferred_log_entry entries[LLAMA_LOGGING_ISOLATION_MAX_DEFERRED_LOGS];
} llama_deferred_logging_buffer;

// ============================================================================
// GLOBAL STATE STRUCTURES
// ============================================================================

/**
 * Complete logging isolation control state
 */
extern llama_logging_isolation_state g_llama_logging_isolation;

/**
 * Deferred logging buffer
 */
extern llama_deferred_logging_buffer g_llama_deferred_logs;

// ============================================================================
// INITIALIZATION AND LIFECYCLE
// ============================================================================

/**
 * Initialize logging isolation system
 * Must be called once at program startup
 * @return 0 on success, -1 on error
 */
int llama_logging_isolation_init(void);

/**
 * Clean up logging isolation system
 * Flushes any deferred logs
 * Must be called at program shutdown
 */
void llama_logging_isolation_fini(void);

/**
 * Enable decode logging blackout
 * Call before starting decode
 * Sets isolation_state -> LLAMA_LOGGING_STATE_DECODE_BLACKOUT
 * @param token_count Expected number of tokens to decode (for metrics)
 */
void llama_logging_isolation_decode_start(uint32_t token_count);

/**
 * Mark first token forward completed
 * Call after first llama_decode() call succeeds
 * Updates first_token_ns timestamp
 */
void llama_logging_isolation_first_token(void);

/**
 * Mark last token selected
 * Call after sampling/selection of final token
 * Updates last_token_ns timestamp
 */
void llama_logging_isolation_last_token(void);

/**
 * Disable decode logging blackout
 * Call after decode completes
 * Sets isolation_state -> LLAMA_LOGGING_STATE_DECODE_COMPLETE
 * Automatically flushes deferred logs if LLAMA_LOGGING_DEFER_FLUSH enabled
 */
void llama_logging_isolation_decode_end(void);

/**
 * Reset to idle state
 * Call after decode result transmission
 * Sets isolation_state -> LLAMA_LOGGING_STATE_IDLE
 */
void llama_logging_isolation_reset(void);

// ============================================================================
// LOGGING GUARDS AND GATES
// ============================================================================

/**
 * Check if logging is currently blackouted
 * @return true if decode is active and logging should be suppressed
 */
inline bool llama_logging_is_blackouted(void) {
#if LLAMA_LOGGING_ISOLATION_ENABLED
    return g_llama_logging_isolation.decode_active.load(std::memory_order_relaxed);
#else
    return false;
#endif
}

/**
 * Check if we're in isolation state
 * @return true if isolation is currently enforced
 */
inline bool llama_logging_isolation_active(void) {
#if LLAMA_LOGGING_ISOLATION_ENABLED
    int32_t state = g_llama_logging_isolation.isolation_state.load(std::memory_order_relaxed);
    return state == LLAMA_LOGGING_STATE_DECODE_BLACKOUT;
#else
    return false;
#endif
}

/**
 * Attempt to log during potential blackout
 * Returns false if logging should be suppressed, true if allowed
 * @return true if logging is allowed, false if blackouted
 */
bool llama_logging_check_allowed(void);

/**
 * Guard macro for conditional logging (compile-time)
 * When LLAMA_LOGGING_ISOLATION_ENABLED=1, logs are compiled out in hot paths
 */
#if LLAMA_LOGGING_ISOLATION_ENABLED && defined(NDEBUG)
#define LLAMA_LOGGING_GUARD_LOG(level, ...) \
    do { \
        if (!llama_logging_is_blackouted()) { \
            /* original logging call */ \
        } \
    } while (0)
#else
#define LLAMA_LOGGING_GUARD_LOG(level, ...) /* no-op */
#endif

// ============================================================================
// DEFERRED LOGGING SYSTEM
// ============================================================================

/**
 * Defer a log message for emission after decode
 * Called instead of immediate logging during blackout
 * Lock-free: uses atomic ring buffer
 * @param level Log level (LLAMA_LOG_LEVEL_*)
 * @param message Formatted log message (max 512 chars)
 * @return true if buffered, false if buffer full
 */
bool llama_logging_defer(int32_t level, const char * message);

/**
 * Deferred printf-style logging
 * @param level Log level
 * @param fmt Format string
 * @param ... Arguments (note: varargs not supported, use defer() with formatted string)
 */
void llama_logging_defer_formatted(int32_t level, const char * fmt, ...);

/**
 * Flush all deferred logs to actual logging system
 * Safe to call from any thread
 * Typically called at decode_end() or shutdown
 * @return Number of logs flushed
 */
uint32_t llama_logging_deferred_flush(void);

/**
 * Get deferred log buffer status
 * @return Number of pending deferred logs
 */
uint32_t llama_logging_deferred_count(void);

/**
 * Clear deferred log buffer (discards pending logs)
 */
void llama_logging_deferred_clear(void);

// ============================================================================
// METRICS AND OBSERVABILITY
// ============================================================================

/**
 * Get current isolation metrics
 * @return Snapshot of isolation state
 */
llama_logging_isolation_state llama_logging_isolation_get_state(void);

/**
 * Get per-token logging metrics
 * @return Metrics structure with accumulated counts
 */
llama_per_token_logging_metrics llama_logging_isolation_get_metrics(void);

/**
 * Report decode performance with logging impact
 * Useful for validation that logging doesn't impact throughput
 * @param tokens_processed Number of tokens in decode
 * @param duration_ns Total decode duration
 * @return Throughput estimate (tokens/sec)
 */
double llama_logging_isolation_report_throughput(uint32_t tokens_processed, uint64_t duration_ns);

/**
 * Validate logging isolation enforcement
 * Checks that no logging occurred during decode
 * @return 0 if valid, -1 if violations detected
 */
int llama_logging_isolation_validate(void);

/**
 * Get human-readable state name
 * @param state LLAMA_LOGGING_STATE_*
 * @return State name string
 */
const char * llama_logging_isolation_state_name(int32_t state);

/**
 * Get human-readable log level name
 * @param level LLAMA_LOG_LEVEL_*
 * @return Level name string
 */
const char * llama_logging_level_name(int32_t level);

// ============================================================================
// CONFIGURATION AND CONTROL
// ============================================================================

/**
 * Enable or disable logging isolation
 * @param enabled true to enable, false to disable
 */
void llama_logging_isolation_set_enabled(bool enabled);

/**
 * Check if logging isolation is enabled
 * @return true if enabled
 */
bool llama_logging_isolation_is_enabled(void);

/**
 * Set whether to abort on isolation violation
 * @param abort_on_violation true to abort(), false to log violation only
 */
void llama_logging_isolation_set_abort_on_violation(bool abort_on_violation);

/**
 * Check if abort-on-violation is enabled
 * @return true if enabled
 */
bool llama_logging_isolation_get_abort_on_violation(void);

/**
 * Enable or disable deferred logging
 * @param defer true to buffer logs, false to discard during blackout
 */
void llama_logging_isolation_set_defer_enabled(bool defer);

/**
 * Check if deferred logging is enabled
 * @return true if enabled
 */
bool llama_logging_isolation_get_defer_enabled(void);

/**
 * Set verbosity level threshold
 * Logs at or below this level will be collected
 * Default: LLAMA_LOG_LEVEL_INFO
 * @param level LLAMA_LOG_LEVEL_*
 */
void llama_logging_isolation_set_verbosity_threshold(int32_t level);

/**
 * Get verbosity level threshold
 * @return Current threshold level
 */
int32_t llama_logging_isolation_get_verbosity_threshold(void);

// ============================================================================
// VALIDATION AND ASSERTIONS
// ============================================================================

/**
 * Assert that no logging is occurring during decode
 * If logging is detected, may abort depending on configuration
 * @param location Source location for error reporting
 * @return true if assertion passed
 */
bool llama_logging_isolation_assert_no_logging(const char * location);

/**
 * Record a logging isolation violation
 * Increments violation counter, optionally aborts
 * @param location Source location of violation
 * @param message Description of violation
 */
void llama_logging_isolation_violation_detected(const char * location, const char * message);

/**
 * Get violation count
 * @return Total number of violations detected
 */
uint32_t llama_logging_isolation_get_violation_count(void);

/**
 * Clear violation counter
 */
void llama_logging_isolation_clear_violations(void);

// ============================================================================
// LOCK CONTENTION DETECTION
// ============================================================================

/**
 * Record logging mutex acquisition
 * Called by logging framework when acquiring logging lock
 * @param wait_time_ns Time spent waiting for lock (0 if no wait)
 */
void llama_logging_record_lock_acquisition(uint64_t wait_time_ns);

/**
 * Get logging lock contention metrics
 * @return Tuple: (total acquisitions, max wait time in ns)
 */
struct llama_logging_lock_metrics {
    uint64_t total_acquisitions;
    uint64_t max_wait_ns;
    uint64_t total_wait_ns;
};
struct llama_logging_lock_metrics llama_logging_isolation_get_lock_metrics(void);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Format a duration in nanoseconds as human-readable string
 * @param ns Duration in nanoseconds
 * @param buf Output buffer (must be >= 32 bytes)
 * @return buf
 */
char * llama_logging_format_duration(uint64_t ns, char * buf);

/**
 * Format throughput as human-readable string
 * @param tokens_per_sec Throughput in tokens/second
 * @param buf Output buffer (must be >= 32 bytes)
 * @return buf
 */
char * llama_logging_format_throughput(double tokens_per_sec, char * buf);

/**
 * Get current timestamp in nanoseconds
 * @return Monotonic timestamp
 */
uint64_t llama_logging_get_timestamp_ns(void);

#ifdef __cplusplus
}
#endif
