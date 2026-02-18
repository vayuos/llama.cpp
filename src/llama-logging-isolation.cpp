#include "llama-logging-isolation.h"

#include <cstring>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>

// ============================================================================
// GLOBAL STATE INITIALIZATION
// ============================================================================

llama_logging_isolation_state g_llama_logging_isolation{};
llama_deferred_logging_buffer g_llama_deferred_logs{};

// Configuration flags
static bool g_logging_isolation_enabled = LLAMA_LOGGING_ISOLATION_ENABLED;
static bool g_abort_on_violation = true;
static bool g_defer_enabled = LLAMA_LOGGING_DEFER_FLUSH;
static int32_t g_verbosity_threshold = LLAMA_LOG_LEVEL_INFO;
static std::mutex g_logging_isolation_mutex;

// ============================================================================
// INITIALIZATION AND LIFECYCLE
// ============================================================================

int llama_logging_isolation_init(void) {
    // Initialize atomic values
    g_llama_logging_isolation.isolation_state.store(LLAMA_LOGGING_STATE_UNINITIALIZED, std::memory_order_release);
    g_llama_logging_isolation.decode_active.store(false, std::memory_order_release);
    g_llama_logging_isolation.logging_disabled.store(false, std::memory_order_release);

    g_llama_logging_isolation.decode_start_ns.store(0, std::memory_order_release);
    g_llama_logging_isolation.decode_end_ns.store(0, std::memory_order_release);
    g_llama_logging_isolation.first_token_ns.store(0, std::memory_order_release);
    g_llama_logging_isolation.last_token_ns.store(0, std::memory_order_release);

    g_llama_logging_isolation.decode_tokens_processed.store(0, std::memory_order_release);
    g_llama_logging_isolation.decode_duration_ns.store(0, std::memory_order_release);

    g_llama_logging_isolation.logging_lock_waits.store(0, std::memory_order_release);
    g_llama_logging_isolation.logging_lock_max_wait_ns.store(0, std::memory_order_release);

    g_llama_logging_isolation.isolation_violation.store(false, std::memory_order_release);
    g_llama_logging_isolation.violation_count.store(0, std::memory_order_release);

    g_llama_logging_isolation.tokens_with_logging.store(0, std::memory_order_release);
    g_llama_logging_isolation.tokens_zero_logging.store(0, std::memory_order_release);

    // Initialize deferred logging buffer
    g_llama_deferred_logs.write_index.store(0, std::memory_order_release);
    g_llama_deferred_logs.read_index.store(0, std::memory_order_release);
    g_llama_deferred_logs.entry_count.store(0, std::memory_order_release);
    g_llama_deferred_logs.overflow.store(false, std::memory_order_release);
    g_llama_deferred_logs.total_deferred.store(0, std::memory_order_release);

    // Clear all entries
    for (size_t i = 0; i < LLAMA_LOGGING_ISOLATION_MAX_DEFERRED_LOGS; i++) {
        g_llama_deferred_logs.entries[i].timestamp_ns = 0;
        g_llama_deferred_logs.entries[i].log_level = LLAMA_LOG_LEVEL_INFO;
        memset(g_llama_deferred_logs.entries[i].message, 0, LLAMA_LOGGING_ISOLATION_MAX_LOG_MESSAGE_LEN);
    }

    // Set initial state to IDLE
    g_llama_logging_isolation.isolation_state.store(LLAMA_LOGGING_STATE_IDLE, std::memory_order_release);

    return 0;
}

void llama_logging_isolation_fini(void) {
    // Flush any pending deferred logs
    if (g_defer_enabled) {
        llama_logging_deferred_flush();
    }

    // Reset state
    g_llama_logging_isolation.isolation_state.store(LLAMA_LOGGING_STATE_UNINITIALIZED, std::memory_order_release);
}

// ============================================================================
// DECODE WINDOW LIFECYCLE
// ============================================================================

void llama_logging_isolation_decode_start(uint32_t token_count) {
    std::lock_guard<std::mutex> lock(g_logging_isolation_mutex);

    if (!g_logging_isolation_enabled) {
        return;
    }

    // Transition to DECODE_BLACKOUT state
    g_llama_logging_isolation.isolation_state.store(LLAMA_LOGGING_STATE_DECODE_BLACKOUT, std::memory_order_release);
    g_llama_logging_isolation.decode_active.store(true, std::memory_order_release);
    g_llama_logging_isolation.logging_disabled.store(true, std::memory_order_release);

    // Record start timestamp
    uint64_t now = llama_logging_get_timestamp_ns();
    g_llama_logging_isolation.decode_start_ns.store(now, std::memory_order_release);

    // Reset metrics for this decode window
    g_llama_logging_isolation.decode_tokens_processed.store(0, std::memory_order_release);
    g_llama_logging_isolation.decode_duration_ns.store(0, std::memory_order_release);
    g_llama_logging_isolation.first_token_ns.store(0, std::memory_order_release);
    g_llama_logging_isolation.last_token_ns.store(0, std::memory_order_release);
    g_llama_logging_isolation.tokens_with_logging.store(0, std::memory_order_release);
    g_llama_logging_isolation.tokens_zero_logging.store(0, std::memory_order_release);
    g_llama_logging_isolation.isolation_violation.store(false, std::memory_order_release);

    // Clamp token count
    if (token_count > 0) {
        g_llama_logging_isolation.decode_tokens_processed.store(token_count, std::memory_order_release);
    }
}

void llama_logging_isolation_first_token(void) {
    if (!g_logging_isolation_enabled || !g_llama_logging_isolation.decode_active.load(std::memory_order_acquire)) {
        return;
    }

    uint64_t now = llama_logging_get_timestamp_ns();
    g_llama_logging_isolation.first_token_ns.store(now, std::memory_order_release);
}

void llama_logging_isolation_last_token(void) {
    if (!g_logging_isolation_enabled || !g_llama_logging_isolation.decode_active.load(std::memory_order_acquire)) {
        return;
    }

    uint64_t now = llama_logging_get_timestamp_ns();
    g_llama_logging_isolation.last_token_ns.store(now, std::memory_order_release);
}

void llama_logging_isolation_decode_end(void) {
    if (!g_logging_isolation_enabled) {
        return;
    }

    // Record end timestamp
    uint64_t now = llama_logging_get_timestamp_ns();
    uint64_t start = g_llama_logging_isolation.decode_start_ns.load(std::memory_order_acquire);
    g_llama_logging_isolation.decode_end_ns.store(now, std::memory_order_release);

    // Calculate duration
    if (start > 0 && now > start) {
        uint64_t duration = now - start;
        g_llama_logging_isolation.decode_duration_ns.store(duration, std::memory_order_release);
    }

    // Transition to DECODE_COMPLETE state
    g_llama_logging_isolation.decode_active.store(false, std::memory_order_release);
    g_llama_logging_isolation.logging_disabled.store(false, std::memory_order_release);
    g_llama_logging_isolation.isolation_state.store(LLAMA_LOGGING_STATE_DECODE_COMPLETE, std::memory_order_release);

    // Flush deferred logs if enabled
    if (g_defer_enabled) {
        llama_logging_deferred_flush();
    }
}

void llama_logging_isolation_reset(void) {
    std::lock_guard<std::mutex> lock(g_logging_isolation_mutex);

    if (!g_logging_isolation_enabled) {
        return;
    }

    g_llama_logging_isolation.isolation_state.store(LLAMA_LOGGING_STATE_IDLE, std::memory_order_release);
    g_llama_logging_isolation.decode_active.store(false, std::memory_order_release);
    g_llama_logging_isolation.logging_disabled.store(false, std::memory_order_release);
}

// ============================================================================
// LOGGING GUARDS
// ============================================================================

bool llama_logging_check_allowed(void) {
    if (!g_logging_isolation_enabled) {
        return true;
    }

    // Check if we're currently in a decode blackout
    int32_t state = g_llama_logging_isolation.isolation_state.load(std::memory_order_acquire);
    if (state == LLAMA_LOGGING_STATE_DECODE_BLACKOUT) {
        // Logging attempted during blackout - record violation
        llama_logging_isolation_violation_detected(__func__, "Logging attempted during decode blackout");
        return false;
    }

    return true;
}

// ============================================================================
// DEFERRED LOGGING SYSTEM
// ============================================================================

bool llama_logging_defer(int32_t level, const char * message) {
    if (!g_defer_enabled || !message) {
        return false;
    }

    // Check if we should buffer this log level
    if (level > g_verbosity_threshold) {
        return false; // Discard below threshold
    }

    // Get atomic write position
    uint32_t write_pos = g_llama_deferred_logs.write_index.load(std::memory_order_acquire);
    uint32_t next_pos = (write_pos + 1) % LLAMA_LOGGING_ISOLATION_MAX_DEFERRED_LOGS;
    uint32_t read_pos = g_llama_deferred_logs.read_index.load(std::memory_order_acquire);

    // Check for buffer full (ring buffer)
    if (next_pos == read_pos) {
        g_llama_deferred_logs.overflow.store(true, std::memory_order_release);
        return false;
    }

    // Write entry
    llama_deferred_log_entry& entry = g_llama_deferred_logs.entries[write_pos];
    entry.timestamp_ns = llama_logging_get_timestamp_ns();
    entry.log_level = level;
    strncpy(entry.message, message, LLAMA_LOGGING_ISOLATION_MAX_LOG_MESSAGE_LEN - 1);
    entry.message[LLAMA_LOGGING_ISOLATION_MAX_LOG_MESSAGE_LEN - 1] = '\0';

    // Advance write position
    g_llama_deferred_logs.write_index.store(next_pos, std::memory_order_release);
    g_llama_deferred_logs.entry_count.store(
        g_llama_deferred_logs.entry_count.load(std::memory_order_acquire) + 1,
        std::memory_order_release
    );
    g_llama_deferred_logs.total_deferred.store(
        g_llama_deferred_logs.total_deferred.load(std::memory_order_acquire) + 1,
        std::memory_order_release
    );

    return true;
}

void llama_logging_defer_formatted(int32_t level, const char * fmt, ...) {
    if (!g_defer_enabled || !fmt) {
        return;
    }

    // Note: Full varargs formatting would require a thread-safe buffer
    // For now, defer the format string itself
    llama_logging_defer(level, fmt);
}

uint32_t llama_logging_deferred_flush(void) {
    if (!g_defer_enabled) {
        return 0;
    }

    uint32_t flushed = 0;
    uint32_t read_pos = g_llama_deferred_logs.read_index.load(std::memory_order_acquire);
    uint32_t write_pos = g_llama_deferred_logs.write_index.load(std::memory_order_acquire);

    // Process all deferred entries
    while (read_pos != write_pos) {
        const llama_deferred_log_entry& entry = g_llama_deferred_logs.entries[read_pos];

        // Print deferred log
        // Note: In production, this would emit to the actual logging system
        // For now, output to stderr with timestamp
        fprintf(stderr, "[DEFERRED] [%s] %s\n",
            llama_logging_level_name(entry.log_level),
            entry.message);

        // Advance position
        read_pos = (read_pos + 1) % LLAMA_LOGGING_ISOLATION_MAX_DEFERRED_LOGS;
        flushed++;
    }

    // Update read index and counters
    g_llama_deferred_logs.read_index.store(write_pos, std::memory_order_release);
    g_llama_deferred_logs.entry_count.store(0, std::memory_order_release);
    g_llama_deferred_logs.overflow.store(false, std::memory_order_release);

    return flushed;
}

uint32_t llama_logging_deferred_count(void) {
    return g_llama_deferred_logs.entry_count.load(std::memory_order_acquire);
}

void llama_logging_deferred_clear(void) {
    g_llama_deferred_logs.write_index.store(0, std::memory_order_release);
    g_llama_deferred_logs.read_index.store(0, std::memory_order_release);
    g_llama_deferred_logs.entry_count.store(0, std::memory_order_release);
    g_llama_deferred_logs.overflow.store(false, std::memory_order_release);
}

// ============================================================================
// METRICS AND OBSERVABILITY
// ============================================================================

llama_logging_isolation_state llama_logging_isolation_get_state(void) {
    return g_llama_logging_isolation;
}

llama_per_token_logging_metrics llama_logging_isolation_get_metrics(void) {
    llama_per_token_logging_metrics metrics{};
    // Metrics would be accumulated by instrumentation
    return metrics;
}

double llama_logging_isolation_report_throughput(uint32_t tokens_processed, uint64_t duration_ns) {
    if (duration_ns == 0) {
        return 0.0;
    }

    double duration_sec = duration_ns / 1e9;
    double throughput = tokens_processed / duration_sec;

    return throughput;
}

int llama_logging_isolation_validate(void) {
    // Check if any logging violations were detected
    if (g_llama_logging_isolation.isolation_violation.load(std::memory_order_acquire)) {
        uint32_t violations = g_llama_logging_isolation.violation_count.load(std::memory_order_acquire);
        fprintf(stderr, "ERROR: Logging isolation violations detected: %u\n", violations);
        return -1;
    }

    return 0;
}

const char * llama_logging_isolation_state_name(int32_t state) {
    switch (state) {
        case LLAMA_LOGGING_STATE_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_LOGGING_STATE_IDLE: return "IDLE";
        case LLAMA_LOGGING_STATE_PREFILL_ACTIVE: return "PREFILL_ACTIVE";
        case LLAMA_LOGGING_STATE_DECODE_BLACKOUT: return "DECODE_BLACKOUT";
        case LLAMA_LOGGING_STATE_DECODE_COMPLETE: return "DECODE_COMPLETE";
        case LLAMA_LOGGING_STATE_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

const char * llama_logging_level_name(int32_t level) {
    switch (level) {
        case LLAMA_LOG_LEVEL_ERROR: return "ERROR";
        case LLAMA_LOG_LEVEL_WARN: return "WARN";
        case LLAMA_LOG_LEVEL_INFO: return "INFO";
        case LLAMA_LOG_LEVEL_DEBUG: return "DEBUG";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// CONFIGURATION AND CONTROL
// ============================================================================

void llama_logging_isolation_set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(g_logging_isolation_mutex);
    g_logging_isolation_enabled = enabled;
}

bool llama_logging_isolation_is_enabled(void) {
    return g_logging_isolation_enabled;
}

void llama_logging_isolation_set_abort_on_violation(bool abort_on_violation) {
    std::lock_guard<std::mutex> lock(g_logging_isolation_mutex);
    g_abort_on_violation = abort_on_violation;
}

bool llama_logging_isolation_get_abort_on_violation(void) {
    return g_abort_on_violation;
}

void llama_logging_isolation_set_defer_enabled(bool defer) {
    std::lock_guard<std::mutex> lock(g_logging_isolation_mutex);
    g_defer_enabled = defer;
}

bool llama_logging_isolation_get_defer_enabled(void) {
    return g_defer_enabled;
}

void llama_logging_isolation_set_verbosity_threshold(int32_t level) {
    std::lock_guard<std::mutex> lock(g_logging_isolation_mutex);
    g_verbosity_threshold = level;
}

int32_t llama_logging_isolation_get_verbosity_threshold(void) {
    return g_verbosity_threshold;
}

// ============================================================================
// VALIDATION AND ASSERTIONS
// ============================================================================

bool llama_logging_isolation_assert_no_logging(const char * location) {
    int32_t state = g_llama_logging_isolation.isolation_state.load(std::memory_order_acquire);
    if (state == LLAMA_LOGGING_STATE_DECODE_BLACKOUT) {
        // Assert violation
        if (g_abort_on_violation) {
            fprintf(stderr, "FATAL: Logging during decode blackout at %s\n", location ? location : "unknown");
            abort();
        }
        return false;
    }
    return true;
}

void llama_logging_isolation_violation_detected(const char * location, const char * message) {
    // Record violation
    g_llama_logging_isolation.isolation_violation.store(true, std::memory_order_release);
    uint32_t count = g_llama_logging_isolation.violation_count.load(std::memory_order_acquire);
    g_llama_logging_isolation.violation_count.store(count + 1, std::memory_order_release);

    if (g_abort_on_violation) {
        fprintf(stderr, "FATAL: Logging isolation violation at %s: %s\n",
            location ? location : "unknown",
            message ? message : "no details");
        abort();
    } else {
        fprintf(stderr, "WARNING: Logging isolation violation at %s: %s\n",
            location ? location : "unknown",
            message ? message : "no details");
    }
}

uint32_t llama_logging_isolation_get_violation_count(void) {
    return g_llama_logging_isolation.violation_count.load(std::memory_order_acquire);
}

void llama_logging_isolation_clear_violations(void) {
    std::lock_guard<std::mutex> lock(g_logging_isolation_mutex);
    g_llama_logging_isolation.violation_count.store(0, std::memory_order_release);
    g_llama_logging_isolation.isolation_violation.store(false, std::memory_order_release);
}

// ============================================================================
// LOCK CONTENTION DETECTION
// ============================================================================

void llama_logging_record_lock_acquisition(uint64_t wait_time_ns) {
    if (!g_logging_isolation_enabled) {
        return;
    }

    // Only track during decode window
    if (!g_llama_logging_isolation.decode_active.load(std::memory_order_acquire)) {
        return;
    }

    // Increment acquisition count
    uint64_t acquisitions = g_llama_logging_isolation.logging_lock_waits.load(std::memory_order_acquire);
    g_llama_logging_isolation.logging_lock_waits.store(acquisitions + 1, std::memory_order_release);

    // Track max wait time
    uint64_t max_wait = g_llama_logging_isolation.logging_lock_max_wait_ns.load(std::memory_order_acquire);
    if (wait_time_ns > max_wait) {
        g_llama_logging_isolation.logging_lock_max_wait_ns.store(wait_time_ns, std::memory_order_release);
    }

    // Record as violation if lock contention detected
    if (wait_time_ns > 0) {
        llama_logging_isolation_violation_detected(__func__, "Logging lock acquisition during decode");
    }
}

struct llama_logging_lock_metrics llama_logging_isolation_get_lock_metrics(void) {
    struct llama_logging_lock_metrics metrics;
    metrics.total_acquisitions = g_llama_logging_isolation.logging_lock_waits.load(std::memory_order_acquire);
    metrics.max_wait_ns = g_llama_logging_isolation.logging_lock_max_wait_ns.load(std::memory_order_acquire);
    metrics.total_wait_ns = 0; // Would need additional tracking
    return metrics;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

char * llama_logging_format_duration(uint64_t ns, char * buf) {
    if (!buf) {
        return nullptr;
    }

    if (ns < 1000) {
        snprintf(buf, 32, "%lu ns", (unsigned long)ns);
    } else if (ns < 1000000) {
        snprintf(buf, 32, "%.2f us", ns / 1000.0);
    } else if (ns < 1000000000) {
        snprintf(buf, 32, "%.2f ms", ns / 1000000.0);
    } else {
        snprintf(buf, 32, "%.2f s", ns / 1000000000.0);
    }

    return buf;
}

char * llama_logging_format_throughput(double tokens_per_sec, char * buf) {
    if (!buf) {
        return nullptr;
    }

    snprintf(buf, 32, "%.2f tokens/sec", tokens_per_sec);
    return buf;
}

uint64_t llama_logging_get_timestamp_ns(void) {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}
