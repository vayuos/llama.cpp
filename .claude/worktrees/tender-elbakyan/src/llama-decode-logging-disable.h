#pragma once

/**
 * Decode Logging Blackout Enforcement for LLAMA
 *
 * Complete removal of logging activity from the decode-critical window.
 * Logging must never execute inside the token-generation dependency chain.
 *
 * Key Properties:
 * - Zero logging during decode (all levels disabled)
 * - Decode-mode logging guard for all logging calls
 * - No per-token log emission
 * - No structured logging serialization during decode
 * - No error logging inside hot path
 * - No verbose server modes during decode
 * - No logging mutex acquisition in decode thread
 * - Deferred logging after decode completion
 * - Compile-time logging elimination option
 * - Atomic counters for metrics-only tracking
 *
 * Blackout Window Definition:
 * decode_start → first_token_forward → ... → last_token_selected → decode_end
 * Any logging inside this boundary is a correctness violation.
 *
 * Expected Outcome:
 * - No mutex contention from logging framework
 * - No I/O syscalls per token
 * - No string formatting in decode path
 * - No CPU jitter from log emission
 * - Stable GPU dispatch cadence
 * - Reduced CPU cache pollution
 * - Per-token performance: consistent microsecond-scale
 * - Throughput: independent of logging configuration
 *
 * 10 Enforcement Rules Implemented:
 * 1. Define Decode Logging Blackout Window
 * 2. Audit All Logging Calls on Decode Path
 * 3. Introduce Decode-Mode Logging Guard
 * 4. Eliminate Per-Token Log Emission
 * 5. Disable Structured Logging During Decode
 * 6. Disable Error Logging Inside Hot Path
 * 7. Disable Verbose Server Modes During Decode
 * 8. Remove Logging Locks from Decode
 * 9. Validate Logging Isolation
 * 10. Measure Performance Impact
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Decode logging phase definition
 */
typedef enum {
    DECODE_LOGGING_UNINITIALIZED = 0,
    DECODE_LOGGING_STARTUP = 1,        // Before decode starts
    DECODE_LOGGING_BLACKOUT_ACTIVE = 2, // Decode in progress - logging disabled
    DECODE_LOGGING_COMPLETE = 3         // Decode finished - logging re-enabled
} decode_logging_phase;

/**
 * Logging call audit entry - records all logging found in decode path
 */
typedef struct {
    const char * file_path;
    int line_number;
    const char * function_name;
    const char * log_call_type;         // LOG, fprintf, std::cout, std::cerr, spdlog
    const char * scope_description;
    const char * reason;
    bool is_on_critical_path;
    bool is_removed;
    int violation_count;
} logging_audit_entry;

/**
 * Decode logging validation result
 */
typedef struct {
    bool is_clean;
    uint32_t remaining_logging_calls;
    uint32_t on_critical_path_count;
    uint64_t total_decode_cycles;
    double avg_decode_time_us;
    uint64_t metrics_atomic_increments;
} decode_logging_validation_result;

/**
 * Per-request logging statistics
 */
typedef struct {
    uint64_t request_id;
    uint32_t tokens_generated;
    double total_decode_time_us;
    uint64_t logging_suppressed_count;
    bool decode_mode_active;
} request_logging_metrics;

/**
 * Decode logging suppression engine
 */
class decode_logging_suppression_engine {
private:
    std::vector<logging_audit_entry> logging_audit_log;
    std::vector<logging_audit_entry> removed_logging_calls;

    std::atomic<decode_logging_phase> current_phase;
    std::atomic<bool> enforcement_enabled;
    std::atomic<bool> decode_mode_active;
    std::atomic<bool> strict_mode;

    // Per-request metrics (using thread-local in implementation)
    std::map<uint64_t, request_logging_metrics> per_request_metrics;

    // Statistics
    std::atomic<uint64_t> total_decode_requests;
    std::atomic<uint64_t> total_tokens_generated;
    std::atomic<uint64_t> total_suppressed_logs;
    std::atomic<uint64_t> metrics_increments;
    std::atomic<uint64_t> accumulated_decode_time_us;

public:
    decode_logging_suppression_engine();

    // Initialization
    bool initialize();
    bool enable_enforcement(bool enable);

    // Decode phase management
    void enter_decode_blackout_window();
    void exit_decode_blackout_window();
    decode_logging_phase get_current_phase() const;
    bool is_decode_mode_active() const;

    // Logging audit
    void audit_logging_call(const char * file, int line, const char * func,
                            const char * log_type, const char * scope,
                            const char * reason, bool on_critical_path);

    size_t get_logging_call_count() const { return logging_audit_log.size(); }
    std::vector<logging_audit_entry> get_logging_audit() const { return logging_audit_log; }

    void record_logging_removal(const logging_audit_entry & entry);
    size_t get_removed_logging_count() const { return removed_logging_calls.size(); }
    std::vector<logging_audit_entry> get_removed_logging_calls() const { return removed_logging_calls; }

    // Per-request metrics
    void initialize_request_metrics(uint64_t request_id);
    void record_token_generated(uint64_t request_id);
    void record_decode_time(uint64_t request_id, double time_us);
    void record_logging_suppressed(uint64_t request_id);
    request_logging_metrics get_request_metrics(uint64_t request_id) const;

    // Statistics
    uint64_t get_total_decode_requests() const { return total_decode_requests.load(); }
    uint64_t get_total_tokens_generated() const { return total_tokens_generated.load(); }
    uint64_t get_total_suppressed_logs() const { return total_suppressed_logs.load(); }
    uint64_t get_metrics_increments() const { return metrics_increments.load(); }
    double get_avg_decode_time_us() const;

    void record_decode_request() { total_decode_requests.fetch_add(1); }
    void record_token() { total_tokens_generated.fetch_add(1); }
    void record_suppressed_log() { total_suppressed_logs.fetch_add(1); }
    void record_metrics_increment() { metrics_increments.fetch_add(1); }

    // Validation
    decode_logging_validation_result validate_logging_blackout() const;
    bool verify_decode_mode_isolation() const;
    bool verify_no_logging_locks_acquired() const;
    bool verify_structured_logging_deferred() const;
    bool verify_throughput_independent_of_logging() const;

    // Enforcement mode
    void set_strict_mode(bool strict) { strict_mode.store(strict); }
    bool get_strict_mode() const { return strict_mode.load(); }
};

/**
 * Logging guard wrapper - prevents logging in decode mode
 */
class decode_mode_logging_guard {
private:
    bool is_inside_decode;
    const char * log_identifier;

public:
    decode_mode_logging_guard(const char * identifier);
    ~decode_mode_logging_guard();

    bool should_log() const;
    void record_suppressed();
};

/**
 * Global engine instance
 */
extern decode_logging_suppression_engine * g_decode_logging_suppression_engine;

// Initialization
bool llama_init_decode_logging_suppression();
bool llama_enable_decode_logging_suppression(bool enable);

// Decode phase tracking
void llama_enter_decode_blackout_window();
void llama_exit_decode_blackout_window();
bool llama_is_decode_mode_active();

// Logging audit
void llama_audit_logging_call(const char * file, int line, const char * func,
                              const char * log_type, const char * scope,
                              const char * reason, bool on_critical_path);

// Per-request metrics
void llama_initialize_request_logging_metrics(uint64_t request_id);
void llama_record_request_token(uint64_t request_id);
void llama_record_request_decode_time(uint64_t request_id, double time_us);

// Validation
bool llama_validate_logging_blackout();
bool llama_validate_decode_logging_isolation();

// Diagnostics
void llama_print_logging_audit_report();
void llama_print_decode_logging_validation_results();
void llama_dump_request_logging_metrics();

// Module initialization
bool llama_init_decode_logging_suppression_module(void);
void llama_cleanup_decode_logging_suppression_module(void);

// Enforcement macros for logging guards
#define DECODE_LOGGING_GUARD(identifier) \
    do { \
        if (g_decode_logging_suppression_engine && llama_is_decode_mode_active()) { \
            /* Log suppressed during decode blackout */ \
            g_decode_logging_suppression_engine->record_suppressed_log(); \
            return; /* Suppress logging emission */ \
        } \
    } while(0)

#define STRUCTURED_LOGGING_DEFER() \
    do { \
        if (g_decode_logging_suppression_engine && llama_is_decode_mode_active()) { \
            /* Defer structured logging until after decode */ \
            g_decode_logging_suppression_engine->record_metrics_increment(); \
            return; /* Do not serialize */ \
        } \
    } while(0)

#define ERROR_LOGGING_PROPAGATE() \
    do { \
        if (g_decode_logging_suppression_engine && llama_is_decode_mode_active()) { \
            /* Convert to error code propagation */ \
            /* Do not emit error log */ \
            return -1; /* Error code return */ \
        } \
    } while(0)

#ifdef __cplusplus
}  // extern "C"
#endif
