#pragma once

/**
 * Server-Decode Thread Isolation Enforcement for LLAMA
 *
 * Physical and logical separation of server-control execution from decode execution.
 * Server infrastructure must never share scheduling resources with the decode-critical path.
 *
 * Key Properties:
 * - Dedicated CPU core set for decode threads
 * - Separate CPU core set for server threads
 * - No shared thread pools between server and decode
 * - No cross-domain mutexes blocking decode
 * - Async streaming without blocking decode thread
 * - No per-token JSON serialization in decode
 * - Decode thread protected from preemption
 * - Admission backpressure when server overloaded
 * - Runtime validation of core affinity
 * - Fast-fail on isolation violations
 *
 * Expected Outcome:
 * - Server activity cannot starve decode
 * - HTTP spikes do not reduce tokens/sec
 * - Decode timing jitter decreases
 * - GPU feed remains stable under server load
 * - Decode remains protected real-time execution domain
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <thread>
#include <vector>
#include <string>
#include <array>
#include <queue>
#include <mutex>
#include <atomic>
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Maximum CPU cores supported in isolation system
 */
#define DECODE_ISOLATION_MAX_CORES 256
#define DECODE_ISOLATION_MAX_THREADS 64
#define DECODE_STREAMING_QUEUE_SIZE 1024

/**
 * Core set representation - bitmask for CPU affinity
 */
struct decode_core_set {
    std::array<uint8_t, DECODE_ISOLATION_MAX_CORES / 8> mask;
    int32_t core_count;

    decode_core_set();

    bool add_core(int32_t core_id);
    bool remove_core(int32_t core_id);
    bool contains(int32_t core_id) const;

    std::vector<int32_t> get_cores() const;
    bool overlaps_with(const decode_core_set & other) const;
    bool is_disjoint_from(const decode_core_set & other) const;

    void * get_native_mask() const;
};

/**
 * Isolation violation record - tracks constraint violations
 */
struct isolation_violation {
    std::string violation_type;
    std::string details;
    int32_t thread_id;
    int64_t timestamp_us;
    bool is_fatal;
};

/**
 * Streaming token event - non-blocking token transfer
 */
struct decode_token_event {
    uint32_t token_id;
    int32_t seq_id;
    bool is_eos;
    int64_t timestamp_ns;
};

#ifdef __cplusplus
}
#endif

/**
 * Streaming queue for lock-free token passing
 */
template<typename T>
class decode_streaming_queue {
private:
    std::vector<T> buffer;
    std::atomic<size_t> read_pos;
    std::atomic<size_t> write_pos;
    size_t capacity;

public:
    decode_streaming_queue(size_t capacity);

    bool try_push(const T & token);
    bool try_pop(T & token);

    bool is_full() const;
    bool is_empty() const;
    size_t size() const;
    void clear();
};

/**
 * Streaming manager - async token consumption without blocking decode
 */
class streaming_manager {
private:
    decode_streaming_queue<decode_token_event> token_queue;
    std::atomic<bool> streaming_active;

public:
    streaming_manager(size_t queue_capacity);

    bool initialize(size_t queue_capacity);

    bool decode_push_token(const decode_token_event & event);
    bool server_consume_token(decode_token_event & event);

    void clear();
};

/**
 * Cross-domain lock detector - prevents decode thread from holding server locks
 */
class cross_domain_lock_detector {
public:
    enum domain_type {
        DOMAIN_DECODE = 0,
        DOMAIN_SERVER = 1
    };

    void enter_critical_section(const std::string & lock_name, int32_t domain);
    void exit_critical_section(const std::string & lock_name);

    bool has_decode_server_contention() const;
    std::vector<std::string> get_contended_locks() const;
};

/**
 * Admission control - backpressure for server load
 */
class admission_control {
private:
    int64_t decode_latency_threshold_us;
    int32_t max_queue_depth;
    std::atomic<int32_t> current_queue_depth;
    std::atomic<int64_t> recent_decode_latency_us;

public:
    admission_control();

    bool initialize(int64_t latency_threshold_us, int32_t max_depth);

    bool try_admit_request();
    void record_decode_latency(int64_t latency_us);
    void update_queue_depth(int32_t new_depth);

    int32_t get_queue_depth() const;
    int64_t get_avg_decode_latency() const;
};

/**
 * Decode isolation engine - main orchestrator
 */
class decode_isolation_engine {
private:
    decode_core_set decode_cores;
    decode_core_set server_cores;

    std::vector<isolation_violation> violations;
    std::atomic<bool> strict_mode;
    std::atomic<bool> is_initialized;

    std::unique_ptr<streaming_manager> streaming;
    std::unique_ptr<cross_domain_lock_detector> lock_detector;
    std::unique_ptr<admission_control> admission;

public:
    decode_isolation_engine();

    // Initialization and configuration
    bool initialize(const std::vector<int32_t> & decode_core_list,
                    const std::vector<int32_t> & server_core_list);

    bool set_decode_cores(const std::vector<int32_t> & core_ids);
    bool set_server_cores(const std::vector<int32_t> & core_ids);

    // Thread pinning
    bool pin_decode_thread(std::thread::id tid, int32_t decode_thread_index);
    bool pin_server_thread(std::thread::id tid, int32_t server_worker_index);

    // Priority control
    bool set_decode_priority(std::thread::id tid, int32_t priority);

    // Validation
    bool validate_configuration() const;
    bool validate_runtime() const;

    // Query
    bool is_thread_on_decode_cores(std::thread::id tid) const;
    decode_core_set get_decode_cores() const { return decode_cores; }
    decode_core_set get_server_cores() const { return server_cores; }

    // Violation tracking
    void record_violation(const std::string & violation_type, const std::string & details);
    void abort_if_violated(bool check_runtime = false) const;

    size_t get_violation_count() const { return violations.size(); }
    std::vector<isolation_violation> get_violations() const { return violations; }

    // Enforcement mode
    void set_strict_mode(bool strict) { strict_mode.store(strict); }
    bool get_strict_mode() const { return strict_mode.load(); }

    // Platform-specific affinity operations
    bool platform_set_affinity(std::thread::id tid, const decode_core_set & cores);
    bool platform_get_affinity(std::thread::id tid, decode_core_set & cores) const;
    bool platform_set_priority(std::thread::id tid, int32_t priority);

private:
    bool validate_non_overlapping_cores() const;
    bool validate_all_cores_assigned() const;
};

/**
 * Global instance management
 */
#ifdef __cplusplus
extern "C" {
#endif
extern decode_isolation_engine * g_decode_isolation_engine;

bool llama_init_server_decode_isolation(const std::vector<int32_t> & decode_cores,
                                         const std::vector<int32_t> & server_cores);

bool llama_pin_thread_to_decode_core(std::thread::id tid, int32_t core_index);
bool llama_pin_thread_to_server_core(std::thread::id tid, int32_t worker_index);

bool llama_validate_server_decode_isolation();

size_t llama_get_isolation_violation_count();
void llama_abort_on_isolation_violation();

// Configuration
void llama_set_isolation_strict_mode(bool strict);
bool llama_get_isolation_strict_mode();

// Streaming support
bool llama_push_token_to_stream(const decode_token_event & event);
bool llama_consume_token_from_stream(decode_token_event & event);

// Admission control
bool llama_try_admit_server_request();
void llama_record_decode_latency(int64_t latency_us);

// Diagnostics
void llama_print_isolation_diagnostics();
void llama_dump_isolation_state();

#ifdef __cplusplus
}  // extern "C"
#endif
