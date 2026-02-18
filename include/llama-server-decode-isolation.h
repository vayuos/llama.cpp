#pragma once

/**
 * llama-server-decode-isolation.h
 *
 * Comprehensive isolation layer separating server control threads from GPU decode threads.
 * Enforces complete physical and logical separation of execution domains with runtime validation.
 *
 * Requirements enforced:
 * - Dedicated CPU core sets for decode vs server
 * - No shared thread pools between domains
 * - No cross-domain lock contention
 * - Decoupled streaming with lock-free queues
 * - Protected decode real-time performance
 * - Server backpressure and admission control
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <array>
#include <memory>
#include <functional>

#ifdef _WIN32
    #include <windows.h>
    #include <process.h>
#else
    #include <pthread.h>
    #include <sched.h>
    #include <unistd.h>
#endif

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

// Maximum number of CPU cores supported
#define DECODE_ISOLATION_MAX_CORES 256

// Maximum decode threads (typically 1-4 for GPU)
#define DECODE_ISOLATION_MAX_DECODE_THREADS 4

// Maximum server worker threads
#define DECODE_ISOLATION_MAX_SERVER_THREADS 512

// Lock-free queue capacity for streaming tokens
#define DECODE_ISOLATION_STREAMING_QUEUE_SIZE 4096

// Monitoring sample window (milliseconds)
#define DECODE_ISOLATION_MONITOR_WINDOW_MS 1000

// Maximum allowed decode thread migration events per monitor window
#define DECODE_ISOLATION_MAX_MIGRATIONS_ALLOWED 0

// ============================================================================
// CORE SET MANAGEMENT
// ============================================================================

/**
 * CPU core set representation for thread affinity.
 * Supports up to 256 cores on modern systems.
 */
struct decode_core_set {
    std::array<uint8_t, DECODE_ISOLATION_MAX_CORES / 8> mask;
    int32_t core_count;

    decode_core_set();

    /**
     * Add a single core to the set
     * @param core_id CPU core identifier (0-based)
     * @return true if added successfully, false if already in set
     */
    bool add_core(int32_t core_id);

    /**
     * Remove a core from the set
     * @param core_id CPU core identifier
     * @return true if removed successfully, false if not in set
     */
    bool remove_core(int32_t core_id);

    /**
     * Check if a core is in this set
     * @param core_id CPU core identifier
     * @return true if core is in set
     */
    bool contains(int32_t core_id) const;

    /**
     * Get all cores in this set as vector
     * @return vector of core IDs
     */
    std::vector<int32_t> get_cores() const;

    /**
     * Check for overlap with another set
     * @param other other core set
     * @return true if sets have common cores
     */
    bool overlaps_with(const decode_core_set & other) const;

    /**
     * Check if sets are disjoint (no overlap)
     * @param other other core set
     * @return true if no common cores
     */
    bool is_disjoint_from(const decode_core_set & other) const;

    /**
     * Get the native OS representation
     * On Linux: cpu_set_t, On Windows: GROUP_AFFINITY
     */
    void * get_native_mask() const;
};

// ============================================================================
// EXECUTION DOMAIN DEFINITIONS
// ============================================================================

/**
 * Decode execution domain - GPU decode threads with real-time priority
 */
struct decode_domain {
    decode_core_set core_set;
    int32_t thread_count;
    int32_t scheduling_priority;  // 0=normal, 1=medium, 2=high, 3=realtime
    bool is_enabled;

    // Metrics tracking
    std::atomic<uint64_t> thread_migrations;
    std::atomic<uint64_t> scheduling_preemptions;
    std::atomic<uint64_t> lock_waits_detected;

    decode_domain();
};

/**
 * Server execution domain - HTTP and control plane threads
 */
struct server_domain {
    decode_core_set core_set;
    int32_t thread_count;
    int32_t scheduling_priority;  // Typically normal (0)
    bool is_enabled;

    // Metrics tracking
    std::atomic<uint64_t> violations_detected;
    std::atomic<uint64_t> threads_on_decode_cores;
    std::atomic<uint64_t> admission_rejections;

    server_domain();
};

// ============================================================================
// ISOLATION ENFORCEMENT ENGINE
// ============================================================================

/**
 * Central isolation enforcement system.
 * Manages domain separation, validates enforcement rules, and tracks violations.
 */
class decode_isolation_engine {
public:
    /**
     * Initialize isolation engine with domain configurations.
     * @param decode_cores vector of core IDs reserved for decode
     * @param server_cores vector of core IDs reserved for server
     * @param decode_priority scheduling priority for decode threads (0-3)
     * @param decode_thread_count expected number of decode threads
     * @param server_thread_count expected number of server worker threads
     * @return true if initialization successful, false if cores overlap or invalid config
     */
    bool initialize(
        const std::vector<int32_t> & decode_cores,
        const std::vector<int32_t> & server_cores,
        int32_t decode_priority = 2,  // High priority for decode
        int32_t decode_thread_count = 1,
        int32_t server_thread_count = 4
    );

    /**
     * Pin a thread to the decode core set.
     * Enforces exclusive core allocation for GPU decode.
     * @param tid thread ID (platform-specific)
     * @param decode_thread_index index in decode domain (0-based)
     * @return true if affinity set successfully
     */
    bool pin_decode_thread(std::thread::id tid, int32_t decode_thread_index);

    /**
     * Pin a thread to the server core set.
     * Prevents server threads from running on decode cores.
     * @param tid thread ID
     * @param server_worker_index index in server domain (0-based)
     * @return true if affinity set successfully
     */
    bool pin_server_thread(std::thread::id tid, int32_t server_worker_index);

    /**
     * Set scheduling priority for decode threads.
     * Raise priority to SCHED_FIFO if available.
     * @param tid thread ID
     * @param priority ggml_sched_priority enum value
     * @return true if priority set successfully
     */
    bool set_decode_priority(std::thread::id tid, int32_t priority);

    /**
     * Validate domain configuration consistency.
     * Checks: no core overlap, cores within system limit, thread counts valid.
     * @return true if configuration is valid
     */
    bool validate_configuration() const;

    /**
     * Perform runtime validation of isolation enforcement.
     * Checks: decode threads on correct cores, no server threads on decode cores,
     * no cross-domain lock contention.
     * @return true if all enforcement rules satisfied
     */
    bool validate_runtime() const;

    /**
     * Check if a thread is currently on a decode core (violation detection).
     * @param tid thread ID
     * @return true if thread found on decode core
     */
    bool is_thread_on_decode_cores(std::thread::id tid) const;

    /**
     * Get current CPU affinity for a thread.
     * @param tid thread ID
     * @return core set representing current affinity
     */
    decode_core_set get_thread_affinity(std::thread::id tid) const;

    /**
     * Record a domain violation for metrics and diagnostics.
     * @param violation_type string describing violation
     * @param details additional context
     */
    void record_violation(const std::string & violation_type, const std::string & details);

    /**
     * Get enforcement metrics.
     * @return struct with current metrics
     */
    struct isolation_metrics get_metrics() const;

    /**
     * Get decode domain configuration.
     * @return const reference to decode domain
     */
    const decode_domain & get_decode_domain() const;

    /**
     * Get server domain configuration.
     * @return const reference to server domain
     */
    const server_domain & get_server_domain() const;

    /**
     * Fail fast if configuration invalid or violation detected.
     * Aborts with descriptive error message.
     * @param check_runtime if true, also check runtime state
     */
    void abort_if_violated(bool check_runtime = false) const;

    /**
     * Singleton accessor
     */
    static decode_isolation_engine & instance();

private:
    decode_domain decode_dom;
    server_domain server_dom;
    std::mutex metrics_mutex;
    std::vector<std::string> violation_log;

    // Platform-specific affinity enforcement
    bool platform_set_affinity(std::thread::id tid, const decode_core_set & cores);
    bool platform_get_affinity(std::thread::id tid, decode_core_set & cores) const;
    bool platform_set_priority(std::thread::id tid, int32_t priority);
};

// ============================================================================
// ISOLATION METRICS
// ============================================================================

/**
 * Real-time metrics for isolation enforcement and performance monitoring.
 */
struct isolation_metrics {
    // Decode domain metrics
    uint64_t decode_migrations;
    uint64_t decode_preemptions;
    uint64_t decode_lock_waits;

    // Server domain metrics
    uint64_t server_violations;
    uint64_t server_threads_on_decode_cores;
    uint64_t server_admission_rejections;

    // Correlation metrics
    float server_load_percent;
    float decode_throughput_tokens_per_sec;
    float decode_latency_variance_us;
};

// ============================================================================
// LOCK-FREE STREAMING QUEUE
// ============================================================================

/**
 * Lock-free queue for streaming decoded tokens from decode thread to server.
 * Uses single-producer (decode) / multiple-consumer (server workers) pattern.
 * Prevents decode thread from waiting on server queue state.
 */
template<typename T>
class decode_streaming_queue {
public:
    decode_streaming_queue(size_t capacity = DECODE_ISOLATION_STREAMING_QUEUE_SIZE);

    /**
     * Try to push token to queue (non-blocking).
     * Called from decode thread.
     * @param token token to enqueue
     * @return true if enqueued successfully, false if queue full (backpressure)
     */
    bool try_push(const T & token);

    /**
     * Try to pop token from queue (non-blocking).
     * Called from server worker threads.
     * @param token output parameter for dequeued token
     * @return true if token dequeued, false if queue empty
     */
    bool try_pop(T & token);

    /**
     * Get current queue depth.
     * @return number of tokens in queue
     */
    size_t depth() const;

    /**
     * Get queue capacity.
     * @return maximum queue depth
     */
    size_t capacity() const;

    /**
     * Check if queue is full (decode should backpressure).
     * @return true if at capacity
     */
    bool is_full() const;

    /**
     * Check if queue is empty.
     * @return true if no tokens available
     */
    bool is_empty() const;

    /**
     * Reset queue to empty state.
     */
    void clear();

private:
    std::vector<T> buffer;
    std::atomic<uint64_t> head;
    std::atomic<uint64_t> tail;
    size_t mask;
};

// ============================================================================
// STREAMING INTEGRATION
// ============================================================================

/**
 * Token produced by decode thread for streaming to server.
 */
struct decode_token_event {
    int32_t token_id;
    float logit;
    bool is_eos;
    int64_t timestamp_us;  // When token was decoded
    uint32_t sequence_id;  // Which decode sequence produced this
};

/**
 * Streaming manager decouples decode from server processing.
 * Decode pushes tokens immediately to non-blocking buffer.
 * Server workers consume asynchronously.
 */
class streaming_manager {
public:
    /**
     * Initialize streaming manager.
     * @param queue_capacity streaming queue size
     * @return true if initialization successful
     */
    bool initialize(size_t queue_capacity = DECODE_ISOLATION_STREAMING_QUEUE_SIZE);

    /**
     * Decode thread: push decoded token to streaming buffer.
     * Must never block decode thread - returns false on full queue only.
     * @param event token event to stream
     * @return true if enqueued, false if backpressure triggered
     */
    bool decode_push_token(const decode_token_event & event);

    /**
     * Server worker: consume token from streaming buffer.
     * Non-blocking - returns false if no tokens available.
     * @param event output parameter for token event
     * @return true if token consumed, false if queue empty
     */
    bool server_consume_token(decode_token_event & event);

    /**
     * Get streaming queue metrics.
     * @return struct with depth, capacity, backpressure events
     */
    struct streaming_metrics get_metrics() const;

    /**
     * Reset streaming state.
     */
    void clear();

    /**
     * Singleton accessor
     */
    static streaming_manager & instance();

private:
    decode_streaming_queue<decode_token_event> queue;
    std::atomic<uint64_t> backpressure_events;
    std::atomic<uint64_t> tokens_produced;
    std::atomic<uint64_t> tokens_consumed;
};

/**
 * Streaming metrics.
 */
struct streaming_metrics {
    size_t queue_depth;
    size_t queue_capacity;
    uint64_t backpressure_events;
    uint64_t total_tokens_produced;
    uint64_t total_tokens_consumed;
    float tokens_per_sec;
};

// ============================================================================
// SYNCHRONIZATION PRIMITIVES
// ============================================================================

/**
 * Cross-domain lock detector.
 * Logs warning when decode thread acquires server-owned locks.
 * Used to identify remaining synchronization coupling.
 */
class cross_domain_lock_detector {
public:
    /**
     * Mark entry to critical section.
     * @param lock_name identifier for the lock
     * @param domain which domain (0=decode, 1=server)
     */
    static void enter_critical_section(const std::string & lock_name, int32_t domain);

    /**
     * Mark exit from critical section.
     * @param lock_name identifier for the lock
     */
    static void exit_critical_section(const std::string & lock_name);

    /**
     * Check if decode thread is holding server locks.
     * @return true if violation detected
     */
    static bool has_decode_server_contention();

    /**
     * Get lock contention statistics.
     * @return number of cross-domain lock acquisitions
     */
    static uint64_t get_contention_count();

    /**
     * Singleton accessor
     */
    static cross_domain_lock_detector & instance();
};

// ============================================================================
// ADMISSION CONTROL
// ============================================================================

/**
 * Server-side admission control prevents overload during decode.
 * Implements backpressure to queue requests when decode has high latency.
 */
class admission_control {
public:
    /**
     * Initialize admission control.
     * @param decode_latency_threshold_us max acceptable decode latency
     * @param max_queue_depth maximum pending request queue depth
     * @return true if initialized
     */
    bool initialize(int64_t decode_latency_threshold_us = 10000, int32_t max_queue_depth = 1000);

    /**
     * Server worker: try to admit new request.
     * Returns false if decode is overloaded and queue is full (apply backpressure).
     * @return true if request should be admitted, false if should be queued/rejected
     */
    bool try_admit_request();

    /**
     * Record decode latency measurement.
     * Used to detect overload and trigger backpressure.
     * @param latency_us measured decode latency in microseconds
     */
    void record_decode_latency(int64_t latency_us);

    /**
     * Get admission queue metrics.
     * @return queue depth, rejection count, etc.
     */
    struct admission_metrics get_metrics() const;

    /**
     * Singleton accessor
     */
    static admission_control & instance();

private:
    std::atomic<int32_t> pending_queue_depth;
    std::atomic<int64_t> last_decode_latency_us;
    int64_t decode_latency_threshold_us;
    int32_t max_queue_depth;
    std::atomic<uint64_t> admissions_rejected;
};

/**
 * Admission control metrics.
 */
struct admission_metrics {
    int32_t pending_queue_depth;
    int64_t current_decode_latency_us;
    uint64_t admissions_rejected;
    float admission_rejection_rate;
};

// ============================================================================
// INITIALIZATION HELPERS
// ============================================================================

/**
 * Helper to initialize full isolation system.
 * Call once at server startup before any threads created.
 *
 * @param total_cores system CPU core count
 * @param decode_core_ids specific cores to reserve for decode
 * @param decode_priority priority level for decode (0-3)
 * @param decode_thread_count number of decode threads (usually 1-2)
 * @param server_thread_count number of HTTP worker threads
 * @return true if fully initialized, false if configuration error
 */
bool initialize_decode_isolation(
    int32_t total_cores,
    const std::vector<int32_t> & decode_core_ids,
    int32_t decode_priority = 2,
    int32_t decode_thread_count = 1,
    int32_t server_thread_count = 4
);

/**
 * Helper to pin current thread to decode domain.
 * Call from decode thread startup.
 *
 * @param thread_index which decode thread (0-based)
 * @return true if successfully pinned
 */
bool pin_current_thread_to_decode(int32_t thread_index);

/**
 * Helper to pin current thread to server domain.
 * Call from HTTP worker thread startup.
 *
 * @param worker_index which server worker (0-based)
 * @return true if successfully pinned
 */
bool pin_current_thread_to_server(int32_t worker_index);

/**
 * Helper to validate full system configuration at startup.
 * Aborts with error if configuration invalid.
 */
void validate_isolation_config();

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

/**
 * Log detailed isolation enforcement state.
 * Shows core assignments, thread affinities, metrics, any violations.
 */
void dump_isolation_state();

/**
 * Get human-readable error message for last violation.
 * @return error message string
 */
const char * get_last_violation_message();

/**
 * Enable verbose logging of isolation enforcement.
 * @param enable true to enable, false to disable
 */
void set_isolation_verbose_logging(bool enable);

#endif // LLAMA_SERVER_DECODE_ISOLATION_H
