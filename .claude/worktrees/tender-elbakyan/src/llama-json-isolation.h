#pragma once

/**
 * JSON Serialization Path Isolation for LLAMA Decode
 *
 * Complete elimination of JSON construction and formatting from decode-critical path.
 * Implements strict architectural isolation: decode produces token IDs + minimal records,
 * server workers handle all JSON serialization asynchronously.
 *
 * Key Properties:
 * - Zero JSON objects created in decode loop
 * - Zero string concatenation per token
 * - No rapidjson/nlohmann usage in decode
 * - No formatting/escaping in decode thread
 * - Decode: token ID + minimal record → lock-free queue → return immediately
 * - Server worker: pop token → text conversion → JSON serialization → HTTP flush
 * - All heap allocation outside decode-critical region
 * - Non-blocking token flow: GPU decode → enqueue → return (microseconds, no blocking)
 * - Formatting/serialization asynchronous on dedicated server workers
 *
 * Expected Outcome:
 * - Zero JSON serialization in decode measurements
 * - Tokens/sec identical with/without streaming enabled
 * - Decode CPU time independent of output bandwidth
 * - Sub-microsecond token enqueue latency
 * - All JSON work measured on server worker thread, not decode thread
 * - Hard assertion failures if JSON library called from decode
 *
 * Enforcement Mechanism:
 * - State machine: UNINITIALIZED → CONFIGURED → ENFORCING → STREAMING
 * - Per-token validation: no JSON calls detected
 * - Decode-critical section guards: asserts on library calls
 * - Metrics tracking: JSON operations, allocations, thread origin
 * - Fast-fail on any critical path violation
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <atomic>
#include <memory>
#include <queue>
#include <vector>
#include <chrono>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONFIGURATION AND LIMITS
// ============================================================================

#define LLAMA_JSON_ISOLATION_MAX_TOKEN_QUEUE    (8192)
#define LLAMA_JSON_ISOLATION_MAX_FORMAT_BUFFERS (256)
#define LLAMA_JSON_ISOLATION_MAX_WORKERS        (16)
#define LLAMA_JSON_ISOLATION_BUFFER_POOL_SIZE   (512 * 1024)  // 512KB pool
#define LLAMA_JSON_ISOLATION_RECORD_SIZE        (128)         // Min token record
#define LLAMA_JSON_ISOLATION_GUARD_INTERVAL     (100)         // Check every N tokens

// ============================================================================
// MINIMAL TOKEN RECORD STRUCTURE
// ============================================================================

/**
 * Minimal token record - produced by decode, consumed by server worker
 * Total: ~64 bytes (cache-line aligned for lock-free efficiency)
 *
 * This is the ONLY data structure created during decode.
 * All JSON serialization deferred to server worker.
 */
typedef struct {
    uint32_t token_id;                      // Token ID (4 bytes)
    uint32_t sequence_id;                   // Sequence identifier (4 bytes)
    uint64_t timestamp_ns;                  // Nanosecond timestamp (8 bytes)
    float logprob;                          // Log probability (4 bytes)
    uint16_t batch_slot;                    // Batch slot index (2 bytes)
    uint16_t reserved0;                     // Padding (2 bytes)
    uint32_t flags;                         // Control flags (4 bytes)
    uint64_t decode_wall_time_ns;           // Decode latency measurement (8 bytes)
    uint64_t reserved1;                     // Future use (8 bytes)
    uint64_t reserved2;                     // Future use (8 bytes)
} llama_minimal_token_record;

/**
 * Flags for token record
 */
#define LLAMA_TOKEN_RECORD_FLAG_STOP_SEQUENCE    (1u << 0)
#define LLAMA_TOKEN_RECORD_FLAG_EOS_TOKEN        (1u << 1)
#define LLAMA_TOKEN_RECORD_FLAG_LOGPROBS_VALID   (1u << 2)
#define LLAMA_TOKEN_RECORD_FLAG_PARTIAL_OUTPUT   (1u << 3)

// ============================================================================
// LOCK-FREE QUEUE FOR TOKEN RECORDS
// ============================================================================

/**
 * Single-Producer, Multiple-Consumer lock-free queue
 * Producer: decode thread (single decode path)
 * Consumers: server worker threads (multiple HTTP handlers)
 *
 * Lock-free enqueue: decode thread never blocks
 * Lock-free dequeue: server workers never block decode
 */
typedef struct {
    llama_minimal_token_record * ring_buffer;   // Pre-allocated ring buffer
    uint32_t capacity;                          // Ring buffer capacity

    std::atomic<uint32_t> write_pos;            // Producer write position
    std::atomic<uint32_t> read_pos;             // Consumer read position
    std::atomic<bool> full_backpressure;        // Signal buffer near-full

    pthread_mutex_t consumer_lock;              // Multiple consumers share this
    pthread_cond_t consumer_signal;             // Signal data available

    uint64_t total_enqueued;                    // Metrics: total enqueued
    uint64_t total_dequeued;                    // Metrics: total dequeued
    uint64_t total_overflows;                   // Metrics: queue overflows
    uint64_t max_utilization;                   // Metrics: peak usage %
} llama_token_record_queue;

// ============================================================================
// FORMATTING BUFFER POOL
// ============================================================================

/**
 * Pre-allocated buffer pool for JSON formatting
 * Avoids per-token malloc in server worker thread
 * Memory reserved at startup, never allocated during streaming
 */
typedef struct {
    char * buffers[LLAMA_JSON_ISOLATION_MAX_FORMAT_BUFFERS];
    size_t buffer_sizes[LLAMA_JSON_ISOLATION_MAX_FORMAT_BUFFERS];
    std::atomic<uint32_t> available_count;  // Buffers available
    pthread_mutex_t lock;

    size_t buffer_size;                     // Size of each buffer
    uint64_t total_bytes;                   // Total allocated
    uint64_t allocations_served;            // Buffers served without malloc
} llama_formatting_buffer_pool;

// ============================================================================
// SERVER WORKER CONTEXT
// ============================================================================

/**
 * Server worker thread context
 * Handles token-to-text conversion and JSON serialization
 */
typedef struct {
    uint32_t worker_id;                     // Worker identifier
    pthread_t thread_handle;                // Worker thread handle
    volatile bool running;                  // Worker running flag

    // Statistics
    uint64_t tokens_processed;              // Total tokens processed
    uint64_t json_objects_created;          // JSON objects created
    uint64_t bytes_serialized;              // JSON bytes produced
    uint64_t total_serialization_time_ns;   // Total JSON time
    uint64_t max_serialization_time_ns;     // Maximum single JSON time

    // Performance tracking
    uint64_t queue_wait_time_ns;            // Time waiting for tokens
    uint64_t conversion_time_ns;            // Token-to-text time

} llama_server_worker_context;

// ============================================================================
// JSON ISOLATION ENFORCEMENT STATE
// ============================================================================

/**
 * Enforcement state machine
 */
typedef enum {
    LLAMA_JSON_ISOLATION_UNINITIALIZED = 0,
    LLAMA_JSON_ISOLATION_CONFIGURED = 1,
    LLAMA_JSON_ISOLATION_ENFORCING = 2,     // Decode loop active, checks enabled
    LLAMA_JSON_ISOLATION_STREAMING = 3,     // Full streaming active
    LLAMA_JSON_ISOLATION_SHUTDOWN = 4
} llama_json_isolation_state_t;

/**
 * Violation type detected
 */
typedef enum {
    LLAMA_JSON_VIOLATION_NONE = 0,
    LLAMA_JSON_VIOLATION_JSON_IN_DECODE = 1,           // JSON library called in decode
    LLAMA_JSON_VIOLATION_STRING_ALLOCATION = 2,        // std::string created in decode
    LLAMA_JSON_VIOLATION_DYNAMIC_ALLOCATION = 3,       // malloc/new in decode
    LLAMA_JSON_VIOLATION_BLOCKING_OUTPUT = 4,          // Output buffer lock in decode
    LLAMA_JSON_VIOLATION_HTTP_FLUSH_IN_DECODE = 5,     // HTTP write in decode
    LLAMA_JSON_VIOLATION_LOGGING_FORMAT = 6            // Formatted logging in decode
} llama_json_violation_type_t;

/**
 * Main isolation enforcement context
 */
typedef struct {
    // State management
    llama_json_isolation_state_t current_state;
    llama_json_isolation_state_t previous_state;
    pthread_mutex_t state_lock;

    // Token queue (core component)
    llama_token_record_queue * token_queue;

    // Server worker threads
    llama_server_worker_context * workers;
    uint32_t num_workers;

    // Formatting buffer pool
    llama_formatting_buffer_pool * buffer_pool;

    // Decode thread tracking
    uint32_t decode_thread_id;
    bool decode_thread_registered;

    // Metrics and statistics
    uint64_t total_tokens_decoded;
    uint64_t total_tokens_emitted;
    uint64_t tokens_in_flight;              // Tokens between decode and server
    uint64_t decode_loop_iterations;
    uint64_t decode_critical_section_exits;

    // Violation tracking
    llama_json_violation_type_t last_violation;
    std::string last_violation_message;
    uint64_t total_violations;
    bool abort_on_violation;                // Hard fail mode

    // Performance measurements
    uint64_t average_token_enqueue_ns;      // Average microseconds per enqueue
    uint64_t max_token_enqueue_ns;          // Maximum enqueue latency
    uint64_t tokens_per_second_with_streaming;
    uint64_t tokens_per_second_without_streaming;

    // Configuration
    bool streaming_enabled;
    bool validate_token_records;
    bool measure_decode_isolation;

    // Initialization timestamp
    uint64_t init_time_ns;
} llama_json_isolation_context_t;

// ============================================================================
// PUBLIC API - INITIALIZATION & LIFECYCLE
// ============================================================================

/**
 * Initialize JSON isolation context
 * Must be called once before any decode operations
 * Sets up token queue, worker threads, buffer pool
 *
 * Returns: 0 on success, negative value on error
 */
int llama_json_isolation_init(
    llama_json_isolation_context_t * ctx,
    uint32_t queue_capacity,
    uint32_t num_server_workers,
    size_t format_buffer_size,
    bool abort_on_violation
);

/**
 * Configure streaming behavior
 * Must be called after init, before decode starts
 *
 * Returns: 0 on success, negative value on error
 */
int llama_json_isolation_configure_streaming(
    llama_json_isolation_context_t * ctx,
    bool streaming_enabled
);

/**
 * Register decode thread
 * Called once per decode thread to establish identity
 * Used for violation detection and metrics
 *
 * Returns: 0 on success, negative value on error
 */
int llama_json_isolation_register_decode_thread(
    llama_json_isolation_context_t * ctx
);

/**
 * Mark start of decode critical section
 * Called before decode loop begins
 * Enables per-token validation and guards
 */
void llama_json_isolation_enter_critical_section(
    llama_json_isolation_context_t * ctx
);

/**
 * Mark end of decode critical section
 * Called after decode loop completes
 * Allows metrics finalization and shutdown
 */
void llama_json_isolation_exit_critical_section(
    llama_json_isolation_context_t * ctx
);

/**
 * Shutdown isolation system
 * Stops server workers, flushes queues, releases resources
 * Must be called once at end of session
 *
 * Returns: 0 on success, negative value on error
 */
int llama_json_isolation_shutdown(
    llama_json_isolation_context_t * ctx
);

// ============================================================================
// PUBLIC API - DECODE PATH OPERATIONS
// ============================================================================

/**
 * Enqueue minimal token record from decode
 * Single operation: create record → lock-free enqueue → return
 * Must complete in < 1 microsecond
 *
 * Returns: 0 on success, 1 if buffer full (backpressure), negative on error
 */
int llama_json_isolation_enqueue_token(
    llama_json_isolation_context_t * ctx,
    uint32_t token_id,
    uint32_t sequence_id,
    float logprob,
    uint16_t batch_slot,
    uint32_t flags
);

/**
 * Validate token record (internal consistency)
 * Checks: token_id in valid range, timestamp reasonable, logprob finite
 * Called only if validation enabled
 *
 * Returns: true if record valid, false if corrupted
 */
bool llama_json_isolation_validate_token_record(
    const llama_minimal_token_record * record
);

/**
 * Get current queue depth
 * Safe for concurrent reads
 *
 * Returns: number of tokens currently in queue
 */
uint32_t llama_json_isolation_get_queue_depth(
    const llama_json_isolation_context_t * ctx
);

/**
 * Check for backpressure condition
 * Returns true if buffer near full, decode should consider slowing
 *
 * Returns: true if backpressure detected
 */
bool llama_json_isolation_check_backpressure(
    const llama_json_isolation_context_t * ctx
);

// ============================================================================
// PUBLIC API - GUARD ASSERTIONS
// ============================================================================

/**
 * Assert that no JSON library is being used in decode
 * Called periodically (every N tokens) to detect violations
 * In debug builds, hard fails if violation detected
 *
 * Returns: 0 if compliant, negative if violation detected
 */
int llama_json_isolation_assert_no_json_in_decode(
    llama_json_isolation_context_t * ctx
);

/**
 * Assert no string allocations in decode
 * Detects std::string construction, concatenation
 *
 * Returns: 0 if compliant, negative if violation detected
 */
int llama_json_isolation_assert_no_string_alloc_in_decode(
    llama_json_isolation_context_t * ctx
);

/**
 * Assert no dynamic memory allocation in decode
 * Detects malloc/new calls in decode thread
 *
 * Returns: 0 if compliant, negative if violation detected
 */
int llama_json_isolation_assert_no_dynamic_alloc_in_decode(
    llama_json_isolation_context_t * ctx
);

/**
 * Assert decode never blocks on output
 * Detects mutex acquisitions that might wait
 *
 * Returns: 0 if compliant, negative if violation detected
 */
int llama_json_isolation_assert_nonblocking_output(
    llama_json_isolation_context_t * ctx
);

/**
 * Combined guard check - all assertions
 * Called at decode start and periodically during execution
 *
 * Returns: 0 if all checks pass, negative if any violation
 */
int llama_json_isolation_guard_all_checks(
    llama_json_isolation_context_t * ctx,
    uint64_t token_index
);

// ============================================================================
// PUBLIC API - METRICS AND VALIDATION
// ============================================================================

/**
 * Get current isolation state
 *
 * Returns: current state enum value
 */
llama_json_isolation_state_t llama_json_isolation_get_state(
    const llama_json_isolation_context_t * ctx
);

/**
 * Get last violation recorded
 *
 * Returns: violation type enum, LLAMA_JSON_VIOLATION_NONE if none
 */
llama_json_violation_type_t llama_json_isolation_get_last_violation(
    const llama_json_isolation_context_t * ctx
);

/**
 * Get violation message as string
 * Safe for concurrent reads
 *
 * Returns: pointer to violation message string (valid until next violation)
 */
const char * llama_json_isolation_get_violation_message(
    const llama_json_isolation_context_t * ctx
);

/**
 * Get performance metrics
 * Aggregates metrics from all workers and queues
 *
 * Returns: 0 on success
 */
typedef struct {
    uint64_t total_tokens_decoded;
    uint64_t total_tokens_emitted;
    uint32_t current_queue_depth;
    uint32_t max_queue_depth;

    uint64_t average_enqueue_ns;
    uint64_t max_enqueue_ns;

    uint64_t total_json_objects_created;
    uint64_t total_json_serialization_time_ns;
    uint64_t average_json_time_per_token_ns;

    uint64_t total_violations;
    double tokens_per_second_with_streaming;
    double tokens_per_second_without_streaming;
    bool streaming_active;
} llama_json_isolation_metrics;

int llama_json_isolation_get_metrics(
    const llama_json_isolation_context_t * ctx,
    llama_json_isolation_metrics * metrics
);

/**
 * Validate isolated architecture compliance
 * Comprehensive check: no JSON in decode, proper flow, timing requirements
 *
 * Returns: 0 if compliant, negative if violations found
 */
int llama_json_isolation_validate_architecture(
    const llama_json_isolation_context_t * ctx
);

/**
 * Validate throughput isolation
 * Measures tokens/sec with and without streaming
 * They must be identical (streaming should not reduce throughput)
 *
 * Returns: 0 if throughput isolated, negative if coupling detected
 */
int llama_json_isolation_validate_throughput_isolation(
    llama_json_isolation_context_t * ctx
);

/**
 * Generate report of JSON isolation status
 * Suitable for logging/debugging
 *
 * Returns: 0 on success
 */
int llama_json_isolation_report_status(
    const llama_json_isolation_context_t * ctx
);

// ============================================================================
// PUBLIC API - TOKEN TEXT CONVERSION (FOR SERVER WORKERS)
// ============================================================================

/**
 * Convert token ID to text string
 * Server workers call this to get printable token text
 * Note: passed as parameter, not defined here (uses llama vocab)
 *
 * This is where token→text conversion happens (on server worker)
 * Decode never calls this
 */
typedef int (*llama_token_to_text_fn)(uint32_t token_id, std::string & output);

/**
 * Set token-to-text conversion function
 * Server workers use this to produce text from token IDs
 *
 * Returns: 0 on success
 */
int llama_json_isolation_set_token_to_text_fn(
    llama_json_isolation_context_t * ctx,
    llama_token_to_text_fn fn
);

// ============================================================================
// GUARD MACROS FOR DEBUG BUILDS
// ============================================================================

/**
 * LLAMA_JSON_ISOLATION_ASSERT_IN_DECODE_ONLY
 * Fails hard if called from non-decode thread or outside critical section
 */
#ifdef LLAMA_DEBUG
#define LLAMA_JSON_ISOLATION_ASSERT_IN_DECODE_ONLY(ctx) \
    do { \
        if (llama_json_isolation_get_state(ctx) != LLAMA_JSON_ISOLATION_ENFORCING) { \
            llama_json_isolation_guard_all_checks(ctx, 0); \
            abort(); \
        } \
    } while(0)
#else
#define LLAMA_JSON_ISOLATION_ASSERT_IN_DECODE_ONLY(ctx) do { } while(0)
#endif

/**
 * LLAMA_JSON_ISOLATION_GUARD_PERIODIC_CHECK
 * Called every N tokens to validate no JSON serialization occurring
 */
#ifdef LLAMA_DEBUG
#define LLAMA_JSON_ISOLATION_GUARD_PERIODIC_CHECK(ctx, token_idx) \
    do { \
        if ((token_idx) % LLAMA_JSON_ISOLATION_GUARD_INTERVAL == 0) { \
            if (llama_json_isolation_guard_all_checks(ctx, token_idx) != 0) { \
                abort(); \
            } \
        } \
    } while(0)
#else
#define LLAMA_JSON_ISOLATION_GUARD_PERIODIC_CHECK(ctx, token_idx) do { } while(0)
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

