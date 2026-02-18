#pragma once

/**
 * llama-streaming-async.h
 *
 * Complete asynchronous streaming decoupling system for llama.cpp server.
 * Enforces strict separation of GPU decode and I/O streaming operations.
 *
 * REQUIREMENT #49: Make Streaming Output Fully Asynchronous
 * ENFORCEMENT RULES (8 total):
 *
 * 1. Separate Decode and Streaming Execution Domains
 *    - Decode: GPU-authoritative, produce tokens, enqueue, continue immediately
 *    - Streaming: I/O-bound, pop tokens, convert text, flush HTTP
 *    - Decode forbidden: HTTP writes, JSON, logging, network, mutexes
 *
 * 2. Lock-Free Token Queue
 *    - Single-producer (decode), single-consumer (streaming)
 *    - Preallocated ring buffer, no dynamic allocation
 *    - No blocking, no mutex in decode path
 *
 * 3. Batched HTTP Flush
 *    - Buffer tokens, batch writes when possible
 *    - Avoid per-token flush() syscalls
 *    - Decode never depends on network behavior
 *
 * 4. No Decode↔Server Cross-Locks
 *    - No shared mutex between domains
 *    - No slot lock held during token emission
 *    - Lock-free queue only communication
 *
 * 5. Non-Blocking Disconnect Handling
 *    - Streaming thread handles client disconnect
 *    - Decode unaware, continues unless cancelled
 *    - Atomic flag check for cancellation only
 *
 * 6. Server Threads Cannot Affect Decode Timing
 *    - Optional isolated CPU core for streaming
 *    - Lower priority than decode threads
 *    - Decode has scheduling dominance
 *
 * 7. Validate Throughput Independence
 *    - t/s (CLI) == t/s (server+streaming) within ±1%
 *    - If streaming reduces throughput: still has coupling
 *
 * 8. Enforce Hard Separation in Code
 *    - Assertion failure if decode does HTTP/JSON/logging
 *    - Debug: hard abort; Production: log and skip
 *
 * This header provides:
 * - Streaming worker abstraction
 * - Lock-free queue infrastructure
 * - Token batching and conversion
 * - Disconnect and cancellation handling
 * - Metrics and domain validation
 * - Request context separation
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <atomic>
#include <memory>
#include <functional>
#include <thread>
#include <chrono>

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

// Maximum streaming worker threads (usually 1-2)
#define STREAMING_WORKER_MAX_THREADS 4

// Lock-free token queue size (single-producer/consumer)
#define STREAMING_TOKEN_QUEUE_SIZE 4096

// Batch size for HTTP flushes (tokens)
#define STREAMING_BATCH_SIZE_MIN 4
#define STREAMING_BATCH_SIZE_MAX 64

// HTTP flush timeout when tokens buffered (milliseconds)
#define STREAMING_FLUSH_TIMEOUT_MS 50

// Maximum buffered bytes before forced flush
#define STREAMING_BUFFER_MAX_BYTES 262144  // 256 KB

// Metrics sampling window (milliseconds)
#define STREAMING_METRICS_WINDOW_MS 1000

// ============================================================================
// DOMAIN SEPARATION GUARDS
// ============================================================================

/**
 * Execution domain identifier.
 * Used to enforce hard separation between decode and streaming contexts.
 */
enum class streaming_execution_domain : int32_t {
    DECODE = 0,      // GPU decode thread
    STREAMING = 1,   // I/O streaming worker
    UNKNOWN = -1
};

/**
 * Get current execution domain for this thread.
 * @return domain identifier for current thread
 */
streaming_execution_domain get_current_domain();

/**
 * Set execution domain for current thread.
 * Called at thread startup by isolation enforcement.
 * @param domain new domain for this thread
 */
void set_current_domain(streaming_execution_domain domain);

/**
 * Guard function: verify code executing in allowed domain.
 * In debug builds: hard assertion on violation.
 * In production: logs violation and continues.
 * @param allowed_domain which domain should execute this code
 * @param operation_name for logging (e.g., "HTTP flush")
 * @return true if domain is correct, false if violation
 */
bool verify_domain(streaming_execution_domain allowed_domain, const char * operation_name);

/**
 * Hard assertion that decode thread never performs I/O operations.
 * Forbidden operations: HTTP writes, network send, JSON serialization,
 * disk I/O initiated by decode, mutex acquisition.
 * @param operation_name what decode tried to do (for error message)
 * @return never returns on violation; logs and aborts
 */
[[noreturn]]
void enforce_decode_purity(const char * operation_name);

// ============================================================================
// TOKEN DEFINITION
// ============================================================================

/**
 * Token produced by decode and streamed asynchronously.
 * Minimal structure to keep lock-free queue efficient.
 */
struct streaming_token {
    int32_t token_id;           // Token ID from model
    int64_t timestamp_us;       // When produced by decode (microseconds)
    uint32_t sequence_id;       // Which decode sequence (slot or request ID)
    bool is_eos;                // End of sequence flag
    uint8_t padding[3];         // Align to 32 bytes
};

/**
 * Convert token ID to text representation.
 * Executed in streaming worker (not decode).
 * @param token_id token ID to convert
 * @param model_vocab token vocabulary (opaque pointer)
 * @param use_special special token mode flag
 * @return text representation of token
 */
std::string streaming_token_to_text(
    int32_t token_id,
    const void * model_vocab,
    bool use_special = true
);

/**
 * Build JSON response chunk for token.
 * Executed in streaming worker (not decode).
 * @param token streaming token
 * @param text_content token text (already converted)
 * @param include_logits if true, include log probability
 * @param include_timing if true, include token timestamp
 * @return JSON string for HTTP chunk
 */
std::string streaming_build_json_chunk(
    const streaming_token & token,
    const std::string & text_content,
    bool include_logits = false,
    bool include_timing = true
);

// ============================================================================
// LOCK-FREE TOKEN QUEUE
// ============================================================================

/**
 * Single-producer (decode) / Single-consumer (streaming) lock-free queue.
 * Uses prealloc ring buffer, no mutex, no condition variables.
 * Decode thread never blocks on this queue.
 */
class streaming_token_queue {
public:
    /**
     * Create queue with specified capacity.
     * @param capacity maximum tokens (must be power of 2)
     */
    explicit streaming_token_queue(size_t capacity = STREAMING_TOKEN_QUEUE_SIZE);

    ~streaming_token_queue();

    /**
     * Try to push token (non-blocking, decode thread).
     * Called from decode thread after producing each token.
     * @param token token to enqueue
     * @return true if enqueued, false if full (backpressure)
     */
    bool try_push(const streaming_token & token);

    /**
     * Try to pop token (non-blocking, streaming thread).
     * Called from streaming worker to get next token.
     * @param token output parameter for dequeued token
     * @return true if token retrieved, false if queue empty
     */
    bool try_pop(streaming_token & token);

    /**
     * Get current queue depth.
     * @return number of tokens currently in queue
     */
    size_t depth() const;

    /**
     * Get queue capacity.
     * @return maximum tokens queue can hold
     */
    size_t capacity() const;

    /**
     * Check if queue is full (triggers backpressure).
     * @return true if depth == capacity
     */
    bool is_full() const;

    /**
     * Check if queue is empty.
     * @return true if depth == 0
     */
    bool is_empty() const;

    /**
     * Clear queue for new request.
     * Only safe if no active producers/consumers.
     */
    void clear();

    /**
     * Get queue metrics snapshot.
     * @return current depth, capacity, overflow count
     */
    struct streaming_queue_metrics get_metrics() const;

private:
    std::vector<streaming_token> buffer;
    std::atomic<uint64_t> head;     // Producer write position
    std::atomic<uint64_t> tail;     // Consumer read position
    size_t mask;                    // Capacity - 1 (for modulo)
    std::atomic<uint64_t> overflows; // Backpressure events
};

/**
 * Queue metrics snapshot.
 */
struct streaming_queue_metrics {
    size_t current_depth;
    size_t capacity;
    uint64_t total_overflow_events;
    float utilization_percent;
};

// ============================================================================
// BATCH ACCUMULATOR
// ============================================================================

/**
 * Accumulates tokens for batched HTTP flush.
 * Executed in streaming worker, maintains local buffer.
 * Prevents per-token flush() syscalls.
 */
class streaming_batch_accumulator {
public:
    /**
     * Create batch accumulator.
     * @param batch_size tokens per batch (4-64)
     * @param buffer_size max buffered bytes before flush (default 256KB)
     */
    streaming_batch_accumulator(
        size_t batch_size = STREAMING_BATCH_SIZE_MIN,
        size_t buffer_size = STREAMING_BUFFER_MAX_BYTES
    );

    ~streaming_batch_accumulator();

    /**
     * Add token to batch.
     * @param token streaming token
     * @param json_chunk JSON formatted token response
     * @return true if batch still has space, false if batch ready to flush
     */
    bool add_token(const streaming_token & token, const std::string & json_chunk);

    /**
     * Check if batch should be flushed.
     * Criteria: count >= batch_size OR buffered_bytes >= buffer_limit
     * @return true if batch ready
     */
    bool should_flush() const;

    /**
     * Get accumulated batch data.
     * Only call after should_flush() returns true.
     * @return accumulated JSON response chunk (may contain multiple tokens)
     */
    std::string get_batch_data() const;

    /**
     * Get current batch token count.
     * @return number of tokens in current batch
     */
    size_t batch_token_count() const;

    /**
     * Get current buffered byte count.
     * @return number of bytes in buffer
     */
    size_t buffered_bytes() const;

    /**
     * Flush batch and return accumulated data.
     * Clears internal state for next batch.
     * @return accumulated data (empty if no tokens)
     */
    std::string flush();

    /**
     * Reset batch without returning data.
     * Used for error cases (e.g., client disconnect).
     */
    void reset();

private:
    std::vector<streaming_token> token_batch;
    std::string json_buffer;
    size_t target_batch_size;
    size_t max_buffer_bytes;
};

// ============================================================================
// CANCELLATION TOKEN
// ============================================================================

/**
 * Non-blocking cancellation signal from streaming to decode.
 * Used for client disconnects, timeouts, explicit cancellation.
 * Decode checks this without waiting (atomic read only).
 */
class streaming_cancellation_token {
public:
    streaming_cancellation_token();

    /**
     * Signal cancellation (from streaming thread).
     * Streaming worker calls this on client disconnect or timeout.
     */
    void cancel();

    /**
     * Check if cancellation requested (from decode thread).
     * Non-blocking atomic read - always returns immediately.
     * @return true if cancel() was called
     */
    bool is_cancelled() const;

    /**
     * Reset for new request.
     * Called when new decode sequence starts.
     */
    void reset();

    /**
     * Get cancellation reason.
     * @return human-readable reason (empty if not cancelled)
     */
    std::string get_reason() const;

private:
    std::atomic<bool> cancelled;
    std::string reason;
    std::atomic<int64_t> cancel_timestamp_us;
};

// ============================================================================
// REQUEST CONTEXT SEPARATION
// ============================================================================

/**
 * Decode request context - isolated from HTTP server context.
 * Decode thread reads this, streaming thread reads/writes metadata.
 * No shared locks between domains.
 */
struct streaming_decode_context {
    uint32_t sequence_id;                       // Unique ID for this decode run
    uint32_t slot_id;                          // Server slot (for reference only)
    streaming_cancellation_token cancellation; // Client disconnect signal
    int64_t start_time_us;                     // When decode started
    std::atomic<uint32_t> tokens_generated;    // Running count
    void * user_context;                       // Opaque pointer to request
};

/**
 * Streaming request context - isolated from decode context.
 * Streaming thread only, no decode access.
 */
struct streaming_http_context {
    uint32_t sequence_id;                      // Matches decode context
    void * http_connection;                    // HTTP connection object
    std::function<bool(const std::string &)> send_chunk; // Callback to send data
    bool client_connected;                     // Is client still connected?
    int64_t last_activity_us;                  // Last send time
    std::atomic<uint32_t> chunks_sent;         // Chunks flushed
};

// ============================================================================
// STREAMING WORKER ABSTRACTION
// ============================================================================

/**
 * Asynchronous streaming worker thread.
 * Consumes tokens from lock-free queue, batches, converts, flushes HTTP.
 * Completely decoupled from decode thread - no shared state/locks.
 */
class streaming_worker {
public:
    /**
     * Create streaming worker.
     * Call from main thread before starting decode.
     * @param worker_index unique ID for this worker (0-based)
     * @param batch_size tokens per batch
     * @param flush_timeout_ms max ms to wait before flushing
     */
    streaming_worker(
        int32_t worker_index,
        size_t batch_size = STREAMING_BATCH_SIZE_MIN,
        int32_t flush_timeout_ms = STREAMING_FLUSH_TIMEOUT_MS
    );

    ~streaming_worker();

    /**
     * Register token queue for this worker.
     * Links worker to lock-free queue where decode enqueues tokens.
     * @param queue shared token queue
     */
    void register_token_queue(streaming_token_queue * queue);

    /**
     * Register model vocabulary for text conversion.
     * Called before any decode starts.
     * @param vocab model vocabulary pointer (opaque)
     */
    void register_vocab(const void * vocab);

    /**
     * Start streaming worker thread.
     * Worker runs in infinite loop: pop token -> convert -> batch -> flush.
     * @return true if thread created successfully
     */
    bool start();

    /**
     * Stop streaming worker thread gracefully.
     * Waits for in-flight batches, then exits loop.
     * @param timeout_ms max time to wait for graceful shutdown
     * @return true if shutdown completed
     */
    bool stop(int32_t timeout_ms = 5000);

    /**
     * Register HTTP context for outbound streaming.
     * Called when new HTTP request arrives that uses streaming.
     * @param context HTTP context with send callback
     * @return registration ID (for later unregister)
     */
    uint32_t register_http_context(const streaming_http_context & context);

    /**
     * Unregister HTTP context (client disconnect or request complete).
     * Signals to worker to stop sending chunks.
     * @param context_id ID from register_http_context
     */
    void unregister_http_context(uint32_t context_id);

    /**
     * Link decode context to active HTTP context.
     * Associates decode sequence with HTTP connection.
     * @param decode_context decode run metadata
     * @param http_context_id from register_http_context
     */
    void link_decode_to_http(
        const streaming_decode_context & decode_context,
        uint32_t http_context_id
    );

    /**
     * Signal decode sequence complete.
     * Flushes any remaining buffered tokens for this sequence.
     * @param sequence_id ID of completed decode
     */
    void signal_sequence_complete(uint32_t sequence_id);

    /**
     * Check if worker is running.
     * @return true if start() succeeded and stop() not called
     */
    bool is_running() const;

    /**
     * Check if worker thread is alive.
     * May return false briefly during startup/shutdown.
     * @return true if worker thread exists and running
     */
    bool is_alive() const;

    /**
     * Get worker metrics snapshot.
     * @return current metrics (depth, tokens, throughput)
     */
    struct streaming_worker_metrics get_metrics() const;

    /**
     * Force flush any pending batches for sequence.
     * Used for testing or explicit flush requests.
     * @param sequence_id which decode sequence to flush
     * @return true if flush succeeded
     */
    bool flush_pending(uint32_t sequence_id);

    /**
     * Force flush all pending batches.
     * Flushes all sequences, waits for completion.
     * @return number of chunks flushed
     */
    uint32_t flush_all();

    /**
     * Get worker index.
     * @return index passed to constructor
     */
    int32_t get_worker_index() const;

private:
    int32_t worker_index;
    size_t target_batch_size;
    int32_t flush_timeout_ms;
    streaming_token_queue * token_queue;
    const void * model_vocab;
    std::thread worker_thread;
    std::atomic<bool> running;
    std::atomic<bool> shutdown_requested;

    // Worker thread main loop
    void worker_main_loop();

    // Helpers
    bool process_token(const streaming_token & token);
    bool flush_batch_to_http(uint32_t sequence_id);
};

/**
 * Streaming worker metrics.
 */
struct streaming_worker_metrics {
    int32_t worker_index;
    bool is_running;
    uint64_t tokens_processed;
    uint64_t chunks_flushed;
    uint64_t batches_created;
    float tokens_per_sec;
    size_t active_sequences;
    int32_t pending_sequences;
};

// ============================================================================
// STREAMING SYSTEM MANAGER
// ============================================================================

/**
 * Central coordinator for asynchronous streaming system.
 * Manages token queue, worker threads, request contexts.
 * Singleton instance.
 */
class streaming_system {
public:
    /**
     * Initialize streaming system.
     * Call once at server startup, before decode starts.
     * @param worker_count number of streaming workers (1-4)
     * @param queue_capacity tokens in lock-free queue
     * @param batch_size tokens per flush batch
     * @return true if initialized successfully
     */
    bool initialize(
        int32_t worker_count = 1,
        size_t queue_capacity = STREAMING_TOKEN_QUEUE_SIZE,
        size_t batch_size = STREAMING_BATCH_SIZE_MIN
    );

    /**
     * Shutdown streaming system.
     * Stops all workers, cleans up resources.
     * @param timeout_ms max time to wait for graceful shutdown
     */
    void shutdown(int32_t timeout_ms = 5000);

    /**
     * Get shared token queue.
     * Decode threads use this to push tokens.
     * @return pointer to singleton token queue
     */
    streaming_token_queue * get_token_queue();

    /**
     * Register model vocabulary globally.
     * @param vocab opaque vocabulary pointer
     */
    void register_vocab(const void * vocab);

    /**
     * Decode thread: emit token to streaming system.
     * Non-blocking - must never stall decode.
     * @param token token produced by decode
     * @param sequence_id which decode sequence
     * @return true if queued, false if backpressure (queue full)
     */
    bool decode_emit_token(const streaming_token & token, uint32_t sequence_id);

    /**
     * Get current token queue depth.
     * For monitoring only.
     * @return current number of tokens in queue
     */
    size_t get_queue_depth() const;

    /**
     * Signal decode sequence starting.
     * Initializes context for new decode.
     * @param sequence_id unique ID for this decode
     * @param slot_id server slot (informational)
     * @param user_context opaque pointer to request
     */
    void signal_decode_start(uint32_t sequence_id, uint32_t slot_id = 0, void * user_context = nullptr);

    /**
     * Signal decode sequence complete.
     * Flushes pending tokens, cleanup context.
     * @param sequence_id ID of completed decode
     */
    void signal_decode_complete(uint32_t sequence_id);

    /**
     * Register HTTP context for streaming response.
     * @param context HTTP connection metadata
     * @return context ID (pass to link_to_decode)
     */
    uint32_t register_http_context(const streaming_http_context & context);

    /**
     * Unregister HTTP context (client disconnected).
     * @param context_id from register_http_context
     */
    void unregister_http_context(uint32_t context_id);

    /**
     * Link decode sequence to HTTP context.
     * Associates produce (decode) with consume (HTTP).
     * @param sequence_id decode sequence
     * @param context_id HTTP context
     */
    void link_decode_to_http(uint32_t sequence_id, uint32_t context_id);

    /**
     * Check if system is initialized and running.
     * @return true if initialize() succeeded
     */
    bool is_initialized() const;

    /**
     * Get total system metrics.
     * @return aggregated metrics from all workers
     */
    struct streaming_system_metrics get_metrics() const;

    /**
     * Flush all pending tokens for all sequences.
     * Waits for workers to complete.
     * @return total chunks flushed
     */
    uint32_t flush_all_pending();

    /**
     * Get singleton instance.
     * @return reference to static streaming system
     */
    static streaming_system & instance();

private:
    streaming_token_queue token_queue;
    std::vector<std::unique_ptr<streaming_worker>> workers;
    const void * model_vocab;
    std::atomic<bool> initialized;
    std::atomic<bool> shutdown_in_progress;

    // Context tracking
    std::map<uint32_t, streaming_decode_context> decode_contexts;
    std::map<uint32_t, streaming_http_context> http_contexts;
};

/**
 * Streaming system metrics.
 */
struct streaming_system_metrics {
    bool initialized;
    int32_t worker_count;
    size_t queue_depth;
    size_t queue_capacity;
    uint64_t total_tokens_produced;
    uint64_t total_tokens_consumed;
    uint64_t backpressure_events;
    float system_throughput_tps;
    std::vector<streaming_worker_metrics> per_worker;
};

// ============================================================================
// VALIDATION AND ENFORCEMENT
// ============================================================================

/**
 * Validate domain separation enforcement.
 * Checks all invariants: no shared locks, token queue is lock-free, etc.
 * @return true if all rules satisfied, false if violation found
 */
bool validate_streaming_domain_separation();

/**
 * Validate throughput independence.
 * Compares decode t/s with and without streaming active.
 * Must be within ±1%.
 * @param cli_tokens_per_sec tokens/sec from CLI (no streaming)
 * @param server_tokens_per_sec tokens/sec from server (with streaming)
 * @return true if throughput matches, false if coupling detected
 */
bool validate_streaming_throughput_independence(
    float cli_tokens_per_sec,
    float server_tokens_per_sec
);

/**
 * Hard assertion that code is executing in decode domain.
 * For enforcement rule #8: detect forbidden operations in decode.
 * @return true if currently in decode domain
 */
bool assert_in_decode_domain();

/**
 * Hard assertion that code is executing in streaming domain.
 * @return true if currently in streaming domain
 */
bool assert_in_streaming_domain();

// ============================================================================
// DIAGNOSTICS
// ============================================================================

/**
 * Dump streaming system state for debugging.
 * Shows queue depth, worker state, active sequences, metrics.
 */
void dump_streaming_state();

/**
 * Get human-readable status of streaming system.
 * @return string describing current state
 */
std::string get_streaming_status();

/**
 * Enable verbose logging of streaming operations.
 * @param enable true to enable debug logging
 */
void set_streaming_verbose_logging(bool enable);

#endif // LLAMA_STREAMING_ASYNC_H
