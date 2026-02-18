#pragma once

/**
 * Asynchronous Streaming Decoupling for LLAMA Decode
 *
 * Complete separation of GPU decode and I/O streaming execution domains.
 * Ensures streaming can never reduce decode throughput, block decode thread,
 * or create dependencies between GPU execution and network I/O.
 *
 * Key Properties:
 * - Two strictly isolated execution domains: DECODE and STREAMING
 * - Lock-free queue for token transfer (single producer, single consumer)
 * - Zero blocking operations in decode path
 * - No shared locks or synchronization between domains
 * - Non-blocking disconnect handling
 * - Decode thread priority dominance
 * - Throughput validation: CLI vs streaming must match
 * - Hard separation enforcement via debug assertions
 *
 * 8 Core Rules Enforced:
 * 1. Separate Decode and Streaming Execution Domains
 * 2. Introduce Lock-Free Token Queue
 * 3. Remove Per-Token HTTP Flush Dependency
 * 4. Eliminate Decode→Server Cross-Locks
 * 5. Make Disconnect Handling Non-Blocking
 * 6. Prevent Server Threads from Affecting Decode Timing
 * 7. Validate Throughput Independence
 * 8. Enforce Hard Separation in Code
 *
 * Expected Outcome:
 * - Streaming cannot reduce decode throughput
 * - Slow clients cannot throttle GPU
 * - Network jitter cannot create GPU idle gaps
 * - Server overhead orthogonal to decode performance
 * - Decode pure compute, streaming pure I/O
 * - Zero interference between domains
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <atomic>
#include <memory>
#include <vector>
#include <queue>
#include <thread>
#include <string>
#include <chrono>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Execution domain enumeration - enforced per-thread
 */
typedef enum {
    UNKNOWN = 0,
    DECODE = 1,
    STREAMING = 2
} streaming_execution_domain;

/**
 * Token record for lock-free queue
 */
typedef struct {
    uint32_t token_id;
    uint32_t sequence_id;
    uint64_t timestamp_ns;
    float logprob;
    uint16_t batch_slot;
    bool is_eos;
} streaming_token_record;

/**
 * Disconnect handler result
 */
typedef enum {
    DISCONNECT_HANDLED_OK = 0,
    DISCONNECT_ERROR = 1,
    CLIENT_STILL_CONNECTED = 2
} disconnect_status;

/**
 * Lock-free ring buffer for token queue
 * Single producer (decode), single consumer (streaming)
 */
class lock_free_token_queue {
private:
    std::vector<streaming_token_record> buffer;
    std::atomic<size_t> write_pos;
    std::atomic<size_t> read_pos;
    size_t capacity;

public:
    lock_free_token_queue(size_t capacity);

    // Producer (decode thread)
    bool try_push(const streaming_token_record & token);
    bool is_full() const;

    // Consumer (streaming thread)
    bool try_pop(streaming_token_record & token);
    bool is_empty() const;

    size_t get_depth() const;
    size_t get_capacity() const { return capacity; }
    void clear();
};

/**
 * Disconnect handler - non-blocking client disconnect management
 */
class disconnect_handler {
private:
    std::atomic<bool> client_connected;
    std::atomic<bool> disconnect_pending;

public:
    disconnect_handler();

    bool is_connected() const { return client_connected.load(); }
    void mark_disconnect_pending() { disconnect_pending.store(true); }
    bool is_disconnect_pending() const { return disconnect_pending.load(); }

    disconnect_status handle_disconnect();
    void reset_connection() { client_connected.store(true); disconnect_pending.store(false); }
};

/**
 * Async streaming engine - orchestrates decode/streaming separation
 */
class async_streaming_engine {
private:
    std::unique_ptr<lock_free_token_queue> token_queue;
    std::unique_ptr<disconnect_handler> disconnect_mgr;

    std::atomic<bool> streaming_active;
    std::atomic<bool> enforcing_separation;

    // Statistics
    std::atomic<uint64_t> total_tokens_decoded;
    std::atomic<uint64_t> total_tokens_streamed;
    std::atomic<uint64_t> domain_violations;
    std::atomic<uint64_t> queue_drops;

public:
    async_streaming_engine();

    // Initialization
    bool initialize(size_t queue_capacity);

    // Domain management
    void set_decode_domain() { set_current_domain(streaming_execution_domain::DECODE); }
    void set_streaming_domain() { set_current_domain(streaming_execution_domain::STREAMING); }

    // Token flow
    bool enqueue_token(const streaming_token_record & token);
    bool dequeue_token(streaming_token_record & token);

    // Queue management
    size_t get_queue_depth() const { return token_queue->get_depth(); }
    bool is_queue_full() const { return token_queue->is_full(); }
    bool is_queue_empty() const { return token_queue->is_empty(); }

    // Disconnect handling
    bool is_client_connected() const { return disconnect_mgr->is_connected(); }
    void mark_disconnect() { disconnect_mgr->mark_disconnect_pending(); }
    disconnect_status handle_disconnect() { return disconnect_mgr->handle_disconnect(); }

    // Enforcement
    bool is_separating_domains() const { return enforcing_separation.load(); }
    void set_enforce_separation(bool enforce) { enforcing_separation.store(enforce); }

    // Statistics
    uint64_t get_tokens_decoded() const { return total_tokens_decoded.load(); }
    uint64_t get_tokens_streamed() const { return total_tokens_streamed.load(); }
    uint64_t get_domain_violations() const { return domain_violations.load(); }
    uint64_t get_queue_drops() const { return queue_drops.load(); }

    void record_token_decoded() { total_tokens_decoded.fetch_add(1); }
    void record_token_streamed() { total_tokens_streamed.fetch_add(1); }
    void record_domain_violation() { domain_violations.fetch_add(1); }
    void record_queue_drop() { queue_drops.fetch_add(1); }

    // Validation
    bool validate_throughput_independence() const;
};

/**
 * Thread-local domain tracking
 */
streaming_execution_domain get_current_domain();
void set_current_domain(streaming_execution_domain domain);

/**
 * Domain verification and enforcement
 */
bool verify_domain(streaming_execution_domain allowed_domain, const char * operation_name);

/**
 * Hard separation enforcement (no-return on violation in debug)
 */
[[noreturn]]
void enforce_decode_purity(const char * operation_name);

/**
 * Streaming operations with domain checks
 */
bool llama_streaming_http_write(const char * data, size_t len);
bool llama_streaming_json_encode(const char * input, std::string & output);
bool llama_streaming_flush();
bool llama_streaming_check_client_connected();

/**
 * Global instance management
 */
extern async_streaming_engine * g_async_streaming;

bool llama_init_async_streaming(size_t queue_capacity);
bool llama_enable_async_streaming(bool enable);

// Domain enforcement macros
#define DECODE_DOMAIN_CHECK() \
    do { \
        if (!verify_domain(streaming_execution_domain::DECODE, __FUNCTION__)) { \
            return false; \
        } \
    } while(0)

#define STREAMING_DOMAIN_CHECK() \
    do { \
        if (!verify_domain(streaming_execution_domain::STREAMING, __FUNCTION__)) { \
            return false; \
        } \
    } while(0)

// Verbose logging control
void set_streaming_verbose_logging(bool enable);

// Diagnostic functions
void llama_print_async_streaming_stats();
void llama_dump_async_streaming_state();
bool llama_validate_streaming_separation();

#ifdef __cplusplus
}  // extern "C"
#endif
