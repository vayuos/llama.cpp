#pragma once

/**
 * Asynchronous Streaming Decoupling for LLAMA Decode
 *
 * Complete separation of GPU decode and I/O streaming execution domains.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <atomic>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Execution domain enumeration - enforced per-thread
 */
typedef enum {
    LLAMA_STREAMING_DOMAIN_UNKNOWN = 0,
    LLAMA_STREAMING_DOMAIN_DECODE = 1,
    LLAMA_STREAMING_DOMAIN_STREAMING = 2
} streaming_execution_domain;

// For compatibility with source using EnumName::Value
#ifdef __cplusplus
#define UNKNOWN LLAMA_STREAMING_DOMAIN_UNKNOWN
#define DECODE LLAMA_STREAMING_DOMAIN_DECODE
#define STREAMING LLAMA_STREAMING_DOMAIN_STREAMING
#endif

/**
 * Token record for lock-free queue
 */
typedef struct {
    int32_t  token_id;
    uint32_t sequence_id;
    uint64_t timestamp_us;
    float    logit;
} streaming_token;

/**
 * Metrics structures
 */
typedef struct {
    size_t current_depth;
    size_t capacity;
    uint64_t total_overflow_events;
    float utilization_percent;
} streaming_queue_metrics;

typedef struct {
    int32_t worker_index;
    bool is_running;
    uint64_t tokens_processed;
    uint64_t chunks_flushed;
    uint64_t batches_created;
    float tokens_per_sec;
    int32_t active_sequences;
    int32_t pending_sequences;
} streaming_worker_metrics;

typedef struct {
    bool initialized;
    int32_t worker_count;
    size_t queue_depth;
    size_t queue_capacity;
    uint64_t total_tokens_produced;
    uint64_t total_tokens_consumed;
    uint64_t backpressure_events;
    float system_throughput_tps;
    std::vector<streaming_worker_metrics> per_worker;
} async_streaming_engine_metrics;

/**
 * Context structures
 */
typedef struct {
    uint32_t sequence_id;
    uint32_t slot_id;
    uint64_t start_time_us;
    uint64_t tokens_generated;
    void * user_context;
} streaming_decode_context;

typedef struct {
    uint32_t sequence_id;
    const char * stream_url;
    void * http_handle;
} streaming_http_context;

#ifdef __cplusplus
} // extern "C"

/**
 * Lock-free ring buffer for token queue
 */
class streaming_token_queue {
private:
    std::vector<streaming_token> buffer;
    std::atomic<uint64_t> head;
    std::atomic<uint64_t> tail;
    std::atomic<uint64_t> overflows;
    size_t mask;

public:
    streaming_token_queue(size_t capacity);
    ~streaming_token_queue();

    bool try_push(const streaming_token & token);
    bool try_pop(streaming_token & token);
    
    size_t depth() const;
    size_t capacity() const;
    bool is_full() const;
    bool is_empty() const;
    void clear();
    
    streaming_queue_metrics get_metrics() const;
};

/**
 * Batch accumulator
 */
class streaming_batch_accumulator {
private:
    std::vector<streaming_token> token_batch;
    std::string json_buffer;
    size_t target_batch_size;
    size_t max_buffer_bytes;

public:
    streaming_batch_accumulator(size_t batch_size, size_t buffer_size);
    ~streaming_batch_accumulator();

    bool add_token(const streaming_token & token, const std::string & json_chunk);
    bool should_flush() const;
    std::string get_batch_data() const;
    size_t batch_token_count() const;
    size_t buffered_bytes() const;
    std::string flush();
    void reset();
};

/**
 * Cancellation token
 */
class streaming_cancellation_token {
private:
    std::atomic<bool> cancelled;
    std::atomic<uint64_t> cancel_timestamp_us;
    std::string reason;

public:
    streaming_cancellation_token();
    void cancel();
    bool is_cancelled() const;
    void reset();
    std::string get_reason() const;
};

/**
 * Streaming worker thread
 */
class streaming_worker {
private:
    int32_t worker_index;
    size_t target_batch_size;
    int32_t flush_timeout_ms;
    
    streaming_token_queue * token_queue;
    const void * model_vocab;
    
    std::atomic<bool> running;
    std::atomic<bool> shutdown_requested;
    std::thread worker_thread;

    void worker_main_loop();
    bool process_token(const streaming_token & token);
    bool flush_batch_to_http(uint32_t sequence_id);

public:
    streaming_worker(int32_t idx, size_t batch_size = 1, int32_t timeout_ms = 100);
    ~streaming_worker();

    void register_token_queue(streaming_token_queue * queue);
    void register_vocab(const void * vocab);
    
    bool start();
    bool stop(int32_t timeout_ms = 1000);
    
    uint32_t register_http_context(const streaming_http_context & context);
    void unregister_http_context(uint32_t context_id);
    void link_decode_to_http(const streaming_decode_context & decode_context, uint32_t http_context_id);
    void signal_sequence_complete(uint32_t sequence_id);
    
    bool is_running() const;
    bool is_alive() const;
    streaming_worker_metrics get_metrics() const;
    
    bool flush_pending(uint32_t sequence_id);
    uint32_t flush_all();
    int32_t get_worker_index() const;
};

/**
 * Main streaming system (Renamed to async_streaming_engine for llama-context.h compatibility)
 */
class async_streaming_engine {
private:
    std::atomic<bool> initialized;
    std::atomic<bool> shutdown_in_progress;
    
    streaming_token_queue token_queue;
    const void * model_vocab;
    
    std::vector<std::unique_ptr<streaming_worker>> workers;
    std::map<uint32_t, streaming_decode_context> decode_contexts;
    std::map<uint32_t, streaming_http_context> http_contexts;

public:
    async_streaming_engine();
    ~async_streaming_engine();

    static async_streaming_engine & instance();

    bool initialize(int32_t worker_count, size_t queue_capacity, size_t batch_size);
    void shutdown(int32_t timeout_ms = 1000);

    streaming_token_queue * get_token_queue();
    void register_vocab(const void * vocab);

    bool decode_emit_token(const streaming_token & token, uint32_t sequence_id);
    size_t get_queue_depth() const;

    void signal_decode_start(uint32_t sequence_id, uint32_t slot_id, void * user_context);
    void signal_decode_complete(uint32_t sequence_id);

    uint32_t register_http_context(const streaming_http_context & context);
    void unregister_http_context(uint32_t context_id);
    void link_decode_to_http(uint32_t sequence_id, uint32_t context_id);

    bool is_initialized() const;
    async_streaming_engine_metrics get_metrics() const;
    uint32_t flush_all_pending();
};

// Typedef for source file compatibility
using streaming_system = async_streaming_engine;

extern "C" {
#endif

// Domain management
streaming_execution_domain get_current_domain();
void set_current_domain(streaming_execution_domain domain);
bool verify_domain(streaming_execution_domain allowed_domain, const char * operation_name);
[[noreturn]]
void enforce_decode_purity(const char * operation_name);

// System interaction
void set_streaming_verbose_logging(bool enable);
bool validate_streaming_domain_separation();
bool validate_streaming_throughput_independence(float cli_tps, float server_tps);
void dump_streaming_state();
std::string get_streaming_status();

#ifdef __cplusplus
}
#endif
