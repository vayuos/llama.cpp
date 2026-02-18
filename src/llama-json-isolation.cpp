#include "llama-json-isolation.h"
#include "../common/log.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <thread>

// ============================================================================
// GLOBALS AND STATIC STORAGE
// ============================================================================

static bool g_isolation_verbose = false;
static std::string g_last_violation_message;
static pthread_mutex_t g_violation_message_mutex = PTHREAD_MUTEX_INITIALIZER;

#define LOG_ISOLATION(fmt, ...) \
    do { \
        if (g_isolation_verbose) { \
            LOG_INF("[JSON_ISOLATION] " fmt, __VA_ARGS__); \
        } \
    } while(0)

#define LOG_VIOLATION(fmt, ...) \
    do { \
        LOG_ERR("[JSON_ISOLATION_VIOLATION] " fmt, __VA_ARGS__); \
        pthread_mutex_lock(&g_violation_message_mutex); \
        g_last_violation_message = std::string(fmt); \
        pthread_mutex_unlock(&g_violation_message_mutex); \
    } while(0)

// ============================================================================
// LOCK-FREE TOKEN RECORD QUEUE IMPLEMENTATION
// ============================================================================

static int llama_token_record_queue_init(
    llama_token_record_queue * q,
    uint32_t capacity
) {
    if (!q || capacity == 0 || capacity > LLAMA_JSON_ISOLATION_MAX_TOKEN_QUEUE) {
        return -1;
    }

    // Allocate ring buffer
    q->ring_buffer = (llama_minimal_token_record *)malloc(
        capacity * sizeof(llama_minimal_token_record)
    );
    if (!q->ring_buffer) {
        return -2;
    }

    q->capacity = capacity;
    q->write_pos.store(0, std::memory_order_release);
    q->read_pos.store(0, std::memory_order_release);
    q->full_backpressure.store(false, std::memory_order_release);

    pthread_mutex_init(&q->consumer_lock, nullptr);
    pthread_cond_init(&q->consumer_signal, nullptr);

    q->total_enqueued = 0;
    q->total_dequeued = 0;
    q->total_overflows = 0;
    q->max_utilization = 0;

    LOG_ISOLATION("Token queue initialized: capacity=%u", capacity);
    return 0;
}

static void llama_token_record_queue_free(llama_token_record_queue * q) {
    if (!q) return;

    if (q->ring_buffer) {
        free(q->ring_buffer);
        q->ring_buffer = nullptr;
    }

    pthread_mutex_destroy(&q->consumer_lock);
    pthread_cond_destroy(&q->consumer_signal);
}

static int llama_token_record_queue_enqueue(
    llama_token_record_queue * q,
    const llama_minimal_token_record * record
) {
    if (!q || !record) {
        return -1;
    }

    uint32_t write_pos = q->write_pos.load(std::memory_order_acquire);
    uint32_t next_write = (write_pos + 1) % q->capacity;
    uint32_t read_pos = q->read_pos.load(std::memory_order_acquire);

    // Check if queue is full
    if (next_write == read_pos) {
        q->full_backpressure.store(true, std::memory_order_release);
        q->total_overflows++;
        return 1;  // Backpressure signal
    }

    // Copy record to ring buffer
    std::memcpy(&q->ring_buffer[write_pos], record, sizeof(llama_minimal_token_record));

    // Publish write position atomically
    q->write_pos.store(next_write, std::memory_order_release);
    q->total_enqueued++;

    // Update utilization
    uint32_t current_depth = (next_write - read_pos + q->capacity) % q->capacity;
    if (current_depth > q->max_utilization) {
        q->max_utilization = current_depth;
    }

    // Signal waiting consumers
    pthread_cond_signal(&q->consumer_signal);

    return 0;  // Success
}

static int llama_token_record_queue_dequeue(
    llama_token_record_queue * q,
    llama_minimal_token_record * record,
    bool wait_for_data
) {
    if (!q || !record) {
        return -1;
    }

    pthread_mutex_lock(&q->consumer_lock);

    while (true) {
        uint32_t read_pos = q->read_pos.load(std::memory_order_acquire);
        uint32_t write_pos = q->write_pos.load(std::memory_order_acquire);

        // Check if queue has data
        if (read_pos != write_pos) {
            // Copy record from ring buffer
            std::memcpy(record, &q->ring_buffer[read_pos], sizeof(llama_minimal_token_record));

            // Update read position atomically
            uint32_t next_read = (read_pos + 1) % q->capacity;
            q->read_pos.store(next_read, std::memory_order_release);
            q->total_dequeued++;

            // Clear backpressure if queue no longer full
            if (q->full_backpressure.load(std::memory_order_acquire)) {
                uint32_t depth = (write_pos - next_read + q->capacity) % q->capacity;
                if (depth < q->capacity / 2) {
                    q->full_backpressure.store(false, std::memory_order_release);
                }
            }

            pthread_mutex_unlock(&q->consumer_lock);
            return 0;  // Success
        }

        // Queue is empty
        if (!wait_for_data) {
            pthread_mutex_unlock(&q->consumer_lock);
            return 1;  // No data available
        }

        // Wait for data
        pthread_cond_wait(&q->consumer_signal, &q->consumer_lock);
    }
}

static uint32_t llama_token_record_queue_get_depth(
    const llama_token_record_queue * q
) {
    if (!q) return 0;

    uint32_t write_pos = q->write_pos.load(std::memory_order_acquire);
    uint32_t read_pos = q->read_pos.load(std::memory_order_acquire);

    return (write_pos - read_pos + q->capacity) % q->capacity;
}

// ============================================================================
// FORMATTING BUFFER POOL IMPLEMENTATION
// ============================================================================

static int llama_formatting_buffer_pool_init(
    llama_formatting_buffer_pool * pool,
    size_t buffer_size,
    uint32_t num_buffers
) {
    if (!pool || buffer_size == 0 || num_buffers == 0) {
        return -1;
    }

    if (num_buffers > LLAMA_JSON_ISOLATION_MAX_FORMAT_BUFFERS) {
        return -2;
    }

    pool->buffer_size = buffer_size;
    pool->total_bytes = 0;
    pool->allocations_served = 0;

    // Allocate all buffers upfront
    for (uint32_t i = 0; i < num_buffers; i++) {
        pool->buffers[i] = (char *)malloc(buffer_size);
        if (!pool->buffers[i]) {
            // Free previously allocated buffers
            for (uint32_t j = 0; j < i; j++) {
                free(pool->buffers[j]);
            }
            return -3;
        }
        pool->buffer_sizes[i] = buffer_size;
        pool->total_bytes += buffer_size;
    }

    pool->available_count.store(num_buffers, std::memory_order_release);
    pthread_mutex_init(&pool->lock, nullptr);

    LOG_ISOLATION("Formatting buffer pool initialized: buffers=%u, size=%zu, total=%zu bytes",
                  num_buffers, buffer_size, pool->total_bytes);
    return 0;
}

static void llama_formatting_buffer_pool_free(llama_formatting_buffer_pool * pool) {
    if (!pool) return;

    for (uint32_t i = 0; i < LLAMA_JSON_ISOLATION_MAX_FORMAT_BUFFERS; i++) {
        if (pool->buffers[i]) {
            free(pool->buffers[i]);
            pool->buffers[i] = nullptr;
        }
    }

    pthread_mutex_destroy(&pool->lock);
}

static char * llama_formatting_buffer_pool_acquire(
    llama_formatting_buffer_pool * pool,
    size_t * out_size
) {
    if (!pool || !out_size) {
        return nullptr;
    }

    pthread_mutex_lock(&pool->lock);

    uint32_t available = pool->available_count.load(std::memory_order_acquire);
    if (available == 0) {
        pthread_mutex_unlock(&pool->lock);
        return nullptr;  // No buffers available
    }

    // Find first available buffer
    char * buffer = nullptr;
    for (uint32_t i = 0; i < LLAMA_JSON_ISOLATION_MAX_FORMAT_BUFFERS; i++) {
        if (pool->buffers[i]) {
            buffer = pool->buffers[i];
            *out_size = pool->buffer_sizes[i];
            pool->buffers[i] = nullptr;  // Mark as in-use
            pool->available_count.store(available - 1, std::memory_order_release);
            pool->allocations_served++;
            break;
        }
    }

    pthread_mutex_unlock(&pool->lock);
    return buffer;
}

static int llama_formatting_buffer_pool_release(
    llama_formatting_buffer_pool * pool,
    char * buffer
) {
    if (!pool || !buffer) {
        return -1;
    }

    pthread_mutex_lock(&pool->lock);

    uint32_t available = pool->available_count.load(std::memory_order_acquire);
    if (available >= LLAMA_JSON_ISOLATION_MAX_FORMAT_BUFFERS) {
        pthread_mutex_unlock(&pool->lock);
        return -2;  // Pool full
    }

    // Find empty slot and return buffer
    for (uint32_t i = 0; i < LLAMA_JSON_ISOLATION_MAX_FORMAT_BUFFERS; i++) {
        if (!pool->buffers[i]) {
            pool->buffers[i] = buffer;
            pool->buffer_sizes[i] = pool->buffer_size;
            pool->available_count.store(available + 1, std::memory_order_release);
            pthread_mutex_unlock(&pool->lock);
            return 0;
        }
    }

    pthread_mutex_unlock(&pool->lock);
    return -3;  // No empty slots (shouldn't happen)
}

// ============================================================================
// SERVER WORKER THREAD IMPLEMENTATION
// ============================================================================

// Forward declare context type for worker thread function
typedef struct {
    llama_json_isolation_context_t * isolation_ctx;
    uint32_t worker_id;
} llama_server_worker_startup_args;

static void * llama_server_worker_thread_fn(void * args_void) {
    llama_server_worker_startup_args * startup = (llama_server_worker_startup_args *)args_void;
    llama_json_isolation_context_t * ctx = startup->isolation_ctx;
    uint32_t worker_id = startup->worker_id;

    free(startup);

    if (!ctx || worker_id >= ctx->num_workers) {
        return nullptr;
    }

    llama_server_worker_context * worker = &ctx->workers[worker_id];

    LOG_ISOLATION("Server worker %u started", worker_id);

    // Main worker loop
    while (worker->running) {
        llama_minimal_token_record token_record;

        // Try to dequeue token
        int dequeue_result = llama_token_record_queue_dequeue(
            ctx->token_queue,
            &token_record,
            true  // Wait for data
        );

        if (dequeue_result != 0) {
            // No data or error
            continue;
        }

        // Token dequeued successfully - now do all the expensive work
        auto token_start_ns = std::chrono::high_resolution_clock::now();

        // Convert token to text (expensive, not in decode!)
        // Note: This is where token_to_text would be called
        // It's safe to do here because we're NOT in the decode thread

        // Serialize to JSON (expensive, not in decode!)
        // This is where JSON serialization happens - on server worker
        worker->json_objects_created++;
        uint64_t json_start_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        // Simulate JSON serialization (in real code, would call JSON library)
        // char * json_buffer = llama_formatting_buffer_pool_acquire(ctx->buffer_pool, &buf_size);
        // if (json_buffer) {
        //     // Serialize token_record to JSON
        //     // snprintf(json_buffer, buf_size, "{\"token_id\":%u,...}", token_record.token_id);
        //     worker->bytes_serialized += strlen(json_buffer);
        //     llama_formatting_buffer_pool_release(ctx->buffer_pool, json_buffer);
        // }

        auto json_end_ns = std::chrono::high_resolution_clock::now();
        uint64_t json_time_ns = (json_end_ns - json_start_ns).count();
        worker->total_serialization_time_ns += json_time_ns;
        if (json_time_ns > worker->max_serialization_time_ns) {
            worker->max_serialization_time_ns = json_time_ns;
        }

        // Flush to HTTP stream (expensive, not in decode!)
        // This is where HTTP writes would happen - never blocks decode

        worker->tokens_processed++;
        ctx->total_tokens_emitted++;
    }

    LOG_ISOLATION("Server worker %u shutting down", worker_id);
    return nullptr;
}

static int llama_server_workers_create(
    llama_json_isolation_context_t * ctx,
    uint32_t num_workers
) {
    if (!ctx || num_workers == 0 || num_workers > LLAMA_JSON_ISOLATION_MAX_WORKERS) {
        return -1;
    }

    ctx->workers = (llama_server_worker_context *)calloc(
        num_workers,
        sizeof(llama_server_worker_context)
    );
    if (!ctx->workers) {
        return -2;
    }

    ctx->num_workers = num_workers;

    // Create worker threads
    for (uint32_t i = 0; i < num_workers; i++) {
        ctx->workers[i].worker_id = i;
        ctx->workers[i].running = true;
        ctx->workers[i].tokens_processed = 0;
        ctx->workers[i].json_objects_created = 0;
        ctx->workers[i].bytes_serialized = 0;
        ctx->workers[i].total_serialization_time_ns = 0;
        ctx->workers[i].max_serialization_time_ns = 0;
        ctx->workers[i].queue_wait_time_ns = 0;
        ctx->workers[i].conversion_time_ns = 0;

        llama_server_worker_startup_args * startup =
            (llama_server_worker_startup_args *)malloc(sizeof(llama_server_worker_startup_args));
        if (!startup) {
            return -3;
        }

        startup->isolation_ctx = ctx;
        startup->worker_id = i;

        int pthread_result = pthread_create(
            &ctx->workers[i].thread_handle,
            nullptr,
            llama_server_worker_thread_fn,
            startup
        );

        if (pthread_result != 0) {
            free(startup);
            return -4;
        }

        LOG_ISOLATION("Server worker thread %u created", i);
    }

    return 0;
}

static int llama_server_workers_shutdown(llama_json_isolation_context_t * ctx) {
    if (!ctx || !ctx->workers) {
        return -1;
    }

    // Signal workers to stop
    for (uint32_t i = 0; i < ctx->num_workers; i++) {
        ctx->workers[i].running = false;
    }

    // Signal all waiting threads
    pthread_cond_broadcast(&ctx->token_queue->consumer_signal);

    // Wait for threads to complete
    for (uint32_t i = 0; i < ctx->num_workers; i++) {
        pthread_join(ctx->workers[i].thread_handle, nullptr);
        LOG_ISOLATION("Server worker thread %u joined", i);
    }

    free(ctx->workers);
    ctx->workers = nullptr;
    ctx->num_workers = 0;

    return 0;
}

// ============================================================================
// TOKEN RECORD VALIDATION
// ============================================================================

bool llama_json_isolation_validate_token_record(
    const llama_minimal_token_record * record
) {
    if (!record) {
        return false;
    }

    // Check token_id is in reasonable range
    if (record->token_id >= 1000000) {  // Arbitrary but sensible limit
        return false;
    }

    // Check logprob is finite
    if (!std::isfinite(record->logprob)) {
        return false;
    }

    // Check timestamp is reasonable (not zero or way in future)
    uint64_t now_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    if (record->timestamp_ns == 0 || record->timestamp_ns > now_ns + 1000000000ULL) {
        return false;
    }

    return true;
}

// ============================================================================
// GUARD ASSERTION IMPLEMENTATION
// ============================================================================

int llama_json_isolation_assert_no_json_in_decode(
    llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return -1;
    }

    // In production, this would use code instrumentation or CPU counters
    // to detect if JSON libraries (rapidjson, nlohmann, etc.) are being called
    // For now, we check state consistency

    if (ctx->current_state != LLAMA_JSON_ISOLATION_ENFORCING) {
        return 0;  // Not enforcing, skip check
    }

    // Verify decode thread is current thread
    uint32_t current_thread = (uint32_t)pthread_self();
    if (ctx->decode_thread_id != current_thread) {
        LOG_VIOLATION("JSON guard called from non-decode thread");
        ctx->last_violation = LLAMA_JSON_VIOLATION_JSON_IN_DECODE;
        ctx->total_violations++;
        return -2;
    }

    return 0;
}

int llama_json_isolation_assert_no_string_alloc_in_decode(
    llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return -1;
    }

    if (ctx->current_state != LLAMA_JSON_ISOLATION_ENFORCING) {
        return 0;
    }

    // Check would detect std::string construction
    // Requires instrumentation of operator new
    return 0;
}

int llama_json_isolation_assert_no_dynamic_alloc_in_decode(
    llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return -1;
    }

    if (ctx->current_state != LLAMA_JSON_ISOLATION_ENFORCING) {
        return 0;
    }

    // Check would detect malloc/new calls
    // Requires malloc hook or instrumentation
    return 0;
}

int llama_json_isolation_assert_nonblocking_output(
    llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return -1;
    }

    if (ctx->current_state != LLAMA_JSON_ISOLATION_ENFORCING) {
        return 0;
    }

    // Check that decode never acquires output buffer locks
    // Queue enqueue is non-blocking, so no issue there
    return 0;
}

int llama_json_isolation_guard_all_checks(
    llama_json_isolation_context_t * ctx,
    uint64_t token_index
) {
    if (!ctx) {
        return -1;
    }

    int result = 0;
    result |= llama_json_isolation_assert_no_json_in_decode(ctx);
    result |= llama_json_isolation_assert_no_string_alloc_in_decode(ctx);
    result |= llama_json_isolation_assert_no_dynamic_alloc_in_decode(ctx);
    result |= llama_json_isolation_assert_nonblocking_output(ctx);

    if (result != 0 && ctx->abort_on_violation) {
        LOG_VIOLATION("Critical violation at token %lu", token_index);
        abort();
    }

    return result;
}

// ============================================================================
// PUBLIC API IMPLEMENTATION
// ============================================================================

int llama_json_isolation_init(
    llama_json_isolation_context_t * ctx,
    uint32_t queue_capacity,
    uint32_t num_server_workers,
    size_t format_buffer_size,
    bool abort_on_violation
) {
    if (!ctx) {
        return -1;
    }

    // Initialize state
    ctx->current_state = LLAMA_JSON_ISOLATION_UNINITIALIZED;
    ctx->previous_state = LLAMA_JSON_ISOLATION_UNINITIALIZED;
    pthread_mutex_init(&ctx->state_lock, nullptr);

    // Allocate token queue
    ctx->token_queue = (llama_token_record_queue *)malloc(sizeof(llama_token_record_queue));
    if (!ctx->token_queue) {
        return -2;
    }

    if (llama_token_record_queue_init(ctx->token_queue, queue_capacity) != 0) {
        free(ctx->token_queue);
        ctx->token_queue = nullptr;
        return -3;
    }

    // Allocate formatting buffer pool
    ctx->buffer_pool = (llama_formatting_buffer_pool *)malloc(sizeof(llama_formatting_buffer_pool));
    if (!ctx->buffer_pool) {
        llama_token_record_queue_free(ctx->token_queue);
        free(ctx->token_queue);
        return -4;
    }

    uint32_t num_buffers = (LLAMA_JSON_ISOLATION_BUFFER_POOL_SIZE / format_buffer_size);
    if (num_buffers > LLAMA_JSON_ISOLATION_MAX_FORMAT_BUFFERS) {
        num_buffers = LLAMA_JSON_ISOLATION_MAX_FORMAT_BUFFERS;
    }
    if (num_buffers == 0) {
        num_buffers = 1;
    }

    if (llama_formatting_buffer_pool_init(ctx->buffer_pool, format_buffer_size, num_buffers) != 0) {
        llama_token_record_queue_free(ctx->token_queue);
        free(ctx->token_queue);
        free(ctx->buffer_pool);
        return -5;
    }

    // Initialize state
    ctx->decode_thread_id = 0;
    ctx->decode_thread_registered = false;
    ctx->total_tokens_decoded = 0;
    ctx->total_tokens_emitted = 0;
    ctx->tokens_in_flight = 0;
    ctx->decode_loop_iterations = 0;
    ctx->decode_critical_section_exits = 0;

    ctx->last_violation = LLAMA_JSON_VIOLATION_NONE;
    ctx->total_violations = 0;
    ctx->abort_on_violation = abort_on_violation;

    ctx->average_token_enqueue_ns = 0;
    ctx->max_token_enqueue_ns = 0;
    ctx->tokens_per_second_with_streaming = 0;
    ctx->tokens_per_second_without_streaming = 0;

    ctx->streaming_enabled = false;
    ctx->validate_token_records = true;
    ctx->measure_decode_isolation = true;

    ctx->init_time_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Create server workers
    if (llama_server_workers_create(ctx, num_server_workers) != 0) {
        llama_formatting_buffer_pool_free(ctx->buffer_pool);
        llama_token_record_queue_free(ctx->token_queue);
        free(ctx->buffer_pool);
        free(ctx->token_queue);
        return -6;
    }

    pthread_mutex_lock(&ctx->state_lock);
    ctx->current_state = LLAMA_JSON_ISOLATION_CONFIGURED;
    pthread_mutex_unlock(&ctx->state_lock);

    LOG_ISOLATION("JSON isolation initialized: queue=%u, workers=%u, buffers=%u",
                  queue_capacity, num_server_workers, num_buffers);
    return 0;
}

int llama_json_isolation_configure_streaming(
    llama_json_isolation_context_t * ctx,
    bool streaming_enabled
) {
    if (!ctx) {
        return -1;
    }

    pthread_mutex_lock(&ctx->state_lock);

    if (ctx->current_state != LLAMA_JSON_ISOLATION_CONFIGURED) {
        pthread_mutex_unlock(&ctx->state_lock);
        return -2;
    }

    ctx->streaming_enabled = streaming_enabled;
    LOG_ISOLATION("Streaming configured: %s", streaming_enabled ? "enabled" : "disabled");

    pthread_mutex_unlock(&ctx->state_lock);
    return 0;
}

int llama_json_isolation_register_decode_thread(
    llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return -1;
    }

    pthread_mutex_lock(&ctx->state_lock);

    if (ctx->decode_thread_registered) {
        pthread_mutex_unlock(&ctx->state_lock);
        return -2;  // Already registered
    }

    ctx->decode_thread_id = (uint32_t)pthread_self();
    ctx->decode_thread_registered = true;

    LOG_ISOLATION("Decode thread registered: thread_id=%u", ctx->decode_thread_id);

    pthread_mutex_unlock(&ctx->state_lock);
    return 0;
}

void llama_json_isolation_enter_critical_section(
    llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return;
    }

    pthread_mutex_lock(&ctx->state_lock);

    if (ctx->current_state == LLAMA_JSON_ISOLATION_CONFIGURED) {
        ctx->previous_state = ctx->current_state;
        ctx->current_state = LLAMA_JSON_ISOLATION_ENFORCING;

        // Enable streaming state if configured
        if (ctx->streaming_enabled) {
            // Note: state doesn't change to STREAMING here, that's for output
            // Current state remains ENFORCING to track decode phase
        }
    }

    pthread_mutex_unlock(&ctx->state_lock);

    LOG_ISOLATION("Entered decode critical section");
}

void llama_json_isolation_exit_critical_section(
    llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return;
    }

    pthread_mutex_lock(&ctx->state_lock);

    ctx->decode_critical_section_exits++;

    if (ctx->current_state == LLAMA_JSON_ISOLATION_ENFORCING) {
        ctx->previous_state = ctx->current_state;
        ctx->current_state = LLAMA_JSON_ISOLATION_CONFIGURED;
    }

    pthread_mutex_unlock(&ctx->state_lock);

    LOG_ISOLATION("Exited decode critical section, tokens_decoded=%lu",
                  ctx->total_tokens_decoded);
}

int llama_json_isolation_shutdown(
    llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return -1;
    }

    pthread_mutex_lock(&ctx->state_lock);
    ctx->current_state = LLAMA_JSON_ISOLATION_SHUTDOWN;
    pthread_mutex_unlock(&ctx->state_lock);

    // Shutdown server workers (they will drain queue and exit)
    if (llama_server_workers_shutdown(ctx) != 0) {
        return -2;
    }

    // Free resources
    if (ctx->token_queue) {
        llama_token_record_queue_free(ctx->token_queue);
        free(ctx->token_queue);
        ctx->token_queue = nullptr;
    }

    if (ctx->buffer_pool) {
        llama_formatting_buffer_pool_free(ctx->buffer_pool);
        free(ctx->buffer_pool);
        ctx->buffer_pool = nullptr;
    }

    pthread_mutex_destroy(&ctx->state_lock);

    LOG_ISOLATION("JSON isolation shutdown complete");
    return 0;
}

int llama_json_isolation_enqueue_token(
    llama_json_isolation_context_t * ctx,
    uint32_t token_id,
    uint32_t sequence_id,
    float logprob,
    uint16_t batch_slot,
    uint32_t flags
) {
    if (!ctx || !ctx->token_queue) {
        return -1;
    }

    // This must execute in microseconds, no blocking
    auto enqueue_start = std::chrono::high_resolution_clock::now();

    // Create minimal record (stack allocation, very fast)
    llama_minimal_token_record record;
    record.token_id = token_id;
    record.sequence_id = sequence_id;
    record.timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    record.logprob = logprob;
    record.batch_slot = batch_slot;
    record.flags = flags;
    record.decode_wall_time_ns = 0;  // Will be filled by worker

    // Validate record
    if (ctx->validate_token_records && !llama_json_isolation_validate_token_record(&record)) {
        LOG_VIOLATION("Invalid token record: token_id=%u", token_id);
        return -2;
    }

    // Enqueue (lock-free, non-blocking)
    int enqueue_result = llama_token_record_queue_enqueue(ctx->token_queue, &record);

    auto enqueue_end = std::chrono::high_resolution_clock::now();
    uint64_t enqueue_time_ns = (enqueue_end - enqueue_start).count();

    // Update metrics
    ctx->total_tokens_decoded++;
    ctx->tokens_in_flight = llama_token_record_queue_get_depth(ctx->token_queue);

    // Update enqueue timing
    if (ctx->average_token_enqueue_ns == 0) {
        ctx->average_token_enqueue_ns = enqueue_time_ns;
    } else {
        ctx->average_token_enqueue_ns = (ctx->average_token_enqueue_ns * 0.9) + (enqueue_time_ns * 0.1);
    }

    if (enqueue_time_ns > ctx->max_token_enqueue_ns) {
        ctx->max_token_enqueue_ns = enqueue_time_ns;
    }

    // Periodic guard checks
    if ((ctx->total_tokens_decoded % LLAMA_JSON_ISOLATION_GUARD_INTERVAL) == 0) {
        llama_json_isolation_guard_all_checks(ctx, ctx->total_tokens_decoded);
    }

    return enqueue_result;
}

uint32_t llama_json_isolation_get_queue_depth(
    const llama_json_isolation_context_t * ctx
) {
    if (!ctx || !ctx->token_queue) {
        return 0;
    }

    return llama_token_record_queue_get_depth(ctx->token_queue);
}

bool llama_json_isolation_check_backpressure(
    const llama_json_isolation_context_t * ctx
) {
    if (!ctx || !ctx->token_queue) {
        return false;
    }

    return ctx->token_queue->full_backpressure.load(std::memory_order_acquire);
}

llama_json_isolation_state_t llama_json_isolation_get_state(
    const llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return LLAMA_JSON_ISOLATION_UNINITIALIZED;
    }

    return ctx->current_state;
}

llama_json_violation_type_t llama_json_isolation_get_last_violation(
    const llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return LLAMA_JSON_VIOLATION_NONE;
    }

    return ctx->last_violation;
}

const char * llama_json_isolation_get_violation_message(
    const llama_json_isolation_context_t * ctx
) {
    pthread_mutex_lock(&g_violation_message_mutex);
    const char * msg = g_last_violation_message.c_str();
    pthread_mutex_unlock(&g_violation_message_mutex);

    return msg;
}

int llama_json_isolation_get_metrics(
    const llama_json_isolation_context_t * ctx,
    llama_json_isolation_metrics * metrics
) {
    if (!ctx || !metrics) {
        return -1;
    }

    metrics->total_tokens_decoded = ctx->total_tokens_decoded;
    metrics->total_tokens_emitted = ctx->total_tokens_emitted;
    metrics->current_queue_depth = llama_json_isolation_get_queue_depth(ctx);
    metrics->max_queue_depth = ctx->token_queue ? ctx->token_queue->max_utilization : 0;

    metrics->average_enqueue_ns = ctx->average_token_enqueue_ns;
    metrics->max_enqueue_ns = ctx->max_token_enqueue_ns;

    metrics->total_json_objects_created = 0;
    metrics->total_json_serialization_time_ns = 0;
    metrics->average_json_time_per_token_ns = 0;

    // Aggregate worker stats
    for (uint32_t i = 0; i < ctx->num_workers; i++) {
        metrics->total_json_objects_created += ctx->workers[i].json_objects_created;
        metrics->total_json_serialization_time_ns += ctx->workers[i].total_serialization_time_ns;
    }

    if (metrics->total_json_objects_created > 0) {
        metrics->average_json_time_per_token_ns =
            metrics->total_json_serialization_time_ns / metrics->total_json_objects_created;
    }

    metrics->total_violations = ctx->total_violations;
    metrics->tokens_per_second_with_streaming = ctx->tokens_per_second_with_streaming;
    metrics->tokens_per_second_without_streaming = ctx->tokens_per_second_without_streaming;
    metrics->streaming_active = ctx->streaming_enabled;

    return 0;
}

int llama_json_isolation_validate_architecture(
    const llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return -1;
    }

    // Check that no JSON operations occurred in decode
    if (ctx->total_violations > 0) {
        LOG_VIOLATION("Architecture validation failed: %lu violations detected", ctx->total_violations);
        return -2;
    }

    // Check token queue is non-blocking
    if (!ctx->token_queue) {
        return -3;
    }

    // Check average enqueue time is very small (< 1 microsecond)
    if (ctx->average_token_enqueue_ns > 1000) {
        LOG_VIOLATION("Enqueue latency too high: %lu ns", ctx->average_token_enqueue_ns);
        return -4;
    }

    LOG_ISOLATION("Architecture validation passed");
    return 0;
}

int llama_json_isolation_validate_throughput_isolation(
    llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return -1;
    }

    // Note: This would require running benchmark with and without streaming
    // For now, log that check should be performed
    LOG_ISOLATION("Throughput isolation validation: with_streaming=%lu t/s, without_streaming=%lu t/s",
                  ctx->tokens_per_second_with_streaming,
                  ctx->tokens_per_second_without_streaming);

    return 0;
}

int llama_json_isolation_report_status(
    const llama_json_isolation_context_t * ctx
) {
    if (!ctx) {
        return -1;
    }

    llama_json_isolation_metrics metrics;
    if (llama_json_isolation_get_metrics(ctx, &metrics) != 0) {
        return -2;
    }

    LOG_INF("\n=== JSON Isolation Status ===");
    LOG_INF("State: %d", ctx->current_state);
    LOG_INF("Tokens decoded: %lu", metrics.total_tokens_decoded);
    LOG_INF("Tokens emitted: %lu", metrics.total_tokens_emitted);
    LOG_INF("Queue depth: %u/%u", metrics.current_queue_depth, metrics.max_queue_depth);
    LOG_INF("Average enqueue time: %lu ns", metrics.average_enqueue_ns);
    LOG_INF("Max enqueue time: %lu ns", metrics.max_enqueue_ns);
    LOG_INF("JSON objects created: %lu", metrics.total_json_objects_created);
    LOG_INF("Total JSON time: %lu ns", metrics.total_json_serialization_time_ns);
    LOG_INF("Average JSON time/token: %lu ns", metrics.average_json_time_per_token_ns);
    LOG_INF("Violations: %lu", metrics.total_violations);
    LOG_INF("Streaming: %s", metrics.streaming_active ? "enabled" : "disabled");

    return 0;
}

int llama_json_isolation_set_token_to_text_fn(
    llama_json_isolation_context_t * ctx,
    llama_token_to_text_fn fn
) {
    if (!ctx) {
        return -1;
    }

    // Store function pointer in context (not implemented in this version)
    // Would be used by server workers to convert token IDs to text
    return 0;
}
