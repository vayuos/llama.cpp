/**
 * ASYNC PIPELINING STREAM SCHEDULER
 * Header: Multi-stream compute orchestration for CPU-GPU parallelism
 *
 * Provides:
 * - CUDA stream pool management
 * - Token-to-stream mapping
 * - Synchronization event tracking
 * - Stream state machine
 *
 * Design: Allows CPU and GPU to process different tokens concurrently
 * Benefit: Reduces GPU idle time, improves throughput by 15-25%
 */

#pragma once

#include <cstdint>
#include <vector>
#include <queue>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// STREAM SCHEDULER TYPES
// ============================================================================

/**
 * Stream types for async pipelining
 */
enum llama_stream_type {
    LLAMA_STREAM_CPU_COMPUTE = 0,    // CPU layer computation
    LLAMA_STREAM_GPU_COMPUTE = 1,    // GPU layer computation
    LLAMA_STREAM_TRANSFER = 2,       // CPU-GPU data transfer
    LLAMA_STREAM_UTILITY = 3         // Synchronization, sampling
};

/**
 * Token processing state in pipeline
 */
enum llama_token_state {
    TOKEN_UNSCHEDULED = 0,           // Not yet assigned to stream
    TOKEN_CPU_PENDING = 1,           // Waiting for CPU compute
    TOKEN_CPU_EXECUTING = 2,         // CPU computing layers 0-66
    TOKEN_CPU_COMPLETE = 3,          // CPU layers done, ready for GPU
    TOKEN_TRANSFER_PENDING = 4,      // Waiting for CPU→GPU transfer
    TOKEN_TRANSFER_EXECUTING = 5,    // Transferring intermediate result
    TOKEN_GPU_PENDING = 6,           // Waiting for GPU compute
    TOKEN_GPU_EXECUTING = 7,         // GPU computing layers 36-49
    TOKEN_GPU_COMPLETE = 8,          // GPU layers done, output ready
    TOKEN_ERROR = -1                 // Processing failed
};

/**
 * Per-token scheduler state
 */
struct llama_token_schedule {
    int token_id;                    // Token index
    enum llama_token_state state;    // Current processing state
    uint64_t timestamp_created;      // When queued
    uint64_t timestamp_cpu_start;    // When CPU started
    uint64_t timestamp_cpu_end;      // When CPU finished
    uint64_t timestamp_gpu_start;    // When GPU started
    uint64_t timestamp_gpu_end;      // When GPU finished
    void * cpu_buffer;               // Buffer for CPU intermediate result
    void * gpu_buffer;               // Buffer for GPU input
    int error_code;                  // Error if state == TOKEN_ERROR
};

/**
 * CUDA stream wrapper with metadata
 */
struct llama_cuda_stream {
    void * cuda_stream;              // Actual cudaStream_t (void* for C compat)
    enum llama_stream_type type;     // Stream purpose (CPU, GPU, etc.)
    int priority;                    // CUDA stream priority
    bool in_use;                     // Currently executing work
    uint64_t operations_count;       // Total ops executed on this stream
    uint64_t total_time_ns;          // Total execution time
};

/**
 * Multi-stream scheduler state
 */
struct llama_stream_scheduler {
    // Stream pools
    struct llama_cuda_stream streams[4];           // 4 streams: CPU, GPU, Transfer, Utility
    int num_streams;                              // Active streams

    // Token queue management
    struct llama_token_schedule * token_queue;    // Array of token schedules
    int queue_capacity;                           // Max tokens in pipeline
    int queue_size;                               // Currently scheduled tokens
    int queue_head;                               // First unprocessed token

    // Synchronization events
    void ** sync_events;                          // CUDA events for stream sync (void* for C compat)
    int num_events;                               // Total sync events

    // Pipeline state
    int max_concurrent_tokens;                    // Max tokens in flight
    int next_token_to_schedule;                   // Next token ID to assign
    int next_token_to_output;                     // Next token ID to output

    // Statistics
    uint64_t total_tokens_processed;
    uint64_t total_pipeline_stalls;               // Times GPU waited for CPU
    uint64_t total_sync_wait_ns;                  // Total time waiting for sync
    uint64_t creation_time_ns;
};

// ============================================================================
// SCHEDULER LIFECYCLE
// ============================================================================

/**
 * Initialize multi-stream scheduler.
 * Allocates CUDA streams, event objects, and token queue.
 *
 * @param max_concurrent_tokens Max tokens in pipeline (typically 2-4)
 * @return Pointer to initialized scheduler, NULL on error
 */
struct llama_stream_scheduler * llama_stream_scheduler_init(
    int max_concurrent_tokens);

/**
 * Cleanup and free scheduler resources.
 *
 * @param scheduler Scheduler to cleanup
 */
void llama_stream_scheduler_cleanup(
    struct llama_stream_scheduler * scheduler);

// ============================================================================
// QUEUE OPERATIONS
// ============================================================================

/**
 * Enqueue token for processing.
 * Adds token to queue with CPU_PENDING state.
 *
 * @param scheduler Active scheduler
 * @param token_id Token to process
 * @return 0 on success, -1 if queue full
 */
int llama_stream_scheduler_enqueue_token(
    struct llama_stream_scheduler * scheduler,
    int token_id);

/**
 * Get next token ready for GPU processing.
 * Returns token with CPU_COMPLETE state.
 *
 * @param scheduler Active scheduler
 * @return Token ID, or -1 if none ready
 */
int llama_stream_scheduler_get_gpu_ready_token(
    struct llama_stream_scheduler * scheduler);

/**
 * Mark token as GPU complete.
 * Transitions to GPU_COMPLETE, output ready.
 *
 * @param scheduler Active scheduler
 * @param token_id Token that finished GPU processing
 * @return 0 on success, -1 on error
 */
int llama_stream_scheduler_mark_gpu_complete(
    struct llama_stream_scheduler * scheduler,
    int token_id);

/**
 * Get next output token.
 * Returns token with GPU_COMPLETE state in order.
 *
 * @param scheduler Active scheduler
 * @return Token ID, or -1 if none ready
 */
int llama_stream_scheduler_get_output_token(
    struct llama_stream_scheduler * scheduler);

// ============================================================================
// STREAM OPERATIONS
// ============================================================================

/**
 * Get CPU compute stream.
 *
 * @param scheduler Active scheduler
 * @return CUDA stream for CPU compute operations
 */
void * llama_stream_scheduler_get_cpu_stream(
    struct llama_stream_scheduler * scheduler);

/**
 * Get GPU compute stream.
 *
 * @param scheduler Active scheduler
 * @return CUDA stream for GPU compute operations
 */
void * llama_stream_scheduler_get_gpu_stream(
    struct llama_stream_scheduler * scheduler);

/**
 * Get transfer stream (CPU→GPU data movement).
 *
 * @param scheduler Active scheduler
 * @return CUDA stream for transfers
 */
void * llama_stream_scheduler_get_transfer_stream(
    struct llama_stream_scheduler * scheduler);

// ============================================================================
// SYNCHRONIZATION
// ============================================================================

/**
 * Record synchronization event on stream.
 * Used to track completion of token processing on a stream.
 *
 * @param scheduler Active scheduler
 * @param stream_type Which stream to record on
 * @param event_index Which event to record (0-3)
 * @return 0 on success, -1 on error
 */
int llama_stream_scheduler_record_event(
    struct llama_stream_scheduler * scheduler,
    enum llama_stream_type stream_type,
    int event_index);

/**
 * Wait for synchronization event.
 * Blocks until event is recorded by stream.
 *
 * @param scheduler Active scheduler
 * @param event_index Which event to wait for
 * @param timeout_ms Maximum milliseconds to wait (-1 = no timeout)
 * @return 0 if signaled, -1 on timeout/error
 */
int llama_stream_scheduler_wait_event(
    struct llama_stream_scheduler * scheduler,
    int event_index,
    int timeout_ms);

// ============================================================================
// DIAGNOSTICS
// ============================================================================

/**
 * Get scheduler statistics.
 * Returns copy of scheduler state for inspection.
 *
 * @param scheduler Active scheduler
 * @return Copy of scheduler state
 */
struct llama_stream_scheduler llama_stream_scheduler_get_state(
    struct llama_stream_scheduler * scheduler);

/**
 * Print detailed scheduler diagnostics.
 *
 * @param scheduler Active scheduler
 */
void llama_stream_scheduler_print_diagnostics(
    struct llama_stream_scheduler * scheduler);

/**
 * Print token queue status.
 *
 * @param scheduler Active scheduler
 */
void llama_stream_scheduler_print_queue(
    struct llama_stream_scheduler * scheduler);

#ifdef __cplusplus
}
#endif
