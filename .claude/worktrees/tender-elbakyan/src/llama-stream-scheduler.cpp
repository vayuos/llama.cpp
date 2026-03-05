/**
 * ASYNC PIPELINING STREAM SCHEDULER
 * Implementation: Multi-stream compute orchestration
 *
 * Manages:
 * - CUDA stream creation and lifecycle
 * - Token-to-stream queue
 * - Synchronization events
 * - Pipeline state transitions
 */

#include "llama-stream-scheduler.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// CUDA event management (for stream synchronization)
#ifdef __CUDACC__
    #include <cuda_runtime.h>
#else
    // Stub for non-CUDA builds
    typedef void* cudaEvent_t;
#endif

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize multi-stream scheduler.
 * Allocates:
 * - 4 CUDA streams (CPU, GPU, Transfer, Utility)
 * - Token queue
 * - Synchronization events
 */
struct llama_stream_scheduler * llama_stream_scheduler_init(
    int max_concurrent_tokens) {

    if (max_concurrent_tokens < 1 || max_concurrent_tokens > 16) {
        fprintf(stderr, "STREAM_SCHEDULER: Invalid max_concurrent_tokens (%d), valid range [1, 16]\n",
                max_concurrent_tokens);
        return NULL;
    }

    // Allocate scheduler structure
    struct llama_stream_scheduler * scheduler =
        (struct llama_stream_scheduler *)malloc(sizeof(struct llama_stream_scheduler));
    if (!scheduler) {
        fprintf(stderr, "STREAM_SCHEDULER: Failed to allocate scheduler\n");
        return NULL;
    }

    // Initialize token queue
    scheduler->token_queue = (struct llama_token_schedule *)calloc(
        max_concurrent_tokens, sizeof(struct llama_token_schedule));
    if (!scheduler->token_queue) {
        fprintf(stderr, "STREAM_SCHEDULER: Failed to allocate token queue\n");
        free(scheduler);
        return NULL;
    }

    // Initialize sync events (4 per stream type = 16 total)
    scheduler->num_events = 16;
    scheduler->sync_events = (void **)calloc(scheduler->num_events, sizeof(void *));
    if (!scheduler->sync_events) {
        fprintf(stderr, "STREAM_SCHEDULER: Failed to allocate sync events\n");
        free(scheduler->token_queue);
        free(scheduler);
        return NULL;
    }

    // Create CUDA streams
    scheduler->num_streams = 4;
    enum llama_stream_type stream_types[] = {
        LLAMA_STREAM_CPU_COMPUTE,
        LLAMA_STREAM_GPU_COMPUTE,
        LLAMA_STREAM_TRANSFER,
        LLAMA_STREAM_UTILITY
    };

    for (int i = 0; i < scheduler->num_streams; i++) {
        scheduler->streams[i].type = stream_types[i];
        scheduler->streams[i].priority = 0;
        scheduler->streams[i].in_use = false;
        scheduler->streams[i].operations_count = 0;
        scheduler->streams[i].total_time_ns = 0;

        // Note: Actual cudaStream_t creation deferred to GPU backend
        // This is set during first use via llama_stream_scheduler_get_*_stream()
        scheduler->streams[i].cuda_stream = NULL;
    }

    // Create CUDA sync events (for stream synchronization)
    // Each event tracks completion of operations on a stream
    for (int i = 0; i < scheduler->num_events; i++) {
#ifdef __CUDACC__
        cudaEvent_t * event = (cudaEvent_t *)malloc(sizeof(cudaEvent_t));
        if (!event) {
            fprintf(stderr, "STREAM_SCHEDULER: Failed to allocate event %d\n", i);
            llama_stream_scheduler_cleanup(scheduler);
            return NULL;
        }

        // Create non-blocking event (allows immediate record/wait without GPU sync)
        // cudaEventDisableTiming reduces memory usage since we don't need timing
        cudaError_t err = cudaEventCreate(event, cudaEventNonBlocking | cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "STREAM_SCHEDULER: Failed to create event %d: %s\n",
                    i, cudaGetErrorString(err));
            free(event);
            llama_stream_scheduler_cleanup(scheduler);
            return NULL;
        }

        scheduler->sync_events[i] = (void *)event;
#else
        scheduler->sync_events[i] = NULL;  // Stub for non-CUDA builds
#endif
    }

    // Initialize queue
    scheduler->queue_capacity = max_concurrent_tokens;
    scheduler->queue_size = 0;
    scheduler->queue_head = 0;
    scheduler->max_concurrent_tokens = max_concurrent_tokens;
    scheduler->next_token_to_schedule = 0;
    scheduler->next_token_to_output = 0;

    // Initialize statistics
    scheduler->total_tokens_processed = 0;
    scheduler->total_pipeline_stalls = 0;
    scheduler->total_sync_wait_ns = 0;
    scheduler->creation_time_ns =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    fprintf(stderr, "STREAM_SCHEDULER: Initialized with %d concurrent token slots\n",
            max_concurrent_tokens);

    return scheduler;
}

// ============================================================================
// CLEANUP
// ============================================================================

/**
 * Cleanup and free scheduler resources.
 * Destroys CUDA streams and events.
 */
void llama_stream_scheduler_cleanup(
    struct llama_stream_scheduler * scheduler) {

    if (!scheduler) {
        return;
    }

    // Destroy CUDA sync events
    if (scheduler->sync_events) {
#ifdef __CUDACC__
        for (int i = 0; i < scheduler->num_events; i++) {
            if (scheduler->sync_events[i]) {
                cudaEvent_t * event = (cudaEvent_t *)scheduler->sync_events[i];
                cudaError_t err = cudaEventDestroy(*event);
                if (err != cudaSuccess) {
                    fprintf(stderr, "STREAM_SCHEDULER: Warning - failed to destroy event %d: %s\n",
                            i, cudaGetErrorString(err));
                }
                free(event);
                scheduler->sync_events[i] = NULL;
            }
        }
#endif
        free(scheduler->sync_events);
        scheduler->sync_events = NULL;
    }

    if (scheduler->token_queue) {
        free(scheduler->token_queue);
        scheduler->token_queue = NULL;
    }

    free(scheduler);

    fprintf(stderr, "STREAM_SCHEDULER: Cleanup complete\n");
}

// ============================================================================
// QUEUE OPERATIONS
// ============================================================================

/**
 * Enqueue token for processing.
 * Adds token to queue with CPU_PENDING state.
 */
int llama_stream_scheduler_enqueue_token(
    struct llama_stream_scheduler * scheduler,
    int token_id) {

    if (!scheduler || !scheduler->token_queue) {
        return -1;
    }

    // Check if queue is full
    if (scheduler->queue_size >= scheduler->queue_capacity) {
        fprintf(stderr, "STREAM_SCHEDULER: Token queue full (%d/%d)\n",
                scheduler->queue_size, scheduler->queue_capacity);
        return -1;
    }

    // Add token to end of queue
    int queue_index = (scheduler->queue_head + scheduler->queue_size) % scheduler->queue_capacity;
    struct llama_token_schedule * token_sched = &scheduler->token_queue[queue_index];

    token_sched->token_id = token_id;
    token_sched->state = TOKEN_CPU_PENDING;
    token_sched->timestamp_created =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    token_sched->timestamp_cpu_start = 0;
    token_sched->timestamp_cpu_end = 0;
    token_sched->timestamp_gpu_start = 0;
    token_sched->timestamp_gpu_end = 0;
    token_sched->cpu_buffer = NULL;
    token_sched->gpu_buffer = NULL;
    token_sched->error_code = 0;

    scheduler->queue_size++;
    scheduler->next_token_to_schedule++;

    return 0;
}

/**
 * Get next token ready for GPU processing.
 * Returns token with CPU_COMPLETE state.
 */
int llama_stream_scheduler_get_gpu_ready_token(
    struct llama_stream_scheduler * scheduler) {

    if (!scheduler || !scheduler->token_queue) {
        return -1;
    }

    // Scan queue for first CPU_COMPLETE token
    for (int i = 0; i < scheduler->queue_size; i++) {
        int idx = (scheduler->queue_head + i) % scheduler->queue_capacity;
        if (scheduler->token_queue[idx].state == TOKEN_CPU_COMPLETE) {
            return scheduler->token_queue[idx].token_id;
        }
    }

    return -1;  // No GPU-ready token
}

/**
 * Mark token as GPU complete.
 * Transitions to GPU_COMPLETE state.
 */
int llama_stream_scheduler_mark_gpu_complete(
    struct llama_stream_scheduler * scheduler,
    int token_id) {

    if (!scheduler || !scheduler->token_queue) {
        return -1;
    }

    // Find token in queue
    for (int i = 0; i < scheduler->queue_size; i++) {
        int idx = (scheduler->queue_head + i) % scheduler->queue_capacity;
        if (scheduler->token_queue[idx].token_id == token_id) {
            scheduler->token_queue[idx].state = TOKEN_GPU_COMPLETE;
            scheduler->token_queue[idx].timestamp_gpu_end =
                std::chrono::high_resolution_clock::now().time_since_epoch().count();
            return 0;
        }
    }

    return -1;  // Token not found
}

/**
 * Get next output token.
 * Returns token with GPU_COMPLETE state, removes from queue.
 */
int llama_stream_scheduler_get_output_token(
    struct llama_stream_scheduler * scheduler) {

    if (!scheduler || !scheduler->token_queue) {
        return -1;
    }

    // Check if head is GPU_COMPLETE
    if (scheduler->queue_size > 0) {
        struct llama_token_schedule * head = &scheduler->token_queue[scheduler->queue_head];
        if (head->state == TOKEN_GPU_COMPLETE) {
            int token_id = head->token_id;

            // Remove from queue
            scheduler->queue_head = (scheduler->queue_head + 1) % scheduler->queue_capacity;
            scheduler->queue_size--;
            scheduler->total_tokens_processed++;
            scheduler->next_token_to_output++;

            return token_id;
        }
    }

    return -1;  // No output token ready
}

// ============================================================================
// STREAM OPERATIONS
// ============================================================================

/**
 * Get CPU compute stream.
 */
void * llama_stream_scheduler_get_cpu_stream(
    struct llama_stream_scheduler * scheduler) {

    if (!scheduler) {
        return NULL;
    }

    return scheduler->streams[LLAMA_STREAM_CPU_COMPUTE].cuda_stream;
}

/**
 * Get GPU compute stream.
 */
void * llama_stream_scheduler_get_gpu_stream(
    struct llama_stream_scheduler * scheduler) {

    if (!scheduler) {
        return NULL;
    }

    return scheduler->streams[LLAMA_STREAM_GPU_COMPUTE].cuda_stream;
}

/**
 * Get transfer stream.
 */
void * llama_stream_scheduler_get_transfer_stream(
    struct llama_stream_scheduler * scheduler) {

    if (!scheduler) {
        return NULL;
    }

    return scheduler->streams[LLAMA_STREAM_TRANSFER].cuda_stream;
}

// ============================================================================
// SYNCHRONIZATION
// ============================================================================

/**
 * Record synchronization event on stream.
 */
int llama_stream_scheduler_record_event(
    struct llama_stream_scheduler * scheduler,
    enum llama_stream_type stream_type,
    int event_index) {

    if (!scheduler || event_index < 0 || event_index >= scheduler->num_events) {
        return -1;
    }

    if (stream_type < 0 || stream_type >= scheduler->num_streams) {
        return -1;
    }

    if (!scheduler->sync_events[event_index]) {
        return -1;  // Event not initialized
    }

#ifdef __CUDACC__
    // Record event on the specified stream
    // This marks the completion point of all operations up to this point
    cudaStream_t stream = (cudaStream_t)scheduler->streams[stream_type].cuda_stream;
    cudaEvent_t * event = (cudaEvent_t *)scheduler->sync_events[event_index];

    if (!stream || !event) {
        return -1;
    }

    cudaError_t err = cudaEventRecord(*event, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "STREAM_SCHEDULER: Failed to record event %d on stream %d: %s\n",
                event_index, stream_type, cudaGetErrorString(err));
        return -1;
    }

    return 0;
#else
    // Stub for non-CUDA builds
    return 0;
#endif
}

/**
 * Wait for synchronization event.
 */
int llama_stream_scheduler_wait_event(
    struct llama_stream_scheduler * scheduler,
    int event_index,
    int timeout_ms) {

    if (!scheduler || event_index < 0 || event_index >= scheduler->num_events) {
        return -1;
    }

    if (!scheduler->sync_events[event_index]) {
        return -1;  // Event not initialized
    }

#ifdef __CUDACC__
    // Wait for the event to be recorded (signals completion of marked operations)
    cudaEvent_t * event = (cudaEvent_t *)scheduler->sync_events[event_index];

    if (!event) {
        return -1;
    }

    // Use cudaEventSynchronize for blocking wait
    // timeout_ms parameter reserved for future stream wait implementation
    cudaError_t err = cudaEventSynchronize(*event);
    if (err != cudaSuccess) {
        fprintf(stderr, "STREAM_SCHEDULER: Failed to wait for event %d: %s\n",
                event_index, cudaGetErrorString(err));
        return -1;
    }

    return 0;
#else
    // Stub for non-CUDA builds
    (void)timeout_ms;  // Suppress unused parameter warning
    return 0;
#endif
}

// ============================================================================
// DIAGNOSTICS
// ============================================================================

/**
 * Get scheduler state snapshot.
 */
struct llama_stream_scheduler llama_stream_scheduler_get_state(
    struct llama_stream_scheduler * scheduler) {

    if (!scheduler) {
        struct llama_stream_scheduler empty = {};
        return empty;
    }

    return *scheduler;
}

/**
 * Print detailed scheduler diagnostics.
 */
void llama_stream_scheduler_print_diagnostics(
    struct llama_stream_scheduler * scheduler) {

    if (!scheduler) {
        fprintf(stderr, "STREAM_SCHEDULER: Null scheduler\n");
        return;
    }

    uint64_t now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    uint64_t uptime_ns = now - scheduler->creation_time_ns;
    double uptime_ms = uptime_ns / 1e6;

    fprintf(stderr, "\n");
    fprintf(stderr, "=== STREAM SCHEDULER DIAGNOSTICS ===\n");
    fprintf(stderr, "Uptime: %.2f ms\n", uptime_ms);
    fprintf(stderr, "Total tokens processed: %lu\n", scheduler->total_tokens_processed);
    fprintf(stderr, "Pipeline stalls: %lu\n", scheduler->total_pipeline_stalls);
    fprintf(stderr, "Sync wait time: %.2f ms\n", scheduler->total_sync_wait_ns / 1e6);
    fprintf(stderr, "Queue: %d/%d tokens\n", scheduler->queue_size, scheduler->queue_capacity);
    fprintf(stderr, "Next to schedule: %d, Next to output: %d\n",
            scheduler->next_token_to_schedule, scheduler->next_token_to_output);
    fprintf(stderr, "\n");

    // Stream statistics
    fprintf(stderr, "Stream Statistics:\n");
    const char * stream_names[] = {"CPU_COMPUTE", "GPU_COMPUTE", "TRANSFER", "UTILITY"};
    for (int i = 0; i < scheduler->num_streams; i++) {
        struct llama_cuda_stream * s = &scheduler->streams[i];
        fprintf(stderr, "  %s: %lu ops, %.2f ms total\n",
                stream_names[i], s->operations_count, s->total_time_ns / 1e6);
    }
}

/**
 * Print token queue status.
 */
void llama_stream_scheduler_print_queue(
    struct llama_stream_scheduler * scheduler) {

    if (!scheduler) {
        fprintf(stderr, "STREAM_SCHEDULER: Null scheduler\n");
        return;
    }

    const char * state_names[] = {
        "UNSCHEDULED", "CPU_PENDING", "CPU_EXECUTING", "CPU_COMPLETE",
        "TRANSFER_PENDING", "TRANSFER_EXECUTING", "GPU_PENDING",
        "GPU_EXECUTING", "GPU_COMPLETE", "ERROR"
    };

    fprintf(stderr, "\n");
    fprintf(stderr, "=== TOKEN QUEUE ===\n");
    fprintf(stderr, "Queue size: %d/%d\n", scheduler->queue_size, scheduler->queue_capacity);

    for (int i = 0; i < scheduler->queue_size; i++) {
        int idx = (scheduler->queue_head + i) % scheduler->queue_capacity;
        struct llama_token_schedule * ts = &scheduler->token_queue[idx];
        fprintf(stderr, "  [%d] Token %d: %s\n", i, ts->token_id,
                state_names[ts->state + 1]);  // +1 to skip ERROR at index 9
    }
    fprintf(stderr, "\n");
}
