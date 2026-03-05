/**
 * SECTION 31: Eliminate Host-Side Token Buffering
 * Header
 *
 * This file implements GPU-exclusive token buffer management.
 * Token queues and buffers are GPU-resident; CPU does not maintain token buffers.
 * CPU cannot queue, enqueue, dequeue, or inspect token buffer state during decode.
 * All token buffering operations occur inside GPU kernels; CPU observes final buffer state only.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TOKEN BUFFER MODE ENUMERATION
// ============================================================================

/**
 * Token buffer management modes
 */
enum llama_token_buffer_mode {
    LLAMA_TOKEN_BUFFER_NONE = 0,
    LLAMA_TOKEN_BUFFER_CPU = 1,       // CPU maintains token buffer (deprecated)
    LLAMA_TOKEN_BUFFER_GPU = 2,       // GPU maintains token buffer
};

// ============================================================================
// GPU TOKEN BUFFER STATE ENUMERATION
// ============================================================================

/**
 * State of GPU token buffer during decode
 */
enum llama_gpu_token_buffer_state {
    LLAMA_GPU_TOKEN_BUFFER_UNINITIALIZED = 0,
    LLAMA_GPU_TOKEN_BUFFER_ALLOCATED = 1,      // GPU token buffer allocated
    LLAMA_GPU_TOKEN_BUFFER_INITIALIZED = 2,    // Token buffer initialized
    LLAMA_GPU_TOKEN_BUFFER_DECODE_ACTIVE = 3,  // Active during decode
    LLAMA_GPU_TOKEN_BUFFER_ENQUEUED = 4,       // Token enqueued on GPU
    LLAMA_GPU_TOKEN_BUFFER_DEQUEUED = 5,       // Token dequeued on GPU
    LLAMA_GPU_TOKEN_BUFFER_SYNCED = 6,         // Synced to CPU (read-only)
    LLAMA_GPU_TOKEN_BUFFER_ERROR = 7,
};

// ============================================================================
// CPU TOKEN BUFFER BYPASS ENUMERATION
// ============================================================================

/**
 * CPU token buffer operations that must be bypassed
 */
enum llama_cpu_token_buffer_bypass {
    LLAMA_CPU_TOKEN_BUFFER_BYPASS_NONE = 0,
    LLAMA_CPU_TOKEN_BUFFER_BYPASS_ENQUEUE = 1,        // Skip CPU token enqueue
    LLAMA_CPU_TOKEN_BUFFER_BYPASS_DEQUEUE = 2,        // Skip CPU token dequeue
    LLAMA_CPU_TOKEN_BUFFER_BYPASS_READ = 3,           // Skip CPU buffer read
    LLAMA_CPU_TOKEN_BUFFER_BYPASS_VALIDATION = 4,     // Skip CPU bounds check
};

// ============================================================================
// TOKEN BUFFER VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-exclusive token buffer ownership
 */
enum llama_token_buffer_violation {
    LLAMA_TOKEN_BUFFER_VIOLATION_NONE = 0,
    LLAMA_TOKEN_BUFFER_VIOLATION_CPU_ENQUEUE = 1,      // CPU enqueued token
    LLAMA_TOKEN_BUFFER_VIOLATION_CPU_DEQUEUE = 2,      // CPU dequeued token
    LLAMA_TOKEN_BUFFER_VIOLATION_CPU_READ = 3,         // CPU read buffer
    LLAMA_TOKEN_BUFFER_VIOLATION_CPU_BOUNDS_CHECK = 4, // CPU checked buffer bounds
    LLAMA_TOKEN_BUFFER_VIOLATION_BUFFER_ON_HOST = 5,   // Buffer materialized on host
    LLAMA_TOKEN_BUFFER_VIOLATION_MIXED_UPDATE = 6,     // Mixed CPU/GPU updates
    LLAMA_TOKEN_BUFFER_VIOLATION_DESYNC = 7,           // CPU/GPU buffer desync
};

// ============================================================================
// GPU TOKEN BUFFER RING STRUCTURE
// ============================================================================

/**
 * Ring buffer structure for GPU token queue
 */
struct llama_gpu_token_buffer_ring {
    uint32_t buffer_capacity;         // Max tokens in buffer
    uint32_t write_pos;               // Current write position
    uint32_t read_pos;                // Current read position
    uint32_t token_count;             // Current tokens in buffer
    uint32_t enqueue_count;           // Total enqueued tokens
    uint32_t dequeue_count;           // Total dequeued tokens
    uint32_t reserved_1;
    uint32_t reserved_2;
};

// ============================================================================
// GPU TOKEN BUFFER CONFIGURATION
// ============================================================================

/**
 * Configuration for GPU token buffer management
 */
struct llama_gpu_token_buffer_config {
    bool gpu_token_buffer_enabled;     // Enable GPU token buffer?
    bool cpu_enqueue_forbidden;        // Forbid CPU token enqueue?
    enum llama_token_buffer_mode mode; // Token buffer mode
    uint32_t buffer_capacity;          // Max tokens in buffer
    uint32_t batch_size;               // Tokens per batch
    bool validate_buffer_bounds;       // Check bounds on each operation?
    bool enforce_gpu_only_buffering;   // Enforce GPU-only buffering?
};

// ============================================================================
// GPU TOKEN BUFFER STATE RECORD
// ============================================================================

/**
 * Current state of GPU token buffer management
 */
struct llama_gpu_token_buffer_state_record {
    enum llama_token_buffer_mode current_mode;         // Current buffer mode
    enum llama_gpu_token_buffer_state buffer_state;    // GPU token buffer state
    uint32_t buffer_capacity;                          // Buffer capacity
    uint32_t current_tokens_in_buffer;                 // Tokens currently in buffer
    uint64_t total_enqueue_operations;                 // GPU kernel enqueue ops
    uint64_t total_dequeue_operations;                 // GPU kernel dequeue ops
    int total_violations;                              // Total violations
    enum llama_token_buffer_violation last_violation;  // Last violation
};

// ============================================================================
// GPU TOKEN BUFFER OPERATION RECORD
// ============================================================================

/**
 * Record of a GPU token buffer operation
 */
struct llama_gpu_token_buffer_operation_record {
    uint32_t tokens_before;            // Tokens in buffer before operation
    uint32_t tokens_after;             // Tokens in buffer after operation
    uint32_t operations_count;         // Number of operations
    uint64_t timestamp_ns;             // When operation occurred
    bool operation_on_gpu;             // Was operation on GPU?
};

// ============================================================================
// TOKEN BUFFER VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU token buffer management
 */
struct llama_gpu_token_buffer_validation_state {
    struct llama_gpu_token_buffer_config config;
    struct llama_gpu_token_buffer_state_record state_record;
    struct llama_gpu_token_buffer_operation_record last_operation;
    int total_buffer_operations;
    int total_violations;
    bool enforcement_strict;           // Abort on violation vs log only
    bool debug_token_buffer;           // Debug output
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_token_buffer_gpu_init(void);
int llama_token_buffer_gpu_configure(
    bool gpu_token_buffer_enabled,
    bool cpu_enqueue_forbidden,
    uint32_t buffer_capacity,
    uint32_t batch_size
);

// Token buffer setup
int llama_token_buffer_gpu_allocate_buffer(uint32_t capacity);
int llama_token_buffer_gpu_initialize_buffer(void);

// GPU token buffer operations (10 enforcement points: 1-10)
int llama_token_buffer_gpu_queue_buffer_kernel(void);
int llama_token_buffer_gpu_keep_buffer_on_device(void);
int llama_token_buffer_gpu_enqueue_token_on_gpu(uint32_t token);
int llama_token_buffer_gpu_dequeue_token_on_gpu(uint32_t* out_token);
int llama_token_buffer_gpu_forbid_cpu_enqueue(void);
int llama_token_buffer_gpu_forbid_cpu_dequeue(void);
int llama_token_buffer_gpu_forbid_cpu_buffer_read(void);
int llama_token_buffer_gpu_validate_buffer_bounds(void);
int llama_token_buffer_gpu_lock_buffer_to_gpu(void);
int llama_token_buffer_gpu_verify_no_cpu_modification(void);

// Buffer state and content operations
int llama_token_buffer_gpu_get_buffer_size(uint32_t* out_size);
int llama_token_buffer_gpu_get_token_count(uint32_t* out_count);
int llama_token_buffer_gpu_peek_token(uint32_t* out_token);

// Violation detection
int llama_token_buffer_gpu_detect_cpu_enqueue(void);
int llama_token_buffer_gpu_detect_cpu_dequeue(void);
int llama_token_buffer_gpu_detect_cpu_buffer_read(void);
int llama_token_buffer_gpu_detect_cpu_bounds_check(void);
int llama_token_buffer_gpu_detect_buffer_on_host(void);
int llama_token_buffer_gpu_detect_mixed_updates(void);
int llama_token_buffer_gpu_detect_desync(void);

// State management
int llama_token_buffer_gpu_set_allocated(void);
int llama_token_buffer_gpu_set_initialized(void);
int llama_token_buffer_gpu_set_decode_active(void);
int llama_token_buffer_gpu_set_enqueued(void);
int llama_token_buffer_gpu_set_dequeued(void);

// Query and verification functions
struct llama_gpu_token_buffer_state_record llama_token_buffer_gpu_get_state_record(void);
struct llama_gpu_token_buffer_operation_record llama_token_buffer_gpu_get_last_operation(void);
enum llama_gpu_token_buffer_state llama_token_buffer_gpu_get_buffer_state(void);

// Verification functions
int llama_token_buffer_gpu_verify_cpu_enqueue_forbidden(void);
int llama_token_buffer_gpu_verify_gpu_token_buffer_active(void);
int llama_token_buffer_gpu_verify_buffer_locked(void);
int llama_token_buffer_gpu_verify_no_cpu_entry_point(void);
int llama_token_buffer_gpu_verify_buffer_within_bounds(void);
int llama_token_buffer_gpu_verify_no_desync(void);
int llama_token_buffer_gpu_verify_no_host_copy(void);

// Diagnostics and logging
void llama_token_buffer_gpu_log_buffer_mode_enabled(void);
void llama_token_buffer_gpu_log_buffer_locked(void);
void llama_token_buffer_gpu_print_state(void);
void llama_token_buffer_gpu_print_execution_stats(void);
void llama_token_buffer_gpu_print_violation_summary(void);

// Violation reporting
void llama_token_buffer_gpu_report_violation(
    enum llama_token_buffer_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_token_buffer_gpu_set_enforcement_strict(bool strict);
bool llama_token_buffer_gpu_get_enforcement_strict(void);
void llama_token_buffer_gpu_set_debug_output(bool debug);

// Self-test suite
int llama_token_buffer_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_token_buffer_mode_name(
    enum llama_token_buffer_mode mode
) {
    switch (mode) {
        case LLAMA_TOKEN_BUFFER_NONE: return "NONE";
        case LLAMA_TOKEN_BUFFER_CPU: return "CPU";
        case LLAMA_TOKEN_BUFFER_GPU: return "GPU";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_token_buffer_state_name(
    enum llama_gpu_token_buffer_state state
) {
    switch (state) {
        case LLAMA_GPU_TOKEN_BUFFER_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_TOKEN_BUFFER_ALLOCATED: return "ALLOCATED";
        case LLAMA_GPU_TOKEN_BUFFER_INITIALIZED: return "INITIALIZED";
        case LLAMA_GPU_TOKEN_BUFFER_DECODE_ACTIVE: return "DECODE_ACTIVE";
        case LLAMA_GPU_TOKEN_BUFFER_ENQUEUED: return "ENQUEUED";
        case LLAMA_GPU_TOKEN_BUFFER_DEQUEUED: return "DEQUEUED";
        case LLAMA_GPU_TOKEN_BUFFER_SYNCED: return "SYNCED";
        case LLAMA_GPU_TOKEN_BUFFER_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_token_buffer_violation_name(
    enum llama_token_buffer_violation violation
) {
    switch (violation) {
        case LLAMA_TOKEN_BUFFER_VIOLATION_NONE: return "NONE";
        case LLAMA_TOKEN_BUFFER_VIOLATION_CPU_ENQUEUE: return "CPU_ENQUEUE";
        case LLAMA_TOKEN_BUFFER_VIOLATION_CPU_DEQUEUE: return "CPU_DEQUEUE";
        case LLAMA_TOKEN_BUFFER_VIOLATION_CPU_READ: return "CPU_READ";
        case LLAMA_TOKEN_BUFFER_VIOLATION_CPU_BOUNDS_CHECK: return "CPU_BOUNDS_CHECK";
        case LLAMA_TOKEN_BUFFER_VIOLATION_BUFFER_ON_HOST: return "BUFFER_ON_HOST";
        case LLAMA_TOKEN_BUFFER_VIOLATION_MIXED_UPDATE: return "MIXED_UPDATE";
        case LLAMA_TOKEN_BUFFER_VIOLATION_DESYNC: return "DESYNC";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif

