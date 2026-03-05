/**
 * SECTION 34: Enforce Stream-Ordered GPU Execution
 * Header
 *
 * This file implements strict single-stream decode execution enforcement.
 * All decode-critical GPU operations execute within single dedicated CUDA stream.
 * Relies exclusively on stream ordering for correctness guarantees.
 *
 * Rules:
 * - Exactly one decode stream per active sequence
 * - No default stream usage
 * - No implicit stream mixing
 * - No per-layer stream switching
 * - All kernels explicitly bound to decode stream
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// STREAM EXECUTION MODE ENUMERATION
// ============================================================================

/**
 * CUDA stream execution modes
 */
enum llama_stream_execution_mode {
    LLAMA_STREAM_EXECUTION_NONE = 0,
    LLAMA_STREAM_EXECUTION_DEFAULT = 1,      // Default stream (deprecated)
    LLAMA_STREAM_EXECUTION_SINGLE = 2,       // Single dedicated stream
    LLAMA_STREAM_EXECUTION_MULTI = 3,        // Multiple streams (forbidden in decode)
};

// ============================================================================
// STREAM ORDERING STATE ENUMERATION
// ============================================================================

/**
 * Current state of stream-ordered execution
 */
enum llama_gpu_stream_ordering_state {
    LLAMA_GPU_STREAM_ORDERING_UNINITIALIZED = 0,
    LLAMA_GPU_STREAM_ORDERING_STREAM_CREATED = 1,
    LLAMA_GPU_STREAM_ORDERING_DECODE_ACTIVE = 2,
    LLAMA_GPU_STREAM_ORDERING_ENFORCED = 3,
    LLAMA_GPU_STREAM_ORDERING_COMPLETE = 4,
    LLAMA_GPU_STREAM_ORDERING_ERROR = 5,
};

// ============================================================================
// STREAM ORDERING VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of single-stream decode execution
 */
enum llama_stream_ordering_violation {
    LLAMA_STREAM_ORDERING_VIOLATION_NONE = 0,
    LLAMA_STREAM_ORDERING_VIOLATION_MULTIPLE_STREAMS = 1,    // Multiple streams detected
    LLAMA_STREAM_ORDERING_VIOLATION_DEFAULT_STREAM = 2,      // Default stream used
    LLAMA_STREAM_ORDERING_VIOLATION_STREAM_DIVERGENCE = 3,   // Stream mismatch
    LLAMA_STREAM_ORDERING_VIOLATION_IMPLICIT_STREAM_MIX = 4, // Implicit stream mixing
    LLAMA_STREAM_ORDERING_VIOLATION_CROSS_STREAM_SYNC = 5,   // Cross-stream sync
    LLAMA_STREAM_ORDERING_VIOLATION_NO_STREAM_BINDING = 6,   // Kernel not bound to stream
    LLAMA_STREAM_ORDERING_VIOLATION_BLOCKED_MEMCPY = 7,      // Blocking cudaMemcpy
    LLAMA_STREAM_ORDERING_VIOLATION_STREAM_SWITCH = 8,       // Per-layer stream switch
};

// ============================================================================
// KERNEL EXECUTION RECORD
// ============================================================================

/**
 * Records execution of a single kernel
 */
struct llama_gpu_kernel_execution_record {
    uint64_t kernel_id;                      // Unique kernel ID
    uint64_t stream_id;                      // Stream ID kernel executed on
    bool explicit_stream_binding;            // Was stream explicitly bound?
    uint64_t issue_order_timestamp;          // When kernel was issued
    uint32_t reserved;
};

// ============================================================================
// STREAM STATE TRACKING
// ============================================================================

/**
 * Tracks state of a single CUDA stream
 */
struct llama_gpu_decode_stream_state {
    uint64_t stream_id;                      // Stream identifier
    bool is_dedicated_decode_stream;         // Is this the decode stream?
    bool is_active;                          // Stream currently active?
    uint64_t num_kernels_launched;           // Total kernels on stream
    uint64_t num_async_memcpy_ops;           // Async memcpy operations
    bool stream_ordered_guaranteed;          // Can guarantee ordering?
};

// ============================================================================
// STREAM ORDERING CONFIGURATION
// ============================================================================

/**
 * Configuration for stream-ordered execution enforcement
 */
struct llama_gpu_stream_ordering_config {
    bool enforce_single_stream;              // Enforce single stream?
    bool forbid_default_stream;              // Forbid default stream?
    bool forbid_cross_stream_sync;           // Forbid cross-stream sync?
    bool validate_stream_binding;            // Validate stream binding?
    bool forbid_stream_switching;            // Forbid per-layer stream switch?
    bool debug_stream_ordering;              // Debug output?
};

// ============================================================================
// STREAM ORDERING STATE RECORD
// ============================================================================

/**
 * Current state of stream-ordered execution
 */
struct llama_gpu_stream_ordering_state_record {
    enum llama_gpu_stream_ordering_state state;         // Current state
    enum llama_stream_execution_mode execution_mode;    // Execution mode
    uint64_t active_decode_stream_id;                   // Active decode stream ID
    uint64_t num_streams_active;                        // Number of active streams
    uint64_t num_kernels_in_decode_stream;              // Kernels in decode stream
    uint64_t total_kernels_during_decode;               // Total kernels launched
    uint64_t kernels_on_wrong_stream;                   // Kernels on wrong stream
    int total_violations;                               // Total violations
    enum llama_stream_ordering_violation last_violation; // Last violation
};

// ============================================================================
// STREAM ORDERING VALIDATION STATE
// ============================================================================

/**
 * Global validation state for stream-ordered execution
 */
struct llama_gpu_stream_ordering_validation_state {
    struct llama_gpu_stream_ordering_config config;
    struct llama_gpu_stream_ordering_state_record state_record;
    struct llama_gpu_decode_stream_state decode_stream_state;
    struct llama_gpu_kernel_execution_record last_kernel_execution;
    int total_kernel_launches;
    int total_violations;
    bool enforcement_strict;                // Abort on violation vs log only
    bool decode_phase_active;               // Is decode phase active?
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_stream_ordering_gpu_init(void);
int llama_stream_ordering_gpu_configure(
    bool enforce_single_stream,
    bool forbid_default_stream,
    bool forbid_cross_stream_sync,
    bool validate_stream_binding
);

// Decode stream management
int llama_stream_ordering_gpu_create_dedicated_decode_stream(uint64_t* out_stream_id);
int llama_stream_ordering_gpu_get_decode_stream_id(uint64_t* out_stream_id);
int llama_stream_ordering_gpu_mark_stream_immutable(uint64_t stream_id);

// Decode phase management
int llama_stream_ordering_gpu_begin_decode_phase(uint64_t decode_stream_id);
int llama_stream_ordering_gpu_end_decode_phase(void);

// Kernel launch validation (10 enforcement points: 1-10)
int llama_stream_ordering_gpu_validate_single_stream_only(void);
int llama_stream_ordering_gpu_record_kernel_launch(uint64_t kernel_id, uint64_t stream_id);
int llama_stream_ordering_gpu_forbid_default_stream_usage(void);
int llama_stream_ordering_gpu_forbid_stream_divergence(void);
int llama_stream_ordering_gpu_forbid_implicit_stream_mixing(void);
int llama_stream_ordering_gpu_forbid_cross_stream_synchronization(void);
int llama_stream_ordering_gpu_verify_stream_binding_explicit(uint64_t kernel_stream_id);
int llama_stream_ordering_gpu_forbid_per_layer_stream_switching(uint32_t layer_id, uint64_t stream_id);
int llama_stream_ordering_gpu_forbid_blocked_memory_operations(void);
int llama_stream_ordering_gpu_verify_stream_ordered_execution_active(void);

// Violation detection
int llama_stream_ordering_gpu_detect_multiple_streams(uint64_t stream_id);
int llama_stream_ordering_gpu_detect_default_stream_usage(void);
int llama_stream_ordering_gpu_detect_stream_divergence(uint64_t expected_stream, uint64_t actual_stream);
int llama_stream_ordering_gpu_detect_implicit_stream_mix(void);
int llama_stream_ordering_gpu_detect_cross_stream_sync(void);
int llama_stream_ordering_gpu_detect_unbound_kernel(void);
int llama_stream_ordering_gpu_detect_blocked_memcpy(void);
int llama_stream_ordering_gpu_detect_stream_switch(uint32_t layer_id);

// Stream state queries
int llama_stream_ordering_gpu_get_num_active_streams(uint64_t* out_count);
int llama_stream_ordering_gpu_get_kernels_on_decode_stream(uint64_t* out_count);
int llama_stream_ordering_gpu_verify_all_kernels_on_decode_stream(void);

// Verification functions
int llama_stream_ordering_gpu_verify_single_stream_decode_active(void);
int llama_stream_ordering_gpu_verify_no_default_stream_usage(void);
int llama_stream_ordering_gpu_verify_no_stream_divergence(void);
int llama_stream_ordering_gpu_verify_stream_binding_complete(void);
int llama_stream_ordering_gpu_verify_no_cross_stream_dependencies(void);
int llama_stream_ordering_gpu_verify_implicit_ordering_guarantee(void);

// Memory operation validation
int llama_stream_ordering_gpu_validate_async_memcpy_binding(uint64_t stream_id);
int llama_stream_ordering_gpu_forbid_blocking_memcpy_in_decode(void);

// Per-layer stream tracking
int llama_stream_ordering_gpu_track_layer_stream(uint32_t layer_id, uint64_t stream_id);
int llama_stream_ordering_gpu_verify_layer_stream_consistency(uint32_t layer_id);
int llama_stream_ordering_gpu_forbid_layer_stream_switching(uint32_t layer_id);

// Query and verification functions
struct llama_gpu_stream_ordering_state_record llama_stream_ordering_gpu_get_state_record(void);
struct llama_gpu_decode_stream_state llama_stream_ordering_gpu_get_decode_stream_state(void);
enum llama_gpu_stream_ordering_state llama_stream_ordering_gpu_get_state(void);
uint64_t llama_stream_ordering_gpu_get_active_decode_stream_id(void);

// Diagnostics and logging
void llama_stream_ordering_gpu_log_single_stream_mode_enabled(void);
void llama_stream_ordering_gpu_log_decode_phase_started(void);
void llama_stream_ordering_gpu_log_stream_ordered_active(void);
void llama_stream_ordering_gpu_print_state(void);
void llama_stream_ordering_gpu_print_kernel_execution_trace(void);
void llama_stream_ordering_gpu_print_violation_summary(void);
void llama_stream_ordering_gpu_print_stream_binding_report(void);

// Violation reporting
void llama_stream_ordering_gpu_report_violation(
    enum llama_stream_ordering_violation violation_type,
    uint64_t kernel_id,
    const char* details
);

// Enforcement mode control
void llama_stream_ordering_gpu_set_enforcement_strict(bool strict);
bool llama_stream_ordering_gpu_get_enforcement_strict(void);
void llama_stream_ordering_gpu_set_debug_output(bool debug);

// Self-test suite
int llama_stream_ordering_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_stream_execution_mode_name(enum llama_stream_execution_mode mode) {
    switch (mode) {
        case LLAMA_STREAM_EXECUTION_NONE: return "NONE";
        case LLAMA_STREAM_EXECUTION_DEFAULT: return "DEFAULT";
        case LLAMA_STREAM_EXECUTION_SINGLE: return "SINGLE";
        case LLAMA_STREAM_EXECUTION_MULTI: return "MULTI";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_stream_ordering_violation_name(enum llama_stream_ordering_violation violation) {
    switch (violation) {
        case LLAMA_STREAM_ORDERING_VIOLATION_NONE: return "NONE";
        case LLAMA_STREAM_ORDERING_VIOLATION_MULTIPLE_STREAMS: return "MULTIPLE_STREAMS";
        case LLAMA_STREAM_ORDERING_VIOLATION_DEFAULT_STREAM: return "DEFAULT_STREAM";
        case LLAMA_STREAM_ORDERING_VIOLATION_STREAM_DIVERGENCE: return "STREAM_DIVERGENCE";
        case LLAMA_STREAM_ORDERING_VIOLATION_IMPLICIT_STREAM_MIX: return "IMPLICIT_STREAM_MIX";
        case LLAMA_STREAM_ORDERING_VIOLATION_CROSS_STREAM_SYNC: return "CROSS_STREAM_SYNC";
        case LLAMA_STREAM_ORDERING_VIOLATION_NO_STREAM_BINDING: return "NO_STREAM_BINDING";
        case LLAMA_STREAM_ORDERING_VIOLATION_BLOCKED_MEMCPY: return "BLOCKED_MEMCPY";
        case LLAMA_STREAM_ORDERING_VIOLATION_STREAM_SWITCH: return "STREAM_SWITCH";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif

