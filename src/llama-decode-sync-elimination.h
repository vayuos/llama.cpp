/**
 * SECTION 32: Remove Decode-Path cudaDeviceSynchronize Calls
 * Header
 *
 * This file implements comprehensive elimination of cudaDeviceSynchronize()
 * from the decode-critical path. Replaces global sync with stream-ordered,
 * GPU-driven execution model.
 *
 * Rules:
 * - cudaDeviceSynchronize() forbidden during decode
 * - Single dedicated decode CUDA stream enforced
 * - CUDA events used only for final token ID signaling
 * - No implicit syncs from host access
 * - Phase-aware synchronization guards
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// SYNCHRONIZATION PHASE ENUMERATION
// ============================================================================

/**
 * Execution phases and their synchronization policies
 */
enum llama_sync_phase {
    LLAMA_SYNC_PHASE_NONE = 0,
    LLAMA_SYNC_PHASE_MODEL_LOAD = 1,      // Global sync allowed (model init)
    LLAMA_SYNC_PHASE_CONTEXT_INIT = 2,    // Global sync allowed (context init)
    LLAMA_SYNC_PHASE_PREFILL = 3,         // Global sync controlled (optional)
    LLAMA_SYNC_PHASE_DECODE = 4,          // Global sync FORBIDDEN
    LLAMA_SYNC_PHASE_COMPLETE = 5,        // Global sync allowed (cleanup)
};

// ============================================================================
// SYNCHRONIZATION STATE ENUMERATION
// ============================================================================

/**
 * Current synchronization state during decode
 */
enum llama_gpu_sync_elimination_state {
    LLAMA_GPU_SYNC_ELIMINATION_UNINITIALIZED = 0,
    LLAMA_GPU_SYNC_ELIMINATION_INITIALIZED = 1,
    LLAMA_GPU_SYNC_ELIMINATION_DECODE_ACTIVE = 2,
    LLAMA_GPU_SYNC_ELIMINATION_STREAM_ORDERED = 3,
    LLAMA_GPU_SYNC_ELIMINATION_COMPLETE = 4,
    LLAMA_GPU_SYNC_ELIMINATION_ERROR = 5,
};

// ============================================================================
// SYNCHRONIZATION VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of sync-free decode execution
 */
enum llama_decode_sync_violation {
    LLAMA_DECODE_SYNC_VIOLATION_NONE = 0,
    LLAMA_DECODE_SYNC_VIOLATION_GLOBAL_SYNC_DECODE = 1,     // cudaDeviceSynchronize in decode
    LLAMA_DECODE_SYNC_VIOLATION_IMPLICIT_SYNC = 2,          // Implicit sync from host access
    LLAMA_DECODE_SYNC_VIOLATION_HOST_MEMORY_READ = 3,       // Host read of device memory
    LLAMA_DECODE_SYNC_VIOLATION_HOST_MEMORY_COPY = 4,       // cudaMemcpyDeviceToHost
    LLAMA_DECODE_SYNC_VIOLATION_UNIFIED_MEMORY_ACCESS = 5,  // Unified memory touch
    LLAMA_DECODE_SYNC_VIOLATION_MULTIPLE_STREAMS = 6,       // Multiple decode streams
    LLAMA_DECODE_SYNC_VIOLATION_DEBUG_SYNC_ENABLED = 7,     // Debug sync in decode
    LLAMA_DECODE_SYNC_VIOLATION_PROFILING_SYNC = 8,         // Profiling sync in decode
};

// ============================================================================
// CUDA STREAM MANAGEMENT ENUMERATION
// ============================================================================

/**
 * CUDA stream management modes
 */
enum llama_cuda_stream_mode {
    LLAMA_CUDA_STREAM_NONE = 0,
    LLAMA_CUDA_STREAM_DEFAULT = 1,           // Default CUDA stream (deprecated)
    LLAMA_CUDA_STREAM_DEDICATED = 2,         // Dedicated decode stream
    LLAMA_CUDA_STREAM_MULTI_STREAM = 3,      // Multiple streams (forbidden in decode)
};

// ============================================================================
// SYNC ELIMINATION CONFIGURATION
// ============================================================================

/**
 * Configuration for sync elimination enforcement
 */
struct llama_gpu_sync_elimination_config {
    bool eliminate_global_sync;              // Eliminate global sync?
    bool enforce_single_stream;              // Enforce single decode stream?
    bool forbid_host_access;                 // Forbid host access during decode?
    bool forbid_debug_sync;                  // Forbid debug sync in decode?
    bool use_stream_events_only;             // Use only stream events (not global)?
    bool debug_sync_elimination;             // Debug output?
};

// ============================================================================
// SYNCHRONIZATION RECORD
// ============================================================================

/**
 * Records a synchronization event
 */
struct llama_gpu_sync_record {
    enum llama_sync_phase phase;             // Phase when sync occurred
    enum llama_decode_sync_violation violation; // Violation type (if any)
    uint64_t timestamp_ns;                   // When sync was called
    bool was_global_sync;                    // Was it global sync?
    bool was_violation;                      // Was it a violation?
};

// ============================================================================
// SYNC ELIMINATION STATE RECORD
// ============================================================================

/**
 * Current state of sync elimination
 */
struct llama_gpu_sync_elimination_state_record {
    enum llama_gpu_sync_elimination_state state;     // Current state
    enum llama_cuda_stream_mode stream_mode;         // Stream mode
    enum llama_sync_phase current_phase;             // Current phase
    uint64_t decode_global_syncs;                    // Global syncs during decode
    uint64_t decode_implicit_syncs;                  // Implicit syncs detected
    uint64_t decode_host_access_syncs;               // Host access syncs
    int total_violations;                            // Total violations
    enum llama_decode_sync_violation last_violation; // Last violation
};

// ============================================================================
// CUDA STREAM STATE
// ============================================================================

/**
 * Tracks CUDA stream management
 */
struct llama_gpu_cuda_stream_state {
    bool dedicated_decode_stream_created;   // Dedicated stream created?
    uint64_t decode_stream_id;              // Stream identifier
    uint64_t num_kernels_in_stream;         // Kernels queued in stream
    uint64_t num_stream_events;             // Stream events recorded
    bool all_kernels_in_single_stream;      // All kernels in one stream?
};

// ============================================================================
// SYNC ELIMINATION VALIDATION STATE
// ============================================================================

/**
 * Global validation state for sync elimination
 */
struct llama_gpu_sync_elimination_validation_state {
    struct llama_gpu_sync_elimination_config config;
    struct llama_gpu_sync_elimination_state_record state_record;
    struct llama_gpu_cuda_stream_state stream_state;
    struct llama_gpu_sync_record last_sync_record;
    int total_sync_events;
    int total_violations;
    bool enforcement_strict;                // Abort on violation vs log only
    bool decode_phase_active;               // Is decode phase active?
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_sync_elimination_gpu_init(void);
int llama_sync_elimination_gpu_configure(
    bool eliminate_global_sync,
    bool enforce_single_stream,
    bool forbid_host_access,
    bool forbid_debug_sync
);

// Phase management
int llama_sync_elimination_gpu_set_phase(enum llama_sync_phase phase);
int llama_sync_elimination_gpu_begin_decode_phase(void);
int llama_sync_elimination_gpu_end_decode_phase(void);

// CUDA stream management (10 enforcement points: 1-10)
int llama_sync_elimination_gpu_create_dedicated_decode_stream(void);
int llama_sync_elimination_gpu_queue_kernel_in_decode_stream(void);
int llama_sync_elimination_gpu_verify_single_stream_only(void);
int llama_sync_elimination_gpu_forbid_global_sync_in_decode(void);
int llama_sync_elimination_gpu_forbid_implicit_sync_from_host_access(void);
int llama_sync_elimination_gpu_forbid_host_memory_reads(void);
int llama_sync_elimination_gpu_record_stream_event_for_token(void);
int llama_sync_elimination_gpu_forbid_debug_sync_in_decode(void);
int llama_sync_elimination_gpu_forbid_profiling_sync_in_decode(void);
int llama_sync_elimination_gpu_verify_stream_ordered_execution(void);

// Synchronization interception
int llama_sync_elimination_gpu_detect_global_sync_call(void);
int llama_sync_elimination_gpu_detect_implicit_sync(void);
int llama_sync_elimination_gpu_detect_host_memory_read(void);
int llama_sync_elimination_gpu_detect_host_memory_copy(void);
int llama_sync_elimination_gpu_detect_unified_memory_access(void);
int llama_sync_elimination_gpu_detect_multi_stream_usage(void);
int llama_sync_elimination_gpu_detect_debug_sync(void);
int llama_sync_elimination_gpu_detect_profiling_sync(void);

// Stream event management
int llama_sync_elimination_gpu_record_stream_event(void);
int llama_sync_elimination_gpu_synchronize_on_stream_event_only(void);

// Query and verification functions
struct llama_gpu_sync_elimination_state_record llama_sync_elimination_gpu_get_state_record(void);
struct llama_gpu_cuda_stream_state llama_sync_elimination_gpu_get_stream_state(void);
enum llama_gpu_sync_elimination_state llama_sync_elimination_gpu_get_state(void);
enum llama_sync_phase llama_sync_elimination_gpu_get_phase(void);

// Verification functions
int llama_sync_elimination_gpu_verify_no_global_sync_in_decode(void);
int llama_sync_elimination_gpu_verify_single_stream_decode(void);
int llama_sync_elimination_gpu_verify_no_implicit_syncs(void);
int llama_sync_elimination_gpu_verify_no_host_access(void);
int llama_sync_elimination_gpu_verify_stream_ordered_execution_active(void);

// Phase checking
int llama_sync_elimination_gpu_allow_global_sync_for_phase(enum llama_sync_phase phase);
int llama_sync_elimination_gpu_forbid_global_sync_for_phase(enum llama_sync_phase phase);

// Diagnostics and logging
void llama_sync_elimination_gpu_log_stream_ordered_execution_enabled(void);
void llama_sync_elimination_gpu_log_decode_phase_started(void);
void llama_sync_elimination_gpu_log_single_stream_decode_active(void);
void llama_sync_elimination_gpu_print_state(void);
void llama_sync_elimination_gpu_print_stream_state(void);
void llama_sync_elimination_gpu_print_violation_summary(void);
void llama_sync_elimination_gpu_print_sync_elimination_stats(void);

// Violation reporting
void llama_sync_elimination_gpu_report_violation(
    enum llama_decode_sync_violation violation_type,
    const char* location,
    const char* details
);

// Enforcement mode control
void llama_sync_elimination_gpu_set_enforcement_strict(bool strict);
bool llama_sync_elimination_gpu_get_enforcement_strict(void);
void llama_sync_elimination_gpu_set_debug_output(bool debug);

// Self-test suite
int llama_sync_elimination_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_sync_phase_name(enum llama_sync_phase phase) {
    switch (phase) {
        case LLAMA_SYNC_PHASE_NONE: return "NONE";
        case LLAMA_SYNC_PHASE_MODEL_LOAD: return "MODEL_LOAD";
        case LLAMA_SYNC_PHASE_CONTEXT_INIT: return "CONTEXT_INIT";
        case LLAMA_SYNC_PHASE_PREFILL: return "PREFILL";
        case LLAMA_SYNC_PHASE_DECODE: return "DECODE";
        case LLAMA_SYNC_PHASE_COMPLETE: return "COMPLETE";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_decode_sync_violation_name(enum llama_decode_sync_violation violation) {
    switch (violation) {
        case LLAMA_DECODE_SYNC_VIOLATION_NONE: return "NONE";
        case LLAMA_DECODE_SYNC_VIOLATION_GLOBAL_SYNC_DECODE: return "GLOBAL_SYNC_IN_DECODE";
        case LLAMA_DECODE_SYNC_VIOLATION_IMPLICIT_SYNC: return "IMPLICIT_SYNC";
        case LLAMA_DECODE_SYNC_VIOLATION_HOST_MEMORY_READ: return "HOST_MEMORY_READ";
        case LLAMA_DECODE_SYNC_VIOLATION_HOST_MEMORY_COPY: return "HOST_MEMORY_COPY";
        case LLAMA_DECODE_SYNC_VIOLATION_UNIFIED_MEMORY_ACCESS: return "UNIFIED_MEMORY_ACCESS";
        case LLAMA_DECODE_SYNC_VIOLATION_MULTIPLE_STREAMS: return "MULTIPLE_STREAMS";
        case LLAMA_DECODE_SYNC_VIOLATION_DEBUG_SYNC_ENABLED: return "DEBUG_SYNC_ENABLED";
        case LLAMA_DECODE_SYNC_VIOLATION_PROFILING_SYNC: return "PROFILING_SYNC";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_cuda_stream_mode_name(enum llama_cuda_stream_mode mode) {
    switch (mode) {
        case LLAMA_CUDA_STREAM_NONE: return "NONE";
        case LLAMA_CUDA_STREAM_DEFAULT: return "DEFAULT";
        case LLAMA_CUDA_STREAM_DEDICATED: return "DEDICATED";
        case LLAMA_CUDA_STREAM_MULTI_STREAM: return "MULTI_STREAM";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif

