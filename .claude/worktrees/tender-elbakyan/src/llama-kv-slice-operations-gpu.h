/**
 * SECTION 33: GPU-Exclusive KV-Cache Slice Operations
 * Header
 *
 * This file implements GPU-exclusive KV-cache slicing and view operations.
 * KV cache slicing (extracting tokens, rows, regions) is GPU-resident.
 * CPU does not maintain, validate, or perform KV cache slice operations during decode.
 * All KV slice operations occur inside GPU kernels; CPU observes final sliced state only.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// KV SLICE OPERATION MODE ENUMERATION
// ============================================================================

/**
 * KV cache slice operation modes
 */
enum llama_kv_slice_mode {
    LLAMA_KV_SLICE_NONE = 0,
    LLAMA_KV_SLICE_CPU = 1,           // CPU performs slice ops (deprecated)
    LLAMA_KV_SLICE_GPU = 2,           // GPU performs slice ops
};

// ============================================================================
// GPU KV SLICE STATE ENUMERATION
// ============================================================================

/**
 * State of GPU KV slice operations during decode
 */
enum llama_gpu_kv_slice_state {
    LLAMA_GPU_KV_SLICE_UNINITIALIZED = 0,
    LLAMA_GPU_KV_SLICE_ALLOCATED = 1,      // GPU slice buffers allocated
    LLAMA_GPU_KV_SLICE_INITIALIZED = 2,    // Slice operations initialized
    LLAMA_GPU_KV_SLICE_DECODE_ACTIVE = 3,  // Active during decode
    LLAMA_GPU_KV_SLICE_EXECUTED = 4,       // Slice operation executed on GPU
    LLAMA_GPU_KV_SLICE_STORED = 5,         // Sliced result stored on GPU
    LLAMA_GPU_KV_SLICE_SYNCED = 6,         // Synced to CPU (read-only)
    LLAMA_GPU_KV_SLICE_ERROR = 7,
};

// ============================================================================
// CPU KV SLICE BYPASS ENUMERATION
// ============================================================================

/**
 * CPU KV slice operations that must be bypassed
 */
enum llama_cpu_kv_slice_bypass {
    LLAMA_CPU_KV_SLICE_BYPASS_NONE = 0,
    LLAMA_CPU_KV_SLICE_BYPASS_ROW_SELECT = 1,  // Skip CPU row selection
    LLAMA_CPU_KV_SLICE_BYPASS_RANGE_EXTRACT = 2, // Skip CPU range extraction
    LLAMA_CPU_KV_SLICE_BYPASS_VIEW_CREATE = 3,  // Skip CPU view creation
    LLAMA_CPU_KV_SLICE_BYPASS_VALIDATION = 4,   // Skip CPU validation
};

// ============================================================================
// KV SLICE VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-exclusive KV slice operations
 */
enum llama_kv_slice_violation {
    LLAMA_KV_SLICE_VIOLATION_NONE = 0,
    LLAMA_KV_SLICE_VIOLATION_CPU_ROW_SELECT = 1,    // CPU selected rows
    LLAMA_KV_SLICE_VIOLATION_CPU_RANGE_EXTRACT = 2, // CPU extracted range
    LLAMA_KV_SLICE_VIOLATION_CPU_VIEW_CREATE = 3,   // CPU created view
    LLAMA_KV_SLICE_VIOLATION_SLICE_ON_HOST = 4,     // Slice materialized on host
    LLAMA_KV_SLICE_VIOLATION_MIXED_OPERATION = 5,   // Mixed CPU/GPU operations
    LLAMA_KV_SLICE_VIOLATION_DESYNC = 6,            // CPU/GPU slice desync
    LLAMA_KV_SLICE_VIOLATION_INVALID_BOUNDS = 7,    // Invalid slice bounds
};

// ============================================================================
// KV SLICE OPERATION RECORD
// ============================================================================

/**
 * Describes a single KV cache slice operation
 */
struct llama_gpu_kv_slice_operation {
    uint32_t source_start;             // Start index in source KV cache
    uint32_t source_end;               // End index in source KV cache
    uint32_t num_tokens;               // Number of tokens sliced
    uint32_t num_layers;               // Layers in slice
    uint32_t operation_type;           // Type of slice (row/range/view)
    uint32_t reserved_1;
    uint32_t reserved_2;
    uint32_t reserved_3;
};

// ============================================================================
// GPU KV SLICE CONFIGURATION
// ============================================================================

/**
 * Configuration for GPU KV slice operations
 */
struct llama_gpu_kv_slice_config {
    bool gpu_kv_slice_enabled;         // Enable GPU slice ops?
    bool cpu_slice_operations_forbidden; // Forbid CPU slice ops?
    enum llama_kv_slice_mode mode;     // KV slice operation mode
    uint32_t max_slice_size;           // Maximum slice size
    uint32_t num_layers;               // Number of KV cache layers
    bool validate_slice_bounds;        // Check bounds on each operation?
    bool enforce_gpu_only_slicing;     // Enforce GPU-only slicing?
};

// ============================================================================
// GPU KV SLICE STATE RECORD
// ============================================================================

/**
 * Current state of GPU KV slice operations
 */
struct llama_gpu_kv_slice_state_record {
    enum llama_kv_slice_mode current_mode;         // Current mode
    enum llama_gpu_kv_slice_state slice_state;     // GPU slice state
    uint32_t max_slice_size;                       // Maximum slice size
    uint64_t total_slice_operations;               // GPU kernel slice ops
    uint64_t total_tokens_sliced;                  // Total tokens processed
    int total_violations;                          // Total violations
    enum llama_kv_slice_violation last_violation;  // Last violation
};

// ============================================================================
// KV SLICE EXECUTION RECORD
// ============================================================================

/**
 * Record of a GPU KV slice execution
 */
struct llama_gpu_kv_slice_execution_record {
    struct llama_gpu_kv_slice_operation operation;
    uint64_t execution_time_ns;        // Execution time in nanoseconds
    bool execution_on_gpu;             // Was execution on GPU?
};

// ============================================================================
// KV SLICE VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU KV slice operations
 */
struct llama_gpu_kv_slice_validation_state {
    struct llama_gpu_kv_slice_config config;
    struct llama_gpu_kv_slice_state_record state_record;
    struct llama_gpu_kv_slice_execution_record last_execution;
    int total_slice_executions;
    int total_violations;
    bool enforcement_strict;           // Abort on violation vs log only
    bool debug_kv_slice;               // Debug output
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_kv_slice_gpu_init(void);
int llama_kv_slice_gpu_configure(
    bool gpu_slice_enabled,
    bool cpu_operations_forbidden,
    uint32_t max_slice_size,
    uint32_t num_layers
);

// KV slice setup
int llama_kv_slice_gpu_allocate_slice_buffers(uint32_t max_slice_size);
int llama_kv_slice_gpu_initialize_slice_operations(void);

// GPU KV slice operations (10 enforcement points: 1-10)
int llama_kv_slice_gpu_queue_slice_kernel(void);
int llama_kv_slice_gpu_keep_slice_on_device(void);
int llama_kv_slice_gpu_select_kv_rows_on_gpu(uint32_t start, uint32_t end);
int llama_kv_slice_gpu_extract_kv_range_on_gpu(uint32_t start, uint32_t end);
int llama_kv_slice_gpu_forbid_cpu_row_select(void);
int llama_kv_slice_gpu_forbid_cpu_range_extract(void);
int llama_kv_slice_gpu_forbid_cpu_view_create(void);
int llama_kv_slice_gpu_validate_slice_bounds(void);
int llama_kv_slice_gpu_lock_slice_to_gpu(void);
int llama_kv_slice_gpu_verify_no_cpu_modification(void);

// Slice retrieval and synchronization
int llama_kv_slice_gpu_read_slice_sync(uint32_t* out_slice_size);
int llama_kv_slice_gpu_read_slice_async(uint32_t* out_slice_size);
int llama_kv_slice_gpu_sync_slice_to_cpu(void);

// Violation detection
int llama_kv_slice_gpu_detect_cpu_row_select(void);
int llama_kv_slice_gpu_detect_cpu_range_extract(void);
int llama_kv_slice_gpu_detect_cpu_view_create(void);
int llama_kv_slice_gpu_detect_slice_on_host(void);
int llama_kv_slice_gpu_detect_mixed_operations(void);
int llama_kv_slice_gpu_detect_desync(void);
int llama_kv_slice_gpu_detect_invalid_bounds(void);

// State management
int llama_kv_slice_gpu_set_allocated(void);
int llama_kv_slice_gpu_set_initialized(void);
int llama_kv_slice_gpu_set_decode_active(void);
int llama_kv_slice_gpu_set_executed(void);
int llama_kv_slice_gpu_set_stored(void);

// Query and verification functions
struct llama_gpu_kv_slice_state_record llama_kv_slice_gpu_get_state_record(void);
struct llama_gpu_kv_slice_execution_record llama_kv_slice_gpu_get_last_execution(void);
enum llama_gpu_kv_slice_state llama_kv_slice_gpu_get_slice_state(void);

// Verification functions
int llama_kv_slice_gpu_verify_cpu_operations_forbidden(void);
int llama_kv_slice_gpu_verify_gpu_slice_active(void);
int llama_kv_slice_gpu_verify_slice_locked(void);
int llama_kv_slice_gpu_verify_no_cpu_entry_point(void);
int llama_kv_slice_gpu_verify_slice_within_bounds(void);
int llama_kv_slice_gpu_verify_no_desync(void);
int llama_kv_slice_gpu_verify_valid_bounds(void);

// Diagnostics and logging
void llama_kv_slice_gpu_log_slice_mode_enabled(void);
void llama_kv_slice_gpu_log_slice_locked(void);
void llama_kv_slice_gpu_print_state(void);
void llama_kv_slice_gpu_print_execution_stats(void);
void llama_kv_slice_gpu_print_violation_summary(void);

// Violation reporting
void llama_kv_slice_gpu_report_violation(
    enum llama_kv_slice_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_kv_slice_gpu_set_enforcement_strict(bool strict);
bool llama_kv_slice_gpu_get_enforcement_strict(void);
void llama_kv_slice_gpu_set_debug_output(bool debug);

// Self-test suite
int llama_kv_slice_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_kv_slice_mode_name(
    enum llama_kv_slice_mode mode
) {
    switch (mode) {
        case LLAMA_KV_SLICE_NONE: return "NONE";
        case LLAMA_KV_SLICE_CPU: return "CPU";
        case LLAMA_KV_SLICE_GPU: return "GPU";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_kv_slice_state_name(
    enum llama_gpu_kv_slice_state state
) {
    switch (state) {
        case LLAMA_GPU_KV_SLICE_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_KV_SLICE_ALLOCATED: return "ALLOCATED";
        case LLAMA_GPU_KV_SLICE_INITIALIZED: return "INITIALIZED";
        case LLAMA_GPU_KV_SLICE_DECODE_ACTIVE: return "DECODE_ACTIVE";
        case LLAMA_GPU_KV_SLICE_EXECUTED: return "EXECUTED";
        case LLAMA_GPU_KV_SLICE_STORED: return "STORED";
        case LLAMA_GPU_KV_SLICE_SYNCED: return "SYNCED";
        case LLAMA_GPU_KV_SLICE_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_kv_slice_violation_name(
    enum llama_kv_slice_violation violation
) {
    switch (violation) {
        case LLAMA_KV_SLICE_VIOLATION_NONE: return "NONE";
        case LLAMA_KV_SLICE_VIOLATION_CPU_ROW_SELECT: return "CPU_ROW_SELECT";
        case LLAMA_KV_SLICE_VIOLATION_CPU_RANGE_EXTRACT: return "CPU_RANGE_EXTRACT";
        case LLAMA_KV_SLICE_VIOLATION_CPU_VIEW_CREATE: return "CPU_VIEW_CREATE";
        case LLAMA_KV_SLICE_VIOLATION_SLICE_ON_HOST: return "SLICE_ON_HOST";
        case LLAMA_KV_SLICE_VIOLATION_MIXED_OPERATION: return "MIXED_OPERATION";
        case LLAMA_KV_SLICE_VIOLATION_DESYNC: return "DESYNC";
        case LLAMA_KV_SLICE_VIOLATION_INVALID_BOUNDS: return "INVALID_BOUNDS";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif

