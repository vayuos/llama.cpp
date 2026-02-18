/**
 * SECTION 36: Enforce GPU-Only Tensor Metadata During Decode
 * Header
 *
 * This file implements GPU-exclusive tensor metadata enforcement. All decode-critical
 * tensor metadata (shape, strides, type, buffer location) remains GPU-resident and
 * immutable during decode. No CPU tensor introspection, shape queries, or metadata
 * modifications permitted during decode phase.
 *
 * Rules:
 * - No CPU read of tensor shape/dims during decode
 * - No CPU read of tensor strides during decode
 * - No CPU read of tensor data type during decode
 * - No CPU buffer pointer queries during decode
 * - All tensor metadata queries result in hard failure in decode
 * - Metadata immutable lock at decode entry
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TENSOR METADATA ACCESS PHASE ENUMERATION
// ============================================================================

/**
 * Execution phases and their tensor metadata access policies
 */
enum llama_metadata_phase {
    LLAMA_METADATA_PHASE_NONE = 0,
    LLAMA_METADATA_PHASE_MODEL_LOAD = 1,      // CPU metadata access allowed (model weights)
    LLAMA_METADATA_PHASE_CONTEXT_INIT = 2,    // CPU metadata access allowed (context setup)
    LLAMA_METADATA_PHASE_PREFILL = 3,         // CPU metadata access controlled (debug allowed)
    LLAMA_METADATA_PHASE_DECODE = 4,          // CPU metadata access FORBIDDEN
    LLAMA_METADATA_PHASE_COMPLETE = 5,        // CPU metadata access allowed (cleanup)
};

// ============================================================================
// TENSOR METADATA STATE ENUMERATION
// ============================================================================

/**
 * Immutability state of tensor metadata
 */
enum llama_gpu_tensor_metadata_state {
    LLAMA_GPU_TENSOR_METADATA_UNINITIALIZED = 0,
    LLAMA_GPU_TENSOR_METADATA_UNLOCKED = 1,        // Metadata mutable
    LLAMA_GPU_TENSOR_METADATA_LOCKED = 2,          // Metadata immutable
    LLAMA_GPU_TENSOR_METADATA_DECODE_ENFORCED = 3, // Decode phase active
    LLAMA_GPU_TENSOR_METADATA_COMPLETE = 4,        // Cleanup complete
    LLAMA_GPU_TENSOR_METADATA_ERROR = 5,
};

// ============================================================================
// TENSOR METADATA VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-only tensor metadata enforcement
 */
enum llama_tensor_metadata_violation {
    LLAMA_TENSOR_METADATA_VIOLATION_NONE = 0,
    LLAMA_TENSOR_METADATA_VIOLATION_CPU_SHAPE_READ = 1,           // CPU reads tensor shape/dims
    LLAMA_TENSOR_METADATA_VIOLATION_CPU_STRIDE_READ = 2,          // CPU reads strides
    LLAMA_TENSOR_METADATA_VIOLATION_CPU_TYPE_READ = 3,            // CPU reads data type
    LLAMA_TENSOR_METADATA_VIOLATION_CPU_BUFFER_QUERY = 4,         // CPU reads buffer pointer
    LLAMA_TENSOR_METADATA_VIOLATION_CPU_METADATA_WRITE = 5,       // CPU modifies metadata
    LLAMA_TENSOR_METADATA_VIOLATION_TYPE_CONVERSION = 6,          // Tensor type changes in decode
    LLAMA_TENSOR_METADATA_VIOLATION_SHAPE_CHANGE = 7,             // Tensor shape changes in decode
    LLAMA_TENSOR_METADATA_VIOLATION_BUFFER_REALLOC = 8,           // Tensor buffer reallocated in decode
};

// ============================================================================
// TENSOR METADATA LOCKING STATE ENUMERATION
// ============================================================================

/**
 * Lock status of individual tensor metadata
 */
enum llama_tensor_metadata_lock_status {
    LLAMA_TENSOR_METADATA_LOCK_NONE = 0,
    LLAMA_TENSOR_METADATA_LOCK_REQUESTED = 1,        // Lock requested
    LLAMA_TENSOR_METADATA_LOCK_ACTIVE = 2,           // Locked during decode
    LLAMA_TENSOR_METADATA_LOCK_IMMUTABLE = 3,        // Permanently immutable
    LLAMA_TENSOR_METADATA_LOCK_ERROR = 4,
};

// ============================================================================
// TENSOR METADATA QUERY TYPE ENUMERATION
// ============================================================================

/**
 * Types of tensor metadata queries attempted
 */
enum llama_tensor_metadata_query_type {
    LLAMA_TENSOR_METADATA_QUERY_NONE = 0,
    LLAMA_TENSOR_METADATA_QUERY_SHAPE = 1,           // Query shape/dims
    LLAMA_TENSOR_METADATA_QUERY_STRIDES = 2,         // Query strides
    LLAMA_TENSOR_METADATA_QUERY_TYPE = 3,            // Query data type
    LLAMA_TENSOR_METADATA_QUERY_BUFFER = 4,          // Query buffer pointer
    LLAMA_TENSOR_METADATA_QUERY_SIZE = 5,            // Query total size
    LLAMA_TENSOR_METADATA_QUERY_BACKEND = 6,         // Query backend location
};

// ============================================================================
// TENSOR METADATA CONFIGURATION
// ============================================================================

/**
 * Configuration for metadata enforcement
 */
struct llama_gpu_tensor_metadata_config {
    bool forbid_cpu_shape_read;          // Forbid shape queries in decode?
    bool forbid_cpu_stride_read;         // Forbid stride queries in decode?
    bool forbid_cpu_type_read;           // Forbid type queries in decode?
    bool forbid_cpu_buffer_query;        // Forbid buffer pointer queries in decode?
    bool enforce_metadata_immutability;  // Enforce metadata immutable during decode?
    bool debug_metadata_tracking;        // Debug output?
};

// ============================================================================
// TENSOR METADATA RECORD
// ============================================================================

/**
 * Records tensor metadata snapshot
 */
struct llama_tensor_metadata_record {
    uint64_t tensor_id;                  // Unique tensor identifier
    uint32_t ndims;                      // Number of dimensions
    uint64_t ne[8];                      // Dimensions (max 8D)
    uint32_t nb[8];                      // Strides
    uint32_t data_type;                  // Data type enum
    uint64_t buffer_address;             // GPU buffer address
    uint64_t total_size_bytes;           // Total size in bytes
    bool is_on_gpu;                      // GPU resident?
    bool is_locked;                      // Metadata locked?
};

// ============================================================================
// TENSOR METADATA QUERY RECORD
// ============================================================================

/**
 * Records tensor metadata query attempt
 */
struct llama_tensor_metadata_query_record {
    uint64_t tensor_id;                  // Tensor queried
    enum llama_tensor_metadata_query_type query_type; // Query type
    enum llama_metadata_phase phase;     // Phase when queried
    uint64_t timestamp_ns;               // When queried
    bool was_blocked;                    // Was query blocked?
    enum llama_tensor_metadata_violation violation;  // Violation type if blocked
};

// ============================================================================
// TENSOR METADATA IMMUTABILITY RECORD
// ============================================================================

/**
 * Records metadata immutability state
 */
struct llama_tensor_metadata_immutability_record {
    uint64_t tensor_id;                  // Tensor
    enum llama_tensor_metadata_lock_status lock_status; // Lock status
    bool shape_locked;                   // Shape immutable?
    bool type_locked;                    // Type immutable?
    bool stride_locked;                  // Strides immutable?
    bool buffer_locked;                  // Buffer immutable?
    uint64_t lock_timestamp_ns;          // When locked
};

// ============================================================================
// TENSOR METADATA STATE RECORD
// ============================================================================

/**
 * Current state of tensor metadata enforcement
 */
struct llama_gpu_tensor_metadata_state_record {
    enum llama_gpu_tensor_metadata_state state;      // Current state
    enum llama_metadata_phase current_phase;         // Current phase
    uint64_t total_tensors_tracked;                  // Total tensors
    uint64_t total_tensors_locked;                   // Tensors with locked metadata
    uint64_t cpu_shape_queries_blocked;              // Shape queries blocked
    uint64_t cpu_stride_queries_blocked;             // Stride queries blocked
    uint64_t cpu_type_queries_blocked;               // Type queries blocked
    uint64_t cpu_buffer_queries_blocked;             // Buffer queries blocked
    uint64_t metadata_modifications_blocked;         // Metadata writes blocked
    int total_violations;                            // Total violations
    enum llama_tensor_metadata_violation last_violation; // Last violation
};

// ============================================================================
// TENSOR METADATA VALIDATION STATE
// ============================================================================

/**
 * Global state for tensor metadata enforcement
 */
struct llama_gpu_tensor_metadata_validation_state {
    struct llama_gpu_tensor_metadata_config config;
    struct llama_gpu_tensor_metadata_state_record state_record;

    // Per-tensor metadata tracking (std::map<tensor_id, metadata_record>)
    void* tensor_metadata_map;  // opaque pointer to std::map

    // Per-tensor immutability (std::map<tensor_id, immutability_record>)
    void* tensor_locks_map;     // opaque pointer to std::map

    // Query history (std::vector<query_record>)
    void* query_history_vector; // opaque pointer to std::vector

    struct llama_tensor_metadata_query_record last_query_record;
    int total_queries;
    int total_violations;
    bool enforcement_strict;    // Abort on violation vs log only
    bool metadata_locked;       // All metadata locked for decode?
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_tensor_metadata_gpu_init(void);
int llama_tensor_metadata_gpu_configure(
    bool forbid_cpu_shape_read,
    bool forbid_cpu_stride_read,
    bool forbid_cpu_type_read,
    bool forbid_cpu_buffer_query,
    bool enforce_metadata_immutability
);

// Phase management
int llama_tensor_metadata_gpu_set_phase(enum llama_metadata_phase phase);
int llama_tensor_metadata_gpu_begin_decode_phase(void);
int llama_tensor_metadata_gpu_end_decode_phase(void);

// Metadata locking (10 enforcement points: 1-10)
int llama_tensor_metadata_gpu_lock_all_tensor_metadata(void);
int llama_tensor_metadata_gpu_lock_tensor_metadata(uint64_t tensor_id);
int llama_tensor_metadata_gpu_forbid_cpu_shape_read_in_decode(void);
int llama_tensor_metadata_gpu_forbid_cpu_stride_read_in_decode(void);
int llama_tensor_metadata_gpu_forbid_cpu_type_read_in_decode(void);
int llama_tensor_metadata_gpu_forbid_cpu_buffer_query_in_decode(void);
int llama_tensor_metadata_gpu_forbid_metadata_write_in_decode(void);
int llama_tensor_metadata_gpu_forbid_type_conversion_in_decode(void);
int llama_tensor_metadata_gpu_forbid_shape_change_in_decode(void);
int llama_tensor_metadata_gpu_verify_all_metadata_locked(void);

// Violation detection
int llama_tensor_metadata_gpu_detect_cpu_shape_query(uint64_t tensor_id);
int llama_tensor_metadata_gpu_detect_cpu_stride_query(uint64_t tensor_id);
int llama_tensor_metadata_gpu_detect_cpu_type_query(uint64_t tensor_id);
int llama_tensor_metadata_gpu_detect_cpu_buffer_query(uint64_t tensor_id);
int llama_tensor_metadata_gpu_detect_metadata_write(uint64_t tensor_id);
int llama_tensor_metadata_gpu_detect_type_conversion(uint64_t tensor_id);
int llama_tensor_metadata_gpu_detect_shape_change(uint64_t tensor_id);
int llama_tensor_metadata_gpu_detect_buffer_realloc(uint64_t tensor_id);

// Tensor metadata tracking
int llama_tensor_metadata_gpu_track_tensor(
    uint64_t tensor_id,
    uint32_t ndims,
    const uint64_t* ne,
    uint32_t data_type,
    uint64_t buffer_address
);
int llama_tensor_metadata_gpu_record_metadata_snapshot(uint64_t tensor_id);
int llama_tensor_metadata_gpu_verify_metadata_immutable(uint64_t tensor_id);

// Verification functions
int llama_tensor_metadata_gpu_verify_decode_metadata_locked(void);
int llama_tensor_metadata_gpu_verify_no_cpu_metadata_access(void);
int llama_tensor_metadata_gpu_verify_metadata_consistency(void);
int llama_tensor_metadata_gpu_verify_all_queries_blocked(void);
int llama_tensor_metadata_gpu_verify_immutability_complete(void);

// Query interception and blocking
int llama_tensor_metadata_gpu_block_shape_query(uint64_t tensor_id);
int llama_tensor_metadata_gpu_block_stride_query(uint64_t tensor_id);
int llama_tensor_metadata_gpu_block_type_query(uint64_t tensor_id);
int llama_tensor_metadata_gpu_block_buffer_query(uint64_t tensor_id);

// Query functions
struct llama_gpu_tensor_metadata_state_record llama_tensor_metadata_gpu_get_state_record(void);
enum llama_gpu_tensor_metadata_state llama_tensor_metadata_gpu_get_state(void);
enum llama_metadata_phase llama_tensor_metadata_gpu_get_phase(void);
struct llama_tensor_metadata_record llama_tensor_metadata_gpu_get_tensor_metadata(uint64_t tensor_id);
uint64_t llama_tensor_metadata_gpu_get_total_tensors_locked(void);

// Diagnostics and logging
void llama_tensor_metadata_gpu_log_metadata_locking_enabled(void);
void llama_tensor_metadata_gpu_log_decode_phase_metadata_locked(void);
void llama_tensor_metadata_gpu_log_all_metadata_immutable(void);
void llama_tensor_metadata_gpu_print_state(void);
void llama_tensor_metadata_gpu_print_metadata_record(const struct llama_tensor_metadata_record* record);
void llama_tensor_metadata_gpu_print_lock_summary(void);
void llama_tensor_metadata_gpu_print_query_history(void);
void llama_tensor_metadata_gpu_print_violation_summary(void);

// Violation reporting
void llama_tensor_metadata_gpu_report_violation(
    enum llama_tensor_metadata_violation violation_type,
    const char* location,
    const char* details
);

// Enforcement mode control
void llama_tensor_metadata_gpu_set_enforcement_strict(bool strict);
bool llama_tensor_metadata_gpu_get_enforcement_strict(void);
void llama_tensor_metadata_gpu_set_debug_output(bool debug);

// Self-test suite
int llama_tensor_metadata_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_metadata_phase_name(enum llama_metadata_phase phase) {
    switch (phase) {
        case LLAMA_METADATA_PHASE_NONE: return "NONE";
        case LLAMA_METADATA_PHASE_MODEL_LOAD: return "MODEL_LOAD";
        case LLAMA_METADATA_PHASE_CONTEXT_INIT: return "CONTEXT_INIT";
        case LLAMA_METADATA_PHASE_PREFILL: return "PREFILL";
        case LLAMA_METADATA_PHASE_DECODE: return "DECODE";
        case LLAMA_METADATA_PHASE_COMPLETE: return "COMPLETE";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_tensor_metadata_violation_name(enum llama_tensor_metadata_violation violation) {
    switch (violation) {
        case LLAMA_TENSOR_METADATA_VIOLATION_NONE: return "NONE";
        case LLAMA_TENSOR_METADATA_VIOLATION_CPU_SHAPE_READ: return "CPU_SHAPE_READ";
        case LLAMA_TENSOR_METADATA_VIOLATION_CPU_STRIDE_READ: return "CPU_STRIDE_READ";
        case LLAMA_TENSOR_METADATA_VIOLATION_CPU_TYPE_READ: return "CPU_TYPE_READ";
        case LLAMA_TENSOR_METADATA_VIOLATION_CPU_BUFFER_QUERY: return "CPU_BUFFER_QUERY";
        case LLAMA_TENSOR_METADATA_VIOLATION_CPU_METADATA_WRITE: return "CPU_METADATA_WRITE";
        case LLAMA_TENSOR_METADATA_VIOLATION_TYPE_CONVERSION: return "TYPE_CONVERSION";
        case LLAMA_TENSOR_METADATA_VIOLATION_SHAPE_CHANGE: return "SHAPE_CHANGE";
        case LLAMA_TENSOR_METADATA_VIOLATION_BUFFER_REALLOC: return "BUFFER_REALLOC";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_tensor_metadata_lock_status_name(enum llama_tensor_metadata_lock_status status) {
    switch (status) {
        case LLAMA_TENSOR_METADATA_LOCK_NONE: return "NONE";
        case LLAMA_TENSOR_METADATA_LOCK_REQUESTED: return "REQUESTED";
        case LLAMA_TENSOR_METADATA_LOCK_ACTIVE: return "ACTIVE";
        case LLAMA_TENSOR_METADATA_LOCK_IMMUTABLE: return "IMMUTABLE";
        case LLAMA_TENSOR_METADATA_LOCK_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_tensor_metadata_query_type_name(enum llama_tensor_metadata_query_type query_type) {
    switch (query_type) {
        case LLAMA_TENSOR_METADATA_QUERY_NONE: return "NONE";
        case LLAMA_TENSOR_METADATA_QUERY_SHAPE: return "SHAPE";
        case LLAMA_TENSOR_METADATA_QUERY_STRIDES: return "STRIDES";
        case LLAMA_TENSOR_METADATA_QUERY_TYPE: return "TYPE";
        case LLAMA_TENSOR_METADATA_QUERY_BUFFER: return "BUFFER";
        case LLAMA_TENSOR_METADATA_QUERY_SIZE: return "SIZE";
        case LLAMA_TENSOR_METADATA_QUERY_BACKEND: return "BACKEND";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
