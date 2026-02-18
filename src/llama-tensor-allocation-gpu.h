/**
 * SECTION 35: Eliminate Host-Side Tensor Allocation During Decode
 * Header
 *
 * This file implements comprehensive elimination of host-side tensor allocation
 * from the decode-critical path. All decode tensors pre-allocated on GPU before
 * decode begins; no runtime host allocation permitted during decode.
 *
 * Rules:
 * - No ggml_new_tensor() during decode phase
 * - No CPU buffer allocation during decode
 * - No ggml_allocr_alloc() for new tensors in decode
 * - All decode tensors pre-sized and reserved before decode
 * - Phase-aware allocation guards
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TENSOR ALLOCATION PHASE ENUMERATION
// ============================================================================

/**
 * Execution phases and their tensor allocation policies
 */
enum llama_allocation_phase {
    LLAMA_ALLOCATION_PHASE_NONE = 0,
    LLAMA_ALLOCATION_PHASE_MODEL_INIT = 1,     // Host allocation allowed (model weights)
    LLAMA_ALLOCATION_PHASE_CONTEXT_INIT = 2,   // GPU allocation for KV cache
    LLAMA_ALLOCATION_PHASE_PREFILL = 3,        // GPU allocation for prefill buffers
    LLAMA_ALLOCATION_PHASE_DECODE = 4,         // Host allocation FORBIDDEN
    LLAMA_ALLOCATION_PHASE_COMPLETE = 5,       // Cleanup allowed
};

// ============================================================================
// TENSOR ALLOCATION STATE ENUMERATION
// ============================================================================

/**
 * Current state of tensor allocation system
 */
enum llama_gpu_tensor_allocation_state {
    LLAMA_GPU_TENSOR_ALLOCATION_UNINITIALIZED = 0,
    LLAMA_GPU_TENSOR_ALLOCATION_CONFIGURED = 1,
    LLAMA_GPU_TENSOR_ALLOCATION_RESERVED = 2,
    LLAMA_GPU_TENSOR_ALLOCATION_DECODE_LOCKED = 3,
    LLAMA_GPU_TENSOR_ALLOCATION_COMPLETE = 4,
    LLAMA_GPU_TENSOR_ALLOCATION_ERROR = 5,
};

// ============================================================================
// TENSOR ALLOCATION VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of pre-allocation decode invariant
 */
enum llama_tensor_allocation_violation {
    LLAMA_TENSOR_ALLOCATION_VIOLATION_NONE = 0,
    LLAMA_TENSOR_ALLOCATION_VIOLATION_NEW_TENSOR_DECODE = 1,        // ggml_new_tensor in decode
    LLAMA_TENSOR_ALLOCATION_VIOLATION_CPU_ALLOCR_DECODE = 2,        // ggml_allocr_alloc in decode
    LLAMA_TENSOR_ALLOCATION_VIOLATION_HOST_BUFFER_DECODE = 3,       // malloc/host buffer in decode
    LLAMA_TENSOR_ALLOCATION_VIOLATION_BUFFER_RESIZE_DECODE = 4,     // Tensor buffer resize in decode
    LLAMA_TENSOR_ALLOCATION_VIOLATION_POOL_ALLOC_DECODE = 5,        // Memory pool allocation in decode
    LLAMA_TENSOR_ALLOCATION_VIOLATION_EXCESSIVE_ALLOCATION = 6,     // Allocation > pre-reserved size
    LLAMA_TENSOR_ALLOCATION_VIOLATION_OUT_OF_BOUNDS = 7,            // Allocation offset exceeds reserved
    LLAMA_TENSOR_ALLOCATION_VIOLATION_UNKNOWN_TENSOR = 8,           // Unknown tensor accessed in decode
};

// ============================================================================
// TENSOR ALLOCATION TRACKING ENUMERATION
// ============================================================================

/**
 * Tensor allocation ownership and lifecycle
 */
enum llama_tensor_allocation_owner {
    LLAMA_TENSOR_OWNER_NONE = 0,
    LLAMA_TENSOR_OWNER_MODEL = 1,         // Model weights (pre-allocated)
    LLAMA_TENSOR_OWNER_KV_CACHE = 2,      // KV cache (pre-allocated)
    LLAMA_TENSOR_OWNER_PREFILL = 3,       // Prefill workspace (pre-allocated)
    LLAMA_TENSOR_OWNER_DECODE = 4,        // Decode workspace (pre-allocated)
    LLAMA_TENSOR_OWNER_TEMPORARY = 5,     // Temporary buffers (not tracked)
};

// ============================================================================
// TENSOR ALLOCATION PRE-RESERVATION ENUMERATION
// ============================================================================

/**
 * Reservation status for decode-critical tensors
 */
enum llama_tensor_reservation_status {
    LLAMA_TENSOR_RESERVATION_NONE = 0,
    LLAMA_TENSOR_RESERVATION_REQUESTED = 1,        // Marked for pre-allocation
    LLAMA_TENSOR_RESERVATION_GPU_ALLOCATED = 2,    // GPU memory reserved
    LLAMA_TENSOR_RESERVATION_LOCKED = 3,           // Locked; no changes allowed
    LLAMA_TENSOR_RESERVATION_ACTIVE_DECODE = 4,    // In use during decode
    LLAMA_TENSOR_RESERVATION_ERROR = 5,            // Allocation error
};

// ============================================================================
// TENSOR ALLOCATION CONFIGURATION
// ============================================================================

/**
 * Configuration for allocation enforcement
 */
struct llama_gpu_tensor_allocation_config {
    bool forbid_host_allocation;         // Forbid malloc during decode?
    bool forbid_ggml_new_tensor;         // Forbid ggml_new_tensor in decode?
    bool forbid_allocr_alloc;            // Forbid allocr_alloc in decode?
    bool enforce_pre_allocation;         // Require pre-allocated tensors?
    bool strict_size_validation;         // Validate allocation stays within reserved?
    bool debug_allocation_tracking;      // Debug output?
};

// ============================================================================
// TENSOR ALLOCATION RECORD
// ============================================================================

/**
 * Records tensor allocation event
 */
struct llama_tensor_allocation_record {
    enum llama_allocation_phase phase;             // Phase when allocated
    enum llama_tensor_allocation_violation violation; // Violation type (if any)
    enum llama_tensor_allocation_owner owner;      // Which subsystem owns it
    uint64_t tensor_id;                            // Unique tensor identifier
    uint64_t size_bytes;                           // Allocation size
    uint64_t timestamp_ns;                         // When allocated
    bool was_violation;                            // Was it a violation?
};

// ============================================================================
// TENSOR RESERVATION RECORD
// ============================================================================

/**
 * Records tensor reservation and pre-allocation
 */
struct llama_tensor_reservation_record {
    uint64_t tensor_id;                                    // Tensor identifier
    enum llama_tensor_reservation_status status;           // Reservation status
    enum llama_tensor_allocation_owner owner;              // Owner subsystem
    uint64_t reserved_size_bytes;                          // Pre-reserved size
    uint64_t actual_size_bytes;                            // Actual usage
    uint64_t gpu_device_ptr;                               // GPU allocation address
    bool is_locked;                                        // Locked during decode?
};

// ============================================================================
// TENSOR ALLOCATION STATE
// ============================================================================

/**
 * Current state of tensor allocation tracking
 */
struct llama_gpu_tensor_allocation_state_record {
    enum llama_gpu_tensor_allocation_state state;      // Current state
    enum llama_allocation_phase current_phase;         // Current phase
    uint64_t total_host_allocations_decode;            // Host allocs during decode
    uint64_t total_ggml_new_tensor_decode;             // ggml_new_tensor calls in decode
    uint64_t total_allocr_alloc_decode;                // allocr_alloc calls in decode
    uint64_t total_decode_tensors_tracked;             // Total decode tensors
    uint64_t total_decode_tensors_reserved;            // Pre-allocated decode tensors
    uint64_t reserved_gpu_memory_bytes;                // Total GPU pre-reserved
    uint64_t active_decode_tensors;                    // In-use during decode
    int total_violations;                              // Total violations
    enum llama_tensor_allocation_violation last_violation; // Last violation type
};

// ============================================================================
// TENSOR ALLOCATION VALIDATION STATE
// ============================================================================

/**
 * Global state for tensor allocation enforcement
 */
struct llama_gpu_tensor_allocation_validation_state {
    struct llama_gpu_tensor_allocation_config config;
    struct llama_gpu_tensor_allocation_state_record state_record;

    // Per-tensor tracking (std::map<tensor_id, allocation_record>)
    void* tensor_allocations_map;  // opaque pointer to std::map

    // Per-owner tracking (std::map<owner, total_size>)
    void* owner_allocations_map;   // opaque pointer to std::map

    // Tensor reservations (std::map<tensor_id, reservation_record>)
    void* tensor_reservations_map; // opaque pointer to std::map

    // Allocation history (std::vector<allocation_record>)
    void* allocation_history_vector; // opaque pointer to std::vector

    struct llama_tensor_allocation_record last_allocation_record;
    int total_allocation_events;
    int total_violations;
    bool enforcement_strict;           // Abort on violation vs log only
    bool decode_phase_locked;          // Is allocation locked for decode?
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_tensor_allocation_gpu_init(void);
int llama_tensor_allocation_gpu_configure(
    bool forbid_host_allocation,
    bool forbid_ggml_new_tensor,
    bool forbid_allocr_alloc,
    bool enforce_pre_allocation
);

// Phase management
int llama_tensor_allocation_gpu_set_phase(enum llama_allocation_phase phase);
int llama_tensor_allocation_gpu_begin_decode_phase(void);
int llama_tensor_allocation_gpu_end_decode_phase(void);

// Pre-allocation and reservation (10 enforcement points: 1-10)
int llama_tensor_allocation_gpu_reserve_decode_tensors(uint64_t total_size_bytes);
int llama_tensor_allocation_gpu_mark_tensor_reserved(uint64_t tensor_id, uint64_t size_bytes);
int llama_tensor_allocation_gpu_lock_allocations_for_decode(void);
int llama_tensor_allocation_gpu_forbid_host_malloc_in_decode(void);
int llama_tensor_allocation_gpu_forbid_ggml_new_tensor_in_decode(void);
int llama_tensor_allocation_gpu_forbid_allocr_alloc_in_decode(void);
int llama_tensor_allocation_gpu_forbid_buffer_resize_in_decode(void);
int llama_tensor_allocation_gpu_forbid_pool_allocation_in_decode(void);
int llama_tensor_allocation_gpu_verify_all_decode_tensors_reserved(void);
int llama_tensor_allocation_gpu_verify_no_allocation_in_decode(void);

// Violation detection
int llama_tensor_allocation_gpu_detect_new_tensor_decode(void);
int llama_tensor_allocation_gpu_detect_cpu_allocr_decode(void);
int llama_tensor_allocation_gpu_detect_host_buffer_decode(void);
int llama_tensor_allocation_gpu_detect_buffer_resize_decode(void);
int llama_tensor_allocation_gpu_detect_pool_alloc_decode(void);
int llama_tensor_allocation_gpu_detect_excessive_allocation(void);
int llama_tensor_allocation_gpu_detect_out_of_bounds(void);
int llama_tensor_allocation_gpu_detect_unknown_tensor(void);

// Tensor tracking
int llama_tensor_allocation_gpu_track_tensor(
    uint64_t tensor_id,
    uint64_t size_bytes,
    enum llama_tensor_allocation_owner owner
);
int llama_tensor_allocation_gpu_track_allocation(
    uint64_t tensor_id,
    uint64_t size_bytes,
    enum llama_allocation_phase phase
);
int llama_tensor_allocation_gpu_verify_tensor_reserved(uint64_t tensor_id);
int llama_tensor_allocation_gpu_verify_tensor_within_bounds(uint64_t tensor_id, uint64_t size_bytes);

// Verification functions
int llama_tensor_allocation_gpu_verify_decode_phase_locked(void);
int llama_tensor_allocation_gpu_verify_all_tensors_on_gpu(void);
int llama_tensor_allocation_gpu_verify_no_host_allocation_decode(void);
int llama_tensor_allocation_gpu_verify_reservation_consistency(void);
int llama_tensor_allocation_gpu_verify_pre_allocation_complete(void);

// Query functions
struct llama_gpu_tensor_allocation_state_record llama_tensor_allocation_gpu_get_state_record(void);
enum llama_gpu_tensor_allocation_state llama_tensor_allocation_gpu_get_state(void);
enum llama_allocation_phase llama_tensor_allocation_gpu_get_phase(void);
uint64_t llama_tensor_allocation_gpu_get_reserved_memory_bytes(void);
uint64_t llama_tensor_allocation_gpu_get_used_memory_bytes(void);

// Diagnostics and logging
void llama_tensor_allocation_gpu_log_pre_allocation_enabled(void);
void llama_tensor_allocation_gpu_log_decode_phase_locked(void);
void llama_tensor_allocation_gpu_log_all_tensors_reserved(void);
void llama_tensor_allocation_gpu_print_state(void);
void llama_tensor_allocation_gpu_print_allocation_record(const struct llama_tensor_allocation_record* record);
void llama_tensor_allocation_gpu_print_reservation_summary(void);
void llama_tensor_allocation_gpu_print_violation_summary(void);
void llama_tensor_allocation_gpu_print_allocation_stats(void);

// Violation reporting
void llama_tensor_allocation_gpu_report_violation(
    enum llama_tensor_allocation_violation violation_type,
    const char* location,
    const char* details
);

// Enforcement mode control
void llama_tensor_allocation_gpu_set_enforcement_strict(bool strict);
bool llama_tensor_allocation_gpu_get_enforcement_strict(void);
void llama_tensor_allocation_gpu_set_debug_output(bool debug);

// Self-test suite
int llama_tensor_allocation_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_allocation_phase_name(enum llama_allocation_phase phase) {
    switch (phase) {
        case LLAMA_ALLOCATION_PHASE_NONE: return "NONE";
        case LLAMA_ALLOCATION_PHASE_MODEL_INIT: return "MODEL_INIT";
        case LLAMA_ALLOCATION_PHASE_CONTEXT_INIT: return "CONTEXT_INIT";
        case LLAMA_ALLOCATION_PHASE_PREFILL: return "PREFILL";
        case LLAMA_ALLOCATION_PHASE_DECODE: return "DECODE";
        case LLAMA_ALLOCATION_PHASE_COMPLETE: return "COMPLETE";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_tensor_allocation_violation_name(enum llama_tensor_allocation_violation violation) {
    switch (violation) {
        case LLAMA_TENSOR_ALLOCATION_VIOLATION_NONE: return "NONE";
        case LLAMA_TENSOR_ALLOCATION_VIOLATION_NEW_TENSOR_DECODE: return "NEW_TENSOR_IN_DECODE";
        case LLAMA_TENSOR_ALLOCATION_VIOLATION_CPU_ALLOCR_DECODE: return "ALLOCR_ALLOC_IN_DECODE";
        case LLAMA_TENSOR_ALLOCATION_VIOLATION_HOST_BUFFER_DECODE: return "HOST_BUFFER_IN_DECODE";
        case LLAMA_TENSOR_ALLOCATION_VIOLATION_BUFFER_RESIZE_DECODE: return "BUFFER_RESIZE_IN_DECODE";
        case LLAMA_TENSOR_ALLOCATION_VIOLATION_POOL_ALLOC_DECODE: return "POOL_ALLOC_IN_DECODE";
        case LLAMA_TENSOR_ALLOCATION_VIOLATION_EXCESSIVE_ALLOCATION: return "EXCESSIVE_ALLOCATION";
        case LLAMA_TENSOR_ALLOCATION_VIOLATION_OUT_OF_BOUNDS: return "OUT_OF_BOUNDS";
        case LLAMA_TENSOR_ALLOCATION_VIOLATION_UNKNOWN_TENSOR: return "UNKNOWN_TENSOR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_tensor_allocation_owner_name(enum llama_tensor_allocation_owner owner) {
    switch (owner) {
        case LLAMA_TENSOR_OWNER_NONE: return "NONE";
        case LLAMA_TENSOR_OWNER_MODEL: return "MODEL";
        case LLAMA_TENSOR_OWNER_KV_CACHE: return "KV_CACHE";
        case LLAMA_TENSOR_OWNER_PREFILL: return "PREFILL";
        case LLAMA_TENSOR_OWNER_DECODE: return "DECODE";
        case LLAMA_TENSOR_OWNER_TEMPORARY: return "TEMPORARY";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_tensor_reservation_status_name(enum llama_tensor_reservation_status status) {
    switch (status) {
        case LLAMA_TENSOR_RESERVATION_NONE: return "NONE";
        case LLAMA_TENSOR_RESERVATION_REQUESTED: return "REQUESTED";
        case LLAMA_TENSOR_RESERVATION_GPU_ALLOCATED: return "GPU_ALLOCATED";
        case LLAMA_TENSOR_RESERVATION_LOCKED: return "LOCKED";
        case LLAMA_TENSOR_RESERVATION_ACTIVE_DECODE: return "ACTIVE_DECODE";
        case LLAMA_TENSOR_RESERVATION_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
