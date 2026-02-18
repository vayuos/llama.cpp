/**
 * SECTION 20: Eliminate CPU tensor metadata updates per token
 * Header
 *
 * This file implements enforcement that CPU tensor metadata mutations are eliminated from decode.
 * All tensor shapes, strides, offsets, and descriptors are frozen before decode begins.
 * CPU cannot update tensor metadata per-token. All per-token variability becomes data-driven on GPU.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CPU TENSOR METADATA MUTATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of CPU tensor metadata mutations (all forbidden during decode)
 */
enum llama_cpu_tensor_metadata_mutation {
    LLAMA_TENSOR_METADATA_MUTATION_NONE = 0,
    LLAMA_TENSOR_METADATA_MUTATION_SHAPE_UPDATE = 1,           // CPU updates tensor shape
    LLAMA_TENSOR_METADATA_MUTATION_STRIDE_UPDATE = 2,          // CPU updates tensor strides
    LLAMA_TENSOR_METADATA_MUTATION_OFFSET_UPDATE = 3,          // CPU updates tensor offset
    LLAMA_TENSOR_METADATA_MUTATION_VIEW_REWIRE = 4,            // CPU rewires tensor view
    LLAMA_TENSOR_METADATA_MUTATION_LAYOUT_CHANGE = 5,          // CPU changes memory layout
    LLAMA_TENSOR_METADATA_MUTATION_RESHAPE = 6,                // CPU reshapes tensor
    LLAMA_TENSOR_METADATA_MUTATION_SLICE = 7,                  // CPU slices tensor per-token
    LLAMA_TENSOR_METADATA_MUTATION_TRANSPOSE = 8,              // CPU transposes tensor
    LLAMA_TENSOR_METADATA_MUTATION_BROADCAST = 9,              // CPU broadcasts tensor
    LLAMA_TENSOR_METADATA_MUTATION_DESCRIPTOR_INIT = 10,       // CPU initializes descriptor
    LLAMA_TENSOR_METADATA_MUTATION_POSITION_ADJUST = 11,       // CPU adjusts for position
    LLAMA_TENSOR_METADATA_MUTATION_TYPE_CAST = 12,             // CPU changes tensor type
};

// ============================================================================
// TENSOR METADATA CATEGORY ENUMERATION
// ============================================================================

/**
 * Categories of tensor metadata mutations by scope and impact
 */
enum llama_tensor_metadata_category {
    LLAMA_TENSOR_META_CAT_NONE = 0,
    LLAMA_TENSOR_META_CAT_SHAPE = 1,                   // Shape/dimension changes
    LLAMA_TENSOR_META_CAT_LAYOUT = 2,                  // Memory layout changes
    LLAMA_TENSOR_META_CAT_POSITIONING = 3,             // Position/offset adjustments
    LLAMA_TENSOR_META_CAT_DESCRIPTOR = 4,              // Descriptor creation/modification
    LLAMA_TENSOR_META_CAT_TYPE_INFO = 5,               // Type/precision changes
};

// ============================================================================
// GPU TENSOR METADATA STATE ENUMERATION
// ============================================================================

/**
 * GPU-resident tensor metadata state during decode
 */
enum llama_gpu_tensor_metadata_state {
    LLAMA_GPU_TENSOR_META_UNINITIALIZED = 0,           // Not started
    LLAMA_GPU_TENSOR_META_PREPARED = 1,                // Tensors prepared
    LLAMA_GPU_TENSOR_META_FROZEN = 2,                  // Metadata frozen
    LLAMA_GPU_TENSOR_META_ACTIVE = 3,                  // Actively used
    LLAMA_GPU_TENSOR_META_ERROR = 4,                   // Metadata error
};

// ============================================================================
// TENSOR METADATA VIOLATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of CPU tensor metadata violations
 */
enum llama_tensor_metadata_violation_type {
    LLAMA_TENSOR_META_VIOLATION_NONE = 0,
    LLAMA_TENSOR_META_VIOLATION_CPU_SHAPE_UPDATE = 1,              // CPU updated shape
    LLAMA_TENSOR_META_VIOLATION_CPU_STRIDE_UPDATE = 2,             // CPU updated strides
    LLAMA_TENSOR_META_VIOLATION_CPU_OFFSET_UPDATE = 3,             // CPU updated offset
    LLAMA_TENSOR_META_VIOLATION_CPU_VIEW_REWIRE = 4,               // CPU rewired view
    LLAMA_TENSOR_META_VIOLATION_PER_TOKEN_RESHAPE = 5,             // Per-token reshape
    LLAMA_TENSOR_META_VIOLATION_POSITION_BASED_SLICE = 6,          // Position-based slicing
    LLAMA_TENSOR_META_VIOLATION_DESCRIPTOR_MUTATION = 7,           // Descriptor changed
    LLAMA_TENSOR_META_VIOLATION_LAYOUT_MISMATCH = 8,               // Layout inconsistency
};

// ============================================================================
// TENSOR METADATA OWNERSHIP MODEL ENUMERATION
// ============================================================================

/**
 * Owner of tensor metadata state during decode
 */
enum llama_tensor_metadata_owner {
    LLAMA_TENSOR_META_OWNER_UNKNOWN = 0,
    LLAMA_TENSOR_META_OWNER_CPU = 1,       // CPU owns (forbidden during decode)
    LLAMA_TENSOR_META_OWNER_GPU = 2,       // GPU owns (required during decode)
    LLAMA_TENSOR_META_OWNER_SHARED = 3,    // Shared ownership (forbidden)
};

// ============================================================================
// TENSOR METADATA FREEZE STATE ENUMERATION
// ============================================================================

/**
 * Freeze state of tensor metadata
 */
enum llama_tensor_metadata_freeze_state {
    LLAMA_TENSOR_META_FREEZE_UNKNOWN = 0,
    LLAMA_TENSOR_META_FREEZE_MUTABLE = 1,              // Mutable (forbidden)
    LLAMA_TENSOR_META_FREEZE_IMMUTABLE = 2,            // Immutable (required)
    LLAMA_TENSOR_META_FREEZE_GPU_MANAGED = 3,          // GPU manages internally
};

// ============================================================================
// TENSOR METADATA OPERATION RECORD
// ============================================================================

/**
 * Record of a tensor metadata mutation attempt
 */
struct llama_tensor_metadata_operation_record {
    enum llama_cpu_tensor_metadata_mutation mutation;          // Mutation type
    enum llama_tensor_metadata_category category;              // Category
    uint64_t timestamp_ns;                                     // When it occurred
    uint32_t tensor_id;                                        // Which tensor
    const char * tensor_name;                                  // Tensor name
    const char * location;                                     // Where it occurred
    enum llama_tensor_metadata_violation_type violation;       // Violation type if any
    bool cpu_initiated;                                        // True if CPU initiated
    bool gpu_authorized;                                       // True if GPU pre-authorized
};

// ============================================================================
// TENSOR METADATA STATE RECORD
// ============================================================================

/**
 * Global state of tensor metadata during decode
 */
struct llama_tensor_metadata_state_record {
    enum llama_tensor_metadata_owner current_owner;            // Current owner
    enum llama_gpu_tensor_metadata_state gpu_state;            // GPU state
    bool cpu_mutations_eliminated;                             // CPU mutations fully removed
    bool metadata_frozen;                                      // Metadata is frozen
    bool all_descriptors_precomputed;                          // All descriptors ready
    int cpu_mutation_violations;                               // Total violations detected
    enum llama_tensor_metadata_violation_type last_violation;  // Last violation type
    uint64_t tensors_validated;                                // Tensors validated
    uint64_t gpu_metadata_start_time_ns;                       // When GPU started managing
};

// ============================================================================
// TENSOR METADATA SNAPSHOT RECORD
// ============================================================================

/**
 * Snapshot of tensor metadata (for consistency checks)
 */
struct llama_tensor_metadata_snapshot {
    int num_tensors;                                // Number of tensors
    int total_tensor_dims;                         // Sum of all dimensions
    size_t total_elements;                         // Total elements across tensors
    bool all_contiguous;                           // All tensors contiguous
    bool all_c_order;                              // All in C order
    enum llama_tensor_metadata_freeze_state freeze_state;  // Freeze state
    uint64_t snapshot_time_ns;                     // When snapshot taken
};

// ============================================================================
// TENSOR METADATA VALIDATION STATE
// ============================================================================

/**
 * Global validation state for tensor metadata elimination
 */
struct llama_tensor_metadata_elimination_validation_state {
    struct llama_tensor_metadata_state_record state_record;
    struct llama_tensor_metadata_snapshot initial_snapshot;
    struct llama_tensor_metadata_snapshot current_snapshot;
    int total_mutation_attempts;
    int total_violations;
    bool metadata_frozen_for_decode;                // Metadata frozen at decode start
    bool enforcement_strict;                        // Abort on violation vs log only
    bool debug_detect_cpu_mutations;                // Debug CPU mutation attempts
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
int llama_tensor_metadata_elimination_init(void);

// Tensor metadata ownership transfer (5 enforcement points: 1-5)
int llama_tensor_metadata_elimination_eliminate_cpu_mutations(void);
int llama_tensor_metadata_elimination_transfer_metadata_to_gpu(void);
int llama_tensor_metadata_elimination_freeze_tensor_descriptors(void);
int llama_tensor_metadata_elimination_forbid_cpu_metadata_updates(void);
int llama_tensor_metadata_elimination_assert_gpu_metadata_owns_state(void);

// Metadata immutability (3 enforcement points: 6-8)
int llama_tensor_metadata_elimination_forbid_per_token_reshapes(void);
int llama_tensor_metadata_elimination_freeze_descriptor_snapshot(void);
int llama_tensor_metadata_elimination_enable_gpu_metadata_control(void);

// Position handling (2 enforcement points: 9-10)
int llama_tensor_metadata_elimination_forbid_position_based_metadata(void);
int llama_tensor_metadata_elimination_assert_gpu_handles_positioning(void);

// CPU mutation violation detection
int llama_tensor_metadata_elimination_detect_cpu_shape_update(void);
int llama_tensor_metadata_elimination_detect_cpu_stride_update(void);
int llama_tensor_metadata_elimination_detect_cpu_offset_update(void);
int llama_tensor_metadata_elimination_detect_cpu_view_rewire(void);
int llama_tensor_metadata_elimination_detect_per_token_reshape(void);
int llama_tensor_metadata_elimination_detect_position_based_slice(void);
int llama_tensor_metadata_elimination_detect_descriptor_mutation(void);
int llama_tensor_metadata_elimination_detect_layout_mismatch(void);

// GPU metadata state management
int llama_tensor_metadata_elimination_set_gpu_metadata_prepared(void);
int llama_tensor_metadata_elimination_set_gpu_metadata_frozen(void);
int llama_tensor_metadata_elimination_signal_metadata_validated(void);
int llama_tensor_metadata_elimination_signal_gpu_active(void);

// Metadata structure control
int llama_tensor_metadata_elimination_snapshot_initial_metadata(void);
int llama_tensor_metadata_elimination_freeze_descriptors(void);
int llama_tensor_metadata_elimination_transfer_metadata_to_gpu_impl(void);

// Query and verification functions
struct llama_tensor_metadata_state_record llama_tensor_metadata_elimination_get_state_record(void);
struct llama_tensor_metadata_snapshot llama_tensor_metadata_elimination_get_current_snapshot(void);
enum llama_tensor_metadata_owner llama_tensor_metadata_elimination_get_metadata_owner(void);
enum llama_gpu_tensor_metadata_state llama_tensor_metadata_elimination_get_gpu_metadata_state(void);

// Verification functions
int llama_tensor_metadata_elimination_verify_cpu_mutations_eliminated(void);
int llama_tensor_metadata_elimination_verify_metadata_frozen(void);
int llama_tensor_metadata_elimination_verify_descriptors_precomputed(void);
int llama_tensor_metadata_elimination_verify_no_per_token_reshapes(void);
int llama_tensor_metadata_elimination_verify_gpu_controls_metadata(void);
int llama_tensor_metadata_elimination_verify_no_position_based_metadata(void);

// Diagnostics and logging
void llama_tensor_metadata_elimination_log_cpu_mutations_eliminated(void);
void llama_tensor_metadata_elimination_log_metadata_frozen(void);
void llama_tensor_metadata_elimination_log_tensors_validated(void);
void llama_tensor_metadata_elimination_print_metadata_state(void);
void llama_tensor_metadata_elimination_print_snapshot_state(void);
void llama_tensor_metadata_elimination_print_violation_summary(void);

// Violation reporting
void llama_tensor_metadata_elimination_report_mutation_violation(
    enum llama_tensor_metadata_violation_type violation_type,
    enum llama_cpu_tensor_metadata_mutation mutation,
    const char* tensor_name,
    const char* details
);

// Enforcement mode control
void llama_tensor_metadata_elimination_set_enforcement_strict(bool strict);
bool llama_tensor_metadata_elimination_get_enforcement_strict(void);
void llama_tensor_metadata_elimination_set_debug_detect_cpu_mutations(bool debug);

// Self-test suite
int llama_tensor_metadata_elimination_selftest(void);

// Helper/inline functions
static inline const char* llama_cpu_tensor_metadata_mutation_name(
    enum llama_cpu_tensor_metadata_mutation mutation
) {
    switch (mutation) {
        case LLAMA_TENSOR_METADATA_MUTATION_NONE: return "NONE";
        case LLAMA_TENSOR_METADATA_MUTATION_SHAPE_UPDATE: return "SHAPE_UPDATE";
        case LLAMA_TENSOR_METADATA_MUTATION_STRIDE_UPDATE: return "STRIDE_UPDATE";
        case LLAMA_TENSOR_METADATA_MUTATION_OFFSET_UPDATE: return "OFFSET_UPDATE";
        case LLAMA_TENSOR_METADATA_MUTATION_VIEW_REWIRE: return "VIEW_REWIRE";
        case LLAMA_TENSOR_METADATA_MUTATION_LAYOUT_CHANGE: return "LAYOUT_CHANGE";
        case LLAMA_TENSOR_METADATA_MUTATION_RESHAPE: return "RESHAPE";
        case LLAMA_TENSOR_METADATA_MUTATION_SLICE: return "SLICE";
        case LLAMA_TENSOR_METADATA_MUTATION_TRANSPOSE: return "TRANSPOSE";
        case LLAMA_TENSOR_METADATA_MUTATION_BROADCAST: return "BROADCAST";
        case LLAMA_TENSOR_METADATA_MUTATION_DESCRIPTOR_INIT: return "DESCRIPTOR_INIT";
        case LLAMA_TENSOR_METADATA_MUTATION_POSITION_ADJUST: return "POSITION_ADJUST";
        case LLAMA_TENSOR_METADATA_MUTATION_TYPE_CAST: return "TYPE_CAST";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_tensor_metadata_violation_type_name(
    enum llama_tensor_metadata_violation_type violation
) {
    switch (violation) {
        case LLAMA_TENSOR_META_VIOLATION_NONE: return "NONE";
        case LLAMA_TENSOR_META_VIOLATION_CPU_SHAPE_UPDATE: return "CPU_SHAPE_UPDATE";
        case LLAMA_TENSOR_META_VIOLATION_CPU_STRIDE_UPDATE: return "CPU_STRIDE_UPDATE";
        case LLAMA_TENSOR_META_VIOLATION_CPU_OFFSET_UPDATE: return "CPU_OFFSET_UPDATE";
        case LLAMA_TENSOR_META_VIOLATION_CPU_VIEW_REWIRE: return "CPU_VIEW_REWIRE";
        case LLAMA_TENSOR_META_VIOLATION_PER_TOKEN_RESHAPE: return "PER_TOKEN_RESHAPE";
        case LLAMA_TENSOR_META_VIOLATION_POSITION_BASED_SLICE: return "POSITION_BASED_SLICE";
        case LLAMA_TENSOR_META_VIOLATION_DESCRIPTOR_MUTATION: return "DESCRIPTOR_MUTATION";
        case LLAMA_TENSOR_META_VIOLATION_LAYOUT_MISMATCH: return "LAYOUT_MISMATCH";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_tensor_metadata_owner_name(
    enum llama_tensor_metadata_owner owner
) {
    switch (owner) {
        case LLAMA_TENSOR_META_OWNER_UNKNOWN: return "UNKNOWN";
        case LLAMA_TENSOR_META_OWNER_CPU: return "CPU";
        case LLAMA_TENSOR_META_OWNER_GPU: return "GPU";
        case LLAMA_TENSOR_META_OWNER_SHARED: return "SHARED";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_tensor_metadata_state_name(
    enum llama_gpu_tensor_metadata_state state
) {
    switch (state) {
        case LLAMA_GPU_TENSOR_META_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_TENSOR_META_PREPARED: return "PREPARED";
        case LLAMA_GPU_TENSOR_META_FROZEN: return "FROZEN";
        case LLAMA_GPU_TENSOR_META_ACTIVE: return "ACTIVE";
        case LLAMA_GPU_TENSOR_META_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_tensor_metadata_category_name(
    enum llama_tensor_metadata_category category
) {
    switch (category) {
        case LLAMA_TENSOR_META_CAT_NONE: return "NONE";
        case LLAMA_TENSOR_META_CAT_SHAPE: return "SHAPE";
        case LLAMA_TENSOR_META_CAT_LAYOUT: return "LAYOUT";
        case LLAMA_TENSOR_META_CAT_POSITIONING: return "POSITIONING";
        case LLAMA_TENSOR_META_CAT_DESCRIPTOR: return "DESCRIPTOR";
        case LLAMA_TENSOR_META_CAT_TYPE_INFO: return "TYPE_INFO";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
