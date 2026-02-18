/**
 * SECTION 19: Remove CPU KV-cache mutation responsibilities
 * Header
 *
 * This file implements enforcement that CPU KV-cache mutations are eliminated from decode.
 * All KV cache management (allocation, updates, expansion, eviction) becomes GPU-resident.
 * CPU cannot mutate KV cache state, update offsets, or expand cache during decode.
 * KV cache becomes GPU-autonomous with CPU as read-only observer of final results.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CPU KV-CACHE MUTATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of CPU KV-cache mutations (all forbidden during decode)
 */
enum llama_cpu_kvcache_mutation {
    LLAMA_KVCACHE_MUTATION_NONE = 0,
    LLAMA_KVCACHE_MUTATION_WRITE = 1,               // CPU writes to KV cache
    LLAMA_KVCACHE_MUTATION_UPDATE = 2,              // CPU updates cache entries
    LLAMA_KVCACHE_MUTATION_EXPAND = 3,              // CPU expands KV cache
    LLAMA_KVCACHE_MUTATION_EVICT = 4,               // CPU evicts cache entries
    LLAMA_KVCACHE_MUTATION_CLEAR = 5,               // CPU clears KV cache
    LLAMA_KVCACHE_MUTATION_OFFSET_UPDATE = 6,       // CPU updates cache offset
    LLAMA_KVCACHE_MUTATION_POSITION_ADVANCE = 7,    // CPU advances position counter
    LLAMA_KVCACHE_MUTATION_RESHAPE = 8,             // CPU reshapes cache tensors
    LLAMA_KVCACHE_MUTATION_LAYOUT_CHANGE = 9,       // CPU changes cache layout
    LLAMA_KVCACHE_MUTATION_DEFRAGMENT = 10,         // CPU defragments cache
    LLAMA_KVCACHE_MUTATION_ALLOCATE = 11,           // CPU allocates new cache
    LLAMA_KVCACHE_MUTATION_DEALLOCATE = 12,         // CPU deallocates cache
};

// ============================================================================
// KV-CACHE MUTATION CATEGORY ENUMERATION
// ============================================================================

/**
 * Categories of KV-cache mutations by scope and impact
 */
enum llama_kvcache_mutation_category {
    LLAMA_KVCACHE_CAT_NONE = 0,
    LLAMA_KVCACHE_CAT_DATA_WRITE = 1,               // Direct data writes
    LLAMA_KVCACHE_CAT_METADATA = 2,                 // Metadata changes (offset, pos)
    LLAMA_KVCACHE_CAT_ALLOCATION = 3,               // Allocation/deallocation
    LLAMA_KVCACHE_CAT_LAYOUT = 4,                   // Layout/reshape operations
    LLAMA_KVCACHE_CAT_EVICTION = 5,                 // Eviction/defragmentation
};

// ============================================================================
// GPU KV-CACHE STATE ENUMERATION
// ============================================================================

/**
 * GPU-resident KV-cache state during decode
 */
enum llama_gpu_kvcache_state {
    LLAMA_GPU_KVCACHE_UNINITIALIZED = 0,            // Not started
    LLAMA_GPU_KVCACHE_PREPARED = 1,                 // Cache prepared
    LLAMA_GPU_KVCACHE_AUTONOMOUS = 2,               // GPU managing cache
    LLAMA_GPU_KVCACHE_UPDATED = 3,                  // Cache updated by GPU
    LLAMA_GPU_KVCACHE_ERROR = 4,                    // Cache error
};

// ============================================================================
// KV-CACHE MUTATION VIOLATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of CPU KV-cache violations
 */
enum llama_kvcache_violation_type {
    LLAMA_KVCACHE_VIOLATION_NONE = 0,
    LLAMA_KVCACHE_VIOLATION_CPU_WRITE = 1,                  // CPU wrote to cache
    LLAMA_KVCACHE_VIOLATION_CPU_UPDATE = 2,                 // CPU updated cache
    LLAMA_KVCACHE_VIOLATION_CPU_EXPAND = 3,                 // CPU expanded cache
    LLAMA_KVCACHE_VIOLATION_CPU_OFFSET_CHANGE = 4,          // CPU changed offset
    LLAMA_KVCACHE_VIOLATION_CPU_POSITION_ADVANCE = 5,       // CPU advanced position
    LLAMA_KVCACHE_VIOLATION_CPU_ALLOCATION = 6,             // CPU allocated/deallocated
    LLAMA_KVCACHE_VIOLATION_CACHE_REALLOCATION = 7,         // Cache reallocated per-token
    LLAMA_KVCACHE_VIOLATION_LAYOUT_MISMATCH = 8,            // Layout inconsistency
};

// ============================================================================
// KV-CACHE OWNERSHIP MODEL ENUMERATION
// ============================================================================

/**
 * Owner of KV-cache mutation authority
 */
enum llama_kvcache_owner {
    LLAMA_KVCACHE_OWNER_UNKNOWN = 0,
    LLAMA_KVCACHE_OWNER_CPU = 1,       // CPU owns (forbidden during decode)
    LLAMA_KVCACHE_OWNER_GPU = 2,       // GPU owns (required during decode)
    LLAMA_KVCACHE_OWNER_SHARED = 3,    // Shared ownership (forbidden)
};

// ============================================================================
// KV-CACHE MUTABILITY ENUMERATION
// ============================================================================

/**
 * Mutability state of KV cache
 */
enum llama_kvcache_mutability {
    LLAMA_KVCACHE_MUTABILITY_UNKNOWN = 0,
    LLAMA_KVCACHE_MUTABILITY_MUTABLE = 1,           // Mutable (forbidden)
    LLAMA_KVCACHE_MUTABILITY_IMMUTABLE = 2,         // Immutable structure
    LLAMA_KVCACHE_MUTABILITY_GPU_MANAGED = 3,       // GPU manages internals
};

// ============================================================================
// KV-CACHE OPERATION RECORD
// ============================================================================

/**
 * Record of a KV-cache mutation attempt
 */
struct llama_kvcache_mutation_record {
    enum llama_cpu_kvcache_mutation mutation;           // Mutation type
    enum llama_kvcache_mutation_category category;      // Category
    uint64_t timestamp_ns;                              // When it occurred
    uint32_t sequence_id;                               // Sequence affected
    const char * location;                              // Where it occurred
    enum llama_kvcache_violation_type violation;        // Violation type if any
    bool cpu_initiated;                                 // True if CPU initiated
    bool gpu_authorized;                                // True if GPU pre-authorized
    int64_t offset_delta;                               // Offset change if applicable
    size_t size_delta;                                  // Size change if applicable
};

// ============================================================================
// KV-CACHE STATE RECORD
// ============================================================================

/**
 * Global state of KV cache during decode
 */
struct llama_kvcache_state_record {
    enum llama_kvcache_owner current_owner;             // Current owner
    enum llama_gpu_kvcache_state gpu_state;             // GPU state
    bool cpu_mutations_eliminated;                      // CPU mutations fully removed
    bool gpu_cache_autonomous;                          // GPU managing cache
    bool cache_layout_immutable;                        // Cache layout fixed
    int cpu_mutation_violations;                        // Total violations detected
    enum llama_kvcache_violation_type last_violation;   // Last violation type
    uint64_t gpu_cache_updates;                         // Cache updates by GPU
    uint64_t gpu_cache_start_time_ns;                   // When GPU started managing
    size_t current_cache_size;                          // Current cache size
    int64_t current_offset;                             // Current offset
};

// ============================================================================
// KV-CACHE SNAPSHOT RECORD
// ============================================================================

/**
 * Snapshot of KV cache state (for consistency checks)
 */
struct llama_kvcache_snapshot {
    size_t cache_size;                      // Total cache size in bytes
    int64_t current_offset;                 // Current read/write offset
    int num_sequences;                      // Number of sequences
    int64_t max_position;                   // Max position in cache
    bool layout_is_interleaved;             // Layout type
    enum llama_kvcache_mutability mutability;  // Can CPU mutate this?
    uint64_t snapshot_time_ns;              // When snapshot taken
};

// ============================================================================
// KV-CACHE VALIDATION STATE
// ============================================================================

/**
 * Global validation state for KV-cache elimination
 */
struct llama_kvcache_elimination_validation_state {
    struct llama_kvcache_state_record state_record;
    struct llama_kvcache_snapshot initial_snapshot;
    struct llama_kvcache_snapshot current_snapshot;
    int total_mutation_attempts;
    int total_violations;
    bool cache_structure_frozen;            // Cache structure immutable
    bool enforcement_strict;                // Abort on violation vs log only
    bool debug_detect_cpu_mutations;        // Debug CPU mutation attempts
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
int llama_kvcache_elimination_init(void);

// KV-cache ownership transfer (5 enforcement points: 1-5)
int llama_kvcache_elimination_eliminate_cpu_mutations(void);
int llama_kvcache_elimination_transfer_cache_to_gpu(void);
int llama_kvcache_elimination_freeze_cache_structure(void);
int llama_kvcache_elimination_forbid_cpu_cache_writes(void);
int llama_kvcache_elimination_assert_gpu_cache_owns_mutations(void);

// Cache immutability (3 enforcement points: 6-8)
int llama_kvcache_elimination_forbid_cpu_offset_changes(void);
int llama_kvcache_elimination_freeze_cache_layout(void);
int llama_kvcache_elimination_enable_gpu_cache_control(void);

// Allocation/deallocation control (2 enforcement points: 9-10)
int llama_kvcache_elimination_forbid_cpu_cache_allocation(void);
int llama_kvcache_elimination_assert_gpu_controls_allocation(void);

// CPU mutation violation detection
int llama_kvcache_elimination_detect_cpu_write(void);
int llama_kvcache_elimination_detect_cpu_update(void);
int llama_kvcache_elimination_detect_cpu_expand(void);
int llama_kvcache_elimination_detect_cpu_offset_change(void);
int llama_kvcache_elimination_detect_cpu_position_advance(void);
int llama_kvcache_elimination_detect_cpu_allocation(void);
int llama_kvcache_elimination_detect_cache_reallocation(void);
int llama_kvcache_elimination_detect_layout_mismatch(void);

// GPU cache state management
int llama_kvcache_elimination_set_gpu_cache_prepared(void);
int llama_kvcache_elimination_set_gpu_cache_autonomous(void);
int llama_kvcache_elimination_signal_gpu_cache_updated(void);
int llama_kvcache_elimination_signal_gpu_cache_complete(void);

// Cache structure control
int llama_kvcache_elimination_snapshot_initial_cache(void);
int llama_kvcache_elimination_freeze_cache_structure_impl(void);
int llama_kvcache_elimination_transfer_cache_to_gpu_impl(void);

// Query and verification functions
struct llama_kvcache_state_record llama_kvcache_elimination_get_state_record(void);
struct llama_kvcache_snapshot llama_kvcache_elimination_get_current_snapshot(void);
enum llama_kvcache_owner llama_kvcache_elimination_get_cache_owner(void);
enum llama_gpu_kvcache_state llama_kvcache_elimination_get_gpu_cache_state(void);

// Verification functions
int llama_kvcache_elimination_verify_cpu_mutations_eliminated(void);
int llama_kvcache_elimination_verify_gpu_cache_autonomous(void);
int llama_kvcache_elimination_verify_cache_structure_frozen(void);
int llama_kvcache_elimination_verify_no_cpu_offset_changes(void);
int llama_kvcache_elimination_verify_gpu_controls_cache(void);
int llama_kvcache_elimination_verify_no_cache_reallocation(void);

// Diagnostics and logging
void llama_kvcache_elimination_log_cpu_mutations_eliminated(void);
void llama_kvcache_elimination_log_gpu_cache_started(void);
void llama_kvcache_elimination_log_cache_updated_by_gpu(void);
void llama_kvcache_elimination_print_cache_state(void);
void llama_kvcache_elimination_print_snapshot_state(void);
void llama_kvcache_elimination_print_violation_summary(void);

// Violation reporting
void llama_kvcache_elimination_report_mutation_violation(
    enum llama_kvcache_violation_type violation_type,
    enum llama_cpu_kvcache_mutation mutation,
    const char* details
);

// Enforcement mode control
void llama_kvcache_elimination_set_enforcement_strict(bool strict);
bool llama_kvcache_elimination_get_enforcement_strict(void);
void llama_kvcache_elimination_set_debug_detect_cpu_mutations(bool debug);

// Self-test suite
int llama_kvcache_elimination_selftest(void);

// Helper/inline functions
static inline const char* llama_cpu_kvcache_mutation_name(
    enum llama_cpu_kvcache_mutation mutation
) {
    switch (mutation) {
        case LLAMA_KVCACHE_MUTATION_NONE: return "NONE";
        case LLAMA_KVCACHE_MUTATION_WRITE: return "WRITE";
        case LLAMA_KVCACHE_MUTATION_UPDATE: return "UPDATE";
        case LLAMA_KVCACHE_MUTATION_EXPAND: return "EXPAND";
        case LLAMA_KVCACHE_MUTATION_EVICT: return "EVICT";
        case LLAMA_KVCACHE_MUTATION_CLEAR: return "CLEAR";
        case LLAMA_KVCACHE_MUTATION_OFFSET_UPDATE: return "OFFSET_UPDATE";
        case LLAMA_KVCACHE_MUTATION_POSITION_ADVANCE: return "POSITION_ADVANCE";
        case LLAMA_KVCACHE_MUTATION_RESHAPE: return "RESHAPE";
        case LLAMA_KVCACHE_MUTATION_LAYOUT_CHANGE: return "LAYOUT_CHANGE";
        case LLAMA_KVCACHE_MUTATION_DEFRAGMENT: return "DEFRAGMENT";
        case LLAMA_KVCACHE_MUTATION_ALLOCATE: return "ALLOCATE";
        case LLAMA_KVCACHE_MUTATION_DEALLOCATE: return "DEALLOCATE";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_kvcache_violation_type_name(
    enum llama_kvcache_violation_type violation
) {
    switch (violation) {
        case LLAMA_KVCACHE_VIOLATION_NONE: return "NONE";
        case LLAMA_KVCACHE_VIOLATION_CPU_WRITE: return "CPU_WRITE";
        case LLAMA_KVCACHE_VIOLATION_CPU_UPDATE: return "CPU_UPDATE";
        case LLAMA_KVCACHE_VIOLATION_CPU_EXPAND: return "CPU_EXPAND";
        case LLAMA_KVCACHE_VIOLATION_CPU_OFFSET_CHANGE: return "CPU_OFFSET_CHANGE";
        case LLAMA_KVCACHE_VIOLATION_CPU_POSITION_ADVANCE: return "CPU_POSITION_ADVANCE";
        case LLAMA_KVCACHE_VIOLATION_CPU_ALLOCATION: return "CPU_ALLOCATION";
        case LLAMA_KVCACHE_VIOLATION_CACHE_REALLOCATION: return "CACHE_REALLOCATION";
        case LLAMA_KVCACHE_VIOLATION_LAYOUT_MISMATCH: return "LAYOUT_MISMATCH";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_kvcache_owner_name(
    enum llama_kvcache_owner owner
) {
    switch (owner) {
        case LLAMA_KVCACHE_OWNER_UNKNOWN: return "UNKNOWN";
        case LLAMA_KVCACHE_OWNER_CPU: return "CPU";
        case LLAMA_KVCACHE_OWNER_GPU: return "GPU";
        case LLAMA_KVCACHE_OWNER_SHARED: return "SHARED";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_kvcache_state_name(
    enum llama_gpu_kvcache_state state
) {
    switch (state) {
        case LLAMA_GPU_KVCACHE_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_KVCACHE_PREPARED: return "PREPARED";
        case LLAMA_GPU_KVCACHE_AUTONOMOUS: return "AUTONOMOUS";
        case LLAMA_GPU_KVCACHE_UPDATED: return "UPDATED";
        case LLAMA_GPU_KVCACHE_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_kvcache_mutation_category_name(
    enum llama_kvcache_mutation_category category
) {
    switch (category) {
        case LLAMA_KVCACHE_CAT_NONE: return "NONE";
        case LLAMA_KVCACHE_CAT_DATA_WRITE: return "DATA_WRITE";
        case LLAMA_KVCACHE_CAT_METADATA: return "METADATA";
        case LLAMA_KVCACHE_CAT_ALLOCATION: return "ALLOCATION";
        case LLAMA_KVCACHE_CAT_LAYOUT: return "LAYOUT";
        case LLAMA_KVCACHE_CAT_EVICTION: return "EVICTION";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
