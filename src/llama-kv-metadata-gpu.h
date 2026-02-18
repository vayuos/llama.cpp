/**
 * SECTION 29: Remove CPU KV Metadata Tracking
 * Header
 *
 * This file implements GPU-exclusive KV-cache metadata management.
 * KV cache state (positions, offsets, validity) is GPU-owned during decode.
 * CPU does not track, maintain, or validate KV metadata during decode.
 * All KV mutations occur inside GPU kernels; CPU observes final KV state only.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// KV METADATA TRACKING MODE ENUMERATION
// ============================================================================

/**
 * KV metadata tracking modes
 */
enum llama_kv_metadata_mode {
    LLAMA_KV_METADATA_NONE = 0,
    LLAMA_KV_METADATA_CPU = 1,       // CPU maintains KV metadata (deprecated)
    LLAMA_KV_METADATA_GPU = 2,       // GPU maintains KV metadata
};

// ============================================================================
// GPU KV METADATA STATE ENUMERATION
// ============================================================================

/**
 * State of GPU KV metadata during decode
 */
enum llama_gpu_kv_metadata_state {
    LLAMA_GPU_KV_METADATA_UNINITIALIZED = 0,
    LLAMA_GPU_KV_METADATA_ALLOCATED = 1,     // GPU KV buffers allocated
    LLAMA_GPU_KV_METADATA_INITIALIZED = 2,   // KV metadata initialized
    LLAMA_GPU_KV_METADATA_DECODE_ACTIVE = 3, // Active during decode
    LLAMA_GPU_KV_METADATA_UPDATED = 4,       // Metadata updated by GPU kernel
    LLAMA_GPU_KV_METADATA_SYNCED = 5,        // Synced to CPU (read-only)
    LLAMA_GPU_KV_METADATA_ERROR = 6,
};

// ============================================================================
// CPU KV METADATA BYPASS ENUMERATION
// ============================================================================

/**
 * CPU KV metadata operations that must be bypassed
 */
enum llama_cpu_kv_metadata_bypass {
    LLAMA_KV_METADATA_BYPASS_NONE = 0,
    LLAMA_KV_METADATA_BYPASS_UPDATE = 1,        // Skip CPU KV metadata update
    LLAMA_KV_METADATA_BYPASS_READ = 2,          // Skip CPU KV metadata read
    LLAMA_KV_METADATA_BYPASS_VALIDATION = 3,    // Skip CPU KV bounds check
    LLAMA_KV_METADATA_BYPASS_SYNC_CHECK = 4,    // Skip CPU KV sync check
};

// ============================================================================
// KV METADATA VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-exclusive KV metadata ownership
 */
enum llama_kv_metadata_violation {
    LLAMA_KV_METADATA_VIOLATION_NONE = 0,
    LLAMA_KV_METADATA_VIOLATION_CPU_UPDATE = 1,      // CPU updated KV metadata
    LLAMA_KV_METADATA_VIOLATION_CPU_READ = 2,        // CPU read KV metadata
    LLAMA_KV_METADATA_VIOLATION_CPU_BOUNDS_CHECK = 3, // CPU checked KV bounds
    LLAMA_KV_METADATA_VIOLATION_CPU_SYNC_CHECK = 4,  // CPU checked KV sync
    LLAMA_KV_METADATA_VIOLATION_MIXED_UPDATE = 5,    // Mixed CPU/GPU updates
    LLAMA_KV_METADATA_VIOLATION_DESYNC = 6,          // CPU/GPU metadata desync
    LLAMA_KV_METADATA_VIOLATION_HYBRID_PATH = 7,     // CPU/GPU hybrid KV cache
};

// ============================================================================
// GPU KV METADATA LAYER RECORD
// ============================================================================

/**
 * Per-layer KV metadata stored on GPU
 */
struct llama_gpu_kv_layer_metadata {
    uint32_t kv_write_offset;        // Current write offset for K/V in this layer
    uint32_t kv_read_offset;         // Current read offset for K/V in this layer
    uint32_t kv_max_tokens;          // Max tokens this layer can store
    uint32_t kv_current_tokens;      // Current tokens in KV cache for this layer
    uint32_t reserved_1;
    uint32_t reserved_2;
    uint32_t reserved_3;
    uint32_t reserved_4;
};

// ============================================================================
// GPU KV METADATA CONFIGURATION
// ============================================================================

/**
 * Configuration for GPU KV metadata management
 */
struct llama_gpu_kv_metadata_config {
    bool gpu_kv_metadata_tracking_enabled;  // Enable GPU KV metadata tracking?
    bool cpu_kv_updates_forbidden;          // Forbid CPU KV metadata updates?
    enum llama_kv_metadata_mode mode;       // KV metadata tracking mode
    uint32_t num_layers;                    // Number of model layers
    uint32_t max_tokens_per_layer;          // Max tokens per layer in KV cache
    bool validate_kv_bounds;                // Check bounds on each update?
    bool enforce_gpu_only_kv;               // Enforce GPU-only KV cache?
};

// ============================================================================
// GPU KV METADATA STATE RECORD
// ============================================================================

/**
 * Current state of GPU KV metadata tracking
 */
struct llama_gpu_kv_metadata_state_record {
    enum llama_kv_metadata_mode current_mode;       // Current tracking mode
    enum llama_gpu_kv_metadata_state gpu_kv_state;  // GPU KV metadata state
    uint32_t num_layers;                            // Number of layers
    uint32_t total_tokens_in_kv;                    // Total tokens in KV cache
    uint64_t metadata_updates_count;                // GPU kernel updates
    int total_violations;                           // Total violations
    enum llama_kv_metadata_violation last_violation; // Last violation
    bool metadata_locked;                           // Locked to GPU?
};

// ============================================================================
// GPU KV METADATA UPDATE RECORD
// ============================================================================

/**
 * Record of a GPU KV metadata update operation
 */
struct llama_gpu_kv_metadata_update_record {
    uint32_t tokens_before;              // Tokens in KV before update
    uint32_t tokens_after;               // Tokens in KV after update
    uint32_t layers_updated;             // Layers updated in this operation
    uint64_t timestamp_ns;               // When update occurred
    bool update_on_gpu;                  // Was update on GPU?
};

// ============================================================================
// KV METADATA VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU KV metadata management
 */
struct llama_gpu_kv_metadata_validation_state {
    struct llama_gpu_kv_metadata_config config;
    struct llama_gpu_kv_metadata_state_record state_record;
    struct llama_gpu_kv_metadata_update_record last_update;
    int total_metadata_updates;
    int total_violations;
    bool enforcement_strict;              // Abort on violation vs log only
    bool debug_kv_metadata;               // Debug output
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_kv_metadata_gpu_init(void);
int llama_kv_metadata_gpu_configure(
    bool gpu_kv_metadata_enabled,
    bool cpu_updates_forbidden,
    uint32_t num_layers,
    uint32_t max_tokens_per_layer
);

// KV metadata setup
int llama_kv_metadata_gpu_allocate_metadata_buffers(uint32_t num_layers);
int llama_kv_metadata_gpu_initialize_metadata(void);

// GPU KV metadata updates (10 enforcement points: 1-10)
int llama_kv_metadata_gpu_queue_metadata_kernel(void);
int llama_kv_metadata_gpu_update_metadata_on_gpu(uint32_t num_tokens);
int llama_kv_metadata_gpu_keep_metadata_on_device(void);
int llama_kv_metadata_gpu_forbid_cpu_metadata_update(void);
int llama_kv_metadata_gpu_forbid_cpu_metadata_read(void);
int llama_kv_metadata_gpu_forbid_cpu_kv_bounds_check(void);
int llama_kv_metadata_gpu_validate_metadata_bounds(void);
int llama_kv_metadata_gpu_lock_metadata_to_gpu(void);
int llama_kv_metadata_gpu_verify_no_cpu_modification(void);
int llama_kv_metadata_gpu_commit_metadata_update(uint32_t new_token_count);

// Metadata retrieval and synchronization
int llama_kv_metadata_gpu_read_metadata_sync(uint32_t* out_token_count);
int llama_kv_metadata_gpu_read_metadata_async(uint32_t* out_token_count);
int llama_kv_metadata_gpu_sync_metadata_to_cpu(void);

// Violation detection
int llama_kv_metadata_gpu_detect_cpu_update(void);
int llama_kv_metadata_gpu_detect_cpu_read(void);
int llama_kv_metadata_gpu_detect_cpu_bounds_check(void);
int llama_kv_metadata_gpu_detect_cpu_sync_check(void);
int llama_kv_metadata_gpu_detect_mixed_updates(void);
int llama_kv_metadata_gpu_detect_desync(void);
int llama_kv_metadata_gpu_detect_hybrid_path(void);

// State management
int llama_kv_metadata_gpu_set_allocated(void);
int llama_kv_metadata_gpu_set_initialized(void);
int llama_kv_metadata_gpu_set_decode_active(void);
int llama_kv_metadata_gpu_set_updated(void);

// Query and verification functions
struct llama_gpu_kv_metadata_state_record llama_kv_metadata_gpu_get_state_record(void);
struct llama_gpu_kv_metadata_update_record llama_kv_metadata_gpu_get_last_update(void);
uint32_t llama_kv_metadata_gpu_get_token_count(void);
enum llama_gpu_kv_metadata_state llama_kv_metadata_gpu_get_kv_state(void);

// Verification functions
int llama_kv_metadata_gpu_verify_cpu_updates_forbidden(void);
int llama_kv_metadata_gpu_verify_gpu_kv_metadata_active(void);
int llama_kv_metadata_gpu_verify_metadata_locked(void);
int llama_kv_metadata_gpu_verify_no_cpu_entry_point(void);
int llama_kv_metadata_gpu_verify_metadata_within_bounds(void);
int llama_kv_metadata_gpu_verify_no_desync(void);
int llama_kv_metadata_gpu_verify_no_hybrid_path(void);

// Diagnostics and logging
void llama_kv_metadata_gpu_log_metadata_mode_enabled(void);
void llama_kv_metadata_gpu_log_metadata_locked(void);
void llama_kv_metadata_gpu_print_state(void);
void llama_kv_metadata_gpu_print_execution_stats(void);
void llama_kv_metadata_gpu_print_violation_summary(void);

// Violation reporting
void llama_kv_metadata_gpu_report_violation(
    enum llama_kv_metadata_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_kv_metadata_gpu_set_enforcement_strict(bool strict);
bool llama_kv_metadata_gpu_get_enforcement_strict(void);
void llama_kv_metadata_gpu_set_debug_output(bool debug);

// Self-test suite
int llama_kv_metadata_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_kv_metadata_mode_name(
    enum llama_kv_metadata_mode mode
) {
    switch (mode) {
        case LLAMA_KV_METADATA_NONE: return "NONE";
        case LLAMA_KV_METADATA_CPU: return "CPU";
        case LLAMA_KV_METADATA_GPU: return "GPU";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_kv_metadata_state_name(
    enum llama_gpu_kv_metadata_state state
) {
    switch (state) {
        case LLAMA_GPU_KV_METADATA_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_KV_METADATA_ALLOCATED: return "ALLOCATED";
        case LLAMA_GPU_KV_METADATA_INITIALIZED: return "INITIALIZED";
        case LLAMA_GPU_KV_METADATA_DECODE_ACTIVE: return "DECODE_ACTIVE";
        case LLAMA_GPU_KV_METADATA_UPDATED: return "UPDATED";
        case LLAMA_GPU_KV_METADATA_SYNCED: return "SYNCED";
        case LLAMA_GPU_KV_METADATA_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_kv_metadata_violation_name(
    enum llama_kv_metadata_violation violation
) {
    switch (violation) {
        case LLAMA_KV_METADATA_VIOLATION_NONE: return "NONE";
        case LLAMA_KV_METADATA_VIOLATION_CPU_UPDATE: return "CPU_UPDATE";
        case LLAMA_KV_METADATA_VIOLATION_CPU_READ: return "CPU_READ";
        case LLAMA_KV_METADATA_VIOLATION_CPU_BOUNDS_CHECK: return "CPU_BOUNDS_CHECK";
        case LLAMA_KV_METADATA_VIOLATION_CPU_SYNC_CHECK: return "CPU_SYNC_CHECK";
        case LLAMA_KV_METADATA_VIOLATION_MIXED_UPDATE: return "MIXED_UPDATE";
        case LLAMA_KV_METADATA_VIOLATION_DESYNC: return "DESYNC";
        case LLAMA_KV_METADATA_VIOLATION_HYBRID_PATH: return "HYBRID_PATH";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
