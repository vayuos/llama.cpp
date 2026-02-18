/**
 * SECTION 28: Enforce GPU-Only Context Position Tracking
 * Header
 *
 * This file implements GPU-exclusive context position (n_past) tracking.
 * Context position state remains GPU-resident during decode.
 * CPU does not update, increment, or manage context position.
 * Only context position value crosses PCIe; state stays on GPU.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONTEXT POSITION MODE ENUMERATION
// ============================================================================

/**
 * Context position (n_past) tracking modes
 */
enum llama_context_position_mode {
    LLAMA_CONTEXT_POSITION_NONE = 0,
    LLAMA_CONTEXT_POSITION_CPU = 1,      // CPU maintains context position (deprecated)
    LLAMA_CONTEXT_POSITION_GPU = 2,      // GPU maintains context position
    LLAMA_CONTEXT_POSITION_FUSED = 3,    // Fused with KV-cache management
};

// ============================================================================
// GPU CONTEXT POSITION STATE ENUMERATION
// ============================================================================

/**
 * State of GPU context position tracking (n_past)
 */
enum llama_gpu_context_position_state {
    LLAMA_GPU_CONTEXT_POS_UNINITIALIZED = 0,
    LLAMA_GPU_CONTEXT_POS_ALLOCATED = 1,    // Position buffer allocated
    LLAMA_GPU_CONTEXT_POS_INITIALIZED = 2,  // Position initialized
    LLAMA_GPU_CONTEXT_POS_ACTIVE = 3,       // Active during decode
    LLAMA_GPU_CONTEXT_POS_UPDATED = 4,      // Position updated
    LLAMA_GPU_CONTEXT_POS_SYNCED = 5,       // Synced to CPU (read-only)
    LLAMA_GPU_CONTEXT_POS_ERROR = 6,
};

// ============================================================================
// CPU CONTEXT POSITION BYPASS ENUMERATION
// ============================================================================

/**
 * CPU context position operations that must be bypassed
 */
enum llama_cpu_context_position_bypass {
    LLAMA_CONTEXT_POS_BYPASS_NONE = 0,
    LLAMA_CONTEXT_POS_BYPASS_UPDATE = 1,         // Skip CPU position update
    LLAMA_CONTEXT_POS_BYPASS_ADVANCE = 2,        // Skip CPU position advance
    LLAMA_CONTEXT_POS_BYPASS_COMPARISON = 3,     // Skip CPU position comparison
};

// ============================================================================
// CONTEXT POSITION VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-exclusive context position tracking
 */
enum llama_context_position_violation {
    LLAMA_CONTEXT_POSITION_VIOLATION_NONE = 0,
    LLAMA_CONTEXT_POSITION_VIOLATION_CPU_UPDATE = 1,      // CPU updated context position
    LLAMA_CONTEXT_POSITION_VIOLATION_CPU_COMPARISON = 2,  // CPU compared context position
    LLAMA_CONTEXT_POSITION_VIOLATION_CONTEXT_POS_ON_HOST = 3, // Position materialized
    LLAMA_CONTEXT_POSITION_VIOLATION_CPU_GATING = 4,      // CPU used position for gating
    LLAMA_CONTEXT_POSITION_VIOLATION_MIXED_UPDATE = 5,    // Mixed CPU/GPU updates
    LLAMA_CONTEXT_POSITION_VIOLATION_DESYNC = 6,          // CPU/GPU desync
};

// ============================================================================
// CONTEXT POSITION CONFIGURATION RECORD
// ============================================================================

/**
 * Configuration for GPU context position tracking
 */
struct llama_gpu_context_position_config {
    bool gpu_context_pos_tracking_enabled;    // Enable GPU context position tracking?
    bool cpu_updates_forbidden;               // Forbid CPU position updates?
    enum llama_context_position_mode mode;    // Position tracking mode
    uint32_t context_length;                  // Maximum context length
    bool validate_position_bounds;            // Check bounds on each update?
};

// ============================================================================
// GPU CONTEXT POSITION STATE RECORD
// ============================================================================

/**
 * Current state of GPU context position tracking (n_past)
 */
struct llama_gpu_context_position_state_record {
    enum llama_context_position_mode current_mode;      // Current mode
    enum llama_gpu_context_position_state gpu_pos_state; // GPU position state
    uint32_t context_position;                           // Current n_past value
    uint32_t context_length;                             // Max context
    uint64_t position_updates_count;                      // GPU updates
    int total_violations;                                 // Total violations
    enum llama_context_position_violation last_violation; // Last violation
    bool position_locked;                                 // Locked to GPU?
};

// ============================================================================
// CONTEXT POSITION UPDATE RECORD
// ============================================================================

/**
 * Record of a single GPU context position update
 */
struct llama_gpu_context_position_update_record {
    uint32_t position_before;                // n_past before
    uint32_t position_after;                 // n_past after
    uint32_t tokens_added;                   // Tokens added in this update
    uint64_t timestamp_ns;                   // When update occurred
    bool update_on_gpu;                      // Was update on GPU?
};

// ============================================================================
// CONTEXT POSITION VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU context position tracking
 */
struct llama_gpu_context_position_validation_state {
    struct llama_gpu_context_position_config config;
    struct llama_gpu_context_position_state_record state_record;
    struct llama_gpu_context_position_update_record last_update;
    int total_position_updates;
    int total_violations;
    bool enforcement_strict;                 // Abort on violation vs log only
    bool debug_context_position;             // Debug output
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_context_position_gpu_init(void);
int llama_context_position_gpu_configure(
    bool gpu_context_pos_enabled,
    bool cpu_updates_forbidden,
    uint32_t context_length
);

// Context position setup
int llama_context_position_gpu_allocate_position_buffer(uint32_t context_length);
int llama_context_position_gpu_initialize_position(uint32_t initial_position);

// GPU context position updates (10 enforcement points: 1-10)
int llama_context_position_gpu_queue_update_kernel(void);
int llama_context_position_gpu_update_on_gpu(uint32_t new_position);
int llama_context_position_gpu_keep_position_on_device(void);
int llama_context_position_gpu_forbid_cpu_update(void);
int llama_context_position_gpu_forbid_cpu_comparison(void);
int llama_context_position_gpu_forbid_cpu_gating(void);
int llama_context_position_gpu_validate_position_bounds(void);
int llama_context_position_gpu_lock_position_to_gpu(void);
int llama_context_position_gpu_verify_no_cpu_modification(void);
int llama_context_position_gpu_commit_position_update(uint32_t new_position);

// Position retrieval and synchronization
int llama_context_position_gpu_read_position_sync(uint32_t* out_position);
int llama_context_position_gpu_read_position_async(uint32_t* out_position);
int llama_context_position_gpu_sync_position_to_cpu(void);

// Violation detection
int llama_context_position_gpu_detect_cpu_update(void);
int llama_context_position_gpu_detect_cpu_comparison(void);
int llama_context_position_gpu_detect_position_on_host(void);
int llama_context_position_gpu_detect_cpu_gating(void);
int llama_context_position_gpu_detect_mixed_updates(void);
int llama_context_position_gpu_detect_desync(void);

// State management
int llama_context_position_gpu_set_allocated(void);
int llama_context_position_gpu_set_initialized(void);
int llama_context_position_gpu_set_active(void);
int llama_context_position_gpu_set_updated(void);

// Query and verification functions
struct llama_gpu_context_position_state_record llama_context_position_gpu_get_state_record(void);
struct llama_gpu_context_position_update_record llama_context_position_gpu_get_last_update(void);
uint32_t llama_context_position_gpu_get_context_position(void);
enum llama_gpu_context_position_state llama_context_position_gpu_get_position_state(void);

// Verification functions
int llama_context_position_gpu_verify_cpu_updates_forbidden(void);
int llama_context_position_gpu_verify_gpu_position_active(void);
int llama_context_position_gpu_verify_position_locked(void);
int llama_context_position_gpu_verify_no_cpu_entry_point(void);
int llama_context_position_gpu_verify_position_within_bounds(void);
int llama_context_position_gpu_verify_no_desync(void);
int llama_context_position_gpu_verify_monotonic_increment(void);

// Diagnostics and logging
void llama_context_position_gpu_log_position_mode_enabled(void);
void llama_context_position_gpu_log_position_locked(void);
void llama_context_position_gpu_print_state(void);
void llama_context_position_gpu_print_execution_stats(void);
void llama_context_position_gpu_print_violation_summary(void);

// Violation reporting
void llama_context_position_gpu_report_violation(
    enum llama_context_position_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_context_position_gpu_set_enforcement_strict(bool strict);
bool llama_context_position_gpu_get_enforcement_strict(void);
void llama_context_position_gpu_set_debug_output(bool debug);

// Self-test suite
int llama_context_position_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_context_position_mode_name(
    enum llama_context_position_mode mode
) {
    switch (mode) {
        case LLAMA_CONTEXT_POSITION_NONE: return "NONE";
        case LLAMA_CONTEXT_POSITION_CPU: return "CPU";
        case LLAMA_CONTEXT_POSITION_GPU: return "GPU";
        case LLAMA_CONTEXT_POSITION_FUSED: return "FUSED";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_context_position_state_name(
    enum llama_gpu_context_position_state state
) {
    switch (state) {
        case LLAMA_GPU_CONTEXT_POS_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_CONTEXT_POS_ALLOCATED: return "ALLOCATED";
        case LLAMA_GPU_CONTEXT_POS_INITIALIZED: return "INITIALIZED";
        case LLAMA_GPU_CONTEXT_POS_ACTIVE: return "ACTIVE";
        case LLAMA_GPU_CONTEXT_POS_UPDATED: return "UPDATED";
        case LLAMA_GPU_CONTEXT_POS_SYNCED: return "SYNCED";
        case LLAMA_GPU_CONTEXT_POS_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_context_position_violation_name(
    enum llama_context_position_violation violation
) {
    switch (violation) {
        case LLAMA_CONTEXT_POSITION_VIOLATION_NONE: return "NONE";
        case LLAMA_CONTEXT_POSITION_VIOLATION_CPU_UPDATE: return "CPU_UPDATE";
        case LLAMA_CONTEXT_POSITION_VIOLATION_CPU_COMPARISON: return "CPU_COMPARISON";
        case LLAMA_CONTEXT_POSITION_VIOLATION_CONTEXT_POS_ON_HOST: return "CONTEXT_POS_ON_HOST";
        case LLAMA_CONTEXT_POSITION_VIOLATION_CPU_GATING: return "CPU_GATING";
        case LLAMA_CONTEXT_POSITION_VIOLATION_MIXED_UPDATE: return "MIXED_UPDATE";
        case LLAMA_CONTEXT_POSITION_VIOLATION_DESYNC: return "DESYNC";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
