/**
 * SECTION 27: Eliminate CPU KV-Cache Position Updates
 * Header
 *
 * This file implements GPU-exclusive KV-cache position tracking.
 * Position state remains GPU-resident during decode.
 * CPU does not update, increment, or re-derive position.
 * Only position value crosses PCIe on read; updates stay on GPU.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// KV-CACHE POSITION MODE ENUMERATION
// ============================================================================

/**
 * KV-cache position tracking modes
 */
enum llama_kvcache_position_mode {
    LLAMA_KVCACHE_POSITION_NONE = 0,
    LLAMA_KVCACHE_POSITION_CPU = 1,          // CPU maintains position (deprecated)
    LLAMA_KVCACHE_POSITION_GPU = 2,          // GPU maintains position
    LLAMA_KVCACHE_POSITION_GPU_SYNC = 3,     // GPU position with periodic sync
};

// ============================================================================
// GPU POSITION STATE ENUMERATION
// ============================================================================

/**
 * State of GPU KV-cache position tracking
 */
enum llama_gpu_position_state {
    LLAMA_GPU_POSITION_UNINITIALIZED = 0,
    LLAMA_GPU_POSITION_ALLOCATED = 1,        // Position buffer allocated on GPU
    LLAMA_GPU_POSITION_INITIALIZED = 2,      // Position initialized to prefill_length
    LLAMA_GPU_POSITION_DECODE_ACTIVE = 3,    // Active during decode phase
    LLAMA_GPU_POSITION_ADVANCED = 4,         // Position advanced by GPU kernel
    LLAMA_GPU_POSITION_SYNCED = 5,           // Position synced to CPU for read-only
    LLAMA_GPU_POSITION_ERROR = 6,
};

// ============================================================================
// CPU POSITION UPDATE BYPASS ENUMERATION
// ============================================================================

/**
 * CPU position update operations that must be bypassed
 */
enum llama_cpu_position_bypass {
    LLAMA_POSITION_BYPASS_NONE = 0,
    LLAMA_POSITION_BYPASS_INCREMENT = 1,         // Skip CPU position increment
    LLAMA_POSITION_BYPASS_UPDATE = 2,            // Skip CPU position update
    LLAMA_POSITION_BYPASS_SYNC = 3,              // Skip CPU-initiated sync
    LLAMA_POSITION_BYPASS_VALIDATION = 4,        // Skip CPU position validation
};

// ============================================================================
// KV-CACHE POSITION VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-exclusive position tracking
 */
enum llama_kvcache_position_violation {
    LLAMA_KVCACHE_POSITION_VIOLATION_NONE = 0,
    LLAMA_KVCACHE_POSITION_VIOLATION_CPU_UPDATE = 1,      // CPU updated position
    LLAMA_KVCACHE_POSITION_VIOLATION_CPU_INCREMENT = 2,   // CPU incremented position
    LLAMA_KVCACHE_POSITION_VIOLATION_POSITION_ON_HOST = 3, // Position materialized on host
    LLAMA_KVCACHE_POSITION_VIOLATION_CPU_SYNC = 4,        // CPU initiated sync
    LLAMA_KVCACHE_POSITION_VIOLATION_CPU_VALIDATION = 5,  // CPU validated position
    LLAMA_KVCACHE_POSITION_VIOLATION_MIXED_UPDATE = 6,    // Mixed CPU/GPU updates
    LLAMA_KVCACHE_POSITION_VIOLATION_DESYNC = 7,          // CPU and GPU positions diverged
};

// ============================================================================
// GPU POSITION UPDATE ENUMERATION
// ============================================================================

/**
 * Types of GPU position updates
 */
enum llama_gpu_position_update_type {
    LLAMA_GPU_POSITION_UPDATE_NONE = 0,
    LLAMA_GPU_POSITION_UPDATE_INCREMENT = 1,   // Increment by 1
    LLAMA_GPU_POSITION_UPDATE_ADVANCE = 2,     // Advance by N tokens
    LLAMA_GPU_POSITION_UPDATE_SET = 3,         // Set to specific value
    LLAMA_GPU_POSITION_UPDATE_RESET = 4,       // Reset to prefill length
};

// ============================================================================
// KV-CACHE POSITION CONFIGURATION RECORD
// ============================================================================

/**
 * Configuration for GPU KV-cache position tracking
 */
struct llama_gpu_kvcache_position_config {
    bool gpu_position_tracking_enabled;    // Enable GPU position tracking?
    bool cpu_updates_forbidden;            // Forbid CPU position updates?
    enum llama_kvcache_position_mode mode; // Position tracking mode
    uint32_t prefill_position;             // Initial position (prefill length)
    uint32_t max_position;                 // Maximum position (context length)
    bool validate_position_bounds;         // Check bounds on each update?
    bool sync_position_periodically;       // Sync position to CPU periodically?
    uint32_t sync_interval_tokens;         // Sync every N tokens (0 = no periodic sync)
};

// ============================================================================
// GPU POSITION STATE RECORD
// ============================================================================

/**
 * Current state of GPU KV-cache position tracking
 */
struct llama_gpu_position_state_record {
    enum llama_kvcache_position_mode current_mode;      // Current tracking mode
    enum llama_gpu_position_state gpu_position_state;   // GPU position state
    uint32_t current_position;                          // Current position value
    uint32_t prefill_position;                          // Prefill length
    uint32_t max_position;                              // Context max
    uint64_t position_updates_count;                    // Total GPU updates
    uint64_t last_update_timestamp_ns;                  // Last update time
    uint64_t last_sync_timestamp_ns;                    // Last sync time
    int total_violations;                               // Total violations
    enum llama_kvcache_position_violation last_violation; // Last violation
    bool position_locked;                               // Position locked to GPU?
};

// ============================================================================
// GPU POSITION UPDATE RECORD
// ============================================================================

/**
 * Record of a single GPU position update operation
 */
struct llama_gpu_position_update_record {
    enum llama_gpu_position_update_type update_type;    // Type of update
    uint32_t position_before;                           // Position before update
    uint32_t position_after;                            // Position after update
    uint32_t tokens_processed;                          // Tokens processed in this update
    uint64_t timestamp_ns;                              // When update occurred
    bool update_on_gpu;                                 // Was update on GPU?
    int cpu_violations_detected;                        // Violations during this update
};

// ============================================================================
// KV-CACHE POSITION VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU KV-cache position tracking
 */
struct llama_gpu_kvcache_position_validation_state {
    struct llama_gpu_kvcache_position_config config;
    struct llama_gpu_position_state_record state_record;
    struct llama_gpu_position_update_record last_update;
    int total_position_updates;
    int total_violations;
    bool enforcement_strict;                 // Abort on violation vs log only
    bool debug_position_tracking;            // Debug output
    bool verify_position_consistency;        // Verify CPU/GPU consistency
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_kvcache_position_gpu_init(void);
int llama_kvcache_position_gpu_configure(
    bool gpu_position_enabled,
    bool cpu_updates_forbidden,
    uint32_t prefill_position,
    uint32_t max_position
);

// Position tracking setup
int llama_kvcache_position_gpu_allocate_position_buffer(uint32_t max_position);
int llama_kvcache_position_gpu_initialize_position(uint32_t prefill_position);

// GPU position updates (10 enforcement points: 1-10)
int llama_kvcache_position_gpu_queue_position_kernel(void);
int llama_kvcache_position_gpu_increment_on_gpu(void);
int llama_kvcache_position_gpu_advance_on_gpu(uint32_t num_tokens);
int llama_kvcache_position_gpu_keep_position_on_device(void);
int llama_kvcache_position_gpu_forbid_cpu_increment(void);
int llama_kvcache_position_gpu_forbid_cpu_update(void);
int llama_kvcache_position_gpu_validate_position_bounds(void);
int llama_kvcache_position_gpu_lock_position_to_gpu(void);
int llama_kvcache_position_gpu_verify_no_cpu_modification(void);
int llama_kvcache_position_gpu_commit_position_advance(uint32_t new_position);

// Position retrieval and synchronization
int llama_kvcache_position_gpu_read_position_sync(uint32_t* out_position);
int llama_kvcache_position_gpu_read_position_async(uint32_t* out_position);
int llama_kvcache_position_gpu_sync_position_to_cpu(void);

// Violation detection
int llama_kvcache_position_gpu_detect_cpu_update(void);
int llama_kvcache_position_gpu_detect_cpu_increment(void);
int llama_kvcache_position_gpu_detect_position_on_host(void);
int llama_kvcache_position_gpu_detect_cpu_sync(void);
int llama_kvcache_position_gpu_detect_cpu_validation(void);
int llama_kvcache_position_gpu_detect_mixed_updates(void);
int llama_kvcache_position_gpu_detect_desync(void);

// State management
int llama_kvcache_position_gpu_set_allocated(void);
int llama_kvcache_position_gpu_set_initialized(void);
int llama_kvcache_position_gpu_set_decode_active(void);
int llama_kvcache_position_gpu_set_advanced(void);

// Query and verification functions
struct llama_gpu_position_state_record llama_kvcache_position_gpu_get_state_record(void);
struct llama_gpu_position_update_record llama_kvcache_position_gpu_get_last_update(void);
uint32_t llama_kvcache_position_gpu_get_current_position(void);
enum llama_gpu_position_state llama_kvcache_position_gpu_get_position_state(void);

// Verification functions
int llama_kvcache_position_gpu_verify_cpu_updates_forbidden(void);
int llama_kvcache_position_gpu_verify_gpu_position_active(void);
int llama_kvcache_position_gpu_verify_position_locked(void);
int llama_kvcache_position_gpu_verify_no_cpu_entry_point(void);
int llama_kvcache_position_gpu_verify_position_within_bounds(void);
int llama_kvcache_position_gpu_verify_position_consistency(void);
int llama_kvcache_position_gpu_verify_monotonic_increment(void);
int llama_kvcache_position_gpu_verify_no_desync(void);

// Diagnostics and logging
void llama_kvcache_position_gpu_log_position_mode_enabled(void);
void llama_kvcache_position_gpu_log_position_locked(void);
void llama_kvcache_position_gpu_print_state(void);
void llama_kvcache_position_gpu_print_execution_stats(void);
void llama_kvcache_position_gpu_print_violation_summary(void);

// Violation reporting
void llama_kvcache_position_gpu_report_violation(
    enum llama_kvcache_position_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_kvcache_position_gpu_set_enforcement_strict(bool strict);
bool llama_kvcache_position_gpu_get_enforcement_strict(void);
void llama_kvcache_position_gpu_set_debug_output(bool debug);
void llama_kvcache_position_gpu_set_verify_consistency(bool verify);

// Self-test suite
int llama_kvcache_position_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_kvcache_position_mode_name(
    enum llama_kvcache_position_mode mode
) {
    switch (mode) {
        case LLAMA_KVCACHE_POSITION_NONE: return "NONE";
        case LLAMA_KVCACHE_POSITION_CPU: return "CPU";
        case LLAMA_KVCACHE_POSITION_GPU: return "GPU";
        case LLAMA_KVCACHE_POSITION_GPU_SYNC: return "GPU_SYNC";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_position_state_name(
    enum llama_gpu_position_state state
) {
    switch (state) {
        case LLAMA_GPU_POSITION_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_POSITION_ALLOCATED: return "ALLOCATED";
        case LLAMA_GPU_POSITION_INITIALIZED: return "INITIALIZED";
        case LLAMA_GPU_POSITION_DECODE_ACTIVE: return "DECODE_ACTIVE";
        case LLAMA_GPU_POSITION_ADVANCED: return "ADVANCED";
        case LLAMA_GPU_POSITION_SYNCED: return "SYNCED";
        case LLAMA_GPU_POSITION_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_kvcache_position_violation_name(
    enum llama_kvcache_position_violation violation
) {
    switch (violation) {
        case LLAMA_KVCACHE_POSITION_VIOLATION_NONE: return "NONE";
        case LLAMA_KVCACHE_POSITION_VIOLATION_CPU_UPDATE: return "CPU_UPDATE";
        case LLAMA_KVCACHE_POSITION_VIOLATION_CPU_INCREMENT: return "CPU_INCREMENT";
        case LLAMA_KVCACHE_POSITION_VIOLATION_POSITION_ON_HOST: return "POSITION_ON_HOST";
        case LLAMA_KVCACHE_POSITION_VIOLATION_CPU_SYNC: return "CPU_SYNC";
        case LLAMA_KVCACHE_POSITION_VIOLATION_CPU_VALIDATION: return "CPU_VALIDATION";
        case LLAMA_KVCACHE_POSITION_VIOLATION_MIXED_UPDATE: return "MIXED_UPDATE";
        case LLAMA_KVCACHE_POSITION_VIOLATION_DESYNC: return "DESYNC";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
