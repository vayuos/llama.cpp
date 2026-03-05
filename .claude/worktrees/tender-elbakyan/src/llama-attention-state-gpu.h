/**
 * SECTION 32: Enforce GPU-Only Attention State Management
 * Header
 *
 * This file implements GPU-exclusive attention state management.
 * Attention state (query/key/value heads, attention scores) is GPU-resident.
 * CPU does not maintain, track, or validate attention state during decode.
 * All attention state mutations occur inside GPU kernels; CPU observes final state only.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ATTENTION STATE MODE ENUMERATION
// ============================================================================

/**
 * Attention state management modes
 */
enum llama_attention_state_mode {
    LLAMA_ATTENTION_STATE_NONE = 0,
    LLAMA_ATTENTION_STATE_CPU = 1,    // CPU maintains attention state (deprecated)
    LLAMA_ATTENTION_STATE_GPU = 2,    // GPU maintains attention state
};

// ============================================================================
// GPU ATTENTION STATE ENUMERATION
// ============================================================================

/**
 * State of GPU attention state during decode
 */
enum llama_gpu_attention_state_status {
    LLAMA_GPU_ATTENTION_STATE_UNINITIALIZED = 0,
    LLAMA_GPU_ATTENTION_STATE_ALLOCATED = 1,      // GPU attention buffers allocated
    LLAMA_GPU_ATTENTION_STATE_INITIALIZED = 2,    // Attention state initialized
    LLAMA_GPU_ATTENTION_STATE_DECODE_ACTIVE = 3,  // Active during decode
    LLAMA_GPU_ATTENTION_STATE_COMPUTED = 4,       // Attention computed on GPU
    LLAMA_GPU_ATTENTION_STATE_STORED = 5,         // State stored in GPU memory
    LLAMA_GPU_ATTENTION_STATE_SYNCED = 6,         // Synced to CPU (read-only)
    LLAMA_GPU_ATTENTION_STATE_ERROR = 7,
};

// ============================================================================
// CPU ATTENTION STATE BYPASS ENUMERATION
// ============================================================================

/**
 * CPU attention state operations that must be bypassed
 */
enum llama_cpu_attention_state_bypass {
    LLAMA_CPU_ATTENTION_STATE_BYPASS_NONE = 0,
    LLAMA_CPU_ATTENTION_STATE_BYPASS_UPDATE = 1,        // Skip CPU attention update
    LLAMA_CPU_ATTENTION_STATE_BYPASS_READ = 2,          // Skip CPU attention read
    LLAMA_CPU_ATTENTION_STATE_BYPASS_VALIDATION = 3,    // Skip CPU attention validation
    LLAMA_CPU_ATTENTION_STATE_BYPASS_SYNC_CHECK = 4,    // Skip CPU sync check
};

// ============================================================================
// ATTENTION STATE VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-exclusive attention state ownership
 */
enum llama_attention_state_violation {
    LLAMA_ATTENTION_STATE_VIOLATION_NONE = 0,
    LLAMA_ATTENTION_STATE_VIOLATION_CPU_UPDATE = 1,      // CPU updated attention state
    LLAMA_ATTENTION_STATE_VIOLATION_CPU_READ = 2,        // CPU read attention state
    LLAMA_ATTENTION_STATE_VIOLATION_CPU_VALIDATION = 3,  // CPU validated attention
    LLAMA_ATTENTION_STATE_VIOLATION_STATE_ON_HOST = 4,   // State materialized on host
    LLAMA_ATTENTION_STATE_VIOLATION_MIXED_UPDATE = 5,    // Mixed CPU/GPU updates
    LLAMA_ATTENTION_STATE_VIOLATION_DESYNC = 6,          // CPU/GPU state desync
    LLAMA_ATTENTION_STATE_VIOLATION_HYBRID_PATH = 7,     // CPU/GPU hybrid attention
};

// ============================================================================
// PER-HEAD ATTENTION STATE RECORD
// ============================================================================

/**
 * Attention state for a single head (GPU-resident)
 */
struct llama_gpu_attention_head_state {
    uint32_t head_id;                  // Head identifier
    uint32_t query_dim;                // Query dimension
    uint32_t key_dim;                  // Key dimension
    uint32_t value_dim;                // Value dimension
    uint32_t seq_len;                  // Sequence length for this head
    uint32_t batch_size;               // Batch size for computation
    uint32_t reserved_1;
    uint32_t reserved_2;
};

// ============================================================================
// GPU ATTENTION STATE CONFIGURATION
// ============================================================================

/**
 * Configuration for GPU attention state management
 */
struct llama_gpu_attention_state_config {
    bool gpu_attention_state_enabled;   // Enable GPU attention state?
    bool cpu_attention_updates_forbidden; // Forbid CPU attention updates?
    enum llama_attention_state_mode mode; // Attention state mode
    uint32_t num_heads;                 // Number of attention heads
    uint32_t head_dim;                  // Dimension per head
    uint32_t num_layers;                // Number of layers
    bool validate_attention_bounds;     // Check bounds on each update?
    bool enforce_gpu_only_attention;    // Enforce GPU-only attention?
};

// ============================================================================
// GPU ATTENTION STATE RECORD
// ============================================================================

/**
 * Current state of GPU attention state management
 */
struct llama_gpu_attention_state_record {
    enum llama_attention_state_mode current_mode;        // Current mode
    enum llama_gpu_attention_state_status attention_state; // GPU attention state
    uint32_t num_heads;                                   // Number of heads
    uint32_t head_dim;                                    // Head dimension
    uint64_t state_updates_count;                         // GPU kernel updates
    uint64_t state_reads_count;                           // GPU state reads
    int total_violations;                                 // Total violations
    enum llama_attention_state_violation last_violation;  // Last violation
};

// ============================================================================
// ATTENTION COMPUTATION RECORD
// ============================================================================

/**
 * Record of a GPU attention computation
 */
struct llama_gpu_attention_computation_record {
    uint32_t sequence_length;          // Sequence processed
    uint32_t batch_size;               // Batch size
    uint32_t heads_computed;           // Heads computed on GPU
    uint64_t timestamp_ns;             // When computed
    bool computation_on_gpu;           // Was computation on GPU?
};

// ============================================================================
// ATTENTION STATE VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU attention state management
 */
struct llama_gpu_attention_state_validation_state {
    struct llama_gpu_attention_state_config config;
    struct llama_gpu_attention_state_record state_record;
    struct llama_gpu_attention_computation_record last_computation;
    int total_attention_computations;
    int total_violations;
    bool enforcement_strict;            // Abort on violation vs log only
    bool debug_attention_state;         // Debug output
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_attention_state_gpu_init(void);
int llama_attention_state_gpu_configure(
    bool gpu_attention_enabled,
    bool cpu_updates_forbidden,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t num_layers
);

// Attention state setup
int llama_attention_state_gpu_allocate_state_buffers(uint32_t num_heads, uint32_t head_dim);
int llama_attention_state_gpu_initialize_state(void);

// GPU attention state operations (10 enforcement points: 1-10)
int llama_attention_state_gpu_queue_attention_kernel(void);
int llama_attention_state_gpu_keep_state_on_device(void);
int llama_attention_state_gpu_compute_attention_on_gpu(uint32_t seq_len, uint32_t batch_size);
int llama_attention_state_gpu_store_attention_on_gpu(void);
int llama_attention_state_gpu_forbid_cpu_attention_update(void);
int llama_attention_state_gpu_forbid_cpu_attention_read(void);
int llama_attention_state_gpu_forbid_cpu_attention_validation(void);
int llama_attention_state_gpu_validate_attention_bounds(void);
int llama_attention_state_gpu_lock_state_to_gpu(void);
int llama_attention_state_gpu_verify_no_cpu_modification(void);

// State retrieval and synchronization
int llama_attention_state_gpu_read_state_sync(uint32_t* out_heads_computed);
int llama_attention_state_gpu_read_state_async(uint32_t* out_heads_computed);
int llama_attention_state_gpu_sync_state_to_cpu(void);

// Violation detection
int llama_attention_state_gpu_detect_cpu_update(void);
int llama_attention_state_gpu_detect_cpu_read(void);
int llama_attention_state_gpu_detect_cpu_validation(void);
int llama_attention_state_gpu_detect_state_on_host(void);
int llama_attention_state_gpu_detect_mixed_updates(void);
int llama_attention_state_gpu_detect_desync(void);
int llama_attention_state_gpu_detect_hybrid_path(void);

// State management
int llama_attention_state_gpu_set_allocated(void);
int llama_attention_state_gpu_set_initialized(void);
int llama_attention_state_gpu_set_decode_active(void);
int llama_attention_state_gpu_set_computed(void);
int llama_attention_state_gpu_set_stored(void);

// Query and verification functions
struct llama_gpu_attention_state_record llama_attention_state_gpu_get_state_record(void);
struct llama_gpu_attention_computation_record llama_attention_state_gpu_get_last_computation(void);
enum llama_gpu_attention_state_status llama_attention_state_gpu_get_state_status(void);

// Verification functions
int llama_attention_state_gpu_verify_cpu_updates_forbidden(void);
int llama_attention_state_gpu_verify_gpu_attention_active(void);
int llama_attention_state_gpu_verify_state_locked(void);
int llama_attention_state_gpu_verify_no_cpu_entry_point(void);
int llama_attention_state_gpu_verify_state_within_bounds(void);
int llama_attention_state_gpu_verify_no_desync(void);
int llama_attention_state_gpu_verify_no_hybrid_path(void);

// Diagnostics and logging
void llama_attention_state_gpu_log_attention_mode_enabled(void);
void llama_attention_state_gpu_log_state_locked(void);
void llama_attention_state_gpu_print_state(void);
void llama_attention_state_gpu_print_execution_stats(void);
void llama_attention_state_gpu_print_violation_summary(void);

// Violation reporting
void llama_attention_state_gpu_report_violation(
    enum llama_attention_state_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_attention_state_gpu_set_enforcement_strict(bool strict);
bool llama_attention_state_gpu_get_enforcement_strict(void);
void llama_attention_state_gpu_set_debug_output(bool debug);

// Self-test suite
int llama_attention_state_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_attention_state_mode_name(
    enum llama_attention_state_mode mode
) {
    switch (mode) {
        case LLAMA_ATTENTION_STATE_NONE: return "NONE";
        case LLAMA_ATTENTION_STATE_CPU: return "CPU";
        case LLAMA_ATTENTION_STATE_GPU: return "GPU";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_attention_state_status_name(
    enum llama_gpu_attention_state_status status
) {
    switch (status) {
        case LLAMA_GPU_ATTENTION_STATE_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_ATTENTION_STATE_ALLOCATED: return "ALLOCATED";
        case LLAMA_GPU_ATTENTION_STATE_INITIALIZED: return "INITIALIZED";
        case LLAMA_GPU_ATTENTION_STATE_DECODE_ACTIVE: return "DECODE_ACTIVE";
        case LLAMA_GPU_ATTENTION_STATE_COMPUTED: return "COMPUTED";
        case LLAMA_GPU_ATTENTION_STATE_STORED: return "STORED";
        case LLAMA_GPU_ATTENTION_STATE_SYNCED: return "SYNCED";
        case LLAMA_GPU_ATTENTION_STATE_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_attention_state_violation_name(
    enum llama_attention_state_violation violation
) {
    switch (violation) {
        case LLAMA_ATTENTION_STATE_VIOLATION_NONE: return "NONE";
        case LLAMA_ATTENTION_STATE_VIOLATION_CPU_UPDATE: return "CPU_UPDATE";
        case LLAMA_ATTENTION_STATE_VIOLATION_CPU_READ: return "CPU_READ";
        case LLAMA_ATTENTION_STATE_VIOLATION_CPU_VALIDATION: return "CPU_VALIDATION";
        case LLAMA_ATTENTION_STATE_VIOLATION_STATE_ON_HOST: return "STATE_ON_HOST";
        case LLAMA_ATTENTION_STATE_VIOLATION_MIXED_UPDATE: return "MIXED_UPDATE";
        case LLAMA_ATTENTION_STATE_VIOLATION_DESYNC: return "DESYNC";
        case LLAMA_ATTENTION_STATE_VIOLATION_HYBRID_PATH: return "HYBRID_PATH";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif

