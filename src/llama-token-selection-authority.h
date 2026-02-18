/**
 * SECTION 26: Enforce GPU-Only Token Selection Authority
 * Header
 *
 * This file implements GPU-exclusive token selection authority for deterministic decode.
 * All token selection logic (sampling, penalties, filtering) becomes GPU-exclusive with no CPU involvement.
 * Logits and sampling remain GPU-resident; only finalized token ID crosses PCIe.
 * CPU observes committed token only after GPU-atomic token commit is complete.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TOKEN SELECTION MODE ENUMERATION
// ============================================================================

/**
 * Token selection execution modes
 */
enum llama_token_selection_mode {
    LLAMA_TOKEN_SELECTION_NONE = 0,
    LLAMA_TOKEN_SELECTION_CPU = 1,          // CPU performs token selection (deprecated)
    LLAMA_TOKEN_SELECTION_GPU_NATIVE = 2,   // GPU-native token selection
    LLAMA_TOKEN_SELECTION_GPU_FUSED = 3,    // Fused GPU sampling pipeline
};

// ============================================================================
// GPU TOKEN SELECTION KERNEL STATE ENUMERATION
// ============================================================================

/**
 * State of GPU token selection kernel execution
 */
enum llama_gpu_token_selection_state {
    LLAMA_GPU_TOKEN_SELECTION_UNINITIALIZED = 0,
    LLAMA_GPU_TOKEN_SELECTION_LOGITS_READY = 1,      // Logits prepared on GPU
    LLAMA_GPU_TOKEN_SELECTION_PENALTIES_APPLIED = 2, // Penalties computed on GPU
    LLAMA_GPU_TOKEN_SELECTION_FILTERED = 3,          // Top-k/top-p filtering done
    LLAMA_GPU_TOKEN_SELECTION_SAMPLED = 4,           // Token sampled on GPU
    LLAMA_GPU_TOKEN_SELECTION_COMMITTED = 5,         // Token committed to decode state
    LLAMA_GPU_TOKEN_SELECTION_ERROR = 6,
};

// ============================================================================
// CPU SAMPLING PATH BYPASS ENUMERATION
// ============================================================================

/**
 * CPU sampling operations that must be bypassed
 */
enum llama_cpu_sampling_bypass {
    LLAMA_SAMPLING_BYPASS_NONE = 0,
    LLAMA_SAMPLING_BYPASS_LOGITS_READ = 1,      // Skip CPU logits read
    LLAMA_SAMPLING_BYPASS_PENALTY_COMPUTATION = 2, // Skip CPU penalties
    LLAMA_SAMPLING_BYPASS_FILTERING = 3,        // Skip CPU top-k/top-p
    LLAMA_SAMPLING_BYPASS_SAMPLING = 4,         // Skip CPU sampling
    LLAMA_SAMPLING_BYPASS_VALIDATION = 5,       // Skip CPU token validation
    LLAMA_SAMPLING_BYPASS_ENTRY_POINT = 6,      // Skip CPU sampling entry
};

// ============================================================================
// TOKEN SELECTION VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-exclusive token selection authority
 */
enum llama_token_selection_violation {
    LLAMA_TOKEN_SELECTION_VIOLATION_NONE = 0,
    LLAMA_TOKEN_SELECTION_VIOLATION_CPU_SAMPLING = 1,      // CPU performed sampling
    LLAMA_TOKEN_SELECTION_VIOLATION_CPU_LOGITS_READ = 2,   // CPU read logits
    LLAMA_TOKEN_SELECTION_VIOLATION_CPU_PENALTIES = 3,     // CPU applied penalties
    LLAMA_TOKEN_SELECTION_VIOLATION_CPU_FILTERING = 4,     // CPU performed filtering
    LLAMA_TOKEN_SELECTION_VIOLATION_CPU_VALIDATION = 5,    // CPU validated token
    LLAMA_TOKEN_SELECTION_VIOLATION_MIXED_PATH = 6,        // Mixed CPU/GPU selection
    LLAMA_TOKEN_SELECTION_VIOLATION_UNCOMMITTED_TOKEN = 7, // Token not committed to GPU state
};

// ============================================================================
// GPU TOKEN COMMITMENT STATE ENUMERATION
// ============================================================================

/**
 * State of GPU-atomic token commit operation
 */
enum llama_gpu_token_commit_state {
    LLAMA_GPU_TOKEN_COMMIT_UNINITIALIZED = 0,
    LLAMA_GPU_TOKEN_COMMIT_PENDING = 1,         // Token selected, awaiting commit
    LLAMA_GPU_TOKEN_COMMIT_WRITTEN = 2,         // Token written to GPU state
    LLAMA_GPU_TOKEN_COMMIT_KV_ADVANCED = 3,     // KV cache state advanced
    LLAMA_GPU_TOKEN_COMMIT_COMPLETE = 4,        // Full commit sequence done
    LLAMA_GPU_TOKEN_COMMIT_ERROR = 5,
};

// ============================================================================
// CPU SAMPLING AUTHORITY ENUMERATION
// ============================================================================

/**
 * Authority for sampling decisions during decode
 */
enum llama_sampling_authority {
    LLAMA_SAMPLING_AUTHORITY_UNINITIALIZED = 0,
    LLAMA_SAMPLING_AUTHORITY_CPU = 1,           // CPU has sampling authority (deprecated)
    LLAMA_SAMPLING_AUTHORITY_GPU = 2,           // GPU has sampling authority
    LLAMA_SAMPLING_AUTHORITY_LOCKED = 3,        // Authority locked to GPU, immutable
};

// ============================================================================
// TOKEN SELECTION CONFIGURATION RECORD
// ============================================================================

/**
 * Configuration for GPU token selection authority
 */
struct llama_gpu_token_selection_config {
    bool token_selection_gpu_enabled;       // Enable GPU token selection?
    bool cpu_sampling_forbidden;            // Forbid CPU sampling during decode?
    enum llama_token_selection_mode mode;   // Execution mode
    enum llama_sampling_authority authority; // Sampling authority
    bool fused_sampling_pipeline;           // Fuse all sampling ops into one kernel?
    bool enforce_gpu_atomic_commit;         // Require GPU-atomic token commit?
    bool use_deterministic_rng;             // Deterministic RNG for stochastic sampling?
    bool validate_gpu_token_authority;      // Verify GPU selected token?
};

// ============================================================================
// TOKEN SELECTION EXECUTION RECORD
// ============================================================================

/**
 * Record of GPU token selection kernel execution
 */
struct llama_gpu_token_selection_execution_record {
    enum llama_token_selection_mode mode;   // Selection mode used
    enum llama_gpu_token_selection_state selection_state; // Kernel execution state
    enum llama_gpu_token_commit_state commit_state; // Token commit state
    uint64_t timestamp_ns;                  // When token selection executed
    uint32_t tokens_processed;              // Tokens selected via GPU
    uint32_t token_selected;                // Last selected token ID
    uint64_t gpu_sampling_ns;               // GPU sampling kernel time
    uint64_t gpu_commit_ns;                 // GPU commit operation time
    int cpu_violations;                     // Violations detected
    enum llama_token_selection_violation last_violation; // Last violation type
};

// ============================================================================
// TOKEN SELECTION STATE RECORD
// ============================================================================

/**
 * Global state of GPU token selection authority during decode
 */
struct llama_gpu_token_selection_state_record {
    enum llama_token_selection_mode current_mode;      // Current selection mode
    enum llama_gpu_token_selection_state selection_state; // GPU selection state
    enum llama_gpu_token_commit_state commit_state;    // Token commit state
    enum llama_sampling_authority current_authority;   // Current sampling authority
    bool gpu_token_selection_active;                   // GPU selection active?
    bool cpu_sampling_bypassed;                        // CPU sampling bypassed?
    bool sampling_authority_locked;                    // Authority locked to GPU?
    int total_violations;                              // Total violations
    enum llama_token_selection_violation last_violation; // Last violation type
    uint64_t total_tokens_selected;                    // GPU-selected token count
    uint64_t total_gpu_time_ns;                        // Cumulative GPU time
    uint64_t total_cpu_time_ns;                        // Cumulative CPU time
};

// ============================================================================
// TOKEN SELECTION VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU token selection authority
 */
struct llama_gpu_token_selection_validation_state {
    struct llama_gpu_token_selection_config config;
    struct llama_gpu_token_selection_state_record state_record;
    struct llama_gpu_token_selection_execution_record last_execution;
    int total_selections;
    int total_violations;
    bool enforcement_strict;                // Abort on violation vs log only
    bool debug_token_selection;             // Debug output
    bool verify_bitwise_identical;          // Verify output equivalence
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_token_selection_gpu_init(void);
int llama_token_selection_gpu_configure(
    bool gpu_token_selection_enabled,
    bool cpu_sampling_forbidden,
    enum llama_sampling_authority authority
);

// Token selection detection and routing
int llama_token_selection_gpu_detect_mode(void);
int llama_token_selection_gpu_should_use_gpu_selection(void);

// GPU token selection enforcement (10 enforcement points: 1-10)
int llama_token_selection_gpu_queue_sampling_kernel(void);
int llama_token_selection_gpu_prepare_logits_on_gpu(void);
int llama_token_selection_gpu_apply_penalties_on_gpu(void);
int llama_token_selection_gpu_filter_candidates_on_gpu(void);
int llama_token_selection_gpu_perform_sampling(void);
int llama_token_selection_gpu_write_token_to_state(uint32_t token_id);
int llama_token_selection_gpu_advance_kv_cache_state(void);
int llama_token_selection_gpu_commit_token_atomic(uint32_t token_id);
int llama_token_selection_gpu_verify_gpu_authority(void);
int llama_token_selection_gpu_forbid_cpu_sampling(void);

// CPU sampling authority management
int llama_token_selection_gpu_lock_authority_to_gpu(void);
int llama_token_selection_gpu_get_sampling_authority(void);
int llama_token_selection_gpu_disable_cpu_sampling_path(void);

// Violation detection
int llama_token_selection_gpu_detect_cpu_sampling(void);
int llama_token_selection_gpu_detect_cpu_logits_read(void);
int llama_token_selection_gpu_detect_cpu_penalties(void);
int llama_token_selection_gpu_detect_cpu_filtering(void);
int llama_token_selection_gpu_detect_cpu_validation(void);
int llama_token_selection_gpu_detect_mixed_path(void);
int llama_token_selection_gpu_detect_uncommitted_token(void);

// GPU state management
int llama_token_selection_gpu_set_logits_ready(void);
int llama_token_selection_gpu_set_penalties_applied(void);
int llama_token_selection_gpu_set_filtered(void);
int llama_token_selection_gpu_set_sampled(void);
int llama_token_selection_gpu_set_committed(void);

// Query and verification functions
struct llama_gpu_token_selection_state_record llama_token_selection_gpu_get_state_record(void);
struct llama_gpu_token_selection_execution_record llama_token_selection_gpu_get_last_execution(void);
enum llama_token_selection_mode llama_token_selection_gpu_get_current_mode(void);
enum llama_gpu_token_selection_state llama_token_selection_gpu_get_selection_state(void);

// Verification functions
int llama_token_selection_gpu_verify_cpu_sampling_bypassed(void);
int llama_token_selection_gpu_verify_gpu_selection_active(void);
int llama_token_selection_gpu_verify_authority_locked(void);
int llama_token_selection_gpu_verify_no_cpu_entry_point(void);
int llama_token_selection_gpu_verify_minimal_cpu_overhead(void);
int llama_token_selection_gpu_verify_token_committed(uint32_t token_id);
int llama_token_selection_gpu_verify_bitwise_identical_output(uint32_t cpu_token, uint32_t gpu_token);
int llama_token_selection_gpu_verify_deterministic_stability(void);

// Diagnostics and logging
void llama_token_selection_gpu_log_selection_mode_enabled(void);
void llama_token_selection_gpu_log_authority_locked(void);
void llama_token_selection_gpu_log_token_selected(uint32_t token_id);
void llama_token_selection_gpu_print_state(void);
void llama_token_selection_gpu_print_execution_stats(void);
void llama_token_selection_gpu_print_violation_summary(void);

// Violation reporting
void llama_token_selection_gpu_report_violation(
    enum llama_token_selection_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_token_selection_gpu_set_enforcement_strict(bool strict);
bool llama_token_selection_gpu_get_enforcement_strict(void);
void llama_token_selection_gpu_set_debug_output(bool debug);
void llama_token_selection_gpu_set_verify_bitwise(bool verify);

// Self-test suite
int llama_token_selection_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_token_selection_mode_name(
    enum llama_token_selection_mode mode
) {
    switch (mode) {
        case LLAMA_TOKEN_SELECTION_NONE: return "NONE";
        case LLAMA_TOKEN_SELECTION_CPU: return "CPU";
        case LLAMA_TOKEN_SELECTION_GPU_NATIVE: return "GPU_NATIVE";
        case LLAMA_TOKEN_SELECTION_GPU_FUSED: return "GPU_FUSED";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_token_selection_state_name(
    enum llama_gpu_token_selection_state state
) {
    switch (state) {
        case LLAMA_GPU_TOKEN_SELECTION_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_TOKEN_SELECTION_LOGITS_READY: return "LOGITS_READY";
        case LLAMA_GPU_TOKEN_SELECTION_PENALTIES_APPLIED: return "PENALTIES_APPLIED";
        case LLAMA_GPU_TOKEN_SELECTION_FILTERED: return "FILTERED";
        case LLAMA_GPU_TOKEN_SELECTION_SAMPLED: return "SAMPLED";
        case LLAMA_GPU_TOKEN_SELECTION_COMMITTED: return "COMMITTED";
        case LLAMA_GPU_TOKEN_SELECTION_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_token_selection_violation_name(
    enum llama_token_selection_violation violation
) {
    switch (violation) {
        case LLAMA_TOKEN_SELECTION_VIOLATION_NONE: return "NONE";
        case LLAMA_TOKEN_SELECTION_VIOLATION_CPU_SAMPLING: return "CPU_SAMPLING";
        case LLAMA_TOKEN_SELECTION_VIOLATION_CPU_LOGITS_READ: return "CPU_LOGITS_READ";
        case LLAMA_TOKEN_SELECTION_VIOLATION_CPU_PENALTIES: return "CPU_PENALTIES";
        case LLAMA_TOKEN_SELECTION_VIOLATION_CPU_FILTERING: return "CPU_FILTERING";
        case LLAMA_TOKEN_SELECTION_VIOLATION_CPU_VALIDATION: return "CPU_VALIDATION";
        case LLAMA_TOKEN_SELECTION_VIOLATION_MIXED_PATH: return "MIXED_PATH";
        case LLAMA_TOKEN_SELECTION_VIOLATION_UNCOMMITTED_TOKEN: return "UNCOMMITTED_TOKEN";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_sampling_authority_name(
    enum llama_sampling_authority authority
) {
    switch (authority) {
        case LLAMA_SAMPLING_AUTHORITY_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_SAMPLING_AUTHORITY_CPU: return "CPU";
        case LLAMA_SAMPLING_AUTHORITY_GPU: return "GPU";
        case LLAMA_SAMPLING_AUTHORITY_LOCKED: return "LOCKED";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
