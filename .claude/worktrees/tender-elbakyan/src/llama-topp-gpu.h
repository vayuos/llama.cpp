/**
 * SECTION 24: Move top-p (Nucleus) filtering to GPU
 * Header
 *
 * This file implements GPU-native top-p (nucleus) filtering for deterministic sampling.
 * All top-p filtering (softmax, sorting, cumulative sum, masking) becomes GPU-exclusive with no CPU involvement.
 * Logits remain GPU-resident; top-p filtering applied in-place in device memory.
 * Only selected token ID crosses PCIe; no probability arrays or candidate sets transferred to CPU.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TOP-P FILTERING MODE ENUMERATION
// ============================================================================

/**
 * Top-p (nucleus) filtering execution modes
 */
enum llama_topp_filtering_mode {
    LLAMA_TOPP_FILTERING_NONE = 0,
    LLAMA_TOPP_FILTERING_ENABLED = 1,       // Top-p filtering active
    LLAMA_TOPP_FILTERING_GPU_NATIVE = 2,    // GPU-native kernel path
    LLAMA_TOPP_FILTERING_FUSED = 3,         // Fused with penalty/temperature/top-k
};

// ============================================================================
// GPU TOP-P KERNEL STATE ENUMERATION
// ============================================================================

/**
 * State of GPU top-p kernel execution
 */
enum llama_gpu_topp_state {
    LLAMA_GPU_TOPP_UNINITIALIZED = 0,
    LLAMA_GPU_TOPP_SOFTMAX_COMPUTED = 1,    // Softmax computed on GPU
    LLAMA_GPU_TOPP_SORTED = 2,              // Candidates sorted on GPU
    LLAMA_GPU_TOPP_CUMSUM_READY = 3,        // Cumulative sum computed
    LLAMA_GPU_TOPP_CANDIDATES_MASKED = 4,   // Candidates masked to nucleus set
    LLAMA_GPU_TOPP_READY_FOR_SAMPLING = 5,  // Ready for final selection
    LLAMA_GPU_TOPP_ERROR = 6,
};

// ============================================================================
// CPU TOP-P PATH BYPASS ENUMERATION
// ============================================================================

/**
 * CPU top-p operations that should be bypassed
 */
enum llama_cpu_topp_bypass {
    LLAMA_TOPP_BYPASS_NONE = 0,
    LLAMA_TOPP_BYPASS_SOFTMAX = 1,          // Skip CPU softmax
    LLAMA_TOPP_BYPASS_SORTING = 2,          // Skip CPU sorting
    LLAMA_TOPP_BYPASS_CUMSUM = 3,           // Skip CPU cumulative sum
    LLAMA_TOPP_BYPASS_MASKING = 4,          // Skip CPU candidate masking
    LLAMA_TOPP_BYPASS_PROBABILITIES_COPY = 5, // Skip probability copy to host
    LLAMA_TOPP_BYPASS_ENTRY_POINT = 6,      // Skip CPU top-p entry point
};

// ============================================================================
// TOP-P VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-exclusive top-p filtering
 */
enum llama_topp_violation {
    LLAMA_TOPP_VIOLATION_NONE = 0,
    LLAMA_TOPP_VIOLATION_CPU_SOFTMAX = 1,       // CPU computed softmax
    LLAMA_TOPP_VIOLATION_CPU_SORTING = 2,       // CPU performed sorting
    LLAMA_TOPP_VIOLATION_CPU_CUMSUM = 3,        // CPU computed cumulative sum
    LLAMA_TOPP_VIOLATION_CPU_MASKING = 4,       // CPU masked candidates
    LLAMA_TOPP_VIOLATION_PROBABILITIES_ON_HOST = 5, // Probabilities on host
    LLAMA_TOPP_VIOLATION_MIXED_PATH = 6,        // Mixed CPU/GPU filtering
};

// ============================================================================
// GPU TOP-P SORTING STRATEGY ENUMERATION
// ============================================================================

/**
 * Sorting strategies for GPU top-p
 */
enum llama_gpu_topp_sort_strategy {
    LLAMA_TOPP_SORT_NONE = 0,
    LLAMA_TOPP_SORT_PARTIAL_RADIX = 1,      // Partial radix sort
    LLAMA_TOPP_SORT_BITONIC_BLOCK = 2,      // Bitonic block sort
    LLAMA_TOPP_SORT_WARP_SELECTION = 3,     // Warp-level selection
    LLAMA_TOPP_SORT_HYBRID_PREFILTER = 4,   // Hybrid top-k + top-p
};

// ============================================================================
// GPU CUMSUM STATE ENUMERATION
// ============================================================================

/**
 * State of GPU parallel scan (prefix sum)
 */
enum llama_gpu_cumsum_state {
    LLAMA_GPU_CUMSUM_UNINITIALIZED = 0,
    LLAMA_GPU_CUMSUM_BLOCK_SCAN = 1,        // Block-level scan done
    LLAMA_GPU_CUMSUM_GLOBAL_READY = 2,      // Global cumulative sum ready
    LLAMA_GPU_CUMSUM_CUTOFF_DETECTED = 3,   // Nucleus cutoff detected
    LLAMA_GPU_CUMSUM_ERROR = 4,
};

// ============================================================================
// TOP-P CONFIGURATION RECORD
// ============================================================================

/**
 * Configuration for GPU top-p filtering execution
 */
struct llama_gpu_topp_config {
    bool topp_filtering_enabled;            // Enable top-p filtering?
    float topp_value;                       // Top-p value (p)
    bool gpu_topp_enabled;                  // Use GPU top-p kernels?
    bool probabilities_on_gpu;              // Keep probabilities on GPU?
    enum llama_topp_filtering_mode mode;    // Execution mode
    enum llama_gpu_topp_sort_strategy sort_strategy; // Sorting strategy
    bool fused_softmax_cumsum;              // Fuse softmax + cumsum?
    bool fused_full_pipeline;               // Fuse penalty + temp + top-k + top-p?
    bool use_deterministic_cumsum;          // Deterministic cumsum?
};

// ============================================================================
// TOP-P EXECUTION RECORD
// ============================================================================

/**
 * Record of GPU top-p kernel execution
 */
struct llama_gpu_topp_execution_record {
    enum llama_topp_filtering_mode mode;    // Filtering mode used
    enum llama_gpu_topp_state topp_state;   // Kernel execution state
    enum llama_gpu_cumsum_state cumsum_state; // Cumulative sum state
    uint64_t timestamp_ns;                  // When top-p executed
    uint32_t tokens_processed;              // Tokens processed
    float topp_value_used;                  // Actual p value used
    uint32_t nucleus_size;                  // Size of nucleus set
    uint64_t gpu_softmax_ns;                // GPU softmax time
    uint64_t gpu_sort_ns;                   // GPU sort time
    uint64_t gpu_cumsum_ns;                 // GPU cumsum time
    int cpu_violations;                     // Violations detected
    enum llama_topp_violation last_violation; // Last violation type
};

// ============================================================================
// TOP-P STATE RECORD
// ============================================================================

/**
 * Global state of GPU top-p filtering during decode
 */
struct llama_gpu_topp_state_record {
    enum llama_topp_filtering_mode current_mode;  // Current filtering mode
    enum llama_gpu_topp_state gpu_topp_state;     // GPU top-p kernel state
    enum llama_gpu_cumsum_state cumsum_state;     // Cumulative sum state
    bool gpu_topp_active;                         // GPU top-p kernel active?
    bool cpu_topp_bypassed;                       // CPU top-p bypassed?
    bool probabilities_gpu_resident;              // Probabilities on GPU?
    int total_violations;                         // Total violations
    enum llama_topp_violation last_violation;     // Last violation type
    uint64_t total_tokens_filtered;               // GPU-filtered token count
    uint64_t total_gpu_time_ns;                   // Cumulative GPU time
    uint64_t total_cpu_time_ns;                   // Cumulative CPU time
};

// ============================================================================
// TOP-P VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU top-p filtering
 */
struct llama_gpu_topp_validation_state {
    struct llama_gpu_topp_config config;
    struct llama_gpu_topp_state_record state_record;
    struct llama_gpu_topp_execution_record last_execution;
    int total_topp_applications;
    int total_violations;
    bool enforcement_strict;                // Abort on violation vs log only
    bool debug_topp_filtering;              // Debug output
    bool verify_bitwise_identical;          // Verify output equivalence
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_topp_gpu_init(void);
int llama_topp_gpu_configure_filtering(
    bool topp_enabled,
    float topp_value,
    enum llama_gpu_topp_sort_strategy sort_strategy
);

// Top-p detection and routing
int llama_topp_gpu_detect_topp_config(float topp_value);
int llama_topp_gpu_should_use_gpu_topp(void);

// GPU top-p kernel execution (5 enforcement points: 1-5)
int llama_topp_gpu_queue_softmax_kernel(void);
int llama_topp_gpu_compute_softmax(void);
int llama_topp_gpu_compute_cumulative_sum(void);
int llama_topp_gpu_detect_nucleus_cutoff(void);
int llama_topp_gpu_mask_nucleus_candidates(void);

// Sorting and ordering (2 enforcement points: 6-7)
int llama_topp_gpu_sort_candidates(void);
int llama_topp_gpu_forbid_cpu_sorting(void);

// CPU bypass enforcement (3 enforcement points: 8-10)
int llama_topp_gpu_forbid_cpu_softmax(void);
int llama_topp_gpu_forbid_cpu_cumsum(void);
int llama_topp_gpu_forbid_cpu_topp_entry_point(void);

// Violation detection
int llama_topp_gpu_detect_cpu_softmax(void);
int llama_topp_gpu_detect_cpu_sorting(void);
int llama_topp_gpu_detect_cpu_cumsum(void);
int llama_topp_gpu_detect_cpu_masking(void);
int llama_topp_gpu_detect_probabilities_on_host(void);
int llama_topp_gpu_detect_mixed_topp_path(void);

// GPU state management
int llama_topp_gpu_set_softmax_computed(void);
int llama_topp_gpu_set_sorted(void);
int llama_topp_gpu_set_cumsum_ready(void);
int llama_topp_gpu_set_masked_ready(void);

// Query and verification functions
struct llama_gpu_topp_state_record llama_topp_gpu_get_state_record(void);
struct llama_gpu_topp_execution_record llama_topp_gpu_get_last_execution(void);
enum llama_topp_filtering_mode llama_topp_gpu_get_current_mode(void);
enum llama_gpu_topp_state llama_topp_gpu_get_topp_state(void);

// Verification functions
int llama_topp_gpu_verify_cpu_topp_bypassed(void);
int llama_topp_gpu_verify_gpu_topp_active(void);
int llama_topp_gpu_verify_probabilities_on_gpu(void);
int llama_topp_gpu_verify_no_cpu_entry_point(void);
int llama_topp_gpu_verify_minimal_cpu_overhead(void);
int llama_topp_gpu_verify_bitwise_identical_output(uint32_t cpu_token, uint32_t gpu_token);
int llama_topp_gpu_verify_deterministic_stability(void);

// Diagnostics and logging
void llama_topp_gpu_log_topp_mode_enabled(void);
void llama_topp_gpu_log_softmax_computed(void);
void llama_topp_gpu_log_nucleus_set_size(uint32_t nucleus_size);
void llama_topp_gpu_print_topp_state(void);
void llama_topp_gpu_print_execution_stats(void);
void llama_topp_gpu_print_violation_summary(void);

// Violation reporting
void llama_topp_gpu_report_violation(
    enum llama_topp_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_topp_gpu_set_enforcement_strict(bool strict);
bool llama_topp_gpu_get_enforcement_strict(void);
void llama_topp_gpu_set_debug_output(bool debug);
void llama_topp_gpu_set_verify_bitwise(bool verify);

// Self-test suite
int llama_topp_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_topp_filtering_mode_name(
    enum llama_topp_filtering_mode mode
) {
    switch (mode) {
        case LLAMA_TOPP_FILTERING_NONE: return "NONE";
        case LLAMA_TOPP_FILTERING_ENABLED: return "ENABLED";
        case LLAMA_TOPP_FILTERING_GPU_NATIVE: return "GPU_NATIVE";
        case LLAMA_TOPP_FILTERING_FUSED: return "FUSED";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_topp_state_name(
    enum llama_gpu_topp_state state
) {
    switch (state) {
        case LLAMA_GPU_TOPP_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_TOPP_SOFTMAX_COMPUTED: return "SOFTMAX_COMPUTED";
        case LLAMA_GPU_TOPP_SORTED: return "SORTED";
        case LLAMA_GPU_TOPP_CUMSUM_READY: return "CUMSUM_READY";
        case LLAMA_GPU_TOPP_CANDIDATES_MASKED: return "CANDIDATES_MASKED";
        case LLAMA_GPU_TOPP_READY_FOR_SAMPLING: return "READY_FOR_SAMPLING";
        case LLAMA_GPU_TOPP_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_topp_violation_name(
    enum llama_topp_violation violation
) {
    switch (violation) {
        case LLAMA_TOPP_VIOLATION_NONE: return "NONE";
        case LLAMA_TOPP_VIOLATION_CPU_SOFTMAX: return "CPU_SOFTMAX";
        case LLAMA_TOPP_VIOLATION_CPU_SORTING: return "CPU_SORTING";
        case LLAMA_TOPP_VIOLATION_CPU_CUMSUM: return "CPU_CUMSUM";
        case LLAMA_TOPP_VIOLATION_CPU_MASKING: return "CPU_MASKING";
        case LLAMA_TOPP_VIOLATION_PROBABILITIES_ON_HOST: return "PROBABILITIES_ON_HOST";
        case LLAMA_TOPP_VIOLATION_MIXED_PATH: return "MIXED_PATH";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
