/**
 * SECTION 23: Move top-k filtering to GPU
 * Header
 *
 * This file implements GPU-native top-k selection for deterministic sampling.
 * All top-k filtering (selection, sorting, masking) becomes GPU-exclusive with no CPU involvement.
 * Logits remain GPU-resident; top-k filtering applied in-place in device memory.
 * Only selected token ID crosses PCIe; no candidate sets or logits arrays transferred to CPU.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TOP-K FILTERING MODE ENUMERATION
// ============================================================================

/**
 * Top-k filtering execution modes
 */
enum llama_topk_filtering_mode {
    LLAMA_TOPK_FILTERING_NONE = 0,
    LLAMA_TOPK_FILTERING_ENABLED = 1,       // Top-k filtering active
    LLAMA_TOPK_FILTERING_GPU_NATIVE = 2,    // GPU-native kernel path
    LLAMA_TOPK_FILTERING_FUSED = 3,         // Fused with penalty/temperature
};

// ============================================================================
// GPU TOP-K KERNEL STATE ENUMERATION
// ============================================================================

/**
 * State of GPU top-k kernel execution
 */
enum llama_gpu_topk_state {
    LLAMA_GPU_TOPK_UNINITIALIZED = 0,
    LLAMA_GPU_TOPK_KERNEL_QUEUED = 1,       // Kernel queued on stream
    LLAMA_GPU_TOPK_KERNEL_RUNNING = 2,      // Kernel executing
    LLAMA_GPU_TOPK_SELECTION_READY = 3,     // Top-k candidates selected
    LLAMA_GPU_TOPK_MASKED_LOGITS_READY = 4, // Logits masked/filtered
    LLAMA_GPU_TOPK_ERROR = 5,
};

// ============================================================================
// CPU TOP-K PATH BYPASS ENUMERATION
// ============================================================================

/**
 * CPU top-k operations that should be bypassed
 */
enum llama_cpu_topk_bypass {
    LLAMA_TOPK_BYPASS_NONE = 0,
    LLAMA_TOPK_BYPASS_PARTIAL_SORT = 1,    // Skip CPU partial sort
    LLAMA_TOPK_BYPASS_CANDIDATE_SELECTION = 2, // Skip CPU candidate selection
    LLAMA_TOPK_BYPASS_LOGITS_FILTERING = 3,    // Skip CPU logits filtering
    LLAMA_TOPK_BYPASS_LOGITS_MASKING = 4,      // Skip CPU logits masking
    LLAMA_TOPK_BYPASS_HOST_COPY = 5,           // Skip copy to host
    LLAMA_TOPK_BYPASS_ENTRY_POINT = 6,         // Skip CPU top-k entry point
};

// ============================================================================
// TOP-K VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-exclusive top-k filtering
 */
enum llama_topk_violation {
    LLAMA_TOPK_VIOLATION_NONE = 0,
    LLAMA_TOPK_VIOLATION_CPU_PARTIAL_SORT = 1,     // CPU performed partial sort
    LLAMA_TOPK_VIOLATION_CPU_CANDIDATE_SELECT = 2, // CPU selected candidates
    LLAMA_TOPK_VIOLATION_CPU_LOGITS_FILTERED = 3,  // CPU filtered logits
    LLAMA_TOPK_VIOLATION_CPU_LOGITS_MASKED = 4,    // CPU masked logits
    LLAMA_TOPK_VIOLATION_CANDIDATES_ON_HOST = 5,   // Candidates on host memory
    LLAMA_TOPK_VIOLATION_MIXED_PATH = 6,           // Mixed CPU/GPU filtering
};

// ============================================================================
// TOP-K BUFFER STATE ENUMERATION
// ============================================================================

/**
 * State of GPU top-k buffers
 */
enum llama_gpu_topk_buffer_state {
    LLAMA_GPU_TOPK_BUFFER_UNINITIALIZED = 0,
    LLAMA_GPU_TOPK_BUFFER_ALLOCATED = 1,   // Buffers allocated on GPU
    LLAMA_GPU_TOPK_BUFFER_POPULATED = 2,   // Buffers contain top-k data
    LLAMA_GPU_TOPK_BUFFER_ACTIVE = 3,      // Buffers actively used per token
    LLAMA_GPU_TOPK_BUFFER_ERROR = 4,
};

// ============================================================================
// TOP-K SELECTION KERNEL FUSION ENUMERATION
// ============================================================================

/**
 * Kernel fusion options for top-k with other operations
 */
enum llama_topk_kernel_fusion {
    LLAMA_TOPK_FUSION_NONE = 0,
    LLAMA_TOPK_FUSION_WITH_PENALTY = 1,     // Fused penalty + top-k
    LLAMA_TOPK_FUSION_WITH_TEMPERATURE = 2, // Fused temperature + top-k
    LLAMA_TOPK_FUSION_FULL_PIPELINE = 3,    // Penalty + temperature + top-k
};

// ============================================================================
// TOP-K CONFIGURATION RECORD
// ============================================================================

/**
 * Configuration for GPU top-k filtering execution
 */
struct llama_gpu_topk_config {
    bool topk_filtering_enabled;        // Enable top-k filtering?
    int topk_value;                     // Top-k value (k)
    bool gpu_topk_enabled;              // Use GPU top-k kernels?
    bool candidates_on_gpu;             // Keep candidates on GPU?
    enum llama_topk_filtering_mode mode; // Execution mode
    enum llama_topk_kernel_fusion fusion; // Kernel fusion strategy
    bool fused_penalty_temp_topk;       // Fuse penalty + temperature + top-k?
    bool use_partial_selection;         // Use partial selection (not full sort)?
};

// ============================================================================
// TOP-K EXECUTION RECORD
// ============================================================================

/**
 * Record of GPU top-k kernel execution
 */
struct llama_gpu_topk_execution_record {
    enum llama_topk_filtering_mode mode; // Filtering mode used
    enum llama_gpu_topk_state topk_state; // Kernel execution state
    enum llama_gpu_topk_buffer_state buffer_state; // Top-k buffer state
    uint64_t timestamp_ns;              // When top-k executed
    uint32_t tokens_processed;          // Tokens processed
    uint32_t topk_value_used;           // Actual k value used
    uint64_t gpu_kernel_ns;             // GPU kernel execution time
    uint64_t candidate_selection_ns;    // Candidate selection time
    int cpu_violations;                 // Violations detected
    enum llama_topk_violation last_violation; // Last violation type
};

// ============================================================================
// TOP-K STATE RECORD
// ============================================================================

/**
 * Global state of GPU top-k filtering during decode
 */
struct llama_gpu_topk_state_record {
    enum llama_topk_filtering_mode current_mode;  // Current filtering mode
    enum llama_gpu_topk_state gpu_topk_state;     // GPU top-k kernel state
    enum llama_gpu_topk_buffer_state buffer_state; // Top-k buffer state
    bool gpu_topk_active;                         // GPU top-k kernel active?
    bool cpu_topk_bypassed;                       // CPU top-k bypassed?
    bool candidates_gpu_resident;                 // Candidates on GPU?
    int total_violations;                         // Total violations
    enum llama_topk_violation last_violation;     // Last violation type
    uint64_t total_tokens_filtered;               // GPU-filtered token count
    uint64_t total_gpu_time_ns;                   // Cumulative GPU time
    uint64_t total_cpu_time_ns;                   // Cumulative CPU time
};

// ============================================================================
// TOP-K VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU top-k filtering
 */
struct llama_gpu_topk_validation_state {
    struct llama_gpu_topk_config config;
    struct llama_gpu_topk_state_record state_record;
    struct llama_gpu_topk_execution_record last_execution;
    int total_topk_applications;
    int total_violations;
    bool enforcement_strict;            // Abort on violation vs log only
    bool debug_topk_filtering;          // Debug output
    bool verify_bitwise_identical;      // Verify output equivalence
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_topk_gpu_init(void);
int llama_topk_gpu_configure_filtering(
    bool topk_enabled,
    int topk_value,
    enum llama_topk_kernel_fusion fusion_mode
);

// Top-k detection and routing
int llama_topk_gpu_detect_topk_config(int topk_value);
int llama_topk_gpu_should_use_gpu_topk(void);

// GPU top-k kernel execution (5 enforcement points: 1-5)
int llama_topk_gpu_queue_topk_kernel(void);
int llama_topk_gpu_launch_topk_kernel(void);
int llama_topk_gpu_wait_topk_result(void);
int llama_topk_gpu_keep_candidates_on_device(void);
int llama_topk_gpu_assert_topk_complete(void);

// Top-k buffer management (3 enforcement points: 6-8)
int llama_topk_gpu_allocate_topk_buffers(uint32_t max_vocab_size);
int llama_topk_gpu_populate_topk_buffers(void);
int llama_topk_gpu_forbid_cpu_candidate_iteration(void);

// CPU bypass enforcement (2 enforcement points: 9-10)
int llama_topk_gpu_forbid_cpu_topk_computation(void);
int llama_topk_gpu_forbid_cpu_topk_entry_point(void);

// Violation detection
int llama_topk_gpu_detect_cpu_partial_sort(void);
int llama_topk_gpu_detect_cpu_candidate_selection(void);
int llama_topk_gpu_detect_cpu_logits_filtering(void);
int llama_topk_gpu_detect_cpu_logits_masking(void);
int llama_topk_gpu_detect_candidates_on_host(void);
int llama_topk_gpu_detect_mixed_topk_path(void);

// GPU state management
int llama_topk_gpu_set_topk_queued(void);
int llama_topk_gpu_set_topk_running(void);
int llama_topk_gpu_set_selection_ready(void);
int llama_topk_gpu_set_masked_logits_ready(void);

// Query and verification functions
struct llama_gpu_topk_state_record llama_topk_gpu_get_state_record(void);
struct llama_gpu_topk_execution_record llama_topk_gpu_get_last_execution(void);
enum llama_topk_filtering_mode llama_topk_gpu_get_current_mode(void);
enum llama_gpu_topk_state llama_topk_gpu_get_topk_state(void);

// Verification functions
int llama_topk_gpu_verify_cpu_topk_bypassed(void);
int llama_topk_gpu_verify_gpu_topk_active(void);
int llama_topk_gpu_verify_candidates_on_gpu(void);
int llama_topk_gpu_verify_no_cpu_entry_point(void);
int llama_topk_gpu_verify_minimal_cpu_overhead(void);
int llama_topk_gpu_verify_bitwise_identical_output(uint32_t cpu_candidate, uint32_t gpu_candidate);
int llama_topk_gpu_verify_deterministic_stability(void);

// Diagnostics and logging
void llama_topk_gpu_log_topk_mode_enabled(void);
void llama_topk_gpu_log_topk_kernel_launched(void);
void llama_topk_gpu_log_candidates_selected(uint32_t num_candidates);
void llama_topk_gpu_print_topk_state(void);
void llama_topk_gpu_print_execution_stats(void);
void llama_topk_gpu_print_violation_summary(void);

// Violation reporting
void llama_topk_gpu_report_violation(
    enum llama_topk_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_topk_gpu_set_enforcement_strict(bool strict);
bool llama_topk_gpu_get_enforcement_strict(void);
void llama_topk_gpu_set_debug_output(bool debug);
void llama_topk_gpu_set_verify_bitwise(bool verify);

// Self-test suite
int llama_topk_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_topk_filtering_mode_name(
    enum llama_topk_filtering_mode mode
) {
    switch (mode) {
        case LLAMA_TOPK_FILTERING_NONE: return "NONE";
        case LLAMA_TOPK_FILTERING_ENABLED: return "ENABLED";
        case LLAMA_TOPK_FILTERING_GPU_NATIVE: return "GPU_NATIVE";
        case LLAMA_TOPK_FILTERING_FUSED: return "FUSED";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_topk_state_name(
    enum llama_gpu_topk_state state
) {
    switch (state) {
        case LLAMA_GPU_TOPK_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_TOPK_KERNEL_QUEUED: return "KERNEL_QUEUED";
        case LLAMA_GPU_TOPK_KERNEL_RUNNING: return "KERNEL_RUNNING";
        case LLAMA_GPU_TOPK_SELECTION_READY: return "SELECTION_READY";
        case LLAMA_GPU_TOPK_MASKED_LOGITS_READY: return "MASKED_LOGITS_READY";
        case LLAMA_GPU_TOPK_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_topk_violation_name(
    enum llama_topk_violation violation
) {
    switch (violation) {
        case LLAMA_TOPK_VIOLATION_NONE: return "NONE";
        case LLAMA_TOPK_VIOLATION_CPU_PARTIAL_SORT: return "CPU_PARTIAL_SORT";
        case LLAMA_TOPK_VIOLATION_CPU_CANDIDATE_SELECT: return "CPU_CANDIDATE_SELECT";
        case LLAMA_TOPK_VIOLATION_CPU_LOGITS_FILTERED: return "CPU_LOGITS_FILTERED";
        case LLAMA_TOPK_VIOLATION_CPU_LOGITS_MASKED: return "CPU_LOGITS_MASKED";
        case LLAMA_TOPK_VIOLATION_CANDIDATES_ON_HOST: return "CANDIDATES_ON_HOST";
        case LLAMA_TOPK_VIOLATION_MIXED_PATH: return "MIXED_PATH";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
