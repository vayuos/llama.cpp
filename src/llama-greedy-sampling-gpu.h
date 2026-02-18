/**
 * SECTION 21: Move greedy argmax sampling to GPU
 * Header
 *
 * This file implements GPU-native greedy argmax sampling for deterministic token selection.
 * All greedy sampling (temperature = 0) becomes GPU-exclusive with no CPU involvement.
 * Logits remain GPU-resident; selected token stays on device until final commit.
 * CPU sampling entry points are completely bypassed for greedy mode.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// GREEDY SAMPLING MODE ENUMERATION
// ============================================================================

/**
 * Greedy sampling execution modes
 */
enum llama_greedy_sampling_mode {
    LLAMA_GREEDY_SAMPLING_NONE = 0,
    LLAMA_GREEDY_SAMPLING_DETERMINISTIC = 1,    // Temperature = 0, no filters
    LLAMA_GREEDY_SAMPLING_GPU_ARGMAX = 2,       // GPU-native argmax path
    LLAMA_GREEDY_SAMPLING_DEVICE_RESIDENT = 3,  // Token stays on GPU
};

// ============================================================================
// CPU SAMPLING BYPASS ENUMERATION
// ============================================================================

/**
 * CPU sampling operations that should be bypassed
 */
enum llama_cpu_sampling_bypass {
    LLAMA_SAMPLING_BYPASS_NONE = 0,
    LLAMA_SAMPLING_BYPASS_LOGIT_ITERATION = 1,      // Skip CPU logit loops
    LLAMA_SAMPLING_BYPASS_ARGMAX_COMPUTATION = 2,   // Skip CPU max-finding
    LLAMA_SAMPLING_BYPASS_PENALTY_APPLICATION = 3,  // Skip CPU penalties
    LLAMA_SAMPLING_BYPASS_LOGIT_BIAS = 4,           // Skip CPU bias
    LLAMA_SAMPLING_BYPASS_HOST_COPY = 5,            // Skip host logits copy
    LLAMA_SAMPLING_BYPASS_SYNCHRONIZATION = 6,      // Minimize CPU waits
    LLAMA_SAMPLING_BYPASS_ENTRY_POINT = 7,          // Redirect sampler call
};

// ============================================================================
// GPU ARGMAX KERNEL STATE ENUMERATION
// ============================================================================

/**
 * State of GPU argmax kernel execution
 */
enum llama_gpu_argmax_state {
    LLAMA_GPU_ARGMAX_UNINITIALIZED = 0,
    LLAMA_GPU_ARGMAX_KERNEL_QUEUED = 1,         // Kernel queued on stream
    LLAMA_GPU_ARGMAX_KERNEL_RUNNING = 2,        // Kernel executing
    LLAMA_GPU_ARGMAX_RESULT_READY = 3,          // Result computed on GPU
    LLAMA_GPU_ARGMAX_COPIED_TO_CPU = 4,         // Token copied to host
    LLAMA_GPU_ARGMAX_ERROR = 5,
};

// ============================================================================
// GREEDY SAMPLING VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of greedy GPU-exclusive sampling
 */
enum llama_greedy_sampling_violation {
    LLAMA_GREEDY_VIOLATION_NONE = 0,
    LLAMA_GREEDY_VIOLATION_CPU_ARGMAX = 1,              // CPU performed argmax
    LLAMA_GREEDY_VIOLATION_LOGITS_COPIED_HOST = 2,      // Logits copied to host
    LLAMA_GREEDY_VIOLATION_CPU_PENALTY_APPLIED = 3,     // CPU applied penalty
    LLAMA_GREEDY_VIOLATION_CPU_LOGIT_BIAS = 4,          // CPU applied bias
    LLAMA_GREEDY_VIOLATION_CPU_ENTRY_POINT = 5,         // CPU sampler called
    LLAMA_GREEDY_VIOLATION_SYNCHRONIZATION_BARRIER = 6, // Unnecessary sync
    LLAMA_GREEDY_VIOLATION_MIXED_PATH = 7,              // Mixed CPU/GPU path
};

// ============================================================================
// GREEDY SAMPLING CONFIGURATION RECORD
// ============================================================================

/**
 * Configuration for greedy sampling execution
 */
struct llama_greedy_sampling_config {
    bool is_greedy_mode;                        // Temperature = 0?
    bool all_filters_disabled;                  // No top-k, top-p?
    bool penalties_disabled;                    // Penalty-free?
    bool logit_bias_disabled;                   // No bias?
    bool gpu_argmax_enabled;                    // Use GPU argmax?
    bool device_resident_token;                 // Keep token on GPU?
    bool async_copy_token;                      // Async copy to host?
    enum llama_greedy_sampling_mode mode;       // Execution mode
};

// ============================================================================
// GREEDY SAMPLING EXECUTION RECORD
// ============================================================================

/**
 * Record of greedy sampling execution
 */
struct llama_greedy_sampling_execution_record {
    enum llama_greedy_sampling_mode mode;                // Execution mode
    enum llama_gpu_argmax_state argmax_state;           // Kernel state
    uint64_t timestamp_ns;                              // When sampled
    uint32_t token_id;                                  // Selected token
    bool token_on_device;                               // Location of token
    uint64_t gpu_kernel_ns;                             // GPU kernel time
    uint64_t cpu_copy_ns;                               // Copy time
    int cpu_violations;                                 // Violations detected
    enum llama_greedy_sampling_violation last_violation; // Last violation
};

// ============================================================================
// GREEDY SAMPLING STATE RECORD
// ============================================================================

/**
 * Global state of greedy sampling during decode
 */
struct llama_greedy_sampling_state_record {
    enum llama_greedy_sampling_mode current_mode;       // Current mode
    enum llama_gpu_argmax_state gpu_state;              // GPU state
    bool gpu_argmax_active;                             // GPU argmax enabled
    bool cpu_sampling_bypassed;                         // CPU sampling bypassed
    bool device_resident_mode;                          // Tokens stay on GPU
    int total_violations;                               // Total violations
    enum llama_greedy_sampling_violation last_violation; // Last violation
    uint64_t total_tokens_sampled;                      // GPU-sampled tokens
    uint64_t total_gpu_time_ns;                         // Cumulative GPU time
    uint64_t total_cpu_time_ns;                         // Cumulative CPU time
};

// ============================================================================
// GREEDY SAMPLING VALIDATION STATE
// ============================================================================

/**
 * Global validation state for greedy GPU sampling
 */
struct llama_greedy_sampling_gpu_validation_state {
    struct llama_greedy_sampling_config config;
    struct llama_greedy_sampling_state_record state_record;
    struct llama_greedy_sampling_execution_record last_execution;
    int total_greedy_samples;
    int total_violations;
    bool enforcement_strict;                    // Abort on violation vs log only
    bool debug_greedy_sampling;                 // Debug output
    bool verify_bitwise_identical;              // Verify output equivalence
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_greedy_sampling_gpu_init(void);
int llama_greedy_sampling_gpu_configure_greedy_mode(bool enable_gpu_argmax);

// Greedy sampling detection and routing
int llama_greedy_sampling_gpu_detect_greedy_config(
    float temperature,
    int top_k,
    float top_p,
    float penalty_repeat,
    float penalty_freq,
    float penalty_pres
);
int llama_greedy_sampling_gpu_should_use_gpu_argmax(void);

// GPU argmax execution (5 enforcement points: 1-5)
int llama_greedy_sampling_gpu_queue_argmax_kernel(void);
int llama_greedy_sampling_gpu_launch_argmax(void);
int llama_greedy_sampling_gpu_wait_argmax_result(void);
int llama_greedy_sampling_gpu_keep_token_on_device(void);
int llama_greedy_sampling_gpu_assert_gpu_argmax_complete(void);

// CPU bypass enforcement (3 enforcement points: 6-8)
int llama_greedy_sampling_gpu_forbid_cpu_logit_iteration(void);
int llama_greedy_sampling_gpu_forbid_cpu_sampling_entry(void);
int llama_greedy_sampling_gpu_eliminate_cpu_penalties(void);

// Synchronization control (2 enforcement points: 9-10)
int llama_greedy_sampling_gpu_minimize_synchronization(void);
int llama_greedy_sampling_gpu_async_copy_token_id(void);

// Violation detection
int llama_greedy_sampling_gpu_detect_cpu_argmax_attempt(void);
int llama_greedy_sampling_gpu_detect_logits_host_copy(void);
int llama_greedy_sampling_gpu_detect_cpu_penalty_application(void);
int llama_greedy_sampling_gpu_detect_cpu_logit_bias(void);
int llama_greedy_sampling_gpu_detect_cpu_sampler_call(void);
int llama_greedy_sampling_gpu_detect_synchronization_barrier(void);
int llama_greedy_sampling_gpu_detect_mixed_path(void);

// GPU state management
int llama_greedy_sampling_gpu_set_argmax_queued(void);
int llama_greedy_sampling_gpu_set_argmax_running(void);
int llama_greedy_sampling_gpu_set_result_ready(uint32_t token_id);
int llama_greedy_sampling_gpu_set_token_on_device(void);

// Query and verification functions
struct llama_greedy_sampling_state_record llama_greedy_sampling_gpu_get_state_record(void);
struct llama_greedy_sampling_execution_record llama_greedy_sampling_gpu_get_last_execution(void);
enum llama_greedy_sampling_mode llama_greedy_sampling_gpu_get_current_mode(void);
enum llama_gpu_argmax_state llama_greedy_sampling_gpu_get_argmax_state(void);

// Verification functions
int llama_greedy_sampling_gpu_verify_cpu_sampling_bypassed(void);
int llama_greedy_sampling_gpu_verify_gpu_argmax_active(void);
int llama_greedy_sampling_gpu_verify_device_resident_tokens(void);
int llama_greedy_sampling_gpu_verify_no_cpu_entry_point(void);
int llama_greedy_sampling_gpu_verify_minimal_synchronization(void);
int llama_greedy_sampling_gpu_verify_bitwise_identical_output(uint32_t cpu_token, uint32_t gpu_token);

// Diagnostics and logging
void llama_greedy_sampling_gpu_log_greedy_mode_enabled(void);
void llama_greedy_sampling_gpu_log_gpu_argmax_launched(void);
void llama_greedy_sampling_gpu_log_token_sampled_by_gpu(uint32_t token);
void llama_greedy_sampling_gpu_print_sampling_state(void);
void llama_greedy_sampling_gpu_print_execution_stats(void);
void llama_greedy_sampling_gpu_print_violation_summary(void);

// Violation reporting
void llama_greedy_sampling_gpu_report_violation(
    enum llama_greedy_sampling_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_greedy_sampling_gpu_set_enforcement_strict(bool strict);
bool llama_greedy_sampling_gpu_get_enforcement_strict(void);
void llama_greedy_sampling_gpu_set_debug_output(bool debug);
void llama_greedy_sampling_gpu_set_verify_bitwise(bool verify);

// Self-test suite
int llama_greedy_sampling_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_greedy_sampling_mode_name(
    enum llama_greedy_sampling_mode mode
) {
    switch (mode) {
        case LLAMA_GREEDY_SAMPLING_NONE: return "NONE";
        case LLAMA_GREEDY_SAMPLING_DETERMINISTIC: return "DETERMINISTIC";
        case LLAMA_GREEDY_SAMPLING_GPU_ARGMAX: return "GPU_ARGMAX";
        case LLAMA_GREEDY_SAMPLING_DEVICE_RESIDENT: return "DEVICE_RESIDENT";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_argmax_state_name(
    enum llama_gpu_argmax_state state
) {
    switch (state) {
        case LLAMA_GPU_ARGMAX_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_ARGMAX_KERNEL_QUEUED: return "KERNEL_QUEUED";
        case LLAMA_GPU_ARGMAX_KERNEL_RUNNING: return "KERNEL_RUNNING";
        case LLAMA_GPU_ARGMAX_RESULT_READY: return "RESULT_READY";
        case LLAMA_GPU_ARGMAX_COPIED_TO_CPU: return "COPIED_TO_CPU";
        case LLAMA_GPU_ARGMAX_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_greedy_sampling_violation_name(
    enum llama_greedy_sampling_violation violation
) {
    switch (violation) {
        case LLAMA_GREEDY_VIOLATION_NONE: return "NONE";
        case LLAMA_GREEDY_VIOLATION_CPU_ARGMAX: return "CPU_ARGMAX";
        case LLAMA_GREEDY_VIOLATION_LOGITS_COPIED_HOST: return "LOGITS_COPIED_HOST";
        case LLAMA_GREEDY_VIOLATION_CPU_PENALTY_APPLIED: return "CPU_PENALTY_APPLIED";
        case LLAMA_GREEDY_VIOLATION_CPU_LOGIT_BIAS: return "CPU_LOGIT_BIAS";
        case LLAMA_GREEDY_VIOLATION_CPU_ENTRY_POINT: return "CPU_ENTRY_POINT";
        case LLAMA_GREEDY_VIOLATION_SYNCHRONIZATION_BARRIER: return "SYNCHRONIZATION_BARRIER";
        case LLAMA_GREEDY_VIOLATION_MIXED_PATH: return "MIXED_PATH";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
