/**
 * SECTION 22: Move penalty application to GPU
 * Header
 *
 * This file implements GPU-native penalty kernels for deterministic token selection.
 * All penalty application (repeat, frequency, presence) becomes GPU-exclusive with no CPU involvement.
 * Logits remain GPU-resident; penalties applied in-place in device memory.
 * Token history stays GPU-resident; CPU has no access to penalty computation.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// PENALTY TYPE ENUMERATION
// ============================================================================

/**
 * Types of penalties applied during sampling
 */
enum llama_penalty_type {
    LLAMA_PENALTY_TYPE_NONE = 0,
    LLAMA_PENALTY_TYPE_REPEAT = 1,      // Repeat penalty (suppress repeated tokens)
    LLAMA_PENALTY_TYPE_FREQUENCY = 2,   // Frequency penalty (suppress frequent tokens)
    LLAMA_PENALTY_TYPE_PRESENCE = 3,    // Presence penalty (suppress seen tokens)
    LLAMA_PENALTY_TYPE_COMBINED = 4,    // All penalties combined in one kernel
};

// ============================================================================
// GPU PENALTY KERNEL STATE ENUMERATION
// ============================================================================

/**
 * State of GPU penalty kernel execution
 */
enum llama_gpu_penalty_state {
    LLAMA_GPU_PENALTY_UNINITIALIZED = 0,
    LLAMA_GPU_PENALTY_KERNEL_QUEUED = 1,      // Kernel queued on stream
    LLAMA_GPU_PENALTY_KERNEL_RUNNING = 2,     // Kernel executing
    LLAMA_GPU_PENALTY_LOGITS_MODIFIED = 3,    // Logits modified in-place
    LLAMA_GPU_PENALTY_READY_FOR_SAMPLING = 4, // Ready for sampling kernel
    LLAMA_GPU_PENALTY_ERROR = 5,
};

// ============================================================================
// CPU PENALTY PATH BYPASS ENUMERATION
// ============================================================================

/**
 * CPU penalty operations that should be bypassed
 */
enum llama_cpu_penalty_bypass {
    LLAMA_PENALTY_BYPASS_NONE = 0,
    LLAMA_PENALTY_BYPASS_REPEAT_ITERATION = 1,    // Skip CPU repeat penalty loop
    LLAMA_PENALTY_BYPASS_FREQUENCY_ITERATION = 2, // Skip CPU frequency penalty loop
    LLAMA_PENALTY_BYPASS_PRESENCE_ITERATION = 3,  // Skip CPU presence penalty loop
    LLAMA_PENALTY_BYPASS_HISTORY_ITERATION = 4,   // Skip CPU token history loop
    LLAMA_PENALTY_BYPASS_LOGITS_MODIFICATION = 5, // Skip CPU logits modification
    LLAMA_PENALTY_BYPASS_ENTRY_POINT = 6,         // Skip CPU penalty entry point
};

// ============================================================================
// PENALTY VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-exclusive penalty enforcement
 */
enum llama_penalty_violation {
    LLAMA_PENALTY_VIOLATION_NONE = 0,
    LLAMA_PENALTY_VIOLATION_CPU_REPEAT = 1,        // CPU applied repeat penalty
    LLAMA_PENALTY_VIOLATION_CPU_FREQUENCY = 2,     // CPU applied frequency penalty
    LLAMA_PENALTY_VIOLATION_CPU_PRESENCE = 3,      // CPU applied presence penalty
    LLAMA_PENALTY_VIOLATION_CPU_HISTORY_LOOP = 4,  // CPU looped over history
    LLAMA_PENALTY_VIOLATION_CPU_LOGITS_MODIFIED = 5, // CPU modified logits
    LLAMA_PENALTY_VIOLATION_HISTORY_ON_HOST = 6,   // Token history on host memory
    LLAMA_PENALTY_VIOLATION_MIXED_PATH = 7,        // Mixed CPU/GPU penalty path
};

// ============================================================================
// TOKEN HISTORY BUFFER STATE ENUMERATION
// ============================================================================

/**
 * State of GPU token history buffer
 */
enum llama_gpu_history_buffer_state {
    LLAMA_GPU_HISTORY_UNINITIALIZED = 0,
    LLAMA_GPU_HISTORY_ALLOCATED = 1,      // Ring buffer allocated on GPU
    LLAMA_GPU_HISTORY_POPULATED = 2,      // Buffer contains token history
    LLAMA_GPU_HISTORY_ACTIVE = 3,         // Buffer actively updated per token
    LLAMA_GPU_HISTORY_ERROR = 4,
};

// ============================================================================
// PENALTY CONFIGURATION RECORD
// ============================================================================

/**
 * Configuration for GPU penalty execution
 */
struct llama_gpu_penalty_config {
    bool repeat_penalty_enabled;         // Apply repeat penalty?
    bool frequency_penalty_enabled;      // Apply frequency penalty?
    bool presence_penalty_enabled;       // Apply presence penalty?
    float repeat_penalty_value;          // Repeat penalty multiplier
    float frequency_penalty_value;       // Frequency penalty coefficient
    float presence_penalty_value;        // Presence penalty coefficient
    bool gpu_penalty_enabled;            // Use GPU penalty kernels?
    bool history_on_gpu;                 // Token history on GPU?
    bool combined_kernel;                // Use combined kernel?
    enum llama_penalty_type penalty_type; // Active penalty type
};

// ============================================================================
// GPU PENALTY EXECUTION RECORD
// ============================================================================

/**
 * Record of GPU penalty kernel execution
 */
struct llama_gpu_penalty_execution_record {
    enum llama_penalty_type penalty_type;          // Penalty type applied
    enum llama_gpu_penalty_state penalty_state;    // Kernel execution state
    enum llama_gpu_history_buffer_state history_state; // History buffer state
    uint64_t timestamp_ns;                         // When penalties applied
    uint32_t tokens_processed;                     // Tokens in penalty kernel
    uint64_t gpu_kernel_ns;                        // GPU kernel execution time
    uint64_t history_update_ns;                    // History buffer update time
    int cpu_violations;                            // Violations detected
    enum llama_penalty_violation last_violation;   // Last violation type
};

// ============================================================================
// GPU PENALTY STATE RECORD
// ============================================================================

/**
 * Global state of GPU penalty during decode
 */
struct llama_gpu_penalty_state_record {
    enum llama_penalty_type current_penalty_type;   // Current penalty type
    enum llama_gpu_penalty_state gpu_penalty_state; // GPU penalty kernel state
    enum llama_gpu_history_buffer_state history_state; // History buffer state
    bool gpu_penalty_active;                        // GPU penalty kernel active?
    bool cpu_penalty_bypassed;                      // CPU penalty bypassed?
    bool history_gpu_resident;                      // History on GPU?
    int total_violations;                           // Total violations
    enum llama_penalty_violation last_violation;    // Last violation type
    uint64_t total_tokens_penalized;                // GPU-penalized token count
    uint64_t total_gpu_time_ns;                     // Cumulative GPU time
    uint64_t total_cpu_time_ns;                     // Cumulative CPU time
};

// ============================================================================
// PENALTY VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU penalty enforcement
 */
struct llama_gpu_penalty_validation_state {
    struct llama_gpu_penalty_config config;
    struct llama_gpu_penalty_state_record state_record;
    struct llama_gpu_penalty_execution_record last_execution;
    int total_penalty_applications;
    int total_violations;
    bool enforcement_strict;                    // Abort on violation vs log only
    bool debug_penalty_application;             // Debug output
    bool verify_bitwise_identical;              // Verify output equivalence
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_penalty_gpu_init(void);
int llama_penalty_gpu_configure_penalties(
    bool repeat_enabled,
    float repeat_value,
    bool frequency_enabled,
    float frequency_value,
    bool presence_enabled,
    float presence_value
);

// Penalty detection and routing
int llama_penalty_gpu_detect_penalty_config(
    int repeat_last_n,
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty
);
int llama_penalty_gpu_should_use_gpu_penalties(void);

// GPU penalty kernel execution (5 enforcement points: 1-5)
int llama_penalty_gpu_queue_penalty_kernel(void);
int llama_penalty_gpu_launch_penalty_kernel(void);
int llama_penalty_gpu_wait_penalty_result(void);
int llama_penalty_gpu_keep_logits_on_device(void);
int llama_penalty_gpu_assert_penalties_complete(void);

// History buffer management (3 enforcement points: 6-8)
int llama_penalty_gpu_allocate_history_buffer(uint32_t max_history_size);
int llama_penalty_gpu_update_history_on_gpu(uint32_t token_id);
int llama_penalty_gpu_forbid_cpu_history_loop(void);

// CPU bypass enforcement (2 enforcement points: 9-10)
int llama_penalty_gpu_forbid_cpu_penalty_computation(void);
int llama_penalty_gpu_forbid_cpu_penalty_entry_point(void);

// Violation detection
int llama_penalty_gpu_detect_cpu_repeat_penalty(void);
int llama_penalty_gpu_detect_cpu_frequency_penalty(void);
int llama_penalty_gpu_detect_cpu_presence_penalty(void);
int llama_penalty_gpu_detect_cpu_history_iteration(void);
int llama_penalty_gpu_detect_cpu_logits_modification(void);
int llama_penalty_gpu_detect_history_on_host(void);
int llama_penalty_gpu_detect_mixed_penalty_path(void);

// GPU state management
int llama_penalty_gpu_set_penalty_queued(void);
int llama_penalty_gpu_set_penalty_running(void);
int llama_penalty_gpu_set_logits_modified(void);
int llama_penalty_gpu_set_ready_for_sampling(void);

// Query and verification functions
struct llama_gpu_penalty_state_record llama_penalty_gpu_get_state_record(void);
struct llama_gpu_penalty_execution_record llama_penalty_gpu_get_last_execution(void);
enum llama_penalty_type llama_penalty_gpu_get_current_penalty_type(void);
enum llama_gpu_penalty_state llama_penalty_gpu_get_penalty_state(void);

// Verification functions
int llama_penalty_gpu_verify_cpu_penalty_bypassed(void);
int llama_penalty_gpu_verify_gpu_penalties_active(void);
int llama_penalty_gpu_verify_history_on_gpu(void);
int llama_penalty_gpu_verify_no_cpu_entry_point(void);
int llama_penalty_gpu_verify_minimal_cpu_overhead(void);
int llama_penalty_gpu_verify_bitwise_identical_output(float cpu_value, float gpu_value);

// Diagnostics and logging
void llama_penalty_gpu_log_penalty_mode_enabled(void);
void llama_penalty_gpu_log_penalty_kernel_launched(void);
void llama_penalty_gpu_log_logits_penalized(uint32_t num_tokens);
void llama_penalty_gpu_print_penalty_state(void);
void llama_penalty_gpu_print_execution_stats(void);
void llama_penalty_gpu_print_violation_summary(void);

// Violation reporting
void llama_penalty_gpu_report_violation(
    enum llama_penalty_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_penalty_gpu_set_enforcement_strict(bool strict);
bool llama_penalty_gpu_get_enforcement_strict(void);
void llama_penalty_gpu_set_debug_output(bool debug);
void llama_penalty_gpu_set_verify_bitwise(bool verify);

// Self-test suite
int llama_penalty_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_penalty_type_name(
    enum llama_penalty_type type
) {
    switch (type) {
        case LLAMA_PENALTY_TYPE_NONE: return "NONE";
        case LLAMA_PENALTY_TYPE_REPEAT: return "REPEAT";
        case LLAMA_PENALTY_TYPE_FREQUENCY: return "FREQUENCY";
        case LLAMA_PENALTY_TYPE_PRESENCE: return "PRESENCE";
        case LLAMA_PENALTY_TYPE_COMBINED: return "COMBINED";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_penalty_state_name(
    enum llama_gpu_penalty_state state
) {
    switch (state) {
        case LLAMA_GPU_PENALTY_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_PENALTY_KERNEL_QUEUED: return "KERNEL_QUEUED";
        case LLAMA_GPU_PENALTY_KERNEL_RUNNING: return "KERNEL_RUNNING";
        case LLAMA_GPU_PENALTY_LOGITS_MODIFIED: return "LOGITS_MODIFIED";
        case LLAMA_GPU_PENALTY_READY_FOR_SAMPLING: return "READY_FOR_SAMPLING";
        case LLAMA_GPU_PENALTY_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_penalty_violation_name(
    enum llama_penalty_violation violation
) {
    switch (violation) {
        case LLAMA_PENALTY_VIOLATION_NONE: return "NONE";
        case LLAMA_PENALTY_VIOLATION_CPU_REPEAT: return "CPU_REPEAT";
        case LLAMA_PENALTY_VIOLATION_CPU_FREQUENCY: return "CPU_FREQUENCY";
        case LLAMA_PENALTY_VIOLATION_CPU_PRESENCE: return "CPU_PRESENCE";
        case LLAMA_PENALTY_VIOLATION_CPU_HISTORY_LOOP: return "CPU_HISTORY_LOOP";
        case LLAMA_PENALTY_VIOLATION_CPU_LOGITS_MODIFIED: return "CPU_LOGITS_MODIFIED";
        case LLAMA_PENALTY_VIOLATION_HISTORY_ON_HOST: return "HISTORY_ON_HOST";
        case LLAMA_PENALTY_VIOLATION_MIXED_PATH: return "MIXED_PATH";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
