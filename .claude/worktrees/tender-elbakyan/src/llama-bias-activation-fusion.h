/**
 * SECTION 38: Enforce Bias + Activation Fusion During Decode
 * Header
 *
 * This file implements mandatory GPU kernel fusion enforcement for bias addition
 * and activation functions. MatMul → Add Bias → Activation sequences must execute
 * as single fused CUDA kernel during decode. Separate execution, intermediate
 * materialization, or host sequencing is forbidden. Unfused execution triggers
 * hard failure.
 *
 * Rules:
 * - MatMul → Add Bias → Activation must fuse into single GPU kernel for decode
 * - No separate bias kernel invocation during decode
 * - No separate activation kernel invocation after matmul in decode
 * - No intermediate tensor materialization (biased or pre-activation)
 * - No host-mediated sequencing between bias and activation
 * - Intermediate stays device-local (registers/shared memory)
 * - Unfused execution during decode results in hard failure
 * - Early failure for unsupported shapes (no fallback)
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// FUSION PHASE ENUMERATION
// ============================================================================

/**
 * Execution phases and their fusion enforcement policies
 */
enum llama_bias_act_fusion_phase {
    LLAMA_BIAS_ACT_FUSION_PHASE_NONE = 0,
    LLAMA_BIAS_ACT_FUSION_PHASE_GRAPH_BUILD = 1,   // Graph construction (fusion detection)
    LLAMA_BIAS_ACT_FUSION_PHASE_PREFILL = 2,       // Prefill phase (fusion optional)
    LLAMA_BIAS_ACT_FUSION_PHASE_DECODE = 3,        // Decode phase (fusion MANDATORY)
    LLAMA_BIAS_ACT_FUSION_PHASE_COMPLETE = 4,      // Cleanup
};

// ============================================================================
// FUSION STATE ENUMERATION
// ============================================================================

/**
 * State of kernel fusion enforcement
 */
enum llama_gpu_bias_act_fusion_state {
    LLAMA_GPU_BIAS_ACT_FUSION_UNINITIALIZED = 0,
    LLAMA_GPU_BIAS_ACT_FUSION_INITIALIZED = 1,
    LLAMA_GPU_BIAS_ACT_FUSION_GRAPH_ANALYZED = 2,
    LLAMA_GPU_BIAS_ACT_FUSION_KERNELS_READY = 3,
    LLAMA_GPU_BIAS_ACT_FUSION_DECODE_ACTIVE = 4,
    LLAMA_GPU_BIAS_ACT_FUSION_COMPLETE = 5,
    LLAMA_GPU_BIAS_ACT_FUSION_ERROR = 6,
};

// ============================================================================
// ACTIVATION TYPE ENUMERATION
// ============================================================================

/**
 * Supported activation functions for fusion
 */
enum llama_activation_function {
    LLAMA_ACTIVATION_NONE = 0,
    LLAMA_ACTIVATION_RELU = 1,              // ReLU
    LLAMA_ACTIVATION_GELU = 2,              // GELU
    LLAMA_ACTIVATION_GELU_APPROX = 3,       // GELU approximate
    LLAMA_ACTIVATION_SILU = 4,              // SiLU (Swish)
    LLAMA_ACTIVATION_TANH = 5,              // Tanh
    LLAMA_ACTIVATION_SIGMOID = 6,           // Sigmoid
    LLAMA_ACTIVATION_MISH = 7,              // Mish
    LLAMA_ACTIVATION_LINEAR = 8,            // Linear (no activation)
};

// ============================================================================
// FUSION VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of bias + activation fusion enforcement
 */
enum llama_bias_activation_violation {
    LLAMA_BIAS_ACTIVATION_VIOLATION_NONE = 0,
    LLAMA_BIAS_ACTIVATION_VIOLATION_UNFUSED_BIAS_DECODE = 1,        // Separate bias in decode
    LLAMA_BIAS_ACTIVATION_VIOLATION_UNFUSED_ACTIVATION_DECODE = 2,   // Separate activation in decode
    LLAMA_BIAS_ACTIVATION_VIOLATION_INTERMEDIATE_BUFFER = 3,         // Biased tensor materialized
    LLAMA_BIAS_ACTIVATION_VIOLATION_HOST_SEQUENCE = 4,               // Host-managed sequencing
    LLAMA_BIAS_ACTIVATION_VIOLATION_UNSUPPORTED_ACTIVATION = 5,      // Unsupported activation type
    LLAMA_BIAS_ACTIVATION_VIOLATION_UNSUPPORTED_SHAPE = 6,           // Incompatible tensor shape
    LLAMA_BIAS_ACTIVATION_VIOLATION_FALLBACK_UNFUSED = 7,            // Silent fallback to unfused
    LLAMA_BIAS_ACTIVATION_VIOLATION_WRONG_BACKEND = 8,               // Non-CUDA backend
};

// ============================================================================
// FUSION OPERATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of fused bias + activation patterns
 */
enum llama_bias_act_fusion_type {
    LLAMA_BIAS_ACT_FUSION_NONE = 0,
    LLAMA_BIAS_ACT_FUSION_FFN_GATE = 1,         // FFN gate (bias + SiLU)
    LLAMA_BIAS_ACT_FUSION_FFN_UP = 2,           // FFN up (bias + GELU)
    LLAMA_BIAS_ACT_FUSION_OUTPUT_PROJ = 3,      // Output proj (bias + linear)
    LLAMA_BIAS_ACT_FUSION_ATTENTION_OUT = 4,    // Attention out (bias + linear)
    LLAMA_BIAS_ACT_FUSION_GATED = 5,            // Gated activation (bias + activation * scale)
    LLAMA_BIAS_ACT_FUSION_CUSTOM = 6,           // Custom pattern
};

// ============================================================================
// KERNEL COMPILATION STATUS ENUMERATION
// ============================================================================

/**
 * Status of fused kernel
 */
enum llama_bias_act_kernel_status {
    LLAMA_BIAS_ACT_KERNEL_UNDETECTED = 0,
    LLAMA_BIAS_ACT_KERNEL_DETECTED = 1,         // Found in graph
    LLAMA_BIAS_ACT_KERNEL_SUPPORTED = 2,        // Supported by backend
    LLAMA_BIAS_ACT_KERNEL_COMPILED = 3,         // Compiled/cached
    LLAMA_BIAS_ACT_KERNEL_ACTIVE_DECODE = 4,    // Running in decode
    LLAMA_BIAS_ACT_KERNEL_FALLBACK = 5,         // Fell back to unfused
    LLAMA_BIAS_ACT_KERNEL_ERROR = 6,
};

// ============================================================================
// FUSION CONFIGURATION
// ============================================================================

/**
 * Configuration for fusion enforcement
 */
struct llama_gpu_bias_act_fusion_config {
    bool enforce_fusion_mandatory;      // Fusion mandatory during decode?
    bool forbid_unfused_bias;           // Hard fail on unfused bias in decode?
    bool forbid_unfused_activation;     // Hard fail on unfused activation in decode?
    bool forbid_intermediate_buffer;    // Forbid biased tensor materialization?
    bool forbid_host_sequencing;        // Forbid host-managed sequencing?
    bool cuda_backend_only;             // Restrict to CUDA backend only?
    bool debug_fusion_tracking;         // Debug output?
};

// ============================================================================
// FUSION OPERATION RECORD
// ============================================================================

/**
 * Records a fused bias + activation operation
 */
struct llama_bias_act_fusion_operation_record {
    uint64_t operation_id;                       // Unique operation ID
    enum llama_bias_act_fusion_type fusion_type; // Operation type
    enum llama_activation_function activation;   // Activation function
    uint32_t layer_idx;                          // Layer index
    uint64_t matmul_output_id;                   // MatMul output tensor ID
    uint64_t bias_tensor_id;                     // Bias tensor ID
    uint64_t final_output_id;                    // Final output tensor ID
    uint64_t element_count;                      // Number of elements
    uint64_t kernel_launch_timestamp_ns;         // When kernel launched
    bool was_fused;                              // Was actually fused?
    bool is_decode_phase;                        // Running in decode phase?
};

// ============================================================================
// FUSION KERNEL RECORD
// ============================================================================

/**
 * Records fused kernel compilation and usage
 */
struct llama_bias_act_kernel_record {
    uint64_t kernel_id;                          // Kernel identifier
    enum llama_bias_act_fusion_type fusion_type; // Fusion type
    enum llama_activation_function activation;   // Activation type
    enum llama_bias_act_kernel_status status;    // Current status
    uint32_t output_channels;                    // Output dimension
    uint32_t batch_size;                         // Batch size
    uint64_t total_launches;                     // Total launches
    uint64_t decode_launches;                    // Launches in decode phase
    bool is_cuda_kernel;                         // CUDA backend?
    char kernel_name[256];                       // Kernel name
};

// ============================================================================
// FUSION STATE RECORD
// ============================================================================

/**
 * Current state of bias + activation fusion enforcement
 */
struct llama_gpu_bias_act_fusion_state_record {
    enum llama_gpu_bias_act_fusion_state state;  // Current state
    enum llama_bias_act_fusion_phase current_phase; // Current phase
    uint64_t total_operations_detected;          // Bias+Act patterns found
    uint64_t total_operations_fused;             // Actually fused
    uint64_t total_operations_unfused;           // Left unfused
    uint64_t fused_kernels_compiled;             // Fused kernels compiled
    uint64_t intermediate_buffers_detected;      // Intermediate tensors found
    uint64_t host_sequences_detected;            // Host-managed sequences found
    uint64_t decode_fused_kernels_invoked;       // Fused kernels in decode
    uint64_t decode_unfused_bias_detected;       // Unfused bias in decode
    uint64_t decode_unfused_activation_detected; // Unfused activation in decode
    int total_violations;                        // Total violations
    enum llama_bias_activation_violation last_violation; // Last violation
};

// ============================================================================
// FUSION VALIDATION STATE
// ============================================================================

/**
 * Global state for bias + activation fusion enforcement
 */
struct llama_gpu_bias_act_fusion_validation_state {
    struct llama_gpu_bias_act_fusion_config config;
    struct llama_gpu_bias_act_fusion_state_record state_record;

    // Per-operation tracking (std::map<operation_id, bias_act_fusion_operation_record>)
    void* fusion_operations_map;  // opaque pointer to std::map

    // Fused kernels (std::map<kernel_id, bias_act_kernel_record>)
    void* fusion_kernels_map;     // opaque pointer to std::map

    // Operation history (std::vector<bias_act_fusion_operation_record>)
    void* operation_history_vector; // opaque pointer to std::vector

    struct llama_bias_act_fusion_operation_record last_operation_record;
    int total_operations;
    int total_violations;
    bool enforcement_strict;       // Abort on violation vs log only
    bool decode_phase_active;      // Is decode phase active?
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_bias_act_fusion_gpu_init(void);
int llama_bias_act_fusion_gpu_configure(
    bool enforce_fusion_mandatory,
    bool forbid_unfused_bias,
    bool forbid_unfused_activation,
    bool forbid_intermediate_buffer,
    bool forbid_host_sequencing,
    bool cuda_backend_only
);

// Phase management
int llama_bias_act_fusion_gpu_set_phase(enum llama_bias_act_fusion_phase phase);
int llama_bias_act_fusion_gpu_begin_decode_phase(void);
int llama_bias_act_fusion_gpu_end_decode_phase(void);

// Graph analysis (10 enforcement points: 1-10)
int llama_bias_act_fusion_gpu_analyze_graph_for_fusion_opportunities(void);
int llama_bias_act_fusion_gpu_detect_bias_activation_patterns(void);
int llama_bias_act_fusion_gpu_validate_activation_support(void);
int llama_bias_act_fusion_gpu_validate_fusion_shapes(void);
int llama_bias_act_fusion_gpu_map_patterns_to_fused_operations(void);
int llama_bias_act_fusion_gpu_compile_fused_kernels(void);
int llama_bias_act_fusion_gpu_forbid_unfused_bias_in_decode(void);
int llama_bias_act_fusion_gpu_forbid_unfused_activation_in_decode(void);
int llama_bias_act_fusion_gpu_verify_all_patterns_fused(void);
int llama_bias_act_fusion_gpu_enforce_fused_execution_in_decode(void);

// Violation detection
int llama_bias_act_fusion_gpu_detect_unfused_bias_decode(void);
int llama_bias_act_fusion_gpu_detect_unfused_activation_decode(void);
int llama_bias_act_fusion_gpu_detect_intermediate_buffer(uint64_t tensor_id);
int llama_bias_act_fusion_gpu_detect_host_sequencing(void);
int llama_bias_act_fusion_gpu_detect_unsupported_activation(enum llama_activation_function act);
int llama_bias_act_fusion_gpu_detect_unsupported_shape(uint32_t output_dim);
int llama_bias_act_fusion_gpu_detect_unfused_fallback(void);
int llama_bias_act_fusion_gpu_detect_wrong_backend(void);

// Fusion operation tracking
int llama_bias_act_fusion_gpu_record_fusion_operation(
    uint64_t operation_id,
    enum llama_bias_act_fusion_type fusion_type,
    enum llama_activation_function activation,
    uint32_t layer_idx,
    uint64_t matmul_output_id,
    uint64_t bias_id,
    uint64_t final_output_id,
    uint64_t element_count
);
int llama_bias_act_fusion_gpu_record_kernel_compilation(
    uint64_t kernel_id,
    enum llama_bias_act_fusion_type fusion_type,
    enum llama_activation_function activation,
    uint32_t output_channels,
    const char* kernel_name
);
int llama_bias_act_fusion_gpu_validate_operation_fused(uint64_t operation_id);

// Verification functions
int llama_bias_act_fusion_gpu_verify_all_patterns_mapped(void);
int llama_bias_act_fusion_gpu_verify_no_intermediate_buffers(void);
int llama_bias_act_fusion_gpu_verify_no_host_sequences(void);
int llama_bias_act_fusion_gpu_verify_kernels_compiled(void);
int llama_bias_act_fusion_gpu_verify_cuda_backend_only(void);

// Activation support queries
int llama_bias_act_fusion_gpu_is_activation_supported(enum llama_activation_function act);
int llama_bias_act_fusion_gpu_get_supported_activations(enum llama_activation_function* activations, uint32_t max_count);

// Query functions
struct llama_gpu_bias_act_fusion_state_record llama_bias_act_fusion_gpu_get_state_record(void);
enum llama_gpu_bias_act_fusion_state llama_bias_act_fusion_gpu_get_state(void);
enum llama_bias_act_fusion_phase llama_bias_act_fusion_gpu_get_phase(void);
uint64_t llama_bias_act_fusion_gpu_get_fused_kernel_count(void);
uint64_t llama_bias_act_fusion_gpu_get_decode_fused_kernels_invoked(void);

// Diagnostics and logging
void llama_bias_act_fusion_gpu_log_fusion_enforcement_enabled(void);
void llama_bias_act_fusion_gpu_log_patterns_detected(void);
void llama_bias_act_fusion_gpu_log_kernels_compiled(void);
void llama_bias_act_fusion_gpu_log_decode_phase_fusion_active(void);
void llama_bias_act_fusion_gpu_print_state(void);
void llama_bias_act_fusion_gpu_print_operation_record(const struct llama_bias_act_fusion_operation_record* record);
void llama_bias_act_fusion_gpu_print_kernel_summary(void);
void llama_bias_act_fusion_gpu_print_violation_summary(void);

// Violation reporting
void llama_bias_act_fusion_gpu_report_violation(
    enum llama_bias_activation_violation violation_type,
    const char* location,
    const char* details
);

// Enforcement mode control
void llama_bias_act_fusion_gpu_set_enforcement_strict(bool strict);
bool llama_bias_act_fusion_gpu_get_enforcement_strict(void);
void llama_bias_act_fusion_gpu_set_debug_output(bool debug);

// Performance validation
int llama_bias_act_fusion_gpu_validate_performance_impact(void);
uint64_t llama_bias_act_fusion_gpu_get_kernel_count_reduction(void);
uint64_t llama_bias_act_fusion_gpu_get_memory_bandwidth_reduction(void);

// Self-test suite
int llama_bias_act_fusion_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_bias_act_fusion_phase_name(enum llama_bias_act_fusion_phase phase) {
    switch (phase) {
        case LLAMA_BIAS_ACT_FUSION_PHASE_NONE: return "NONE";
        case LLAMA_BIAS_ACT_FUSION_PHASE_GRAPH_BUILD: return "GRAPH_BUILD";
        case LLAMA_BIAS_ACT_FUSION_PHASE_PREFILL: return "PREFILL";
        case LLAMA_BIAS_ACT_FUSION_PHASE_DECODE: return "DECODE";
        case LLAMA_BIAS_ACT_FUSION_PHASE_COMPLETE: return "COMPLETE";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_activation_function_name(enum llama_activation_function act) {
    switch (act) {
        case LLAMA_ACTIVATION_NONE: return "NONE";
        case LLAMA_ACTIVATION_RELU: return "RELU";
        case LLAMA_ACTIVATION_GELU: return "GELU";
        case LLAMA_ACTIVATION_GELU_APPROX: return "GELU_APPROX";
        case LLAMA_ACTIVATION_SILU: return "SILU";
        case LLAMA_ACTIVATION_TANH: return "TANH";
        case LLAMA_ACTIVATION_SIGMOID: return "SIGMOID";
        case LLAMA_ACTIVATION_MISH: return "MISH";
        case LLAMA_ACTIVATION_LINEAR: return "LINEAR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_bias_activation_violation_name(enum llama_bias_activation_violation violation) {
    switch (violation) {
        case LLAMA_BIAS_ACTIVATION_VIOLATION_NONE: return "NONE";
        case LLAMA_BIAS_ACTIVATION_VIOLATION_UNFUSED_BIAS_DECODE: return "UNFUSED_BIAS_IN_DECODE";
        case LLAMA_BIAS_ACTIVATION_VIOLATION_UNFUSED_ACTIVATION_DECODE: return "UNFUSED_ACTIVATION_IN_DECODE";
        case LLAMA_BIAS_ACTIVATION_VIOLATION_INTERMEDIATE_BUFFER: return "INTERMEDIATE_BUFFER";
        case LLAMA_BIAS_ACTIVATION_VIOLATION_HOST_SEQUENCE: return "HOST_SEQUENCE";
        case LLAMA_BIAS_ACTIVATION_VIOLATION_UNSUPPORTED_ACTIVATION: return "UNSUPPORTED_ACTIVATION";
        case LLAMA_BIAS_ACTIVATION_VIOLATION_UNSUPPORTED_SHAPE: return "UNSUPPORTED_SHAPE";
        case LLAMA_BIAS_ACTIVATION_VIOLATION_FALLBACK_UNFUSED: return "FALLBACK_UNFUSED";
        case LLAMA_BIAS_ACTIVATION_VIOLATION_WRONG_BACKEND: return "WRONG_BACKEND";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_bias_act_fusion_type_name(enum llama_bias_act_fusion_type fusion_type) {
    switch (fusion_type) {
        case LLAMA_BIAS_ACT_FUSION_NONE: return "NONE";
        case LLAMA_BIAS_ACT_FUSION_FFN_GATE: return "FFN_GATE";
        case LLAMA_BIAS_ACT_FUSION_FFN_UP: return "FFN_UP";
        case LLAMA_BIAS_ACT_FUSION_OUTPUT_PROJ: return "OUTPUT_PROJ";
        case LLAMA_BIAS_ACT_FUSION_ATTENTION_OUT: return "ATTENTION_OUT";
        case LLAMA_BIAS_ACT_FUSION_GATED: return "GATED";
        case LLAMA_BIAS_ACT_FUSION_CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_bias_act_kernel_status_name(enum llama_bias_act_kernel_status status) {
    switch (status) {
        case LLAMA_BIAS_ACT_KERNEL_UNDETECTED: return "UNDETECTED";
        case LLAMA_BIAS_ACT_KERNEL_DETECTED: return "DETECTED";
        case LLAMA_BIAS_ACT_KERNEL_SUPPORTED: return "SUPPORTED";
        case LLAMA_BIAS_ACT_KERNEL_COMPILED: return "COMPILED";
        case LLAMA_BIAS_ACT_KERNEL_ACTIVE_DECODE: return "ACTIVE_DECODE";
        case LLAMA_BIAS_ACT_KERNEL_FALLBACK: return "FALLBACK";
        case LLAMA_BIAS_ACT_KERNEL_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
