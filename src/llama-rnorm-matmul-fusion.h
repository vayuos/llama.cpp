/**
 * SECTION 37: Enforce RMSNorm + MatMul Fusion During Decode
 * Header
 *
 * This file implements mandatory GPU kernel fusion enforcement. RMSNorm followed by
 * MatMul operations must execute as a single fused CUDA kernel during decode. Separate
 * execution, intermediate materialization, or CPU sequencing between these operations
 * is forbidden. Unfused execution detected during decode triggers hard failure.
 *
 * Rules:
 * - RMSNorm + MatMul must fuse into single GPU kernel for decode
 * - No separate RMSNorm kernel invocation during decode
 * - No host-visible intermediate normalized buffer
 * - No CPU sequencing between RMSNorm and MatMul
 * - Normalized vector stays in register/shared memory
 * - Graph builder must detect and map to fused op
 * - Unfused execution during decode results in hard failure
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
enum llama_fusion_phase {
    LLAMA_FUSION_PHASE_NONE = 0,
    LLAMA_FUSION_PHASE_GRAPH_BUILD = 1,    // Graph construction (fusion detection)
    LLAMA_FUSION_PHASE_PREFILL = 2,        // Prefill phase (fusion optional)
    LLAMA_FUSION_PHASE_DECODE = 3,         // Decode phase (fusion MANDATORY)
    LLAMA_FUSION_PHASE_COMPLETE = 4,       // Cleanup
};

// ============================================================================
// FUSION OPERATION STATE ENUMERATION
// ============================================================================

/**
 * State of kernel fusion enforcement
 */
enum llama_gpu_fusion_state {
    LLAMA_GPU_FUSION_UNINITIALIZED = 0,
    LLAMA_GPU_FUSION_INITIALIZED = 1,
    LLAMA_GPU_FUSION_GRAPH_ANALYZED = 2,
    LLAMA_GPU_FUSION_FUSED_KERNELS_READY = 3,
    LLAMA_GPU_FUSION_DECODE_ACTIVE = 4,
    LLAMA_GPU_FUSION_COMPLETE = 5,
    LLAMA_GPU_FUSION_ERROR = 6,
};

// ============================================================================
// FUSION VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of RMSNorm + MatMul fusion enforcement
 */
enum llama_rnorm_matmul_violation {
    LLAMA_RNORM_MATMUL_VIOLATION_NONE = 0,
    LLAMA_RNORM_MATMUL_VIOLATION_UNFUSED_RNORM_DECODE = 1,        // Separate RMSNorm in decode
    LLAMA_RNORM_MATMUL_VIOLATION_INTERMEDIATE_BUFFER = 2,         // Normalized output to global memory
    LLAMA_RNORM_MATMUL_VIOLATION_HOST_SEQUENCE = 3,               // Host-managed sequencing
    LLAMA_RNORM_MATMUL_VIOLATION_UNSUPPORTED_SHAPE = 4,           // Shape incompatible with fusion
    LLAMA_RNORM_MATMUL_VIOLATION_FALLBACK_UNFUSED = 5,            // Silent fallback to unfused
    LLAMA_RNORM_MATMUL_VIOLATION_WRONG_BACKEND = 6,               // Non-CUDA backend for fusion
    LLAMA_RNORM_MATMUL_VIOLATION_INTERMEDIATE_D2H = 7,            // Device-to-host copy of intermediate
    LLAMA_RNORM_MATMUL_VIOLATION_CPU_NORM_ACCESS = 8,             // CPU reads normalized tensor
};

// ============================================================================
// FUSION OPERATION TYPE ENUMERATION
// ============================================================================

/**
 * Types of fused operation patterns
 */
enum llama_fusion_operation_type {
    LLAMA_FUSION_OP_NONE = 0,
    LLAMA_FUSION_OP_RNORM_QKV = 1,         // RMSNorm + QKV projection
    LLAMA_FUSION_OP_RNORM_FFN_GATE = 2,    // RMSNorm + FFN gate projection
    LLAMA_FUSION_OP_RNORM_FFN_UP = 3,      // RMSNorm + FFN up projection
    LLAMA_FUSION_OP_RNORM_OUTPUT = 4,      // RMSNorm + output projection
    LLAMA_FUSION_OP_RNORM_CUSTOM = 5,      // Custom fusion pattern
};

// ============================================================================
// FUSION KERNEL STATUS ENUMERATION
// ============================================================================

/**
 * Status of individual fused kernel instantiation
 */
enum llama_fusion_kernel_status {
    LLAMA_FUSION_KERNEL_UNDETECTED = 0,
    LLAMA_FUSION_KERNEL_DETECTED = 1,           // Found in graph
    LLAMA_FUSION_KERNEL_SUPPORTED = 2,          // Supported by backend
    LLAMA_FUSION_KERNEL_COMPILED = 3,           // Compiled/cached
    LLAMA_FUSION_KERNEL_ACTIVE_DECODE = 4,      // Running in decode
    LLAMA_FUSION_KERNEL_UNFUSED_FALLBACK = 5,   // Fell back to unfused
    LLAMA_FUSION_KERNEL_ERROR = 6,              // Error state
};

// ============================================================================
// FUSION CONFIGURATION
// ============================================================================

/**
 * Configuration for fusion enforcement
 */
struct llama_gpu_fusion_config {
    bool enforce_fusion_mandatory;      // Fusion mandatory during decode?
    bool forbid_unfused_execution;      // Hard fail on unfused RMSNorm in decode?
    bool forbid_intermediate_buffer;    // Forbid normalized tensor materialization?
    bool forbid_host_sequencing;        // Forbid host-managed sequencing?
    bool cuda_backend_only;             // Restrict to CUDA backend only?
    bool debug_fusion_tracking;         // Debug output?
};

// ============================================================================
// FUSION OPERATION RECORD
// ============================================================================

/**
 * Records a fused operation instance
 */
struct llama_fusion_operation_record {
    uint64_t operation_id;                           // Unique operation ID
    enum llama_fusion_operation_type fusion_type;    // Operation type (RMSNorm+QKV, etc.)
    uint32_t layer_idx;                              // Layer index
    uint64_t input_tensor_id;                        // Input tensor ID
    uint64_t output_tensor_id;                       // Output tensor ID
    uint32_t normalized_dim;                         // Normalization dimension
    uint32_t projected_dim;                          // Projection output dimension
    uint64_t kernel_launch_timestamp_ns;             // When kernel launched
    bool was_fused;                                  // Was actually fused?
    bool is_decode_phase;                            // Running in decode phase?
};

// ============================================================================
// FUSION KERNEL RECORD
// ============================================================================

/**
 * Records fused kernel compilation and usage
 */
struct llama_fusion_kernel_record {
    uint64_t kernel_id;                              // Kernel identifier
    enum llama_fusion_operation_type fusion_type;    // Fusion type
    enum llama_fusion_kernel_status status;          // Current status
    uint32_t in_channels;                            // Input channels
    uint32_t out_channels;                           // Output channels
    uint32_t batch_size;                             // Batch size
    uint64_t total_launches;                         // Total launches
    uint64_t decode_launches;                        // Launches in decode phase
    bool is_cuda_kernel;                             // CUDA backend?
    char kernel_name[256];                           // Kernel name
};

// ============================================================================
// FUSION VALIDATION STATE RECORD
// ============================================================================

/**
 * Current state of kernel fusion enforcement
 */
struct llama_gpu_fusion_state_record {
    enum llama_gpu_fusion_state state;               // Current state
    enum llama_fusion_phase current_phase;           // Current phase
    uint64_t total_operations_detected;              // RMSNorm+MatMul patterns found
    uint64_t total_operations_fused;                 // Actually fused
    uint64_t total_operations_unfused;               // Left unfused
    uint64_t fused_kernels_compiled;                 // Fused kernels compiled
    uint64_t intermediate_buffers_detected;          // Intermediate tensors found
    uint64_t host_sequences_detected;                // Host-managed sequences found
    uint64_t decode_fused_kernels_invoked;           // Fused kernels in decode
    uint64_t decode_unfused_rnorm_detected;          // Unfused RMSNorm in decode
    int total_violations;                            // Total violations
    enum llama_rnorm_matmul_violation last_violation; // Last violation
};

// ============================================================================
// FUSION VALIDATION STATE
// ============================================================================

/**
 * Global state for kernel fusion enforcement
 */
struct llama_gpu_fusion_validation_state {
    struct llama_gpu_fusion_config config;
    struct llama_gpu_fusion_state_record state_record;

    // Per-operation tracking (std::map<operation_id, fusion_operation_record>)
    void* fusion_operations_map;  // opaque pointer to std::map

    // Fused kernels (std::map<kernel_id, fusion_kernel_record>)
    void* fusion_kernels_map;     // opaque pointer to std::map

    // Operation history (std::vector<fusion_operation_record>)
    void* operation_history_vector; // opaque pointer to std::vector

    struct llama_fusion_operation_record last_operation_record;
    int total_operations;
    int total_violations;
    bool enforcement_strict;       // Abort on violation vs log only
    bool decode_phase_active;      // Is decode phase active?
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_fusion_gpu_init(void);
int llama_fusion_gpu_configure(
    bool enforce_fusion_mandatory,
    bool forbid_unfused_execution,
    bool forbid_intermediate_buffer,
    bool forbid_host_sequencing,
    bool cuda_backend_only
);

// Phase management
int llama_fusion_gpu_set_phase(enum llama_fusion_phase phase);
int llama_fusion_gpu_begin_decode_phase(void);
int llama_fusion_gpu_end_decode_phase(void);

// Graph analysis (10 enforcement points: 1-10)
int llama_fusion_gpu_analyze_graph_for_fusion_opportunities(void);
int llama_fusion_gpu_detect_rnorm_matmul_patterns(void);
int llama_fusion_gpu_validate_fusion_shapes(void);
int llama_fusion_gpu_map_patterns_to_fused_operations(void);
int llama_fusion_gpu_compile_fused_kernels(void);
int llama_fusion_gpu_forbid_unfused_rnorm_in_decode(void);
int llama_fusion_gpu_forbid_intermediate_buffer_in_decode(void);
int llama_fusion_gpu_forbid_host_sequencing_in_decode(void);
int llama_fusion_gpu_verify_all_patterns_fused(void);
int llama_fusion_gpu_enforce_fused_execution_in_decode(void);

// Violation detection
int llama_fusion_gpu_detect_unfused_rnorm_decode(void);
int llama_fusion_gpu_detect_intermediate_buffer(uint64_t tensor_id);
int llama_fusion_gpu_detect_host_sequencing(void);
int llama_fusion_gpu_detect_unsupported_shape(uint32_t in_channels, uint32_t out_channels);
int llama_fusion_gpu_detect_unfused_fallback(void);
int llama_fusion_gpu_detect_wrong_backend(void);
int llama_fusion_gpu_detect_intermediate_d2h_copy(uint64_t tensor_id);
int llama_fusion_gpu_detect_cpu_normalized_access(uint64_t tensor_id);

// Fusion operation tracking
int llama_fusion_gpu_record_fusion_operation(
    uint64_t operation_id,
    enum llama_fusion_operation_type fusion_type,
    uint32_t layer_idx,
    uint64_t input_id,
    uint64_t output_id,
    uint32_t norm_dim,
    uint32_t proj_dim
);
int llama_fusion_gpu_record_kernel_compilation(
    uint64_t kernel_id,
    enum llama_fusion_operation_type fusion_type,
    uint32_t in_channels,
    uint32_t out_channels,
    const char* kernel_name
);
int llama_fusion_gpu_validate_operation_fused(uint64_t operation_id);

// Verification functions
int llama_fusion_gpu_verify_all_patterns_mapped(void);
int llama_fusion_gpu_verify_no_intermediate_buffers(void);
int llama_fusion_gpu_verify_no_host_sequences(void);
int llama_fusion_gpu_verify_kernels_compiled(void);
int llama_fusion_gpu_verify_cuda_backend_only(void);

// Query functions
struct llama_gpu_fusion_state_record llama_fusion_gpu_get_state_record(void);
enum llama_gpu_fusion_state llama_fusion_gpu_get_state(void);
enum llama_fusion_phase llama_fusion_gpu_get_phase(void);
uint64_t llama_fusion_gpu_get_fused_kernel_count(void);
uint64_t llama_fusion_gpu_get_decode_fused_kernels_invoked(void);

// Diagnostics and logging
void llama_fusion_gpu_log_fusion_enforcement_enabled(void);
void llama_fusion_gpu_log_patterns_detected(void);
void llama_fusion_gpu_log_kernels_compiled(void);
void llama_fusion_gpu_log_decode_phase_fusion_active(void);
void llama_fusion_gpu_print_state(void);
void llama_fusion_gpu_print_operation_record(const struct llama_fusion_operation_record* record);
void llama_fusion_gpu_print_kernel_summary(void);
void llama_fusion_gpu_print_violation_summary(void);

// Violation reporting
void llama_fusion_gpu_report_violation(
    enum llama_rnorm_matmul_violation violation_type,
    const char* location,
    const char* details
);

// Enforcement mode control
void llama_fusion_gpu_set_enforcement_strict(bool strict);
bool llama_fusion_gpu_get_enforcement_strict(void);
void llama_fusion_gpu_set_debug_output(bool debug);

// Performance validation
int llama_fusion_gpu_validate_performance_impact(void);
uint64_t llama_fusion_gpu_get_kernel_count_reduction(void);
uint64_t llama_fusion_gpu_get_memory_bandwidth_reduction(void);

// Self-test suite
int llama_fusion_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_fusion_phase_name(enum llama_fusion_phase phase) {
    switch (phase) {
        case LLAMA_FUSION_PHASE_NONE: return "NONE";
        case LLAMA_FUSION_PHASE_GRAPH_BUILD: return "GRAPH_BUILD";
        case LLAMA_FUSION_PHASE_PREFILL: return "PREFILL";
        case LLAMA_FUSION_PHASE_DECODE: return "DECODE";
        case LLAMA_FUSION_PHASE_COMPLETE: return "COMPLETE";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_rnorm_matmul_violation_name(enum llama_rnorm_matmul_violation violation) {
    switch (violation) {
        case LLAMA_RNORM_MATMUL_VIOLATION_NONE: return "NONE";
        case LLAMA_RNORM_MATMUL_VIOLATION_UNFUSED_RNORM_DECODE: return "UNFUSED_RNORM_IN_DECODE";
        case LLAMA_RNORM_MATMUL_VIOLATION_INTERMEDIATE_BUFFER: return "INTERMEDIATE_BUFFER";
        case LLAMA_RNORM_MATMUL_VIOLATION_HOST_SEQUENCE: return "HOST_SEQUENCE";
        case LLAMA_RNORM_MATMUL_VIOLATION_UNSUPPORTED_SHAPE: return "UNSUPPORTED_SHAPE";
        case LLAMA_RNORM_MATMUL_VIOLATION_FALLBACK_UNFUSED: return "FALLBACK_UNFUSED";
        case LLAMA_RNORM_MATMUL_VIOLATION_WRONG_BACKEND: return "WRONG_BACKEND";
        case LLAMA_RNORM_MATMUL_VIOLATION_INTERMEDIATE_D2H: return "INTERMEDIATE_D2H";
        case LLAMA_RNORM_MATMUL_VIOLATION_CPU_NORM_ACCESS: return "CPU_NORM_ACCESS";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_fusion_operation_type_name(enum llama_fusion_operation_type op_type) {
    switch (op_type) {
        case LLAMA_FUSION_OP_NONE: return "NONE";
        case LLAMA_FUSION_OP_RNORM_QKV: return "RNORM_QKV";
        case LLAMA_FUSION_OP_RNORM_FFN_GATE: return "RNORM_FFN_GATE";
        case LLAMA_FUSION_OP_RNORM_FFN_UP: return "RNORM_FFN_UP";
        case LLAMA_FUSION_OP_RNORM_OUTPUT: return "RNORM_OUTPUT";
        case LLAMA_FUSION_OP_RNORM_CUSTOM: return "RNORM_CUSTOM";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_fusion_kernel_status_name(enum llama_fusion_kernel_status status) {
    switch (status) {
        case LLAMA_FUSION_KERNEL_UNDETECTED: return "UNDETECTED";
        case LLAMA_FUSION_KERNEL_DETECTED: return "DETECTED";
        case LLAMA_FUSION_KERNEL_SUPPORTED: return "SUPPORTED";
        case LLAMA_FUSION_KERNEL_COMPILED: return "COMPILED";
        case LLAMA_FUSION_KERNEL_ACTIVE_DECODE: return "ACTIVE_DECODE";
        case LLAMA_FUSION_KERNEL_UNFUSED_FALLBACK: return "UNFUSED_FALLBACK";
        case LLAMA_FUSION_KERNEL_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
