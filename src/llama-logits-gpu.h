/**
 * SECTION 25: Eliminate CPU logits reads during decode
 * Header
 *
 * This file implements GPU-exclusive logits management for deterministic sampling.
 * All logits remain GPU-resident during decode; no CPU reads, inspections, or materializations.
 * Logits are classified as decode-critical GPU-only data with hard enforcement.
 * Only final selected token ID crosses PCIe; no logits data ever transferred to CPU.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// DECODE PHASE ENUMERATION
// ============================================================================

/**
 * Execution phase affecting logits visibility
 */
enum llama_decode_phase {
    LLAMA_DECODE_PHASE_UNINITIALIZED = 0,
    LLAMA_DECODE_PHASE_PREFILL = 1,     // Prefill phase (CPU logits access allowed)
    LLAMA_DECODE_PHASE_DECODE = 2,      // Decode phase (CPU logits access forbidden)
    LLAMA_DECODE_PHASE_COMPLETE = 3,    // Decode complete
};

// ============================================================================
// LOGITS ACCESS MODE ENUMERATION
// ============================================================================

/**
 * Modes of logits access
 */
enum llama_logits_access_mode {
    LLAMA_LOGITS_ACCESS_NONE = 0,
    LLAMA_LOGITS_ACCESS_GPU_RESIDENT = 1,    // GPU-resident (decode phase)
    LLAMA_LOGITS_ACCESS_CPU_READABLE = 2,    // CPU-readable (prefill, debug)
    LLAMA_LOGITS_ACCESS_CPU_FORBIDDEN = 3,   // CPU access forbidden
};

// ============================================================================
// CPU LOGITS ACCESS OPERATION ENUMERATION
// ============================================================================

/**
 * CPU operations on logits that should be blocked
 */
enum llama_cpu_logits_operation {
    LLAMA_LOGITS_OP_NONE = 0,
    LLAMA_LOGITS_OP_GET_DATA = 1,           // ggml_get_data
    LLAMA_LOGITS_OP_GET_DATA_F32 = 2,       // ggml_get_data_f32
    LLAMA_LOGITS_OP_BACKEND_TENSOR_GET = 3, // ggml_backend_tensor_get
    LLAMA_LOGITS_OP_CPU_BUFFER_VIEW = 4,    // CPU buffer view mapping
    LLAMA_LOGITS_OP_HOST_COPY = 5,          // cudaMemcpy to host
    LLAMA_LOGITS_OP_INSPECTION = 6,         // CPU inspection/debug read
};

// ============================================================================
// LOGITS VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-exclusive logits enforcement
 */
enum llama_logits_violation {
    LLAMA_LOGITS_VIOLATION_NONE = 0,
    LLAMA_LOGITS_VIOLATION_CPU_READ = 1,           // CPU read during decode
    LLAMA_LOGITS_VIOLATION_HOST_COPY = 2,          // Host copy during decode
    LLAMA_LOGITS_VIOLATION_CPU_VIEW_MAP = 3,       // CPU view mapping during decode
    LLAMA_LOGITS_VIOLATION_GET_DATA_CALLED = 4,    // get_data() during decode
    LLAMA_LOGITS_VIOLATION_DEBUG_DUMP = 5,         // Debug dump during decode
    LLAMA_LOGITS_VIOLATION_MATERIALIZATION = 6,    // Logits materialization on CPU
    LLAMA_LOGITS_VIOLATION_PHASE_MISMATCH = 7,     // Phase/access mode mismatch
};

// ============================================================================
// GPU LOGITS BUFFER STATE ENUMERATION
// ============================================================================

/**
 * State of GPU logits buffer
 */
enum llama_gpu_logits_buffer_state {
    LLAMA_GPU_LOGITS_UNINITIALIZED = 0,
    LLAMA_GPU_LOGITS_ALLOCATED = 1,       // Buffer allocated on GPU
    LLAMA_GPU_LOGITS_POPULATED = 2,       // Buffer contains logits
    LLAMA_GPU_LOGITS_READY_FOR_SAMPLING = 3, // Ready for sampling kernels
    LLAMA_GPU_LOGITS_ERROR = 4,
};

// ============================================================================
// LOGITS MATERIALIZATION STATE ENUMERATION
// ============================================================================

/**
 * State of logits materialization status
 */
enum llama_logits_materialization_state {
    LLAMA_LOGITS_MATERIALIZATION_NONE = 0,
    LLAMA_LOGITS_MATERIALIZATION_ALLOWED = 1,   // Materialization allowed (prefill)
    LLAMA_LOGITS_MATERIALIZATION_BLOCKED = 2,   // Materialization blocked (decode)
    LLAMA_LOGITS_MATERIALIZATION_ATTEMPTED = 3, // Materialization attempted illegally
};

// ============================================================================
// LOGITS CONFIGURATION RECORD
// ============================================================================

/**
 * Configuration for GPU-exclusive logits management
 */
struct llama_gpu_logits_config {
    bool gpu_exclusive_logits;              // Enforce GPU-exclusive logits?
    bool logits_cpu_access_forbidden;       // Forbid CPU access during decode?
    enum llama_decode_phase current_phase;  // Current execution phase
    enum llama_logits_access_mode access_mode; // Current access mode
    bool cpu_logits_materialization_allowed; // Allow CPU materialization?
    bool enforce_gpu_resident_only;         // Enforce GPU-resident only?
    bool phase_aware_access;                // Phase-aware access control?
};

// ============================================================================
// LOGITS EXECUTION RECORD
// ============================================================================

/**
 * Record of logits access during execution
 */
struct llama_gpu_logits_execution_record {
    enum llama_decode_phase phase;         // Execution phase
    enum llama_logits_access_mode access_mode; // Access mode used
    enum llama_gpu_logits_buffer_state buffer_state; // Buffer state
    uint64_t timestamp_ns;                 // When accessed
    uint32_t tokens_processed;             // Tokens processed
    int cpu_violations;                    // Violations detected
    enum llama_logits_violation last_violation; // Last violation type
    bool cpu_attempted_read;               // CPU attempted read?
    bool gpu_resident_maintained;          // GPU residency maintained?
};

// ============================================================================
// LOGITS STATE RECORD
// ============================================================================

/**
 * Global state of GPU-exclusive logits during decode
 */
struct llama_gpu_logits_state_record {
    enum llama_decode_phase current_phase;       // Current execution phase
    enum llama_logits_access_mode current_access_mode; // Current access mode
    enum llama_gpu_logits_buffer_state buffer_state;   // Logits buffer state
    enum llama_logits_materialization_state materialization_state; // Materialization state
    bool cpu_logits_access_blocked;              // CPU access blocked?
    bool gpu_logits_resident;                    // Logits GPU-resident?
    int total_violations;                        // Total violations
    enum llama_logits_violation last_violation;  // Last violation type
    uint64_t total_tokens_processed;             // Total tokens processed
    uint64_t total_gpu_residency_ns;             // Total GPU residency time
};

// ============================================================================
// LOGITS VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU-exclusive logits
 */
struct llama_gpu_logits_validation_state {
    struct llama_gpu_logits_config config;
    struct llama_gpu_logits_state_record state_record;
    struct llama_gpu_logits_execution_record last_execution;
    int total_logits_accesses;
    int total_violations;
    bool enforcement_strict;                // Abort on violation vs log only
    bool debug_logits_access;               // Debug output
    bool verify_gpu_residency;              // Verify GPU residency?
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_logits_gpu_init(void);
int llama_logits_gpu_configure_exclusive(
    bool gpu_exclusive,
    bool cpu_forbidden,
    bool phase_aware
);

// Phase management
int llama_logits_gpu_set_decode_phase(enum llama_decode_phase phase);
enum llama_decode_phase llama_logits_gpu_get_current_phase(void);
int llama_logits_gpu_is_decode_phase(void);

// Logits access control (5 enforcement points: 1-5)
int llama_logits_gpu_queue_logits_computation(void);
int llama_logits_gpu_keep_logits_on_gpu(void);
int llama_logits_gpu_forbid_cpu_logits_read(void);
int llama_logits_gpu_forbid_cpu_logits_materialization(void);
int llama_logits_gpu_assert_logits_gpu_resident(void);

// CPU logits operation blocking (3 enforcement points: 6-8)
int llama_logits_gpu_forbid_get_data(void);
int llama_logits_gpu_forbid_backend_tensor_get(void);
int llama_logits_gpu_forbid_cpu_buffer_view(void);

// GPU residency enforcement (2 enforcement points: 9-10)
int llama_logits_gpu_verify_no_host_copy(void);
int llama_logits_gpu_verify_gpu_exclusive_access(void);

// Violation detection
int llama_logits_gpu_detect_cpu_read(void);
int llama_logits_gpu_detect_host_copy(void);
int llama_logits_gpu_detect_cpu_view_map(void);
int llama_logits_gpu_detect_get_data_call(void);
int llama_logits_gpu_detect_debug_dump(void);
int llama_logits_gpu_detect_materialization_attempt(void);
int llama_logits_gpu_detect_phase_mismatch(void);

// GPU state management
int llama_logits_gpu_set_buffer_allocated(void);
int llama_logits_gpu_set_buffer_populated(void);
int llama_logits_gpu_set_ready_for_sampling(void);

// Query and verification functions
struct llama_gpu_logits_state_record llama_logits_gpu_get_state_record(void);
struct llama_gpu_logits_execution_record llama_logits_gpu_get_last_execution(void);
enum llama_logits_access_mode llama_logits_gpu_get_current_access_mode(void);
enum llama_gpu_logits_buffer_state llama_logits_gpu_get_buffer_state(void);

// Verification functions
int llama_logits_gpu_verify_gpu_resident(void);
int llama_logits_gpu_verify_cpu_access_forbidden(void);
int llama_logits_gpu_verify_no_host_materializations(void);
int llama_logits_gpu_verify_decode_phase_compliance(void);
int llama_logits_gpu_verify_minimal_cpu_overhead(void);
int llama_logits_gpu_verify_logits_only_token_crosses_pcie(void);

// Diagnostics and logging
void llama_logits_gpu_log_decode_phase_started(void);
void llama_logits_gpu_log_gpu_exclusive_mode_enabled(void);
void llama_logits_gpu_log_cpu_read_blocked(void);
void llama_logits_gpu_print_logits_state(void);
void llama_logits_gpu_print_execution_stats(void);
void llama_logits_gpu_print_violation_summary(void);

// Violation reporting
void llama_logits_gpu_report_violation(
    enum llama_logits_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_logits_gpu_set_enforcement_strict(bool strict);
bool llama_logits_gpu_get_enforcement_strict(void);
void llama_logits_gpu_set_debug_output(bool debug);
void llama_logits_gpu_set_verify_gpu_residency(bool verify);

// Self-test suite
int llama_logits_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_decode_phase_name(
    enum llama_decode_phase phase
) {
    switch (phase) {
        case LLAMA_DECODE_PHASE_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_DECODE_PHASE_PREFILL: return "PREFILL";
        case LLAMA_DECODE_PHASE_DECODE: return "DECODE";
        case LLAMA_DECODE_PHASE_COMPLETE: return "COMPLETE";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_logits_access_mode_name(
    enum llama_logits_access_mode mode
) {
    switch (mode) {
        case LLAMA_LOGITS_ACCESS_NONE: return "NONE";
        case LLAMA_LOGITS_ACCESS_GPU_RESIDENT: return "GPU_RESIDENT";
        case LLAMA_LOGITS_ACCESS_CPU_READABLE: return "CPU_READABLE";
        case LLAMA_LOGITS_ACCESS_CPU_FORBIDDEN: return "CPU_FORBIDDEN";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_logits_violation_name(
    enum llama_logits_violation violation
) {
    switch (violation) {
        case LLAMA_LOGITS_VIOLATION_NONE: return "NONE";
        case LLAMA_LOGITS_VIOLATION_CPU_READ: return "CPU_READ";
        case LLAMA_LOGITS_VIOLATION_HOST_COPY: return "HOST_COPY";
        case LLAMA_LOGITS_VIOLATION_CPU_VIEW_MAP: return "CPU_VIEW_MAP";
        case LLAMA_LOGITS_VIOLATION_GET_DATA_CALLED: return "GET_DATA_CALLED";
        case LLAMA_LOGITS_VIOLATION_DEBUG_DUMP: return "DEBUG_DUMP";
        case LLAMA_LOGITS_VIOLATION_MATERIALIZATION: return "MATERIALIZATION";
        case LLAMA_LOGITS_VIOLATION_PHASE_MISMATCH: return "PHASE_MISMATCH";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
