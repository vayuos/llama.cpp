/**
 * SECTION 3: Formalize Decode Admission Control (GPU-only eligibility)
 *
 * This file implements a strict decode admission control mechanism that allows
 * decode execution to begin ONLY if GPU eligibility is fully satisfied.
 * Decode must never start in a configuration where any decode-critical work
 * could execute on CPU.
 *
 * Core Principle:
 * "Decode execution is admitted only when GPU-exclusive execution is guaranteed.
 *  No decode begins in hybrid or degraded mode. Failure is immediate and final."
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>

// ============================================================================
// DECODE ADMISSION CONTROL ENUMS
// ============================================================================

/**
 * Enum defining the state of decode admission
 */
enum llama_decode_admission_state {
    LLAMA_ADMISSION_STATE_UNINITIALIZED = 0,    // Not yet checked
    LLAMA_ADMISSION_STATE_ELIGIBLE = 1,         // GPU-only eligibility satisfied
    LLAMA_ADMISSION_STATE_INELIGIBLE = 2,       // Fails at least one GPU criterion
    LLAMA_ADMISSION_STATE_ADMITTED = 3,         // Eligible AND decode has begun (locked)
    LLAMA_ADMISSION_STATE_TERMINATED = 4,       // Admitted decode was terminated
};

/**
 * Enum defining the specific eligibility failure reason
 */
enum llama_admission_failure_reason {
    LLAMA_ADMISSION_FAIL_UNKNOWN = 0,
    LLAMA_ADMISSION_FAIL_NO_GPU_BACKEND = 1,        // No GPU backend available
    LLAMA_ADMISSION_FAIL_DECODE_OP_CPU = 2,         // Decode-critical op resolves to CPU
    LLAMA_ADMISSION_FAIL_INVALID_CUDA_FEATURES = 3, // CUDA features missing
    LLAMA_ADMISSION_FAIL_KV_CACHE_NOT_GPU = 4,      // KV cache not GPU-resident
    LLAMA_ADMISSION_FAIL_BACKEND_NOT_FROZEN = 5,    // Backend selection not locked
    LLAMA_ADMISSION_FAIL_MIXED_CLASSIFICATION = 6,  // Task classification is ambiguous
    LLAMA_ADMISSION_FAIL_NON_CRITICAL_BLOCKING = 7, // NON_CRITICAL task blocking decode
};

// ============================================================================
// GPU ELIGIBILITY CRITERIA
// ============================================================================

/**
 * Structure defining GPU-only eligibility criteria for decode
 *
 * All criteria must be satisfied for decode to be admitted.
 */
struct llama_gpu_eligibility_criteria {
    // Criterion 1: GPU backend availability
    bool has_valid_gpu_backend;                 // At least one GPU backend available
    const char* available_gpu_backend;          // Name of available GPU backend

    // Criterion 2: Decode-critical ops backend resolution
    bool all_decode_critical_ops_gpu;           // All decode-critical ops have GPU backend
    int decode_critical_ops_on_cpu;             // Count of decode-critical ops on CPU (should be 0)
    const char* first_cpu_decode_op;            // First op found on CPU (if any)

    // Criterion 3: CUDA/GPU feature validation
    bool cuda_features_available;               // Required CUDA features present
    const char* missing_cuda_feature;           // Which feature is missing (if any)

    // Criterion 4: KV cache residency
    bool kv_cache_gpu_resident;                 // KV cache fully on GPU
    const char* kv_cache_location;              // Where KV cache is stored

    // Criterion 5: Backend selection frozen
    bool backend_selection_frozen;              // Backend choices cannot change
    const char* backend_freeze_reason;          // Why backend is or isn't frozen

    // Hierarchical Priority Info (for Advice)
    int32_t current_n_ctx;                      // Current context size
    int32_t current_n_batch;                    // Current batch size

    // Overall: All criteria met?
    bool all_criteria_satisfied;                // True only if all above are true
};

/**
 * Structure holding decode admission control state
 */
struct llama_decode_admission_control {
    // Admission state machine
    enum llama_decode_admission_state state;    // Current admission state
    enum llama_admission_failure_reason failure_reason;  // Why admission failed (if failed)

    // Eligibility information
    struct llama_gpu_eligibility_criteria eligibility;   // Current eligibility status

    // Admission lock (once admitted, cannot re-check eligibility)
    bool admission_locked;                      // True once decode begins
    bool decode_has_started;                    // True once first token generated

    // Diagnostics and logging
    std::string detailed_failure_message;       // Human-readable failure explanation
    int eligibility_check_count;                // How many times eligibility was checked
    uint64_t admission_time_us;                 // When decode was admitted (microseconds)
};

// ============================================================================
// ELIGIBILITY CHECK FUNCTIONS
// ============================================================================

/**
 * Check GPU backend availability
 * Criterion 1: At least one valid GPU backend exists
 */
int llama_admission_check_gpu_backend_available(
    struct llama_gpu_eligibility_criteria* criteria,
    const char** available_backends,
    int num_backends
);

/**
 * Check that all decode-critical ops are GPU-bound
 * Criterion 2: No decode-critical op resolves to CPU
 */
int llama_admission_check_no_cpu_decode_ops(
    struct llama_gpu_eligibility_criteria* criteria,
    const char** decode_critical_ops,
    const char** op_backends,
    int num_ops
);

/**
 * Check CUDA/GPU feature availability
 * Criterion 3: Required features present
 */
int llama_admission_check_cuda_features(
    struct llama_gpu_eligibility_criteria* criteria,
    const char** required_features,
    const char** available_features,
    int num_required,
    int num_available
);

/**
 * Check KV cache residency
 * Criterion 4: KV cache is on GPU
 */
int llama_admission_check_kv_cache_gpu_resident(
    struct llama_gpu_eligibility_criteria* criteria,
    const char* kv_cache_location
);

/**
 * Check backend selection is frozen
 * Criterion 5: Backend choices cannot change
 */
int llama_admission_check_backend_frozen(
    struct llama_gpu_eligibility_criteria* criteria,
    bool backend_is_frozen,
    const char* freeze_reason
);

/**
 * Perform exhaustive GPU-only eligibility check
 * All five criteria must pass
 * Returns 0 if eligible, -1 if ineligible
 */
int llama_admission_check_gpu_eligibility(
    struct llama_gpu_eligibility_criteria* criteria
);

// ============================================================================
// ADMISSION CONTROL FUNCTIONS
// ============================================================================

/**
 * Initialize decode admission control
 * Called once per context creation
 */
int llama_decode_admission_init(
    struct llama_decode_admission_control* admission
);

/**
 * Perform exhaustive GPU eligibility check and gate decode
 *
 * This is the main admission gate function.
 * Called exactly once, before the first decode token.
 *
 * Returns:
 *  0 = Decode ADMITTED (GPU-exclusive path guaranteed)
 * -1 = Decode INELIGIBLE (would violate GPU-exclusive invariant)
 */
int llama_decode_admission_check_and_gate(
    struct llama_decode_admission_control* admission,
    struct llama_gpu_eligibility_criteria* criteria
);

/**
 * Lock decode admission (prevent re-checking eligibility mid-run)
 * Called immediately after first token is generated
 *
 * Once locked:
 * - No new eligibility checks are allowed
 * - Backend selection cannot change
 * - If conditions change (OOM, backend invalidation), decode terminates
 */
int llama_decode_admission_lock(
    struct llama_decode_admission_control* admission
);

/**
 * Verify that decode is admitted and locked
 * Called at key decode checkpoints to ensure invariant holds
 */
int llama_decode_admission_verify_locked(
    const struct llama_decode_admission_control* admission
);

/**
 * Terminate decode session due to mid-run condition change
 * If GPU-exclusive conditions are violated during decode, terminate immediately
 * Do not degrade to CPU execution
 */
int llama_decode_admission_terminate_session(
    struct llama_decode_admission_control* admission,
    const char* termination_reason
);

// ============================================================================
// DIAGNOSTICS AND FAILURE REPORTING
// ============================================================================

/**
 * Convert admission failure reason to human-readable string
 */
static inline const char* llama_admission_failure_name(enum llama_admission_failure_reason reason) {
    switch (reason) {
        case LLAMA_ADMISSION_FAIL_UNKNOWN:
            return "UNKNOWN_REASON";
        case LLAMA_ADMISSION_FAIL_NO_GPU_BACKEND:
            return "NO_GPU_BACKEND";
        case LLAMA_ADMISSION_FAIL_DECODE_OP_CPU:
            return "DECODE_CRITICAL_OP_ON_CPU";
        case LLAMA_ADMISSION_FAIL_INVALID_CUDA_FEATURES:
            return "INVALID_CUDA_FEATURES";
        case LLAMA_ADMISSION_FAIL_KV_CACHE_NOT_GPU:
            return "KV_CACHE_NOT_GPU_RESIDENT";
        case LLAMA_ADMISSION_FAIL_BACKEND_NOT_FROZEN:
            return "BACKEND_SELECTION_NOT_FROZEN";
        case LLAMA_ADMISSION_FAIL_MIXED_CLASSIFICATION:
            return "MIXED_TASK_CLASSIFICATION";
        case LLAMA_ADMISSION_FAIL_NON_CRITICAL_BLOCKING:
            return "NON_CRITICAL_TASK_BLOCKING_DECODE";
        default:
            return "(invalid)";
    }
}

/**
 * Convert admission state to human-readable string
 */
static inline const char* llama_admission_state_name(enum llama_decode_admission_state state) {
    switch (state) {
        case LLAMA_ADMISSION_STATE_UNINITIALIZED:
            return "UNINITIALIZED";
        case LLAMA_ADMISSION_STATE_ELIGIBLE:
            return "ELIGIBLE";
        case LLAMA_ADMISSION_STATE_INELIGIBLE:
            return "INELIGIBLE";
        case LLAMA_ADMISSION_STATE_ADMITTED:
            return "ADMITTED";
        case LLAMA_ADMISSION_STATE_TERMINATED:
            return "TERMINATED";
        default:
            return "(invalid)";
    }
}

/**
 * Print detailed admission failure diagnostics
 * On admission failure, report:
 * - Which op failed eligibility
 * - Why GPU execution was unavailable
 * - Which invariant would have been violated
 */
void llama_admission_print_failure_diagnostics(
    const struct llama_decode_admission_control* admission,
    const struct llama_gpu_eligibility_criteria* criteria
);

/**
 * Print admission status summary
 * Shows:
 * - Decode admitted: YES/NO
 * - GPU-exclusive path confirmed
 * - All five eligibility criteria status
 */
void llama_admission_print_status_summary(
    const struct llama_decode_admission_control* admission
);

// ============================================================================
// EXPLICIT ADMISSION CONTROL STATEMENT
// ============================================================================

/**
 * Print the decode admission control statement explicitly
 */
void llama_print_decode_admission_statement(void);

// ============================================================================
// VALIDATION AND SELF-TEST
// ============================================================================

/**
 * Self-test: verify admission control mechanism works correctly
 */
int llama_decode_admission_selftest(void);

