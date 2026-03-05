/**
 * SECTION 31: Eliminate Hybrid KV Cache Modes
 * Header
 *
 * This file implements GPU-only KV cache mode enforcement.
 * Hybrid KV cache modes (CPU+GPU split) are forbidden during decode.
 * Decode uses one and only one KV cache backend: GPU.
 * CPU-resident KV cache is not permitted once decode begins.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// KV BACKEND MODE ENUMERATION
// ============================================================================

/**
 * KV cache backend modes (single, not hybrid)
 */
enum llama_kv_backend_mode {
    LLAMA_KV_BACKEND_NONE = 0,
    LLAMA_KV_BACKEND_CPU = 1,           // CPU-only (prefill only)
    LLAMA_KV_BACKEND_GPU = 2,           // GPU-only (decode required)
    LLAMA_KV_BACKEND_HYBRID = 3,        // CPU+GPU hybrid (forbidden in decode)
};

// ============================================================================
// GPU KV EXCLUSIVITY STATE ENUMERATION
// ============================================================================

/**
 * State of GPU-only KV cache enforcement during decode
 */
enum llama_gpu_kv_exclusivity_state {
    LLAMA_GPU_KV_EXCLUSIVITY_UNINITIALIZED = 0,
    LLAMA_GPU_KV_EXCLUSIVITY_PREFILL_PHASE = 1,  // Prefill allowed hybrid
    LLAMA_GPU_KV_EXCLUSIVITY_DECODE_READY = 2,   // Ready to enforce GPU-only
    LLAMA_GPU_KV_EXCLUSIVITY_DECODE_ACTIVE = 3,  // Decode phase: GPU-only
    LLAMA_GPU_KV_EXCLUSIVITY_COMPLETE = 4,
    LLAMA_GPU_KV_EXCLUSIVITY_ERROR = 5,
};

// ============================================================================
// HYBRID KV VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of GPU-only KV cache requirement
 */
enum llama_hybrid_kv_violation {
    LLAMA_HYBRID_KV_VIOLATION_NONE = 0,
    LLAMA_HYBRID_KV_VIOLATION_HYBRID_MODE_DECODE = 1,   // Hybrid mode in decode
    LLAMA_HYBRID_KV_VIOLATION_CPU_KV_RESIDENCY = 2,     // CPU KV during decode
    LLAMA_HYBRID_KV_VIOLATION_CPU_KV_ACCESS = 3,        // CPU accessed KV
    LLAMA_HYBRID_KV_VIOLATION_PER_LAYER_BRANCHING = 4,  // Per-layer CPU/GPU branch
    LLAMA_HYBRID_KV_VIOLATION_KV_FALLBACK = 5,          // CPU fallback under pressure
    LLAMA_HYBRID_KV_VIOLATION_HOST_VISIBLE_POINTER = 6, // Host-visible KV pointer
    LLAMA_HYBRID_KV_VIOLATION_HYBRID_PATH_SELECTED = 7, // Hybrid path selected
    LLAMA_HYBRID_KV_VIOLATION_INCOMPLETE_GPU_ALLOCATION = 8, // Incomplete GPU KV
};

// ============================================================================
// KV BACKEND CONFIGURATION
// ============================================================================

/**
 * Configuration for GPU-only KV cache enforcement
 */
struct llama_gpu_kv_backend_config {
    bool enforce_gpu_only_decode;       // Enforce GPU-only during decode?
    bool forbid_hybrid_modes;           // Forbid hybrid modes completely?
    bool fail_on_incomplete_gpu_alloc;  // Fail if GPU KV incomplete?
    bool validate_kv_residency;         // Validate KV residency at decode start?
    uint32_t num_layers;                // Number of transformer layers
    bool debug_kv_backend;              // Debug output?
};

// ============================================================================
// PER-LAYER KV RESIDENCY RECORD
// ============================================================================

/**
 * Tracks KV residency for a single layer
 */
struct llama_gpu_kv_layer_residency {
    uint32_t layer_id;                  // Layer identifier
    enum llama_kv_backend_mode backend; // Where KV is resident
    bool gpu_allocated;                 // Is GPU KV allocated?
    bool cpu_allocated;                 // Is CPU KV allocated?
    uint64_t gpu_size_bytes;            // GPU KV size
    uint64_t cpu_size_bytes;            // CPU KV size
};

// ============================================================================
// KV BACKEND STATE RECORD
// ============================================================================

/**
 * Current state of GPU-only KV enforcement
 */
struct llama_gpu_kv_backend_state_record {
    enum llama_gpu_kv_exclusivity_state state;         // Current state
    enum llama_kv_backend_mode decode_backend_mode;    // Backend for decode phase
    uint32_t num_layers;                               // Number of layers
    uint32_t layers_gpu_only;                          // Layers with GPU-only KV
    uint32_t layers_with_cpu_kv;                       // Layers still with CPU KV
    uint64_t total_gpu_kv_bytes;                       // Total GPU KV allocated
    int total_violations;                              // Total violations
    enum llama_hybrid_kv_violation last_violation;    // Last violation
};

// ============================================================================
// HYBRID PATH TRACKING
// ============================================================================

/**
 * Tracks usage of hybrid KV code paths
 */
struct llama_gpu_hybrid_kv_path_record {
    uint64_t hybrid_path_attempts;      // Attempts to use hybrid path
    uint64_t per_layer_branch_attempts; // Per-layer branching attempts
    uint64_t cpu_fallback_attempts;     // CPU fallback attempts
    uint64_t host_visible_pointer_attempts; // Host pointer creation attempts
    uint32_t reserved_1;
};

// ============================================================================
// GPU-ONLY KV VALIDATION STATE
// ============================================================================

/**
 * Global validation state for GPU-only KV enforcement
 */
struct llama_gpu_kv_hybrid_elimination_validation_state {
    struct llama_gpu_kv_backend_config config;
    struct llama_gpu_kv_backend_state_record state_record;
    struct llama_gpu_hybrid_kv_path_record hybrid_path_record;
    int total_decode_starts;
    int total_violations;
    bool enforcement_strict;            // Abort on violation vs log only
    bool decode_phase_active;           // Is decode phase active?
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_hybrid_kv_elimination_gpu_init(void);
int llama_hybrid_kv_elimination_gpu_configure(
    bool enforce_gpu_only_decode,
    bool forbid_hybrid_modes,
    bool fail_on_incomplete_gpu_alloc,
    uint32_t num_layers
);

// Prefill and decode phase management
int llama_hybrid_kv_elimination_gpu_begin_prefill_phase(void);
int llama_hybrid_kv_elimination_gpu_end_prefill_phase(void);
int llama_hybrid_kv_elimination_gpu_begin_decode_phase(void);
int llama_hybrid_kv_elimination_gpu_end_decode_phase(void);

// KV backend validation (10 enforcement points: 1-10)
int llama_hybrid_kv_elimination_gpu_validate_gpu_only_kv_at_decode_start(void);
int llama_hybrid_kv_elimination_gpu_forbid_hybrid_kv_modes_in_decode(void);
int llama_hybrid_kv_elimination_gpu_forbid_cpu_kv_residency_in_decode(void);
int llama_hybrid_kv_elimination_gpu_forbid_per_layer_kv_branching(void);
int llama_hybrid_kv_elimination_gpu_forbid_cpu_kv_fallback_under_pressure(void);
int llama_hybrid_kv_elimination_gpu_forbid_host_visible_kv_pointers(void);
int llama_hybrid_kv_elimination_gpu_lock_kv_to_gpu_only(void);
int llama_hybrid_kv_elimination_gpu_verify_all_layers_gpu_kv(void);
int llama_hybrid_kv_elimination_gpu_verify_no_hybrid_paths_in_decode(void);
int llama_hybrid_kv_elimination_gpu_verify_gpu_kv_allocation_complete(void);

// Layer-specific KV validation
int llama_hybrid_kv_elimination_gpu_validate_layer_kv_backend(uint32_t layer_id);
int llama_hybrid_kv_elimination_gpu_check_layer_gpu_kv_allocated(uint32_t layer_id);
int llama_hybrid_kv_elimination_gpu_check_layer_no_cpu_kv(uint32_t layer_id);

// Hybrid path detection and blocking
int llama_hybrid_kv_elimination_gpu_detect_hybrid_mode_attempt(void);
int llama_hybrid_kv_elimination_gpu_detect_per_layer_branch_attempt(void);
int llama_hybrid_kv_elimination_gpu_detect_cpu_fallback_attempt(void);
int llama_hybrid_kv_elimination_gpu_detect_host_pointer_attempt(void);

// KV backend lock and enforcement
int llama_hybrid_kv_elimination_gpu_lock_kv_backend_to_gpu(void);
int llama_hybrid_kv_elimination_gpu_lock_all_layers_to_gpu_kv(void);
int llama_hybrid_kv_elimination_gpu_verify_kv_backend_locked(void);

// GPU KV allocation validation
int llama_hybrid_kv_elimination_gpu_validate_all_layers_gpu_allocated(void);
int llama_hybrid_kv_elimination_gpu_validate_total_gpu_kv_size(uint64_t total_bytes);
int llama_hybrid_kv_elimination_gpu_calculate_required_gpu_kv_bytes(uint64_t* out_bytes);

// Query and verification functions
struct llama_gpu_kv_backend_state_record llama_hybrid_kv_elimination_gpu_get_state_record(void);
struct llama_gpu_hybrid_kv_path_record llama_hybrid_kv_elimination_gpu_get_hybrid_path_record(void);
enum llama_gpu_kv_exclusivity_state llama_hybrid_kv_elimination_gpu_get_kv_state(void);
enum llama_kv_backend_mode llama_hybrid_kv_elimination_gpu_get_decode_backend_mode(void);

// Verification functions
int llama_hybrid_kv_elimination_gpu_verify_gpu_only_decode_ready(void);
int llama_hybrid_kv_elimination_gpu_verify_no_hybrid_modes_active(void);
int llama_hybrid_kv_elimination_gpu_verify_all_kv_gpu_resident(void);
int llama_hybrid_kv_elimination_gpu_verify_no_cpu_kv_present(void);
int llama_hybrid_kv_elimination_gpu_verify_no_hybrid_paths_reachable(void);

// Diagnostics and logging
void llama_hybrid_kv_elimination_gpu_log_gpu_only_kv_enforced(void);
void llama_hybrid_kv_elimination_gpu_log_decode_phase_started(void);
void llama_hybrid_kv_elimination_gpu_log_kv_backend_locked(void);
void llama_hybrid_kv_elimination_gpu_print_state(void);
void llama_hybrid_kv_elimination_gpu_print_layer_residency_status(void);
void llama_hybrid_kv_elimination_gpu_print_violation_summary(void);
void llama_hybrid_kv_elimination_gpu_print_hybrid_path_attempts(void);

// Violation reporting
void llama_hybrid_kv_elimination_gpu_report_violation(
    enum llama_hybrid_kv_violation violation_type,
    uint32_t layer_id,
    const char* details
);

// Enforcement mode control
void llama_hybrid_kv_elimination_gpu_set_enforcement_strict(bool strict);
bool llama_hybrid_kv_elimination_gpu_get_enforcement_strict(void);
void llama_hybrid_kv_elimination_gpu_set_debug_output(bool debug);

// Self-test suite
int llama_hybrid_kv_elimination_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_kv_backend_mode_name(enum llama_kv_backend_mode mode) {
    switch (mode) {
        case LLAMA_KV_BACKEND_NONE: return "NONE";
        case LLAMA_KV_BACKEND_CPU: return "CPU";
        case LLAMA_KV_BACKEND_GPU: return "GPU";
        case LLAMA_KV_BACKEND_HYBRID: return "HYBRID";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_gpu_kv_exclusivity_state_name(enum llama_gpu_kv_exclusivity_state state) {
    switch (state) {
        case LLAMA_GPU_KV_EXCLUSIVITY_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_KV_EXCLUSIVITY_PREFILL_PHASE: return "PREFILL_PHASE";
        case LLAMA_GPU_KV_EXCLUSIVITY_DECODE_READY: return "DECODE_READY";
        case LLAMA_GPU_KV_EXCLUSIVITY_DECODE_ACTIVE: return "DECODE_ACTIVE";
        case LLAMA_GPU_KV_EXCLUSIVITY_COMPLETE: return "COMPLETE";
        case LLAMA_GPU_KV_EXCLUSIVITY_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_hybrid_kv_violation_name(enum llama_hybrid_kv_violation violation) {
    switch (violation) {
        case LLAMA_HYBRID_KV_VIOLATION_NONE: return "NONE";
        case LLAMA_HYBRID_KV_VIOLATION_HYBRID_MODE_DECODE: return "HYBRID_MODE_IN_DECODE";
        case LLAMA_HYBRID_KV_VIOLATION_CPU_KV_RESIDENCY: return "CPU_KV_RESIDENCY";
        case LLAMA_HYBRID_KV_VIOLATION_CPU_KV_ACCESS: return "CPU_KV_ACCESS";
        case LLAMA_HYBRID_KV_VIOLATION_PER_LAYER_BRANCHING: return "PER_LAYER_BRANCHING";
        case LLAMA_HYBRID_KV_VIOLATION_KV_FALLBACK: return "KV_FALLBACK";
        case LLAMA_HYBRID_KV_VIOLATION_HOST_VISIBLE_POINTER: return "HOST_VISIBLE_POINTER";
        case LLAMA_HYBRID_KV_VIOLATION_HYBRID_PATH_SELECTED: return "HYBRID_PATH_SELECTED";
        case LLAMA_HYBRID_KV_VIOLATION_INCOMPLETE_GPU_ALLOCATION: return "INCOMPLETE_GPU_ALLOCATION";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif

