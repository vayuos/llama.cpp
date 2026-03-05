/**
 * SECTION 30: Freeze KV Cache Layout Before Decode
 * Header
 *
 * This file implements immutable KV-cache layout enforcement.
 * KV cache layout (shape, strides, offsets, residency) is fully determined before decode.
 * Layout cannot change during decode phase; CPU cannot resize, repartition, or adjust KV cache.
 * GPU operates on fixed KV layout for all tokens; no runtime reconfiguration allowed.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// KV LAYOUT FREEZE MODE ENUMERATION
// ============================================================================

/**
 * KV layout freeze modes
 */
enum llama_kv_layout_freeze_mode {
    LLAMA_KV_LAYOUT_FREEZE_NONE = 0,
    LLAMA_KV_LAYOUT_FREEZE_DISABLED = 1,  // Layout can change (deprecated)
    LLAMA_KV_LAYOUT_FREEZE_ENABLED = 2,   // Layout immutable during decode
};

// ============================================================================
// KV LAYOUT PHASE ENUMERATION
// ============================================================================

/**
 * KV layout phase tracking
 */
enum llama_kv_layout_phase {
    LLAMA_KV_LAYOUT_PHASE_UNINITIALIZED = 0,
    LLAMA_KV_LAYOUT_PHASE_SETUP = 1,      // Layout being configured
    LLAMA_KV_LAYOUT_PHASE_FROZEN = 2,     // Layout frozen before decode
    LLAMA_KV_LAYOUT_PHASE_DECODE = 3,     // Decode in progress (layout immutable)
    LLAMA_KV_LAYOUT_PHASE_COMPLETE = 4,   // Decode complete
};

// ============================================================================
// KV LAYOUT VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of KV layout immutability
 */
enum llama_kv_layout_freeze_violation {
    LLAMA_KV_LAYOUT_FREEZE_VIOLATION_NONE = 0,
    LLAMA_KV_LAYOUT_FREEZE_VIOLATION_CPU_RESIZE = 1,       // CPU attempted resize
    LLAMA_KV_LAYOUT_FREEZE_VIOLATION_CPU_REPARTITION = 2,  // CPU attempted repartition
    LLAMA_KV_LAYOUT_FREEZE_VIOLATION_CPU_REALLOC = 3,      // CPU attempted realloc
    LLAMA_KV_LAYOUT_FREEZE_VIOLATION_LAYOUT_CHECK = 4,     // CPU checked layout bounds
    LLAMA_KV_LAYOUT_FREEZE_VIOLATION_HYBRID_PATH = 5,      // Hybrid CPU/GPU KV attempted
    LLAMA_KV_LAYOUT_FREEZE_VIOLATION_WINDOWING_CHANGE = 6, // Windowing mode changed
    LLAMA_KV_LAYOUT_FREEZE_VIOLATION_POINTER_CHANGE = 7,   // KV pointer changed
};

// ============================================================================
// KV LAYOUT DESCRIPTOR
// ============================================================================

/**
 * Immutable KV cache layout descriptor
 */
struct llama_kv_layout_descriptor {
    uint32_t context_length;              // Max context length
    uint32_t num_layers;                  // Number of model layers
    uint32_t num_heads;                   // Attention heads
    uint32_t head_dim;                    // Head dimension
    uint32_t vocab_size;                  // Vocabulary size
    uint64_t kv_cache_size_bytes;         // Total KV cache size
    uint64_t per_layer_size_bytes;        // Size per layer
    uint64_t per_token_size_bytes;        // Size per token per layer
    uint32_t max_seq_len;                 // Maximum sequence length
    uint32_t rope_base;                   // RoPE base for positioning
    uint32_t reserved_1;
    uint32_t reserved_2;
};

// ============================================================================
// KV LAYOUT STATE RECORD
// ============================================================================

/**
 * Current state of KV layout freeze
 */
struct llama_kv_layout_freeze_state_record {
    enum llama_kv_layout_freeze_mode mode;      // Freeze mode
    enum llama_kv_layout_phase phase;           // Current phase
    struct llama_kv_layout_descriptor layout;   // Immutable layout descriptor
    bool layout_locked;                         // Layout locked to GPU?
    bool cpu_modifications_forbidden;           // CPU changes forbidden?
    int total_violations;                       // Total violations detected
    enum llama_kv_layout_freeze_violation last_violation; // Last violation
    uint64_t freeze_timestamp_ns;               // When layout was frozen
};

// ============================================================================
// KV LAYOUT VALIDATION STATE
// ============================================================================

/**
 * Global validation state for KV layout freeze
 */
struct llama_kv_layout_freeze_validation_state {
    struct llama_kv_layout_freeze_state_record state_record;
    int total_freeze_checks;
    int total_violations;
    bool enforcement_strict;                // Abort on violation vs log only
    bool debug_kv_layout_freeze;            // Debug output
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_kv_layout_freeze_init(void);
int llama_kv_layout_freeze_configure(
    bool freeze_enabled,
    bool cpu_modifications_forbidden
);

// KV layout setup and freezing
int llama_kv_layout_freeze_compute_layout(
    uint32_t context_length,
    uint32_t num_layers,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t vocab_size,
    uint32_t max_seq_len
);

int llama_kv_layout_freeze_allocate_kv_cache(void);
int llama_kv_layout_freeze_freeze_layout_before_decode(void);

// Decode-time enforcement (10 enforcement points: 1-10)
int llama_kv_layout_freeze_queue_decode_kernel(void);
int llama_kv_layout_freeze_keep_layout_immutable(void);
int llama_kv_layout_freeze_forbid_cpu_resize(void);
int llama_kv_layout_freeze_forbid_cpu_repartition(void);
int llama_kv_layout_freeze_forbid_cpu_realloc(void);
int llama_kv_layout_freeze_forbid_layout_checks(void);
int llama_kv_layout_freeze_forbid_hybrid_kv_path(void);
int llama_kv_layout_freeze_forbid_windowing_change(void);
int llama_kv_layout_freeze_verify_no_pointer_change(void);
int llama_kv_layout_freeze_verify_layout_immutable(void);

// Violation detection
int llama_kv_layout_freeze_detect_cpu_resize(void);
int llama_kv_layout_freeze_detect_cpu_repartition(void);
int llama_kv_layout_freeze_detect_cpu_realloc(void);
int llama_kv_layout_freeze_detect_layout_check(void);
int llama_kv_layout_freeze_detect_hybrid_path(void);
int llama_kv_layout_freeze_detect_windowing_change(void);
int llama_kv_layout_freeze_detect_pointer_change(void);

// Phase management
int llama_kv_layout_freeze_enter_setup_phase(void);
int llama_kv_layout_freeze_exit_setup_enter_frozen(void);
int llama_kv_layout_freeze_enter_decode_phase(void);
int llama_kv_layout_freeze_exit_decode_enter_complete(void);

// Query and verification
struct llama_kv_layout_freeze_state_record llama_kv_layout_freeze_get_state_record(void);
struct llama_kv_layout_descriptor llama_kv_layout_freeze_get_layout_descriptor(void);
enum llama_kv_layout_phase llama_kv_layout_freeze_get_current_phase(void);

// Verification functions
int llama_kv_layout_freeze_verify_layout_frozen(void);
int llama_kv_layout_freeze_verify_cpu_modifications_forbidden(void);
int llama_kv_layout_freeze_verify_layout_locked(void);
int llama_kv_layout_freeze_verify_no_cpu_entry_point(void);
int llama_kv_layout_freeze_verify_layout_consistency(void);
int llama_kv_layout_freeze_verify_no_hybrid_path(void);
int llama_kv_layout_freeze_verify_no_violations(void);

// Diagnostics and logging
void llama_kv_layout_freeze_log_layout_frozen(void);
void llama_kv_layout_freeze_log_decode_entered(void);
void llama_kv_layout_freeze_print_state(void);
void llama_kv_layout_freeze_print_layout_descriptor(void);
void llama_kv_layout_freeze_print_violation_summary(void);

// Violation reporting
void llama_kv_layout_freeze_report_violation(
    enum llama_kv_layout_freeze_violation violation_type,
    const char* details
);

// Enforcement mode control
void llama_kv_layout_freeze_set_enforcement_strict(bool strict);
bool llama_kv_layout_freeze_get_enforcement_strict(void);
void llama_kv_layout_freeze_set_debug_output(bool debug);

// Self-test suite
int llama_kv_layout_freeze_selftest(void);

// Helper/inline functions
static inline const char* llama_kv_layout_freeze_mode_name(
    enum llama_kv_layout_freeze_mode mode
) {
    switch (mode) {
        case LLAMA_KV_LAYOUT_FREEZE_NONE: return "NONE";
        case LLAMA_KV_LAYOUT_FREEZE_DISABLED: return "DISABLED";
        case LLAMA_KV_LAYOUT_FREEZE_ENABLED: return "ENABLED";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_kv_layout_phase_name(
    enum llama_kv_layout_phase phase
) {
    switch (phase) {
        case LLAMA_KV_LAYOUT_PHASE_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_KV_LAYOUT_PHASE_SETUP: return "SETUP";
        case LLAMA_KV_LAYOUT_PHASE_FROZEN: return "FROZEN";
        case LLAMA_KV_LAYOUT_PHASE_DECODE: return "DECODE";
        case LLAMA_KV_LAYOUT_PHASE_COMPLETE: return "COMPLETE";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_kv_layout_freeze_violation_name(
    enum llama_kv_layout_freeze_violation violation
) {
    switch (violation) {
        case LLAMA_KV_LAYOUT_FREEZE_VIOLATION_NONE: return "NONE";
        case LLAMA_KV_LAYOUT_FREEZE_VIOLATION_CPU_RESIZE: return "CPU_RESIZE";
        case LLAMA_KV_LAYOUT_FREEZE_VIOLATION_CPU_REPARTITION: return "CPU_REPARTITION";
        case LLAMA_KV_LAYOUT_FREEZE_VIOLATION_CPU_REALLOC: return "CPU_REALLOC";
        case LLAMA_KV_LAYOUT_FREEZE_VIOLATION_LAYOUT_CHECK: return "LAYOUT_CHECK";
        case LLAMA_KV_LAYOUT_FREEZE_VIOLATION_HYBRID_PATH: return "HYBRID_PATH";
        case LLAMA_KV_LAYOUT_FREEZE_VIOLATION_WINDOWING_CHANGE: return "WINDOWING_CHANGE";
        case LLAMA_KV_LAYOUT_FREEZE_VIOLATION_POINTER_CHANGE: return "POINTER_CHANGE";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
