/**
 * SECTION 30: Prohibit Per-Token Host↔Device Transfers
 * Header
 *
 * This file implements comprehensive transfer prohibition enforcement.
 * No decode-critical tensor or buffer may cross PCIe during decode.
 * Only the final selected token ID is permitted to cross PCIe per token.
 * All other data remains GPU-resident throughout decode execution.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TRANSFER PROHIBITION MODE ENUMERATION
// ============================================================================

/**
 * Transfer prohibition enforcement modes
 */
enum llama_transfer_prohibition_mode {
    LLAMA_TRANSFER_PROHIBITION_NONE = 0,
    LLAMA_TRANSFER_PROHIBITION_DISABLED = 1,  // Disabled (deprecated)
    LLAMA_TRANSFER_PROHIBITION_ENABLED = 2,   // Enabled (GPU-exclusive decode)
};

// ============================================================================
// GPU TRANSFER PROHIBITION STATE ENUMERATION
// ============================================================================

/**
 * State of transfer prohibition enforcement during decode
 */
enum llama_gpu_transfer_prohibition_state {
    LLAMA_GPU_TRANSFER_PROHIBITION_UNINITIALIZED = 0,
    LLAMA_GPU_TRANSFER_PROHIBITION_INITIALIZED = 1,
    LLAMA_GPU_TRANSFER_PROHIBITION_DECODE_ACTIVE = 2,
    LLAMA_GPU_TRANSFER_PROHIBITION_MONITORING = 3,
    LLAMA_GPU_TRANSFER_PROHIBITION_COMPLETE = 4,
    LLAMA_GPU_TRANSFER_PROHIBITION_ERROR = 5,
};

// ============================================================================
// TRANSFER TYPE ENUMERATION
// ============================================================================

/**
 * Types of host↔device transfers
 */
enum llama_transfer_type {
    LLAMA_TRANSFER_TYPE_NONE = 0,
    LLAMA_TRANSFER_TYPE_H2D = 1,              // Host to Device
    LLAMA_TRANSFER_TYPE_D2H = 2,              // Device to Host
    LLAMA_TRANSFER_TYPE_P2P = 3,              // Peer-to-Peer
    LLAMA_TRANSFER_TYPE_UNIFIED_READ = 4,     // Unified memory read
    LLAMA_TRANSFER_TYPE_UNIFIED_WRITE = 5,    // Unified memory write
    LLAMA_TRANSFER_TYPE_MAPPED_ACCESS = 6,    // Mapped buffer access
};

// ============================================================================
// TRANSFER VIOLATION ENUMERATION
// ============================================================================

/**
 * Violations of per-token transfer prohibition
 */
enum llama_transfer_violation {
    LLAMA_TRANSFER_VIOLATION_NONE = 0,
    LLAMA_TRANSFER_VIOLATION_LOGITS_D2H = 1,            // Logits Device→Host
    LLAMA_TRANSFER_VIOLATION_LOGITS_READ = 2,           // Host read of logits
    LLAMA_TRANSFER_VIOLATION_KV_CACHE_TRANSFER = 3,     // KV cache D2H/H2D
    LLAMA_TRANSFER_VIOLATION_ACTIVATIONS_TRANSFER = 4,  // Intermediate activation transfer
    LLAMA_TRANSFER_VIOLATION_SAMPLING_BUFFER_TRANSFER = 5, // Sampling buffer transfer
    LLAMA_TRANSFER_VIOLATION_CANDIDATE_TRANSFER = 6,    // Candidate array transfer
    LLAMA_TRANSFER_VIOLATION_EXCESSIVE_TRANSFER = 7,    // Transfer > sizeof(token_id)
    LLAMA_TRANSFER_VIOLATION_UNIFIED_MEMORY_ACCESS = 8, // Unified memory in decode
    LLAMA_TRANSFER_VIOLATION_MAPPED_BUFFER_ACCESS = 9,  // Mapped buffer in decode
    LLAMA_TRANSFER_VIOLATION_IMPLICIT_SYNC_TRANSFER = 10, // Implicit sync transfer
};

// ============================================================================
// TRANSFER RECORD STRUCTURE
// ============================================================================

/**
 * Records details of a single transfer
 */
struct llama_gpu_transfer_record {
    enum llama_transfer_type transfer_type;     // Type of transfer
    uint64_t transfer_size_bytes;               // Size of transfer
    bool is_decode_critical;                    // Is decode-critical?
    bool during_decode_phase;                   // Occurred during decode?
    uint64_t timestamp_ns;                      // When transfer occurred
    uint32_t reserved;
};

// ============================================================================
// TRANSFER PROHIBITION CONFIGURATION
// ============================================================================

/**
 * Configuration for transfer prohibition enforcement
 */
struct llama_gpu_transfer_prohibition_config {
    bool transfer_prohibition_enabled;          // Enable enforcement?
    bool preallocate_all_buffers;               // Preallocate all device buffers?
    bool forbid_implicit_syncs;                 // Forbid implicit syncs?
    bool forbid_unified_memory;                 // Forbid unified memory?
    bool forbid_mapped_access;                  // Forbid mapped buffer access?
    uint64_t max_transfer_per_token_bytes;      // Max transfer size per token (should be ~4-8)
    bool debug_transfer_prohibition;            // Debug output?
};

// ============================================================================
// TRANSFER PROHIBITION STATE RECORD
// ============================================================================

/**
 * Current state of transfer prohibition enforcement
 */
struct llama_gpu_transfer_prohibition_state_record {
    enum llama_gpu_transfer_prohibition_state state;     // Current state
    enum llama_transfer_prohibition_mode mode;           // Current mode
    uint64_t total_transfers_during_decode;              // Total transfers in decode phase
    uint64_t total_transfer_bytes_during_decode;         // Total bytes transferred
    uint64_t total_violations;                           // Total violations
    enum llama_transfer_violation last_violation;        // Last violation type
    uint32_t reserved_1;
};

// ============================================================================
// TRANSFER BUFFER PREALLOCATE RECORD
// ============================================================================

/**
 * Records pre-allocated GPU buffers
 */
struct llama_gpu_preallocated_buffers {
    bool logits_buffer_allocated;               // Logits buffer allocated?
    bool sampling_workspace_allocated;          // Sampling workspace allocated?
    bool topk_buffer_allocated;                 // Top-k buffer allocated?
    bool topp_buffer_allocated;                 // Top-p buffer allocated?
    bool kv_cache_allocated;                    // KV cache allocated?
    bool attention_state_allocated;             // Attention state allocated?
    bool penalty_buffer_allocated;              // Penalty buffer allocated?
    bool candidate_buffer_allocated;            // Candidate buffer allocated?
    uint64_t total_preallocated_bytes;          // Total preallocated bytes
    uint32_t reserved_1;
};

// ============================================================================
// TRANSFER PROHIBITION VALIDATION STATE
// ============================================================================

/**
 * Global validation state for transfer prohibition enforcement
 */
struct llama_gpu_transfer_prohibition_validation_state {
    struct llama_gpu_transfer_prohibition_config config;
    struct llama_gpu_transfer_prohibition_state_record state_record;
    struct llama_gpu_preallocated_buffers preallocated_buffers;
    struct llama_gpu_transfer_record last_transfer;
    int total_transfer_events;
    int total_violations;
    bool enforcement_strict;                    // Abort on violation vs log only
    bool decode_phase_active;                   // Is decode phase active?
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization and configuration
int llama_transfer_prohibition_gpu_init(void);
int llama_transfer_prohibition_gpu_configure(
    bool transfer_prohibition_enabled,
    bool preallocate_all_buffers,
    bool forbid_implicit_syncs,
    bool forbid_unified_memory,
    uint64_t max_transfer_per_token_bytes
);

// Decode phase management (10 enforcement points: 1-10)
int llama_transfer_prohibition_gpu_begin_decode_phase(void);
int llama_transfer_prohibition_gpu_end_decode_phase(void);
int llama_transfer_prohibition_gpu_verify_all_buffers_preallocated(void);
int llama_transfer_prohibition_gpu_forbid_implicit_synchronization(void);
int llama_transfer_prohibition_gpu_forbid_unified_memory_access(void);
int llama_transfer_prohibition_gpu_forbid_mapped_buffer_access(void);
int llama_transfer_prohibition_gpu_forbid_logits_host_reads(void);
int llama_transfer_prohibition_gpu_forbid_kv_cache_transfers(void);
int llama_transfer_prohibition_gpu_allow_token_id_only(void);
int llama_transfer_prohibition_gpu_verify_single_stream_decode(void);

// Transfer monitoring
int llama_transfer_prohibition_gpu_record_transfer(
    enum llama_transfer_type transfer_type,
    uint64_t transfer_size_bytes,
    bool is_decode_critical
);

int llama_transfer_prohibition_gpu_check_transfer_allowed(
    enum llama_transfer_type transfer_type,
    uint64_t transfer_size_bytes,
    bool is_decode_critical
);

// Buffer preallocate operations
int llama_transfer_prohibition_gpu_preallocate_logits_buffer(uint64_t size);
int llama_transfer_prohibition_gpu_preallocate_sampling_workspace(uint64_t size);
int llama_transfer_prohibition_gpu_preallocate_topk_buffer(uint64_t size);
int llama_transfer_prohibition_gpu_preallocate_topp_buffer(uint64_t size);
int llama_transfer_prohibition_gpu_preallocate_kv_cache(uint64_t size);
int llama_transfer_prohibition_gpu_preallocate_attention_state(uint64_t size);
int llama_transfer_prohibition_gpu_preallocate_penalty_buffer(uint64_t size);
int llama_transfer_prohibition_gpu_preallocate_candidate_buffer(uint64_t size);

// Violation detection
int llama_transfer_prohibition_gpu_detect_logits_d2h_transfer(void);
int llama_transfer_prohibition_gpu_detect_logits_host_read(void);
int llama_transfer_prohibition_gpu_detect_kv_cache_transfer(void);
int llama_transfer_prohibition_gpu_detect_activation_transfer(void);
int llama_transfer_prohibition_gpu_detect_sampling_buffer_transfer(void);
int llama_transfer_prohibition_gpu_detect_candidate_transfer(void);
int llama_transfer_prohibition_gpu_detect_excessive_transfer(uint64_t transfer_size);
int llama_transfer_prohibition_gpu_detect_unified_memory_access(void);
int llama_transfer_prohibition_gpu_detect_mapped_buffer_access(void);
int llama_transfer_prohibition_gpu_detect_implicit_sync_transfer(void);

// Query and verification functions
struct llama_gpu_transfer_prohibition_state_record llama_transfer_prohibition_gpu_get_state_record(void);
struct llama_gpu_preallocated_buffers llama_transfer_prohibition_gpu_get_preallocated_buffers(void);
struct llama_gpu_transfer_record llama_transfer_prohibition_gpu_get_last_transfer(void);

// Verification functions
int llama_transfer_prohibition_gpu_verify_no_transfers_during_decode(void);
int llama_transfer_prohibition_gpu_verify_all_buffers_persistent(void);
int llama_transfer_prohibition_gpu_verify_single_stream_execution(void);
int llama_transfer_prohibition_gpu_verify_no_unified_memory_used(void);
int llama_transfer_prohibition_gpu_verify_no_mapped_buffers_used(void);
int llama_transfer_prohibition_gpu_verify_only_token_id_transferred(void);

// Diagnostics and logging
void llama_transfer_prohibition_gpu_log_prohibition_enabled(void);
void llama_transfer_prohibition_gpu_log_decode_phase_started(void);
void llama_transfer_prohibition_gpu_log_decode_phase_ended(void);
void llama_transfer_prohibition_gpu_print_state(void);
void llama_transfer_prohibition_gpu_print_transfer_stats(void);
void llama_transfer_prohibition_gpu_print_violation_summary(void);
void llama_transfer_prohibition_gpu_print_preallocated_buffers(void);

// Violation reporting
void llama_transfer_prohibition_gpu_report_violation(
    enum llama_transfer_violation violation_type,
    const char* details,
    uint64_t transfer_size
);

// Enforcement mode control
void llama_transfer_prohibition_gpu_set_enforcement_strict(bool strict);
bool llama_transfer_prohibition_gpu_get_enforcement_strict(void);
void llama_transfer_prohibition_gpu_set_debug_output(bool debug);

// Self-test suite
int llama_transfer_prohibition_gpu_selftest(void);

// Helper/inline functions
static inline const char* llama_transfer_type_name(enum llama_transfer_type type) {
    switch (type) {
        case LLAMA_TRANSFER_TYPE_NONE: return "NONE";
        case LLAMA_TRANSFER_TYPE_H2D: return "HOST_TO_DEVICE";
        case LLAMA_TRANSFER_TYPE_D2H: return "DEVICE_TO_HOST";
        case LLAMA_TRANSFER_TYPE_P2P: return "PEER_TO_PEER";
        case LLAMA_TRANSFER_TYPE_UNIFIED_READ: return "UNIFIED_READ";
        case LLAMA_TRANSFER_TYPE_UNIFIED_WRITE: return "UNIFIED_WRITE";
        case LLAMA_TRANSFER_TYPE_MAPPED_ACCESS: return "MAPPED_ACCESS";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_transfer_violation_name(enum llama_transfer_violation violation) {
    switch (violation) {
        case LLAMA_TRANSFER_VIOLATION_NONE: return "NONE";
        case LLAMA_TRANSFER_VIOLATION_LOGITS_D2H: return "LOGITS_D2H";
        case LLAMA_TRANSFER_VIOLATION_LOGITS_READ: return "LOGITS_HOST_READ";
        case LLAMA_TRANSFER_VIOLATION_KV_CACHE_TRANSFER: return "KV_CACHE_TRANSFER";
        case LLAMA_TRANSFER_VIOLATION_ACTIVATIONS_TRANSFER: return "ACTIVATIONS_TRANSFER";
        case LLAMA_TRANSFER_VIOLATION_SAMPLING_BUFFER_TRANSFER: return "SAMPLING_BUFFER_TRANSFER";
        case LLAMA_TRANSFER_VIOLATION_CANDIDATE_TRANSFER: return "CANDIDATE_TRANSFER";
        case LLAMA_TRANSFER_VIOLATION_EXCESSIVE_TRANSFER: return "EXCESSIVE_TRANSFER";
        case LLAMA_TRANSFER_VIOLATION_UNIFIED_MEMORY_ACCESS: return "UNIFIED_MEMORY_ACCESS";
        case LLAMA_TRANSFER_VIOLATION_MAPPED_BUFFER_ACCESS: return "MAPPED_BUFFER_ACCESS";
        case LLAMA_TRANSFER_VIOLATION_IMPLICIT_SYNC_TRANSFER: return "IMPLICIT_SYNC_TRANSFER";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_transfer_prohibition_state_name(
    enum llama_gpu_transfer_prohibition_state state
) {
    switch (state) {
        case LLAMA_GPU_TRANSFER_PROHIBITION_UNINITIALIZED: return "UNINITIALIZED";
        case LLAMA_GPU_TRANSFER_PROHIBITION_INITIALIZED: return "INITIALIZED";
        case LLAMA_GPU_TRANSFER_PROHIBITION_DECODE_ACTIVE: return "DECODE_ACTIVE";
        case LLAMA_GPU_TRANSFER_PROHIBITION_MONITORING: return "MONITORING";
        case LLAMA_GPU_TRANSFER_PROHIBITION_COMPLETE: return "COMPLETE";
        case LLAMA_GPU_TRANSFER_PROHIBITION_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif

