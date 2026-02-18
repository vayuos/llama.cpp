/**
 * SECTION 13: Cache backend decisions per decode graph
 * Header
 *
 * This file implements enforcement that backend selection is resolved once during
 * decode graph construction and then cached permanently for that graph. During decode
 * execution, no backend re-evaluation or dynamic dispatch is allowed. Backend decisions
 * are immutable, traceable, and optimized for deterministic execution.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BACKEND RESOLUTION TIMING ENUMERATION
// ============================================================================

/**
 * Timing of backend resolution decision
 */
enum llama_backend_resolution_timing {
    LLAMA_BACKEND_RESOLUTION_NONE = 0,          // Not resolved
    LLAMA_BACKEND_RESOLUTION_GRAPH_BUILD = 1,   // Resolved during graph construction
    LLAMA_BACKEND_RESOLUTION_PREFILL = 2,       // Resolved before prefill (wrong - late)
    LLAMA_BACKEND_RESOLUTION_DECODE_START = 3,  // Resolved at decode start (wrong - too late)
    LLAMA_BACKEND_RESOLUTION_RUNTIME = 4,       // Resolved at runtime (wrong - forbidden)
};

// ============================================================================
// BACKEND ATTACHMENT STATE ENUMERATION
// ============================================================================

/**
 * State of backend attachment to graph nodes
 */
enum llama_backend_attachment_state {
    LLAMA_BACKEND_ATTACH_UNATTACHED = 0,       // No backend attached
    LLAMA_BACKEND_ATTACH_ATTACHING = 1,        // Attachment in progress
    LLAMA_BACKEND_ATTACH_ATTACHED = 2,         // Backend immutably attached
    LLAMA_BACKEND_ATTACH_FROZEN = 3,           // Backend attachment frozen (no changes)
    LLAMA_BACKEND_ATTACH_INVALID = 4,          // Invalid attachment state
};

// ============================================================================
// BACKEND CACHE INTEGRITY STATE ENUMERATION
// ============================================================================

/**
 * Integrity of cached backend decisions
 */
enum llama_backend_cache_integrity {
    LLAMA_BACKEND_CACHE_VALID = 0,              // Cache valid and trusted
    LLAMA_BACKEND_CACHE_STALE = 1,              // Cache became stale
    LLAMA_BACKEND_CACHE_CORRUPTED = 2,          // Cache corrupted
    LLAMA_BACKEND_CACHE_INVALIDATED = 3,        // Cache explicitly invalidated
    LLAMA_BACKEND_CACHE_DRIFT_DETECTED = 4,     // Backend drift from cache detected
};

// ============================================================================
// BACKEND RESOLUTION FAILURE REASON ENUMERATION
// ============================================================================

/**
 * Reasons backend resolution might fail during graph build
 */
enum llama_backend_resolution_failure {
    LLAMA_BACKEND_RESOLVE_OK = 0,
    LLAMA_BACKEND_RESOLVE_NOT_AVAILABLE = 1,    // Backend not available for this operation
    LLAMA_BACKEND_RESOLVE_NO_CAPABILITY = 2,    // Backend lacks required capability
    LLAMA_BACKEND_RESOLVE_INCOMPATIBLE = 3,     // Backend incompatible with tensor/operation
    LLAMA_BACKEND_RESOLVE_DEFERRED = 4,         // Resolution was deferred (forbidden)
    LLAMA_BACKEND_RESOLVE_LATE_QUERY = 5,       // Late backend query during decode
    LLAMA_BACKEND_RESOLVE_RUNTIME_MISMATCH = 6, // Runtime backend doesn't match cache
    LLAMA_BACKEND_RESOLVE_UNKNOWN = 7,
};

// ============================================================================
// BACKEND DISPATCH TYPE ENUMERATION
// ============================================================================

/**
 * Types of backend dispatch operations that must be cached
 */
enum llama_backend_dispatch_type {
    LLAMA_BACKEND_DISPATCH_NONE = 0,
    LLAMA_BACKEND_DISPATCH_NODE_EXECUTION = 1,   // Node execution backend
    LLAMA_BACKEND_DISPATCH_KERNEL_LAUNCH = 2,    // Kernel launch backend
    LLAMA_BACKEND_DISPATCH_TENSOR_OP = 3,        // Tensor operation backend
    LLAMA_BACKEND_DISPATCH_MEMORY_OP = 4,        // Memory operation backend
    LLAMA_BACKEND_DISPATCH_ASYNC_OP = 5,         // Async operation backend
};

// ============================================================================
// BACKEND DRIFT REASON ENUMERATION
// ============================================================================

/**
 * Reasons why backend might drift from cached decision at runtime
 */
enum llama_backend_drift_reason {
    LLAMA_BACKEND_DRIFT_NONE = 0,
    LLAMA_BACKEND_DRIFT_TENSOR_RELOCATED = 1,    // Tensor moved to different device
    LLAMA_BACKEND_DRIFT_CONTEXT_GROWTH = 2,      // Context grew, new backend needed
    LLAMA_BACKEND_DRIFT_MEMORY_PRESSURE = 3,     // Memory pressure forced backend change
    LLAMA_BACKEND_DRIFT_CAPABILITY_LOST = 4,     // GPU capability became unavailable
    LLAMA_BACKEND_DRIFT_IMPLICIT_FALLBACK = 5,   // Implicit fallback to CPU detected
};

// ============================================================================
// BACKEND QUERY VIOLATION ENUMERATION
// ============================================================================

/**
 * Types of backend query violations during decode
 */
enum llama_backend_query_violation {
    LLAMA_BACKEND_QUERY_NONE = 0,
    LLAMA_BACKEND_QUERY_RUNTIME_DECISION = 1,    // Runtime backend decision attempt
    LLAMA_BACKEND_QUERY_DISPATCH_LOOKUP = 2,     // Backend dispatch lookup during exec
    LLAMA_BACKEND_QUERY_CAPABILITY_CHECK = 3,    // Capability check during decode
    LLAMA_BACKEND_QUERY_DEVICE_CHANGE = 4,       // Device re-query during decode
    LLAMA_BACKEND_QUERY_VIRTUAL_DISPATCH = 5,    // Virtual dispatch call during hot path
};

// ============================================================================
// BACKEND CACHE ENTRY RECORD
// ============================================================================

/**
 * Single cached backend decision for a graph node
 */
struct llama_backend_cache_entry {
    uint64_t node_id;                            // Graph node ID
    uint64_t graph_id;                           // Parent graph ID
    enum ggml_backend_type cached_backend;       // Cached backend for this node
    enum llama_backend_attachment_state attachment_state; // Attachment state
    uint64_t resolution_time_ns;                 // When resolved
    bool backend_immutable;                      // True = immutable after freeze
    int query_count;                             // Number of times queried
    int cache_hits;                              // Number of cache hits
    int dispatch_violations;                     // Number of dispatch violations
};

// ============================================================================
// BACKEND CACHE GLOBAL RECORD
// ============================================================================

/**
 * Global record for backend cache state
 */
struct llama_backend_cache_record {
    uint64_t graph_id;                           // Current graph ID
    int total_cached_decisions;                  // Total backend decisions cached
    int total_nodes_resolved;                    // Total nodes backend resolved
    int total_runtime_queries;                   // Total forbidden runtime queries
    enum llama_backend_cache_integrity cache_integrity; // Overall cache state
    enum llama_backend_resolution_failure last_resolution_failure; // Last failure reason
    enum llama_backend_drift_reason last_drift_reason; // Last drift reason detected
    enum llama_backend_query_violation last_query_violation; // Last query violation
    bool cache_frozen;                           // True = cache immutable
    bool virtual_dispatch_eliminated;            // True = hot path dispatch removed
    uint64_t cache_creation_time_ns;             // When cache created
};

// ============================================================================
// BACKEND CACHE VALIDATION STATE
// ============================================================================

/**
 * Global state for backend cache enforcement
 */
struct llama_graph_backend_cache_validation_state {
    struct llama_backend_cache_record cache_record;
    int total_resolution_failures;
    int total_cache_misses;
    int total_drift_detections;
    bool enforcement_strict;                     // Abort on violation vs log only
    bool debug_verify_cache_consistency;         // Debug: verify cache at each step
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Initialization
int llama_backend_cache_init(void);

// Backend resolution at graph build time (5 enforcement points: 1-5)
int llama_backend_cache_resolve_at_graph_build(
    uint64_t graph_id,
    uint64_t node_id,
    enum ggml_backend_type * out_backend
);
int llama_backend_cache_resolve_all_nodes_upfront(uint64_t graph_id);
int llama_backend_cache_forbid_deferred_resolution(void);
int llama_backend_cache_attach_backend_to_node(
    uint64_t node_id,
    enum ggml_backend_type backend
);
int llama_backend_cache_freeze_backend_assignment(void);

// Cache immutability enforcement (3 enforcement points: 6-8)
int llama_backend_cache_disable_runtime_queries(void);
int llama_backend_cache_eliminate_virtual_dispatch(void);
int llama_backend_cache_assert_no_dispatch_during_decode(void);

// Backend consistency verification (2 enforcement points: 9-10)
int llama_backend_cache_verify_cache_before_freeze(void);
int llama_backend_cache_detect_backend_drift(
    uint64_t node_id,
    enum ggml_backend_type actual_backend
);

// Query and lookup functions (cached - no dynamic dispatch)
enum ggml_backend_type llama_backend_cache_lookup_cached(uint64_t node_id);
bool llama_backend_cache_has_cached_decision(uint64_t node_id);

// Diagnostic functions
struct llama_backend_cache_entry llama_backend_cache_get_entry(uint64_t node_id);
struct llama_backend_cache_record llama_backend_cache_get_record(void);
int llama_backend_cache_get_cache_hit_rate(void);
int llama_backend_cache_get_total_cached_decisions(void);

// Violation detection
int llama_backend_cache_detect_late_query(
    uint64_t node_id,
    enum llama_backend_query_violation violation_type
);
int llama_backend_cache_detect_backend_change(
    uint64_t node_id,
    enum ggml_backend_type old_backend,
    enum ggml_backend_type new_backend
);

// Diagnostics and logging
void llama_backend_cache_log_resolution_complete(uint64_t graph_id);
void llama_backend_cache_log_dispatch_eliminated(void);
void llama_backend_cache_print_cache_status(void);
void llama_backend_cache_print_backend_mapping(void);

// Violation reporting
void llama_backend_cache_report_runtime_query(
    uint64_t node_id,
    enum llama_backend_query_violation violation_type,
    const char* reason
);
void llama_backend_cache_report_drift(
    uint64_t node_id,
    enum llama_backend_drift_reason drift_reason
);

// Enforcement mode control
void llama_backend_cache_set_enforcement_strict(bool strict);
bool llama_backend_cache_get_enforcement_strict(void);
void llama_backend_cache_set_debug_verify_consistency(bool debug);

// Cache verification
int llama_backend_cache_verify_all_nodes_resolved(uint64_t graph_id);
int llama_backend_cache_verify_no_late_resolution(void);
int llama_backend_cache_verify_immutability_invariant(void);

// Self-test suite
int llama_backend_cache_selftest(void);

// Helper/inline functions
static inline const char* llama_backend_resolution_timing_name(
    enum llama_backend_resolution_timing timing
) {
    switch (timing) {
        case LLAMA_BACKEND_RESOLUTION_NONE: return "NONE";
        case LLAMA_BACKEND_RESOLUTION_GRAPH_BUILD: return "GRAPH_BUILD";
        case LLAMA_BACKEND_RESOLUTION_PREFILL: return "PREFILL";
        case LLAMA_BACKEND_RESOLUTION_DECODE_START: return "DECODE_START";
        case LLAMA_BACKEND_RESOLUTION_RUNTIME: return "RUNTIME";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_backend_attachment_state_name(
    enum llama_backend_attachment_state state
) {
    switch (state) {
        case LLAMA_BACKEND_ATTACH_UNATTACHED: return "UNATTACHED";
        case LLAMA_BACKEND_ATTACH_ATTACHING: return "ATTACHING";
        case LLAMA_BACKEND_ATTACH_ATTACHED: return "ATTACHED";
        case LLAMA_BACKEND_ATTACH_FROZEN: return "FROZEN";
        case LLAMA_BACKEND_ATTACH_INVALID: return "INVALID";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_backend_cache_integrity_name(
    enum llama_backend_cache_integrity integrity
) {
    switch (integrity) {
        case LLAMA_BACKEND_CACHE_VALID: return "VALID";
        case LLAMA_BACKEND_CACHE_STALE: return "STALE";
        case LLAMA_BACKEND_CACHE_CORRUPTED: return "CORRUPTED";
        case LLAMA_BACKEND_CACHE_INVALIDATED: return "INVALIDATED";
        case LLAMA_BACKEND_CACHE_DRIFT_DETECTED: return "DRIFT_DETECTED";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_backend_resolution_failure_name(
    enum llama_backend_resolution_failure failure
) {
    switch (failure) {
        case LLAMA_BACKEND_RESOLVE_OK: return "OK";
        case LLAMA_BACKEND_RESOLVE_NOT_AVAILABLE: return "NOT_AVAILABLE";
        case LLAMA_BACKEND_RESOLVE_NO_CAPABILITY: return "NO_CAPABILITY";
        case LLAMA_BACKEND_RESOLVE_INCOMPATIBLE: return "INCOMPATIBLE";
        case LLAMA_BACKEND_RESOLVE_DEFERRED: return "DEFERRED";
        case LLAMA_BACKEND_RESOLVE_LATE_QUERY: return "LATE_QUERY";
        case LLAMA_BACKEND_RESOLVE_RUNTIME_MISMATCH: return "RUNTIME_MISMATCH";
        case LLAMA_BACKEND_RESOLVE_UNKNOWN: return "UNKNOWN";
        default: return "INVALID";
    }
}

static inline const char* llama_backend_query_violation_name(
    enum llama_backend_query_violation violation
) {
    switch (violation) {
        case LLAMA_BACKEND_QUERY_NONE: return "NONE";
        case LLAMA_BACKEND_QUERY_RUNTIME_DECISION: return "RUNTIME_DECISION";
        case LLAMA_BACKEND_QUERY_DISPATCH_LOOKUP: return "DISPATCH_LOOKUP";
        case LLAMA_BACKEND_QUERY_CAPABILITY_CHECK: return "CAPABILITY_CHECK";
        case LLAMA_BACKEND_QUERY_DEVICE_CHANGE: return "DEVICE_CHANGE";
        case LLAMA_BACKEND_QUERY_VIRTUAL_DISPATCH: return "VIRTUAL_DISPATCH";
        default: return "UNKNOWN";
    }
}

static inline const char* llama_backend_drift_reason_name(
    enum llama_backend_drift_reason drift
) {
    switch (drift) {
        case LLAMA_BACKEND_DRIFT_NONE: return "NONE";
        case LLAMA_BACKEND_DRIFT_TENSOR_RELOCATED: return "TENSOR_RELOCATED";
        case LLAMA_BACKEND_DRIFT_CONTEXT_GROWTH: return "CONTEXT_GROWTH";
        case LLAMA_BACKEND_DRIFT_MEMORY_PRESSURE: return "MEMORY_PRESSURE";
        case LLAMA_BACKEND_DRIFT_CAPABILITY_LOST: return "CAPABILITY_LOST";
        case LLAMA_BACKEND_DRIFT_IMPLICIT_FALLBACK: return "IMPLICIT_FALLBACK";
        default: return "UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif
