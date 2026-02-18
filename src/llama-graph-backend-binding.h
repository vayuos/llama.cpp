/**
 * SECTION 7: Enforce Single Backend Binding at Decode Graph Build
 *
 * This file implements single backend binding enforcement at decode graph construction.
 * A single backend is bound to the entire decode graph at build time. This binding is
 * exclusive, immutable, and GPU-only. No per-node or per-op backend overrides permitted.
 *
 * Core Principle:
 * "Decode graphs are single-backend by construction. Backend ownership is fixed at
 *  graph build time before the first token. CPU fallback and hybrid execution are
 *  impossible by design."
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <map>
#include "llama-backend-immutability-enforce.h"
#include <vector>

// ============================================================================
// GRAPH BACKEND BINDING STATE DEFINITION
// ============================================================================

/**
 * Enum defining the backend binding state of a graph
 */
enum llama_graph_backend_binding_state {
    LLAMA_GRAPH_BINDING_UNBOUND = 0,        // No backend bound yet
    LLAMA_GRAPH_BINDING_SELECTING = 1,      // Backend selection in progress
    LLAMA_GRAPH_BINDING_BOUND = 2,          // Backend bound (immutable)
    LLAMA_GRAPH_BINDING_FINALIZED = 3,      // Graph finalized with verified binding
    LLAMA_GRAPH_BINDING_INVALID = 4,        // Binding invalid or graph rejected
};

/**
 * Enum defining backend binding violation types
 */
enum llama_graph_binding_violation_type {
    LLAMA_BINDING_VIOLATION_UNKNOWN = 0,
    LLAMA_BINDING_VIOLATION_PER_NODE_OVERRIDE = 1,     // Node attempted backend override
    LLAMA_BINDING_VIOLATION_MIXED_BACKENDS = 2,        // Mixed backends in graph
    LLAMA_BINDING_VIOLATION_CPU_OWNERSHIP = 3,         // CPU selected as graph owner
    LLAMA_BINDING_VIOLATION_REBINDING_ATTEMPT = 4,     // Attempted backend rebinding
    LLAMA_BINDING_VIOLATION_NODE_MISMATCH = 5,         // Node backend doesn't match graph
    LLAMA_BINDING_VIOLATION_CONSISTENCY_CHECK = 6,     // Consistency check failed
    LLAMA_BINDING_VIOLATION_EXECUTION_MISMATCH = 7,    // Executing backend doesn't match binding
    LLAMA_BINDING_VIOLATION_PARTIAL_GPU_BINDING = 8,   // Only partial GPU binding achieved
};

/**
 * Enum defining graph classification
 */
enum llama_graph_classification {
    LLAMA_GRAPH_CLASS_UNKNOWN = 0,
    LLAMA_GRAPH_CLASS_DECODE = 1,           // Decode phase graph (single-backend)
    LLAMA_GRAPH_CLASS_PREFILL = 2,          // Prefill graph (flexible backend)
    LLAMA_GRAPH_CLASS_SETUP = 3,            // Setup/background graph (flexible)
};

// ============================================================================
// GRAPH BACKEND BINDING STRUCTURES
// ============================================================================

/**
 * Structure recording backend binding for a specific node
 */
struct llama_node_backend_binding {
    const char* node_name;                  // Node identifier
    enum llama_backend_type expected_backend;  // Expected backend from graph
    enum llama_backend_type actual_backend;    // Actual backend assigned
    bool matches_graph;                     // Does node backend match graph backend?
    bool attempted_override;                // Did node attempt override?
};

/**
 * Structure tracking backend binding state for a graph
 */
struct llama_graph_backend_binding_record {
    uint64_t graph_id;                      // Unique graph identifier
    enum llama_graph_classification graph_class;  // Graph type (decode/prefill/setup)
    enum llama_graph_backend_binding_state binding_state;  // Current binding state
    enum llama_backend_type bound_backend;  // The bound backend (GPU-only for decode)

    // Node tracking
    uint64_t node_count;                    // Total nodes in graph
    struct llama_node_backend_binding* nodes;  // Array of node bindings
    int max_nodes;                          // Capacity

    // Binding verification
    bool all_nodes_match;                   // All nodes match graph backend?
    bool graph_finalized;                   // Graph finalization complete?
    uint64_t finalization_time_us;          // When graph was finalized

    // Violation tracking
    int violation_count;                    // Violations detected
    enum llama_graph_binding_violation_type last_violation_type;
    const char* last_violation_message;
    const char* last_violation_node;
};

/**
 * Global graph backend binding registry
 */
struct llama_graph_binding_registry {
    struct llama_graph_backend_binding_record* graphs;
    int graph_count;
    int max_graphs;

    // Violation tracking
    int total_violations;
    int rejection_count;
};

// ============================================================================
// GRAPH BACKEND BINDING CONTROL
// ============================================================================

/**
 * Initialize graph backend binding tracking
 */
int llama_graph_backend_binding_init(void);

/**
 * Register a graph and begin backend selection
 * Returns: 0 = success, -1 = FATAL (graph registration failed)
 */
int llama_graph_backend_binding_register(
    uint64_t graph_id,
    enum llama_graph_classification graph_class
);

/**
 * Bind a backend to the graph (single-backend, GPU-only for decode)
 * Returns: 0 = success, -1 = FATAL (invalid backend or rebinding attempt)
 */
int llama_graph_backend_binding_bind(
    uint64_t graph_id,
    enum llama_backend_type backend
);

/**
 * Get the bound backend for a graph
 */
enum llama_backend_type llama_graph_backend_binding_get_backend(uint64_t graph_id);

/**
 * Check if graph is already bound
 */
bool llama_graph_backend_binding_is_bound(uint64_t graph_id);

/**
 * Check if graph binding is finalized
 */
bool llama_graph_backend_binding_is_finalized(uint64_t graph_id);

// ============================================================================
// PER-NODE BACKEND OVERRIDE PREVENTION
// ============================================================================

/**
 * Register a node in the graph and verify backend matches graph binding
 * Returns: 0 = node acceptable, -1 = FATAL (backend mismatch or override attempted)
 */
int llama_graph_node_verify_backend_matches_graph(
    uint64_t graph_id,
    const char* node_name,
    enum llama_backend_type node_backend
);

/**
 * Detect if a node is attempting backend override (different from graph binding)
 * Returns: 0 = no override, -1 = FATAL (override detected)
 */
int llama_graph_detect_node_backend_override(
    uint64_t graph_id,
    const char* node_name,
    enum llama_backend_type node_attempted_backend
);

/**
 * Assert that nodes cannot select their own backend for decode graphs
 * Returns: 0 = assertion passes, -1 = FATAL (node attempting selection)
 */
int llama_assert_no_per_node_backend_selection_decode(
    uint64_t graph_id,
    bool node_backend_selection_attempted
);

// ============================================================================
// MIXED-BACKEND GRAPH DETECTION
// ============================================================================

/**
 * Detect if graph has mixed backends (some GPU, some CPU)
 * Returns: 0 = single backend, -1 = FATAL (mixed backends detected)
 */
int llama_detect_mixed_backend_decode_graph(
    uint64_t graph_id,
    const char** node_names,
    enum llama_backend_type* node_backends,
    int num_nodes
);

/**
 * Verify all decode-critical nodes in graph use same backend
 * Returns: 0 = uniform backend, -1 = FATAL (backend mismatch)
 */
int llama_verify_graph_backend_uniformity(
    uint64_t graph_id,
    const char** decode_critical_nodes,
    enum llama_backend_type* node_backends,
    int num_nodes
);

// ============================================================================
// BACKEND CONSISTENCY CHECKS
// ============================================================================

/**
 * ENFORCEMENT POINT 1: Graph finalization backend consistency check
 * At graph finalization, validate that every decode-critical node matches graph backend.
 * Returns: 0 = consistent, -1 = FATAL (mismatch found)
 */
int llama_enforce_backend_consistency_at_graph_finalization(
    uint64_t graph_id,
    const char** node_names,
    enum llama_backend_type* node_backends,
    int num_nodes
);

/**
 * ENFORCEMENT POINT 2: Graph build backend validation
 * Verify graph can be built with single GPU backend.
 * Returns: 0 = buildable, -1 = FATAL (incompatible with single backend)
 */
int llama_enforce_graph_single_backend_buildability(
    uint64_t graph_id
);

/**
 * ENFORCEMENT POINT 3: Abort graph creation on binding failure
 * If graph cannot be bound to single GPU backend, abort immediately.
 * Returns: 0 = graph buildable, -1 = FATAL (graph rejected)
 */
int llama_enforce_graph_rejection_on_binding_failure(
    uint64_t graph_id
);

/**
 * ENFORCEMENT POINT 4: Graph finalization lock
 * Finalize graph binding and lock against further changes.
 * Returns: 0 = finalized, -1 = FATAL (already finalized or invalid)
 */
int llama_enforce_graph_finalization_and_lock(
    uint64_t graph_id
);

/**
 * ENFORCEMENT POINT 5: Backend rebinding prevention
 * Prevent any backend reassignment after graph build.
 * Returns: 0 = no rebinding, -1 = FATAL (rebinding attempted)
 */
int llama_enforce_no_backend_rebinding_after_build(
    uint64_t graph_id,
    bool rebinding_attempted
);

/**
 * ENFORCEMENT POINT 6: Execution backend verification (debug builds)
 * Confirm executing backend matches graph's bound backend.
 * Returns: 0 = matches, -1 = FATAL (mismatch detected)
 */
int llama_enforce_execution_backend_matches_binding(
    uint64_t graph_id,
    enum llama_backend_type executing_backend
);

// ============================================================================
// DECODE vs NON-DECODE GRAPH CLASSIFICATION
// ============================================================================

/**
 * Classify graph as decode or non-decode
 * Decode graphs: strict single-backend rule applies
 * Non-decode graphs: flexible backend selection allowed
 */
int llama_graph_backend_binding_classify(
    uint64_t graph_id,
    enum llama_graph_classification classification
);

/**
 * Check if graph is decode graph (requires single-backend binding)
 */
bool llama_graph_backend_binding_is_decode_graph(uint64_t graph_id);

/**
 * Check if graph is non-decode graph (flexible backend allowed)
 */
bool llama_graph_backend_binding_is_nondecode_graph(uint64_t graph_id);

// ============================================================================
// RUNTIME VERIFICATION
// ============================================================================

/**
 * Runtime verification: confirm binding matches execution
 * Optional check for debug builds to verify no backend slippage.
 * Returns: 0 = binding valid, -1 = FATAL (binding violated)
 */
int llama_graph_verify_binding_at_execution(
    uint64_t graph_id,
    enum llama_backend_type executing_backend
);

/**
 * Enable/disable runtime verification checks
 */
void llama_set_graph_binding_runtime_verification(bool enabled);

/**
 * Check if runtime verification is enabled
 */
bool llama_get_graph_binding_runtime_verification(void);

// ============================================================================
// FAILURE DIAGNOSTICS & REPORTING
// ============================================================================

/**
 * Record a graph backend binding violation
 */
void llama_record_graph_binding_violation(
    uint64_t graph_id,
    enum llama_graph_binding_violation_type violation_type,
    const char* violation_message,
    const char* offending_node
);

/**
 * Print comprehensive backend binding violation diagnostics
 */
void llama_print_graph_binding_violation_diagnostics(
    const struct llama_graph_backend_binding_record* binding,
    enum llama_graph_binding_violation_type violation_type,
    const char* violation_message
);

/**
 * Convert violation type to human-readable string
 */
static inline const char* llama_graph_binding_violation_name(
    enum llama_graph_binding_violation_type violation_type
) {
    switch (violation_type) {
        case LLAMA_BINDING_VIOLATION_UNKNOWN:
            return "UNKNOWN";
        case LLAMA_BINDING_VIOLATION_PER_NODE_OVERRIDE:
            return "PER_NODE_OVERRIDE";
        case LLAMA_BINDING_VIOLATION_MIXED_BACKENDS:
            return "MIXED_BACKENDS";
        case LLAMA_BINDING_VIOLATION_CPU_OWNERSHIP:
            return "CPU_OWNERSHIP";
        case LLAMA_BINDING_VIOLATION_REBINDING_ATTEMPT:
            return "REBINDING_ATTEMPT";
        case LLAMA_BINDING_VIOLATION_NODE_MISMATCH:
            return "NODE_MISMATCH";
        case LLAMA_BINDING_VIOLATION_CONSISTENCY_CHECK:
            return "CONSISTENCY_CHECK";
        case LLAMA_BINDING_VIOLATION_EXECUTION_MISMATCH:
            return "EXECUTION_MISMATCH";
        case LLAMA_BINDING_VIOLATION_PARTIAL_GPU_BINDING:
            return "PARTIAL_GPU_BINDING";
        default:
            return "(invalid)";
    }
}

/**
 * Convert graph classification to human-readable string
 */
static inline const char* llama_graph_classification_name(
    enum llama_graph_classification classification
) {
    switch (classification) {
        case LLAMA_GRAPH_CLASS_UNKNOWN:
            return "UNKNOWN";
        case LLAMA_GRAPH_CLASS_DECODE:
            return "DECODE (single-backend)";
        case LLAMA_GRAPH_CLASS_PREFILL:
            return "PREFILL (flexible)";
        case LLAMA_GRAPH_CLASS_SETUP:
            return "SETUP (flexible)";
        default:
            return "(invalid)";
    }
}

// ============================================================================
// DIAGNOSTIC LOGGING
// ============================================================================

/**
 * Log backend binding confirmation (once per decode session)
 */
void llama_log_graph_backend_binding_confirmation(
    uint64_t graph_id,
    enum llama_backend_type bound_backend
);

/**
 * Log graph binding state (for validation and benchmarking)
 */
void llama_log_graph_binding_state(uint64_t graph_id);

/**
 * Print explicit graph backend binding principle
 */
void llama_print_graph_backend_binding_principle(void);

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

/**
 * Enable/disable strict graph backend binding enforcement
 * When enabled, any violation causes immediate failure.
 */
void llama_set_graph_binding_enforcement_strict(bool enforce_strict);

/**
 * Get current enforcement mode
 */
bool llama_get_graph_binding_enforcement_strict(void);

/**
 * Get total graph binding violations
 */
int llama_get_graph_binding_violation_count(void);

/**
 * Get graph rejection count
 */
int llama_get_graph_binding_rejection_count(void);

/**
 * Reset graph binding violation counters
 */
void llama_reset_graph_binding_counters(void);

// ============================================================================
// VALIDATION AND SELF-TEST
// ============================================================================

/**
 * Self-test: verify graph backend binding mechanism works correctly
 */
int llama_graph_backend_binding_selftest(void);

