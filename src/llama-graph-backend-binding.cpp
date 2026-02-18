/**
 * SECTION 7 IMPLEMENTATION: Enforce Single Backend Binding at Decode Graph Build
 *
 * This file implements single backend binding enforcement at decode graph construction.
 * Decode graphs are bound to a single GPU backend at build time with no per-node overrides.
 */

#include "llama-graph-backend-binding.h"
#include <cstring>
#include <cstdio>
#include <ctime>
#include <map>
#include <vector>

// ============================================================================
// GLOBAL STATE
// ============================================================================

static struct llama_graph_binding_registry g_graph_binding_registry = {
    .graphs = NULL,
    .graph_count = 0,
    .max_graphs = 256,
    .total_violations = 0,
    .rejection_count = 0,
};

static bool g_enforce_strict = true;
static bool g_runtime_verification_enabled = true;

// Graph ID to registry index mapping for fast lookup
static std::map<uint64_t, int> g_graph_id_index_map;

// ============================================================================
// INITIALIZATION
// ============================================================================

int llama_graph_backend_binding_init(void) {
    if (g_graph_binding_registry.graphs == NULL) {
        g_graph_binding_registry.graphs =
            (struct llama_graph_backend_binding_record*)malloc(
                sizeof(struct llama_graph_backend_binding_record) *
                g_graph_binding_registry.max_graphs
            );
        if (g_graph_binding_registry.graphs == NULL) {
            fprintf(stderr, "[GRAPH_BINDING] FATAL: Failed to allocate graph registry\n");
            return -1;
        }
    }

    g_graph_binding_registry.graph_count = 0;
    g_graph_binding_registry.total_violations = 0;
    g_graph_binding_registry.rejection_count = 0;
    g_graph_id_index_map.clear();

    fprintf(stderr, "[GRAPH_BINDING] Initialized: Graph backend binding tracking ready\n");
    return 0;
}

// ============================================================================
// GRAPH REGISTRATION AND BINDING
// ============================================================================

int llama_graph_backend_binding_register(
    uint64_t graph_id,
    enum llama_graph_classification graph_class
) {
    if (g_graph_binding_registry.graph_count >= g_graph_binding_registry.max_graphs) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Graph registry full (max=%d)\n",
                g_graph_binding_registry.max_graphs);
        return -1;
    }

    if (g_graph_id_index_map.find(graph_id) != g_graph_id_index_map.end()) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Graph %lu already registered\n", graph_id);
        return -1;
    }

    int idx = g_graph_binding_registry.graph_count;
    struct llama_graph_backend_binding_record* record = &g_graph_binding_registry.graphs[idx];

    // Initialize record
    record->graph_id = graph_id;
    record->graph_class = graph_class;
    record->binding_state = LLAMA_GRAPH_BINDING_UNBOUND;
    record->bound_backend = LLAMA_BACKEND_UNKNOWN;
    record->node_count = 0;
    record->nodes = (struct llama_node_backend_binding*)malloc(
        sizeof(struct llama_node_backend_binding) * 1024
    );
    record->max_nodes = 1024;
    record->all_nodes_match = false;
    record->graph_finalized = false;
    record->finalization_time_us = 0;
    record->violation_count = 0;
    record->last_violation_type = LLAMA_BINDING_VIOLATION_UNKNOWN;
    record->last_violation_message = NULL;
    record->last_violation_node = NULL;

    if (record->nodes == NULL) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Failed to allocate node tracking for graph %lu\n",
                graph_id);
        return -1;
    }

    g_graph_id_index_map[graph_id] = idx;
    g_graph_binding_registry.graph_count++;

    fprintf(stderr, "[GRAPH_BINDING] Registered graph %lu (%s)\n",
            graph_id, llama_graph_classification_name(graph_class));
    return 0;
}

int llama_graph_backend_binding_bind(
    uint64_t graph_id,
    enum llama_backend_type backend
) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Graph %lu not registered\n", graph_id);
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    // Decode graphs must use GPU backend
    if (record->graph_class == LLAMA_GRAPH_CLASS_DECODE) {
        if (backend == LLAMA_BACKEND_CPU || backend == LLAMA_BACKEND_UNKNOWN) {
            fprintf(stderr, "[GRAPH_BINDING] FATAL: Decode graph cannot use backend %s\n",
                    llama_backend_type_name(backend));
            llama_record_graph_binding_violation(
                graph_id,
                LLAMA_BINDING_VIOLATION_CPU_OWNERSHIP,
                "Decode graph attempted to use non-GPU backend",
                "graph"
            );
            return -1;
        }
    }

    // Check if already bound
    if (record->binding_state == LLAMA_GRAPH_BINDING_BOUND ||
        record->binding_state == LLAMA_GRAPH_BINDING_FINALIZED) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Graph %lu already bound to %s, rebinding to %s not allowed\n",
                graph_id,
                llama_backend_type_name(record->bound_backend),
                llama_backend_type_name(backend));
        llama_record_graph_binding_violation(
            graph_id,
            LLAMA_BINDING_VIOLATION_REBINDING_ATTEMPT,
            "Attempted to rebind graph to different backend",
            "graph"
        );
        return -1;
    }

    // Bind the backend
    record->bound_backend = backend;
    record->binding_state = LLAMA_GRAPH_BINDING_BOUND;

    fprintf(stderr, "[GRAPH_BINDING] Graph %lu bound to backend %s\n",
            graph_id, llama_backend_type_name(backend));
    return 0;
}

enum llama_backend_type llama_graph_backend_binding_get_backend(uint64_t graph_id) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return LLAMA_BACKEND_UNKNOWN;
    }
    return g_graph_binding_registry.graphs[it->second].bound_backend;
}

bool llama_graph_backend_binding_is_bound(uint64_t graph_id) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return false;
    }
    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];
    return record->binding_state != LLAMA_GRAPH_BINDING_UNBOUND;
}

bool llama_graph_backend_binding_is_finalized(uint64_t graph_id) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return false;
    }
    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];
    return record->binding_state == LLAMA_GRAPH_BINDING_FINALIZED;
}

// ============================================================================
// PER-NODE BACKEND OVERRIDE PREVENTION
// ============================================================================

int llama_graph_node_verify_backend_matches_graph(
    uint64_t graph_id,
    const char* node_name,
    enum llama_backend_type node_backend
) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Graph %lu not registered\n", graph_id);
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    if (record->binding_state == LLAMA_GRAPH_BINDING_UNBOUND) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Graph %lu not bound yet\n", graph_id);
        return -1;
    }

    if (node_backend != record->bound_backend) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Node %s backend mismatch in graph %lu\n"
                "  Expected: %s, Got: %s\n",
                node_name, graph_id,
                llama_backend_type_name(record->bound_backend),
                llama_backend_type_name(node_backend));
        llama_record_graph_binding_violation(
            graph_id,
            LLAMA_BINDING_VIOLATION_NODE_MISMATCH,
            "Node backend doesn't match graph binding",
            node_name
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    // Record node binding
    if (record->node_count < (uint64_t)record->max_nodes) {
        struct llama_node_backend_binding* node_rec = &record->nodes[record->node_count];
        node_rec->node_name = node_name;
        node_rec->expected_backend = record->bound_backend;
        node_rec->actual_backend = node_backend;
        node_rec->matches_graph = (node_backend == record->bound_backend);
        node_rec->attempted_override = false;
        record->node_count++;
    }

    return 0;
}

int llama_graph_detect_node_backend_override(
    uint64_t graph_id,
    const char* node_name,
    enum llama_backend_type node_attempted_backend
) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    if (node_attempted_backend != record->bound_backend) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Node %s attempted backend override in graph %lu\n"
                "  Graph backend: %s, Node attempted: %s\n",
                node_name, graph_id,
                llama_backend_type_name(record->bound_backend),
                llama_backend_type_name(node_attempted_backend));
        llama_record_graph_binding_violation(
            graph_id,
            LLAMA_BINDING_VIOLATION_PER_NODE_OVERRIDE,
            "Node attempted to override graph backend binding",
            node_name
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_assert_no_per_node_backend_selection_decode(
    uint64_t graph_id,
    bool node_backend_selection_attempted
) {
    if (!node_backend_selection_attempted) {
        return 0;
    }

    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    if (record->graph_class != LLAMA_GRAPH_CLASS_DECODE) {
        return 0; // OK for non-decode graphs
    }

    fprintf(stderr, "[GRAPH_BINDING] FATAL: Node attempted backend selection in decode graph %lu\n",
            graph_id);
    llama_record_graph_binding_violation(
        graph_id,
        LLAMA_BINDING_VIOLATION_PER_NODE_OVERRIDE,
        "Node attempted independent backend selection in decode graph",
        "node"
    );
    if (g_enforce_strict) {
        return -1;
    }
    return 0;
}

// ============================================================================
// MIXED-BACKEND GRAPH DETECTION
// ============================================================================

int llama_detect_mixed_backend_decode_graph(
    uint64_t graph_id,
    const char** node_names,
    enum llama_backend_type* node_backends,
    int num_nodes
) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    if (record->graph_class != LLAMA_GRAPH_CLASS_DECODE) {
        return 0; // OK for non-decode graphs
    }

    // Check for mixed backends
    enum llama_backend_type first_backend = LLAMA_BACKEND_UNKNOWN;
    bool has_gpu = false;
    bool has_cpu = false;

    for (int i = 0; i < num_nodes; i++) {
        if (node_backends[i] == LLAMA_BACKEND_CPU) {
            has_cpu = true;
        } else if (node_backends[i] != LLAMA_BACKEND_UNKNOWN &&
                   node_backends[i] != LLAMA_BACKEND_CPU) {
            has_gpu = true;
        }

        if (first_backend == LLAMA_BACKEND_UNKNOWN) {
            first_backend = node_backends[i];
        } else if (node_backends[i] != first_backend) {
            fprintf(stderr, "[GRAPH_BINDING] FATAL: Mixed backends detected in decode graph %lu\n",
                    graph_id);
            llama_record_graph_binding_violation(
                graph_id,
                LLAMA_BINDING_VIOLATION_MIXED_BACKENDS,
                "Graph contains nodes with different backends",
                node_names[i]
            );
            if (g_enforce_strict) {
                return -1;
            }
        }
    }

    if (has_gpu && has_cpu) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Mixed GPU/CPU backends in decode graph %lu\n",
                graph_id);
        llama_record_graph_binding_violation(
            graph_id,
            LLAMA_BINDING_VIOLATION_MIXED_BACKENDS,
            "Graph contains both GPU and CPU backends",
            "graph"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_verify_graph_backend_uniformity(
    uint64_t graph_id,
    const char** decode_critical_nodes,
    enum llama_backend_type* node_backends,
    int num_nodes
) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    enum llama_backend_type expected_backend = record->bound_backend;
    int mismatches = 0;

    for (int i = 0; i < num_nodes; i++) {
        if (node_backends[i] != expected_backend) {
            fprintf(stderr, "[GRAPH_BINDING] FATAL: Node %s backend mismatch\n"
                    "  Expected: %s, Got: %s\n",
                    decode_critical_nodes[i],
                    llama_backend_type_name(expected_backend),
                    llama_backend_type_name(node_backends[i]));
            mismatches++;
        }
    }

    if (mismatches > 0) {
        llama_record_graph_binding_violation(
            graph_id,
            LLAMA_BINDING_VIOLATION_NODE_MISMATCH,
            "Multiple node backend mismatches found",
            "graph"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// BACKEND CONSISTENCY CHECKS
// ============================================================================

int llama_enforce_backend_consistency_at_graph_finalization(
    uint64_t graph_id,
    const char** node_names,
    enum llama_backend_type* node_backends,
    int num_nodes
) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    if (record->binding_state != LLAMA_GRAPH_BINDING_BOUND) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Graph %lu not in BOUND state\n", graph_id);
        return -1;
    }

    // Verify every decode-critical node matches graph backend
    int all_match = 1;
    for (int i = 0; i < num_nodes; i++) {
        if (node_backends[i] != record->bound_backend) {
            fprintf(stderr, "[GRAPH_BINDING] Consistency check FAILED for node %s\n",
                    node_names[i]);
            all_match = 0;
            if (g_enforce_strict) {
                llama_record_graph_binding_violation(
                    graph_id,
                    LLAMA_BINDING_VIOLATION_CONSISTENCY_CHECK,
                    "Node backend inconsistent with graph binding",
                    node_names[i]
                );
                return -1;
            }
        }
    }

    record->all_nodes_match = all_match;
    return 0;
}

int llama_enforce_graph_single_backend_buildability(uint64_t graph_id) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    if (record->graph_class != LLAMA_GRAPH_CLASS_DECODE) {
        return 0; // OK for non-decode graphs
    }

    if (record->bound_backend == LLAMA_BACKEND_UNKNOWN ||
        record->bound_backend == LLAMA_BACKEND_CPU) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Decode graph cannot be built with backend %s\n",
                llama_backend_type_name(record->bound_backend));
        llama_record_graph_binding_violation(
            graph_id,
            LLAMA_BINDING_VIOLATION_PARTIAL_GPU_BINDING,
            "Graph not buildable with single GPU backend",
            "graph"
        );
        return -1;
    }

    return 0;
}

int llama_enforce_graph_rejection_on_binding_failure(uint64_t graph_id) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    if (record->graph_class != LLAMA_GRAPH_CLASS_DECODE) {
        return 0; // OK for non-decode graphs
    }

    if (record->binding_state == LLAMA_GRAPH_BINDING_INVALID ||
        record->violation_count > 0) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Decode graph %lu rejected (violations=%d)\n",
                graph_id, record->violation_count);
        g_graph_binding_registry.rejection_count++;
        record->binding_state = LLAMA_GRAPH_BINDING_INVALID;
        return -1;
    }

    return 0;
}

int llama_enforce_graph_finalization_and_lock(uint64_t graph_id) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    if (record->binding_state == LLAMA_GRAPH_BINDING_FINALIZED) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Graph %lu already finalized\n", graph_id);
        return -1;
    }

    if (record->binding_state != LLAMA_GRAPH_BINDING_BOUND) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Graph %lu not in BOUND state\n", graph_id);
        return -1;
    }

    // Finalize
    record->binding_state = LLAMA_GRAPH_BINDING_FINALIZED;
    record->graph_finalized = true;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    record->finalization_time_us = (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;

    fprintf(stderr, "[GRAPH_BINDING] Graph %lu finalized with backend %s\n",
            graph_id, llama_backend_type_name(record->bound_backend));
    return 0;
}

int llama_enforce_no_backend_rebinding_after_build(
    uint64_t graph_id,
    bool rebinding_attempted
) {
    if (!rebinding_attempted) {
        return 0;
    }

    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    if (record->binding_state == LLAMA_GRAPH_BINDING_FINALIZED) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Attempted to rebind finalized graph %lu\n",
                graph_id);
        llama_record_graph_binding_violation(
            graph_id,
            LLAMA_BINDING_VIOLATION_REBINDING_ATTEMPT,
            "Attempted backend rebinding after graph finalization",
            "graph"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_enforce_execution_backend_matches_binding(
    uint64_t graph_id,
    enum llama_backend_type executing_backend
) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    if (executing_backend != record->bound_backend) {
        fprintf(stderr, "[GRAPH_BINDING] FATAL: Execution backend mismatch for graph %lu\n"
                "  Bound: %s, Executing: %s\n",
                graph_id,
                llama_backend_type_name(record->bound_backend),
                llama_backend_type_name(executing_backend));
        llama_record_graph_binding_violation(
            graph_id,
            LLAMA_BINDING_VIOLATION_EXECUTION_MISMATCH,
            "Executing backend doesn't match graph binding",
            "execution"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// DECODE vs NON-DECODE GRAPH CLASSIFICATION
// ============================================================================

int llama_graph_backend_binding_classify(
    uint64_t graph_id,
    enum llama_graph_classification classification
) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        fprintf(stderr, "[GRAPH_BINDING] Warning: Graph %lu not registered for classification\n",
                graph_id);
        return -1;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    record->graph_class = classification;
    fprintf(stderr, "[GRAPH_BINDING] Graph %lu classified as %s\n",
            graph_id, llama_graph_classification_name(classification));
    return 0;
}

bool llama_graph_backend_binding_is_decode_graph(uint64_t graph_id) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return false;
    }
    return g_graph_binding_registry.graphs[it->second].graph_class == LLAMA_GRAPH_CLASS_DECODE;
}

bool llama_graph_backend_binding_is_nondecode_graph(uint64_t graph_id) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return false;
    }
    enum llama_graph_classification gc = g_graph_binding_registry.graphs[it->second].graph_class;
    return gc == LLAMA_GRAPH_CLASS_PREFILL || gc == LLAMA_GRAPH_CLASS_SETUP;
}

// ============================================================================
// RUNTIME VERIFICATION
// ============================================================================

int llama_graph_verify_binding_at_execution(
    uint64_t graph_id,
    enum llama_backend_type executing_backend
) {
    if (!g_runtime_verification_enabled) {
        return 0;
    }

    return llama_enforce_execution_backend_matches_binding(graph_id, executing_backend);
}

void llama_set_graph_binding_runtime_verification(bool enabled) {
    g_runtime_verification_enabled = enabled;
    fprintf(stderr, "[GRAPH_BINDING] Runtime verification: %s\n",
            enabled ? "ENABLED" : "DISABLED");
}

bool llama_get_graph_binding_runtime_verification(void) {
    return g_runtime_verification_enabled;
}

// ============================================================================
// FAILURE DIAGNOSTICS & REPORTING
// ============================================================================

void llama_record_graph_binding_violation(
    uint64_t graph_id,
    enum llama_graph_binding_violation_type violation_type,
    const char* violation_message,
    const char* offending_node
) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it != g_graph_id_index_map.end()) {
        struct llama_graph_backend_binding_record* record =
            &g_graph_binding_registry.graphs[it->second];
        record->violation_count++;
        record->last_violation_type = violation_type;
        record->last_violation_message = violation_message;
        record->last_violation_node = offending_node;
    }

    g_graph_binding_registry.total_violations++;

    fprintf(stderr, "[GRAPH_BINDING] Violation recorded:\n");
    fprintf(stderr, "  Graph: %lu\n", graph_id);
    fprintf(stderr, "  Type: %s\n", llama_graph_binding_violation_name(violation_type));
    fprintf(stderr, "  Message: %s\n", violation_message);
    fprintf(stderr, "  Node: %s\n", offending_node);
}

void llama_print_graph_binding_violation_diagnostics(
    const struct llama_graph_backend_binding_record* binding,
    enum llama_graph_binding_violation_type violation_type,
    const char* violation_message
) {
    fprintf(stderr, "\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "GRAPH BACKEND BINDING VIOLATION DIAGNOSTICS\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Graph ID: %lu\n", binding->graph_id);
    fprintf(stderr, "Graph Class: %s\n", llama_graph_classification_name(binding->graph_class));
    fprintf(stderr, "Violation Type: %s\n", llama_graph_binding_violation_name(violation_type));
    fprintf(stderr, "Violation Message: %s\n", violation_message);
    fprintf(stderr, "\n");
    fprintf(stderr, "Binding State:\n");
    fprintf(stderr, "  Bound Backend: %s\n", llama_backend_type_name(binding->bound_backend));
    fprintf(stderr, "  Binding State: %d (0=unbound, 1=selecting, 2=bound, 3=finalized, 4=invalid)\n",
            binding->binding_state);
    fprintf(stderr, "  Graph Finalized: %s\n", binding->graph_finalized ? "YES" : "NO");
    fprintf(stderr, "  All Nodes Match: %s\n", binding->all_nodes_match ? "YES" : "NO");
    fprintf(stderr, "  Node Count: %lu\n", binding->node_count);
    fprintf(stderr, "  Violation Count: %d\n", binding->violation_count);
    fprintf(stderr, "\n");
    fprintf(stderr, "Graph Backend Binding Principle:\n");
    fprintf(stderr, "  Decode graphs are single-backend by construction.\n");
    fprintf(stderr, "  Backend ownership is fixed at graph build time.\n");
    fprintf(stderr, "  Mixed or dynamic backend binding during decode is forbidden.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "\n");
}

// ============================================================================
// DIAGNOSTIC LOGGING
// ============================================================================

void llama_log_graph_backend_binding_confirmation(
    uint64_t graph_id,
    enum llama_backend_type bound_backend
) {
    fprintf(stderr, "[GRAPH_BINDING] CONFIRMATION: Graph %lu is single-backend decode graph\n",
            graph_id);
    fprintf(stderr, "[GRAPH_BINDING] Bound Backend: %s\n", llama_backend_type_name(bound_backend));
    fprintf(stderr, "[GRAPH_BINDING] No per-node overrides permitted\n");
    fprintf(stderr, "[GRAPH_BINDING] CPU fallback impossible by design\n");
}

void llama_log_graph_binding_state(uint64_t graph_id) {
    auto it = g_graph_id_index_map.find(graph_id);
    if (it == g_graph_id_index_map.end()) {
        return;
    }

    struct llama_graph_backend_binding_record* record =
        &g_graph_binding_registry.graphs[it->second];

    fprintf(stderr, "\n[GRAPH_BINDING] Graph %lu Binding State:\n", graph_id);
    fprintf(stderr, "  Class: %s\n", llama_graph_classification_name(record->graph_class));
    fprintf(stderr, "  Backend: %s\n", llama_backend_type_name(record->bound_backend));
    fprintf(stderr, "  State: %d\n", record->binding_state);
    fprintf(stderr, "  Nodes: %lu\n", record->node_count);
    fprintf(stderr, "  Finalized: %s\n", record->graph_finalized ? "YES" : "NO");
    fprintf(stderr, "  All Match: %s\n", record->all_nodes_match ? "YES" : "NO");
    fprintf(stderr, "  Violations: %d\n", record->violation_count);
    fprintf(stderr, "\n");
}

void llama_print_graph_backend_binding_principle(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "GRAPH BACKEND BINDING PRINCIPLE STATEMENT\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Core Principle:\n");
    fprintf(stderr, "\"Decode graphs are single-backend by construction. Backend ownership is\n");
    fprintf(stderr, " fixed at graph build time. CPU fallback and hybrid execution are\n");
    fprintf(stderr, " impossible by design.\"\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Enforcement Strategy:\n");
    fprintf(stderr, "1. Bind backend ownership at graph construction\n");
    fprintf(stderr, "2. Remove per-node backend selection for decode\n");
    fprintf(stderr, "3. Reject mixed-backend decode graphs\n");
    fprintf(stderr, "4. Enforce backend consistency checks at finalization\n");
    fprintf(stderr, "5. Prevent backend rebinding after graph build\n");
    fprintf(stderr, "6. Separate decode vs non-decode graphs\n");
    fprintf(stderr, "7. Add runtime verification (debug builds)\n");
    fprintf(stderr, "8. Fail fast on backend incompatibility\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Violations Are FATAL:\n");
    fprintf(stderr, "- Per-node backend override attempts\n");
    fprintf(stderr, "- Mixed-backend graphs (GPU + CPU nodes)\n");
    fprintf(stderr, "- CPU ownership for decode graphs\n");
    fprintf(stderr, "- Backend rebinding after finalization\n");
    fprintf(stderr, "- Node backend mismatches with graph binding\n");
    fprintf(stderr, "- Consistency check failures\n");
    fprintf(stderr, "- Execution backend mismatches\n");
    fprintf(stderr, "- Partial GPU binding (not full GPU)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "\n");
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

void llama_set_graph_binding_enforcement_strict(bool enforce_strict) {
    g_enforce_strict = enforce_strict;
    fprintf(stderr, "[GRAPH_BINDING] Enforcement mode: %s\n",
            enforce_strict ? "STRICT" : "PERMISSIVE");
}

bool llama_get_graph_binding_enforcement_strict(void) {
    return g_enforce_strict;
}

int llama_get_graph_binding_violation_count(void) {
    return g_graph_binding_registry.total_violations;
}

int llama_get_graph_binding_rejection_count(void) {
    return g_graph_binding_registry.rejection_count;
}

void llama_reset_graph_binding_counters(void) {
    g_graph_binding_registry.total_violations = 0;
    g_graph_binding_registry.rejection_count = 0;
    fprintf(stderr, "[GRAPH_BINDING] Counters reset\n");
}

// ============================================================================
// VALIDATION AND SELF-TEST
// ============================================================================

int llama_graph_backend_binding_selftest(void) {
    fprintf(stderr, "\n[GRAPH_BINDING] Running self-test...\n");

    // Test 1: Initialization
    fprintf(stderr, "[TEST 1] Initialization\n");
    if (llama_graph_backend_binding_init() != 0) {
        fprintf(stderr, "  FAILED: Initialization\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 2: Register graph
    fprintf(stderr, "[TEST 2] Register graph\n");
    if (llama_graph_backend_binding_register(1001, LLAMA_GRAPH_CLASS_DECODE) != 0) {
        fprintf(stderr, "  FAILED: Register graph\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 3: Bind backend
    fprintf(stderr, "[TEST 3] Bind backend\n");
    if (llama_graph_backend_binding_bind(1001, LLAMA_BACKEND_CUDA) != 0) {
        fprintf(stderr, "  FAILED: Bind backend\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 4: Verify binding
    fprintf(stderr, "[TEST 4] Verify binding\n");
    if (llama_graph_backend_binding_get_backend(1001) != LLAMA_BACKEND_CUDA) {
        fprintf(stderr, "  FAILED: Get backend\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 5: Finalize
    fprintf(stderr, "[TEST 5] Finalize graph\n");
    if (llama_enforce_graph_finalization_and_lock(1001) != 0) {
        fprintf(stderr, "  FAILED: Finalize\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 6: Verify finalization
    fprintf(stderr, "[TEST 6] Verify finalization\n");
    if (!llama_graph_backend_binding_is_finalized(1001)) {
        fprintf(stderr, "  FAILED: Not finalized\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 7: Prevent rebinding
    fprintf(stderr, "[TEST 7] Prevent rebinding\n");
    if (llama_graph_backend_binding_bind(1001, LLAMA_BACKEND_HIP) == 0) {
        fprintf(stderr, "  FAILED: Allowed rebinding\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 8: Mixed backend detection
    fprintf(stderr, "[TEST 8] Mixed backend detection\n");
    const char* mixed_nodes[] = { "node1", "node2" };
    enum llama_backend_type mixed_backends[] = { LLAMA_BACKEND_CUDA, LLAMA_BACKEND_CPU };
    if (llama_detect_mixed_backend_decode_graph(1001, mixed_nodes, mixed_backends, 2) == 0) {
        fprintf(stderr, "  FAILED: Didn't detect mixed\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    fprintf(stderr, "\n[GRAPH_BINDING] Self-test completed successfully!\n\n");
    return 0;
}
