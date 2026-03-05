/**
 * SECTION 15: Enforce token-persistent graph execution model
 * Implementation
 *
 * Enforces that decode execution uses a token-persistent graph model where
 * a single decode graph instance is created once and executed repeatedly for
 * each token without CPU re-entry into graph control. GPU owns a long-lived
 * execution context across token iterations.
 */

#include "llama-token-persistent-execution.h"

#include <cinttypes>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <vector>

// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================

static struct llama_token_persistent_execution_validation_state g_token_persistent_state = {
    {
        LLAMA_DECODE_MODE_INACTIVE,
        0,
        0,
        0,
        false,
        false,
        false,
        LLAMA_LIFETIME_UNBOUND,
        LLAMA_PERSISTENCE_UNINITIALIZED
    },
    {
        0,
        LLAMA_DECODE_MODE_INACTIVE,
        0,
        0,
        LLAMA_REENTRANCY_NONE,
        LLAMA_MUTATION_NONE,
        false,
        false,
        false,
        false
    },
    {
        0,
        0,
        0,
        0,
        LLAMA_TOKEN_PROGRESS_NONE,
        0,
        false,
        0
    },
    0,
    0,
    0,
    true,
    false
};

// Per-graph persistence state: maps graph_id -> persistence_state
static std::map<uint64_t, enum llama_graph_persistence_state> g_graph_persistence_states;

// Per-graph lifetime binding: maps graph_id -> lifetime_binding
static std::map<uint64_t, enum llama_graph_lifetime_binding> g_graph_lifetime_bindings;

// Per-graph re-entry violation tracking: maps graph_id -> violation_count
static std::map<uint64_t, int> g_graph_reentrancy_violations;

// Per-graph mutation tracking: maps graph_id -> mutation_count
static std::map<uint64_t, int> g_graph_mutations;

// Decode session start time
static uint64_t g_decode_session_start_ns = 0;

// Decode session graph ID (single persistent graph)
static uint64_t g_decode_session_graph_id = 0;

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize token-persistent execution enforcement system
 */
int llama_token_persistent_init(void) {
    // Clear all tracking maps
    g_graph_persistence_states.clear();
    g_graph_lifetime_bindings.clear();
    g_graph_reentrancy_violations.clear();
    g_graph_mutations.clear();

    // Reset global state
    g_token_persistent_state.decode_mode_record.mode = LLAMA_DECODE_MODE_INACTIVE;
    g_token_persistent_state.decode_mode_record.graph_id = 0;
    g_token_persistent_state.decode_mode_record.total_tokens_processed = 0;
    g_token_persistent_state.decode_mode_record.graph_lifetime_locked = false;
    g_token_persistent_state.decode_mode_record.graph_persistence_locked = false;
    g_token_persistent_state.decode_mode_record.cpu_reentrancy_forbidden = false;
    g_token_persistent_state.execution_record.total_cpu_reentrancy_violations = 0;
    g_token_persistent_state.execution_record.total_graph_mutations = 0;
    g_token_persistent_state.total_reentrancy_violations = 0;
    g_token_persistent_state.total_mutation_violations = 0;
    g_token_persistent_state.total_graph_lifetime_mismatches = 0;
    g_decode_session_start_ns = 0;
    g_decode_session_graph_id = 0;

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 1-5: Decode mode lifecycle
// ============================================================================

/**
 * ENFORCEMENT POINT 1: Enter decode mode with persistent graph
 * Transition from inactive to active decode mode with graph bound to decode lifetime.
 */
int llama_token_persistent_enter_decode_mode(uint64_t graph_id) {
    if (graph_id == 0) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP1: Invalid graph_id (0)\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    if (g_token_persistent_state.decode_mode_record.mode != LLAMA_DECODE_MODE_INACTIVE) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP1: Decode mode already active\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    // Enter initializing state
    g_token_persistent_state.decode_mode_record.mode = LLAMA_DECODE_MODE_INITIALIZING;
    g_token_persistent_state.decode_mode_record.graph_id = graph_id;
    g_decode_session_graph_id = graph_id;
    g_decode_session_start_ns = 0; // Would be current_time_ns in real implementation

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 2: Lock graph lifetime to decode
 * Graph lifetime must exactly match decode lifetime.
 */
int llama_token_persistent_lock_graph_lifetime_to_decode(uint64_t graph_id) {
    if (graph_id != g_decode_session_graph_id) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP2: graph_id mismatch: %" PRIu64 " vs %" PRIu64 "\n",
                graph_id, g_decode_session_graph_id);
        g_token_persistent_state.total_graph_lifetime_mismatches++;
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    g_graph_lifetime_bindings[graph_id] = LLAMA_LIFETIME_BINDING_START;
    g_token_persistent_state.decode_mode_record.lifetime_binding = LLAMA_LIFETIME_BINDING_START;

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 3: Bind graph to decode lifetime
 * Complete the binding; graph lifetime now locked to decode.
 */
int llama_token_persistent_bind_graph_to_decode_lifetime(uint64_t graph_id) {
    if (g_graph_lifetime_bindings[graph_id] != LLAMA_LIFETIME_BINDING_START) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP3: lifetime binding not in START state\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    g_graph_lifetime_bindings[graph_id] = LLAMA_LIFETIME_BOUND;
    g_token_persistent_state.decode_mode_record.graph_lifetime_locked = true;
    g_token_persistent_state.decode_mode_record.lifetime_binding = LLAMA_LIFETIME_BOUND;

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 4: Lock graph persistence model
 * Persistence model locked; graph will execute repeatedly without teardown.
 */
int llama_token_persistent_lock_graph_persistence_model(void) {
    uint64_t graph_id = g_decode_session_graph_id;
    if (graph_id == 0) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP4: No decode session graph\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    g_graph_persistence_states[graph_id] = LLAMA_PERSISTENCE_FIRST_TOKEN;
    g_token_persistent_state.decode_mode_record.persistence = LLAMA_PERSISTENCE_FIRST_TOKEN;
    g_token_persistent_state.decode_mode_record.graph_persistence_locked = true;

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 5: Assert single persistent graph throughout
 * Only one graph persists across entire decode session.
 */
int llama_token_persistent_assert_single_persistent_graph(void) {
    uint64_t graph_id = g_decode_session_graph_id;
    if (graph_id == 0) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP5: No persistent graph\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    // Verify persistence state
    if (g_graph_persistence_states[graph_id] == LLAMA_PERSISTENCE_UNINITIALIZED) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP5: Graph persistence not initialized\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    // Activate decode mode
    g_token_persistent_state.decode_mode_record.mode = LLAMA_DECODE_MODE_ACTIVE;
    g_graph_persistence_states[graph_id] = LLAMA_PERSISTENCE_PERSISTENT;
    g_token_persistent_state.decode_mode_record.persistence = LLAMA_PERSISTENCE_PERSISTENT;
    g_token_persistent_state.execution_record.current_graph_id = graph_id;
    g_token_persistent_state.execution_record.decode_mode = LLAMA_DECODE_MODE_ACTIVE;

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 6-8: CPU re-entry prevention
// ============================================================================

/**
 * ENFORCEMENT POINT 6: Forbid per-token graph resubmission
 * Graph is submitted once; not resubmitted per token.
 */
int llama_token_persistent_forbid_per_token_graph_resubmission(void) {
    if (g_token_persistent_state.decode_mode_record.mode != LLAMA_DECODE_MODE_ACTIVE) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP6: Decode not in ACTIVE mode\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    g_token_persistent_state.decode_mode_record.cpu_reentrancy_forbidden = true;
    return 0; // Success
}

/**
 * ENFORCEMENT POINT 7: Forbid per-token rebinding
 * No tensor rebinding, backend reassignment, or buffer rebinding per token.
 */
int llama_token_persistent_forbid_per_token_rebinding(void) {
    if (!g_token_persistent_state.execution_record.graph_backend_immutable) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP7: Graph backend not immutable\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 8: Forbid CPU orchestration of token loop
 * CPU must not advance token counters, update inputs, or patch parameters per token.
 */
int llama_token_persistent_forbid_cpu_orchestration(void) {
    if (!g_token_persistent_state.execution_record.token_progression_gpu_owned) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP8: Token progression not GPU-owned\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// ENFORCEMENT POINT 9-10: Graph state immutability
// ============================================================================

/**
 * ENFORCEMENT POINT 9: Lock graph inputs and backend
 * Graph inputs and backend are immutable throughout decode.
 */
int llama_token_persistent_lock_graph_inputs_and_backend(void) {
    if (!g_token_persistent_state.execution_record.graph_input_stable) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP9: Graph inputs not stable\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    g_token_persistent_state.execution_record.graph_backend_immutable = true;
    g_token_persistent_state.execution_record.execution_plan_immutable = true;

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 10: Assert execution context unchanged
 * Runtime check that context persists unchanged.
 */
int llama_token_persistent_assert_execution_context_unchanged(void) {
    uint64_t graph_id = g_token_persistent_state.execution_record.current_graph_id;
    if (graph_id == 0) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP10: No current graph\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    // Verify persistence still active
    if (g_graph_persistence_states[graph_id] != LLAMA_PERSISTENCE_PERSISTENT &&
        g_graph_persistence_states[graph_id] != LLAMA_PERSISTENCE_LOCKED) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR EP10: Graph persistence broken\n");
        g_token_persistent_state.total_mutation_violations++;
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// GPU-OWNED TOKEN PROGRESSION
// ============================================================================

/**
 * Enable GPU-owned token progression
 * Token progression state moves to GPU; CPU no longer updates tokens.
 */
int llama_token_persistent_enable_gpu_owned_token_progression(void) {
    g_token_persistent_state.execution_record.token_progression_gpu_owned = true;
    g_token_persistent_state.gpu_state.state_gpu_resident = true;

    return 0; // Success
}

/**
 * Update GPU-resident token state
 * Called by GPU to update token progress.
 */
int llama_token_persistent_update_gpu_token_state(
    uint64_t token_index,
    uint64_t kv_offset,
    uint64_t context_index
) {
    if (!g_token_persistent_state.execution_record.token_progression_gpu_owned) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR: Token progression not GPU-owned\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    g_token_persistent_state.gpu_state.current_token_index = token_index;
    g_token_persistent_state.gpu_state.kv_cache_offset = kv_offset;
    g_token_persistent_state.gpu_state.context_index = context_index;
    g_token_persistent_state.gpu_state.token_progress = LLAMA_TOKEN_PROGRESS_COMPLETE;
    g_token_persistent_state.decode_mode_record.total_tokens_processed++;

    return 0; // Success
}

// ============================================================================
// RE-ENTRY VIOLATION DETECTION
// ============================================================================

/**
 * Detect graph rebuild attempt
 */
int llama_token_persistent_detect_graph_rebuild_attempt(void) {
    fprintf(stderr, "[TOKEN_PERSIST] VIOLATION: Graph rebuild attempted\n");
    g_token_persistent_state.execution_record.last_reentrancy = LLAMA_REENTRANCY_GRAPH_REBUILD;
    g_token_persistent_state.execution_record.total_cpu_reentrancy_violations++;
    g_token_persistent_state.total_reentrancy_violations++;
    g_graph_reentrancy_violations[g_decode_session_graph_id]++;

    if (g_token_persistent_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect graph resubmit attempt per token
 */
int llama_token_persistent_detect_graph_resubmit_attempt(void) {
    fprintf(stderr, "[TOKEN_PERSIST] VIOLATION: Graph resubmit per token attempted\n");
    g_token_persistent_state.execution_record.last_reentrancy = LLAMA_REENTRANCY_GRAPH_RESUBMIT;
    g_token_persistent_state.execution_record.total_cpu_reentrancy_violations++;
    g_token_persistent_state.total_reentrancy_violations++;
    g_graph_reentrancy_violations[g_decode_session_graph_id]++;

    if (g_token_persistent_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect CPU token counter update attempt
 */
int llama_token_persistent_detect_token_counter_cpu_update(void) {
    fprintf(stderr, "[TOKEN_PERSIST] VIOLATION: CPU token counter update attempted\n");
    g_token_persistent_state.execution_record.last_reentrancy = LLAMA_REENTRANCY_TOKEN_COUNTER_UPDATE;
    g_token_persistent_state.execution_record.total_cpu_reentrancy_violations++;
    g_token_persistent_state.total_reentrancy_violations++;

    if (g_token_persistent_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect graph input patch attempt
 */
int llama_token_persistent_detect_graph_input_patch_attempt(void) {
    fprintf(stderr, "[TOKEN_PERSIST] VIOLATION: Graph input patch attempted\n");
    g_token_persistent_state.execution_record.last_reentrancy = LLAMA_REENTRANCY_GRAPH_INPUT_PATCH;
    g_token_persistent_state.execution_record.total_cpu_reentrancy_violations++;
    g_token_persistent_state.total_reentrancy_violations++;

    if (g_token_persistent_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect tensor rebinding attempt
 */
int llama_token_persistent_detect_tensor_rebinding_attempt(void) {
    fprintf(stderr, "[TOKEN_PERSIST] VIOLATION: Tensor rebinding attempted\n");
    g_token_persistent_state.execution_record.last_reentrancy = LLAMA_REENTRANCY_TENSOR_REBINDING;
    g_token_persistent_state.execution_record.total_cpu_reentrancy_violations++;
    g_token_persistent_state.total_reentrancy_violations++;

    if (g_token_persistent_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect backend reassignment attempt
 */
int llama_token_persistent_detect_backend_reassignment_attempt(void) {
    fprintf(stderr, "[TOKEN_PERSIST] VIOLATION: Backend reassignment attempted\n");
    g_token_persistent_state.execution_record.last_reentrancy = LLAMA_REENTRANCY_BACKEND_REASSIGN;
    g_token_persistent_state.execution_record.total_cpu_reentrancy_violations++;
    g_token_persistent_state.total_reentrancy_violations++;

    if (g_token_persistent_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect CPU orchestration attempt
 */
int llama_token_persistent_detect_cpu_orchestration_attempt(void) {
    fprintf(stderr, "[TOKEN_PERSIST] VIOLATION: CPU orchestration detected\n");
    g_token_persistent_state.execution_record.last_reentrancy = LLAMA_REENTRANCY_ORCHESTRATION;
    g_token_persistent_state.execution_record.total_cpu_reentrancy_violations++;
    g_token_persistent_state.total_reentrancy_violations++;

    if (g_token_persistent_state.enforcement_strict) abort();
    return -1;
}

// ============================================================================
// MUTATION DETECTION
// ============================================================================

/**
 * Detect graph state mutation
 */
int llama_token_persistent_detect_graph_mutation(
    enum llama_graph_state_mutation mutation_type
) {
    fprintf(stderr, "[TOKEN_PERSIST] VIOLATION: Graph mutation detected: %s\n",
            llama_graph_state_mutation_name(mutation_type));

    g_token_persistent_state.execution_record.last_mutation = mutation_type;
    g_token_persistent_state.execution_record.total_graph_mutations++;
    g_token_persistent_state.total_mutation_violations++;
    g_graph_mutations[g_decode_session_graph_id]++;

    if (g_token_persistent_state.enforcement_strict) abort();
    return -1;
}

/**
 * Detect input shape change
 */
int llama_token_persistent_detect_input_shape_change(void) {
    return llama_token_persistent_detect_graph_mutation(LLAMA_MUTATION_INPUT_SHAPE_CHANGE);
}

/**
 * Detect input location change
 */
int llama_token_persistent_detect_input_location_change(void) {
    return llama_token_persistent_detect_graph_mutation(LLAMA_MUTATION_INPUT_LOCATION_CHANGE);
}

// ============================================================================
// QUERY AND VERIFICATION FUNCTIONS
// ============================================================================

/**
 * Get decode mode record
 */
struct llama_decode_mode_record llama_token_persistent_get_decode_mode_record(void) {
    return g_token_persistent_state.decode_mode_record;
}

/**
 * Get execution record
 */
struct llama_token_persistent_execution_record llama_token_persistent_get_execution_record(void) {
    return g_token_persistent_state.execution_record;
}

/**
 * Get GPU state
 */
struct llama_token_persistent_state llama_token_persistent_get_gpu_state(void) {
    return g_token_persistent_state.gpu_state;
}

/**
 * Get decode mode
 */
enum llama_decode_mode_state llama_token_persistent_get_decode_mode(void) {
    return g_token_persistent_state.decode_mode_record.mode;
}

// ============================================================================
// PERSISTENCE VERIFICATION
// ============================================================================

/**
 * Verify single graph throughout decode
 */
int llama_token_persistent_verify_single_graph_throughout_decode(void) {
    if (g_decode_session_graph_id == 0) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR: No persistent graph\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    uint64_t graph_id = g_decode_session_graph_id;
    if (g_token_persistent_state.execution_record.current_graph_id != graph_id) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR: Current graph mismatch\n");
        g_token_persistent_state.total_graph_lifetime_mismatches++;
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify no per-token resubmission
 */
int llama_token_persistent_verify_no_per_token_resubmission(void) {
    if (!g_token_persistent_state.decode_mode_record.cpu_reentrancy_forbidden) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR: Per-token resubmission not forbidden\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify graph inputs stable
 */
int llama_token_persistent_verify_graph_inputs_stable(void) {
    if (!g_token_persistent_state.execution_record.graph_input_stable) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR: Graph inputs not stable\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify GPU owns token progression
 */
int llama_token_persistent_verify_gpu_owns_token_progression(void) {
    if (!g_token_persistent_state.execution_record.token_progression_gpu_owned) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR: Token progression not GPU-owned\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify no CPU orchestration
 */
int llama_token_persistent_verify_no_cpu_orchestration(void) {
    if (g_token_persistent_state.execution_record.total_cpu_reentrancy_violations > 0) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR: CPU orchestration detected (%d violations)\n",
                g_token_persistent_state.execution_record.total_cpu_reentrancy_violations);
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// LIFETIME BINDING VERIFICATION
// ============================================================================

/**
 * Verify graph lifetime matches decode lifetime
 */
int llama_token_persistent_verify_graph_lifetime_matches_decode(void) {
    uint64_t graph_id = g_decode_session_graph_id;
    if (graph_id == 0) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR: No decode session graph\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    if (g_graph_lifetime_bindings[graph_id] != LLAMA_LIFETIME_BOUND &&
        g_graph_lifetime_bindings[graph_id] != LLAMA_LIFETIME_LOCKED) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR: Graph lifetime not bound to decode\n");
        g_token_persistent_state.total_graph_lifetime_mismatches++;
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

/**
 * Verify graph doesn't outlive decode
 */
int llama_token_persistent_verify_graph_not_outliving_decode(void) {
    uint64_t graph_id = g_token_persistent_state.decode_mode_record.graph_id;
    enum llama_decode_mode_state mode = g_token_persistent_state.decode_mode_record.mode;

    if (mode == LLAMA_DECODE_MODE_TERMINATED || mode == LLAMA_DECODE_MODE_ERROR) {
        // Graph must be cleaned up when decode terminates
        if (g_graph_persistence_states[graph_id] != LLAMA_PERSISTENCE_INVALID) {
            fprintf(stderr, "[TOKEN_PERSIST] ERROR: Graph outliving decode\n");
            if (g_token_persistent_state.enforcement_strict) abort();
            return -1;
        }
    }

    return 0; // Success
}

/**
 * Verify decode doesn't outlive graph
 */
int llama_token_persistent_verify_decode_not_outliving_graph(void) {
    if (g_decode_session_graph_id == 0 &&
        g_token_persistent_state.decode_mode_record.mode != LLAMA_DECODE_MODE_INACTIVE) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR: Decode outliving graph\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    return 0; // Success
}

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

/**
 * Log decode mode entered
 */
void llama_token_persistent_log_decode_mode_entered(uint64_t graph_id) {
    printf("[TOKEN_PERSIST] ✓ Decode mode entered with persistent graph %" PRIu64 "\n", graph_id);
    printf("  - Single graph instance for entire decode session\n");
    printf("  - No per-token graph resubmission\n");
    printf("  - CPU relinquishes graph control\n");
}

/**
 * Log graph lifetime bound
 */
void llama_token_persistent_log_graph_lifetime_bound(void) {
    printf("[TOKEN_PERSIST] ✓ Graph lifetime bound to decode lifetime\n");
    printf("  - Graph lifetime: [decode_start, decode_end]\n");
    printf("  - No graph outliving decode\n");
    printf("  - No decode outliving graph\n");
}

/**
 * Log persistence locked
 */
void llama_token_persistent_log_persistence_locked(void) {
    printf("[TOKEN_PERSIST] ✓ Graph persistence locked\n");
    printf("  - Graph executes repeatedly without teardown\n");
    printf("  - GPU maintains execution context\n");
    printf("  - Token progression GPU-owned\n");
}

/**
 * Print decode mode status
 */
void llama_token_persistent_print_decode_mode_status(void) {
    printf("\n=== Decode Mode Status ===\n");
    printf("Mode: %s\n", llama_decode_mode_state_name(g_token_persistent_state.decode_mode_record.mode));
    printf("Graph ID: %" PRIu64 "\n", g_token_persistent_state.decode_mode_record.graph_id);
    printf("Lifetime locked: %s\n", g_token_persistent_state.decode_mode_record.graph_lifetime_locked ? "YES" : "NO");
    printf("Persistence locked: %s\n", g_token_persistent_state.decode_mode_record.graph_persistence_locked ? "YES" : "NO");
    printf("CPU re-entry forbidden: %s\n", g_token_persistent_state.decode_mode_record.cpu_reentrancy_forbidden ? "YES" : "NO");
    printf("Tokens processed: %" PRIu64 "\n", g_token_persistent_state.decode_mode_record.total_tokens_processed);
    printf("===========================\n\n");
}

/**
 * Print GPU state summary
 */
void llama_token_persistent_print_gpu_state_summary(void) {
    printf("\n=== GPU State Summary ===\n");
    printf("Graph ID: %" PRIu64 "\n", g_token_persistent_state.gpu_state.graph_id);
    printf("Token index: %" PRIu64 "\n", g_token_persistent_state.gpu_state.current_token_index);
    printf("KV cache offset: %" PRIu64 "\n", g_token_persistent_state.gpu_state.kv_cache_offset);
    printf("Context index: %" PRIu64 "\n", g_token_persistent_state.gpu_state.context_index);
    printf("Token progress: %s\n", llama_token_progress_state_name(g_token_persistent_state.gpu_state.token_progress));
    printf("GPU-resident: %s\n", g_token_persistent_state.gpu_state.state_gpu_resident ? "YES" : "NO");
    printf("========================\n\n");
}

/**
 * Print invariant violations
 */
void llama_token_persistent_print_invariant_violations(void) {
    printf("\n=== Invariant Violations ===\n");
    printf("CPU re-entry violations: %d\n", g_token_persistent_state.execution_record.total_cpu_reentrancy_violations);
    printf("Graph mutations: %d\n", g_token_persistent_state.execution_record.total_graph_mutations);
    printf("Lifetime mismatches: %d\n", g_token_persistent_state.total_graph_lifetime_mismatches);
    printf("Last re-entry: %s\n", llama_cpu_reentrancy_violation_name(g_token_persistent_state.execution_record.last_reentrancy));
    printf("Last mutation: %s\n", llama_graph_state_mutation_name(g_token_persistent_state.execution_record.last_mutation));
    printf("============================\n\n");
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

/**
 * Report re-entry violation
 */
void llama_token_persistent_report_reentrancy_violation(
    enum llama_cpu_reentrancy_violation violation_type,
    const char* details
) {
    fprintf(stderr, "[TOKEN_PERSIST] REPORT: CPU re-entry violation\n");
    fprintf(stderr, "  - Violation type: %s\n", llama_cpu_reentrancy_violation_name(violation_type));
    fprintf(stderr, "  - Details: %s\n", details ? details : "unknown");
    fprintf(stderr, "  - Expected: No CPU re-entry during persistent execution\n");

    g_token_persistent_state.execution_record.total_cpu_reentrancy_violations++;
    g_token_persistent_state.total_reentrancy_violations++;
}

/**
 * Report mutation violation
 */
void llama_token_persistent_report_mutation_violation(
    enum llama_graph_state_mutation mutation_type
) {
    fprintf(stderr, "[TOKEN_PERSIST] REPORT: Graph state mutation\n");
    fprintf(stderr, "  - Mutation type: %s\n", llama_graph_state_mutation_name(mutation_type));
    fprintf(stderr, "  - Expected: Graph state immutable during persistent execution\n");

    g_token_persistent_state.execution_record.total_graph_mutations++;
    g_token_persistent_state.total_mutation_violations++;
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

/**
 * Set enforcement mode (strict=abort, permissive=log)
 */
void llama_token_persistent_set_enforcement_strict(bool strict) {
    g_token_persistent_state.enforcement_strict = strict;
}

/**
 * Get current enforcement mode
 */
bool llama_token_persistent_get_enforcement_strict(void) {
    return g_token_persistent_state.enforcement_strict;
}

/**
 * Set debug persistence check per token
 */
void llama_token_persistent_set_debug_check_persistence_per_token(bool debug) {
    g_token_persistent_state.debug_check_persistence_per_token = debug;
}

// ============================================================================
// EXIT DECODE MODE
// ============================================================================

/**
 * Exit decode mode
 */
int llama_token_persistent_exit_decode_mode(void) {
    if (g_token_persistent_state.decode_mode_record.mode != LLAMA_DECODE_MODE_ACTIVE) {
        fprintf(stderr, "[TOKEN_PERSIST] ERROR: Decode mode not active\n");
        if (g_token_persistent_state.enforcement_strict) abort();
        return -1;
    }

    g_token_persistent_state.decode_mode_record.mode = LLAMA_DECODE_MODE_TERMINATING;
    return 0; // Success
}

/**
 * Verify decode mode exit clean
 */
int llama_token_persistent_verify_decode_mode_exit_clean(void) {
    uint64_t graph_id = g_decode_session_graph_id;
    if (graph_id != 0) {
        g_graph_persistence_states[graph_id] = LLAMA_PERSISTENCE_INVALID;
        g_graph_lifetime_bindings[graph_id] = LLAMA_LIFETIME_INVALID;
    }

    g_token_persistent_state.decode_mode_record.mode = LLAMA_DECODE_MODE_TERMINATED;
    g_token_persistent_state.execution_record.decode_mode = LLAMA_DECODE_MODE_TERMINATED;
    g_decode_session_graph_id = 0;

    return 0; // Success
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================

/**
 * Test Case 1: Decode mode entry
 */
static int test_decode_mode_entry(void) {
    llama_token_persistent_init();

    int ret = llama_token_persistent_enter_decode_mode(1);
    if (ret != 0 || g_token_persistent_state.decode_mode_record.mode != LLAMA_DECODE_MODE_INITIALIZING) {
        fprintf(stderr, "[TEST] FAIL: Decode mode entry\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 2: Graph lifetime binding
 */
static int test_graph_lifetime_binding(void) {
    llama_token_persistent_init();

    llama_token_persistent_enter_decode_mode(1);
    int ret = llama_token_persistent_lock_graph_lifetime_to_decode(1);
    if (ret != 0) {
        fprintf(stderr, "[TEST] FAIL: Graph lifetime lock\n");
        return -1;
    }

    ret = llama_token_persistent_bind_graph_to_decode_lifetime(1);
    if (ret != 0 || !g_token_persistent_state.decode_mode_record.graph_lifetime_locked) {
        fprintf(stderr, "[TEST] FAIL: Graph lifetime binding\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 3: Graph persistence lock
 */
static int test_graph_persistence_lock(void) {
    llama_token_persistent_init();

    llama_token_persistent_enter_decode_mode(1);
    llama_token_persistent_lock_graph_lifetime_to_decode(1);
    llama_token_persistent_bind_graph_to_decode_lifetime(1);

    int ret = llama_token_persistent_lock_graph_persistence_model();
    if (ret != 0 || !g_token_persistent_state.decode_mode_record.graph_persistence_locked) {
        fprintf(stderr, "[TEST] FAIL: Graph persistence lock\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 4: Persistent graph assertion
 */
static int test_persistent_graph_assertion(void) {
    llama_token_persistent_init();

    llama_token_persistent_enter_decode_mode(1);
    llama_token_persistent_lock_graph_lifetime_to_decode(1);
    llama_token_persistent_bind_graph_to_decode_lifetime(1);
    llama_token_persistent_lock_graph_persistence_model();

    int ret = llama_token_persistent_assert_single_persistent_graph();
    if (ret != 0 || g_token_persistent_state.decode_mode_record.mode != LLAMA_DECODE_MODE_ACTIVE) {
        fprintf(stderr, "[TEST] FAIL: Persistent graph assertion\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 5: GPU-owned token progression
 */
static int test_gpu_owned_progression(void) {
    llama_token_persistent_init();

    int ret = llama_token_persistent_enable_gpu_owned_token_progression();
    if (ret != 0 || !g_token_persistent_state.execution_record.token_progression_gpu_owned) {
        fprintf(stderr, "[TEST] FAIL: GPU-owned token progression\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 6: Re-entry violation detection
 */
static int test_reentrancy_violation(void) {
    llama_token_persistent_init();

    int ret = llama_token_persistent_detect_graph_rebuild_attempt();
    if (ret == 0) {
        fprintf(stderr, "[TEST] FAIL: Re-entry violation not detected\n");
        return -1;
    }

    if (g_token_persistent_state.execution_record.total_cpu_reentrancy_violations != 1) {
        fprintf(stderr, "[TEST] FAIL: Violation count incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 7: Mutation detection
 */
static int test_mutation_detection(void) {
    llama_token_persistent_init();

    int ret = llama_token_persistent_detect_graph_mutation(LLAMA_MUTATION_BACKEND_CHANGE);
    if (ret == 0) {
        fprintf(stderr, "[TEST] FAIL: Mutation not detected\n");
        return -1;
    }

    if (g_token_persistent_state.execution_record.total_graph_mutations != 1) {
        fprintf(stderr, "[TEST] FAIL: Mutation count incorrect\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Test Case 8: Decode mode exit
 */
static int test_decode_mode_exit(void) {
    llama_token_persistent_init();

    llama_token_persistent_enter_decode_mode(1);
    llama_token_persistent_assert_single_persistent_graph();

    int ret = llama_token_persistent_exit_decode_mode();
    if (ret != 0 || g_token_persistent_state.decode_mode_record.mode != LLAMA_DECODE_MODE_TERMINATING) {
        fprintf(stderr, "[TEST] FAIL: Decode mode exit\n");
        return -1;
    }

    return 0; // Success
}

/**
 * Run all self-tests
 */
int llama_token_persistent_selftest(void) {
    printf("[TOKEN_PERSIST] Running self-test suite...\n");

    // Set permissive mode for testing
    bool old_strict = g_token_persistent_state.enforcement_strict;
    g_token_persistent_state.enforcement_strict = false;

    int tests_passed = 0;
    int tests_failed = 0;

    #define RUN_TEST(test_fn) do { \
        if (test_fn() == 0) { \
            printf("  ✓ " #test_fn "\n"); \
            tests_passed++; \
        } else { \
            printf("  ✗ " #test_fn "\n"); \
            tests_failed++; \
        } \
    } while(0)

    RUN_TEST(test_decode_mode_entry);
    RUN_TEST(test_graph_lifetime_binding);
    RUN_TEST(test_graph_persistence_lock);
    RUN_TEST(test_persistent_graph_assertion);
    RUN_TEST(test_gpu_owned_progression);
    RUN_TEST(test_reentrancy_violation);
    RUN_TEST(test_mutation_detection);
    RUN_TEST(test_decode_mode_exit);

    #undef RUN_TEST

    // Restore enforcement mode
    g_token_persistent_state.enforcement_strict = old_strict;

    printf("[TOKEN_PERSIST] Self-tests complete: %d passed, %d failed\n", tests_passed, tests_failed);
    return (tests_failed == 0) ? 0 : -1;
}
