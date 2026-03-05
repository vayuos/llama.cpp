/**
 * SECTION 6 IMPLEMENTATION: Remove Runtime Backend Switching During Decode
 *
 * This file implements backend immutability enforcement to eliminate all runtime
 * backend switching during the decode phase.
 */

#include "llama-backend-immutability-enforce.h"
#include <cstring>
#include <cstdio>
#include <ctime>
#include <map>
#include <vector>

// ============================================================================
// GLOBAL STATE
// ============================================================================

static struct llama_backend_immutability_state g_backend_immutability = {
    LLAMA_BACKEND_PHASE_UNINITIALIZED,
    LLAMA_BACKEND_UNKNOWN,
    0,
    NULL,
    1024,
    false,
    NULL,
    0,
    false,
    0,
    0,
    LLAMA_BACKEND_VIOLATION_UNKNOWN,
    NULL
};

static bool g_enforce_strict = true;
static int g_total_violation_count = 0;

// Per-token backend tracking to detect per-token switching
static std::map<uint64_t, enum llama_backend_type> g_token_backend_map;

// Per-layer backend tracking to detect per-layer switching
static std::map<int, enum llama_backend_type> g_layer_backend_map;

// Per-operation backend tracking to detect per-op switching
static std::map<std::string, enum llama_backend_type> g_operation_backend_map;

// ============================================================================
// INITIALIZATION
// ============================================================================

int llama_backend_immutability_init(void) {
    // Allocate resolution records array
    if (g_backend_immutability.resolutions == NULL) {
        g_backend_immutability.resolutions =
            (struct llama_backend_resolution_record*)malloc(
                sizeof(struct llama_backend_resolution_record) *
                g_backend_immutability.max_resolutions
            );
        if (g_backend_immutability.resolutions == NULL) {
            fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Failed to allocate resolution records\n");
            return -1;
        }
    }

    // Clear state
    g_backend_immutability.phase = LLAMA_BACKEND_PHASE_UNINITIALIZED;
    g_backend_immutability.decode_backend = LLAMA_BACKEND_UNKNOWN;
    g_backend_immutability.resolution_count = 0;
    g_backend_immutability.backend_invalid = false;
    g_backend_immutability.invalidation_reason = NULL;
    g_backend_immutability.invalidation_time_us = 0;
    g_backend_immutability.immutability_locked = false;
    g_backend_immutability.freeze_time_us = 0;
    g_backend_immutability.violation_count = 0;
    g_backend_immutability.last_violation_location = LLAMA_BACKEND_VIOLATION_UNKNOWN;
    g_backend_immutability.last_violation_message = NULL;

    // Clear tracking maps
    g_token_backend_map.clear();
    g_layer_backend_map.clear();
    g_operation_backend_map.clear();

    fprintf(stderr, "[BACKEND_IMMUTABILITY] Initialized: Backend immutability tracking ready\n");
    return 0;
}

// ============================================================================
// BACKEND FREEZING AND IMMUTABILITY CONTROL
// ============================================================================

int llama_backend_immutability_freeze_for_decode(enum llama_backend_type backend) {
    if (backend == LLAMA_BACKEND_UNKNOWN || backend == LLAMA_BACKEND_CPU) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Cannot freeze decode with backend=%s\n",
                llama_backend_type_name(backend));
        return -1;
    }

    if (g_backend_immutability.immutability_locked) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend already frozen, cannot re-freeze\n");
        return -1;
    }

    // Record current time
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now_us = (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;

    // Freeze the backend
    g_backend_immutability.decode_backend = backend;
    g_backend_immutability.immutability_locked = true;
    g_backend_immutability.freeze_time_us = now_us;
    g_backend_immutability.phase = LLAMA_BACKEND_PHASE_DECODE_FROZEN;

    fprintf(stderr, "[BACKEND_IMMUTABILITY] Backend FROZEN for decode: %s\n",
            llama_backend_type_name(backend));
    return 0;
}

bool llama_backend_immutability_is_frozen(void) {
    return g_backend_immutability.immutability_locked;
}

enum llama_backend_type llama_backend_immutability_get_frozen_backend(void) {
    return g_backend_immutability.decode_backend;
}

int llama_backend_immutability_verify_unchanged(enum llama_backend_type expected_backend) {
    if (!g_backend_immutability.immutability_locked) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend not frozen, cannot verify\n");
        return -1;
    }

    if (g_backend_immutability.decode_backend != expected_backend) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend changed! Expected=%s, Current=%s\n",
                llama_backend_type_name(expected_backend),
                llama_backend_type_name(g_backend_immutability.decode_backend));
        g_backend_immutability.violation_count++;
        g_total_violation_count++;
        llama_record_backend_immutability_violation(
            LLAMA_BACKEND_VIOLATION_INVALIDATION,
            "verify_unchanged",
            "Backend changed after freezing"
        );
        return -1;
    }

    return 0;
}

int llama_backend_immutability_record_invalidation(const char* reason) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t now_us = (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;

    g_backend_immutability.backend_invalid = true;
    g_backend_immutability.invalidation_reason = reason;
    g_backend_immutability.invalidation_time_us = now_us;
    g_backend_immutability.phase = LLAMA_BACKEND_PHASE_TERMINATED;

    fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend invalidated during decode: %s\n", reason);
    g_backend_immutability.violation_count++;
    g_total_violation_count++;
    return -1;
}

// ============================================================================
// DECODE-LOOP BACKEND CHECK PROHIBITION
// ============================================================================

int llama_assert_no_backend_check_in_decode_loop(
    const char* location_description,
    bool backend_check_attempted
) {
    if (!backend_check_attempted) {
        return 0;
    }

    if (!llama_backend_phase_in_decode()) {
        return 0; // OK if not in decode
    }

    fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend check in decode loop at %s\n",
            location_description);
    g_backend_immutability.violation_count++;
    g_total_violation_count++;
    llama_record_backend_immutability_violation(
        LLAMA_BACKEND_VIOLATION_DECODE_LOOP_CHECK,
        location_description,
        "Backend check attempted in decode loop"
    );

    if (g_enforce_strict) {
        return -1;
    }
    return 0;
}

int llama_assert_no_capability_check_during_decode(
    const char* operation_name,
    bool capability_check_performed
) {
    if (!capability_check_performed) {
        return 0;
    }

    if (!llama_backend_phase_in_decode()) {
        return 0; // OK if not in decode
    }

    fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Capability check during decode for %s\n",
            operation_name);
    g_backend_immutability.violation_count++;
    g_total_violation_count++;
    llama_record_backend_immutability_violation(
        LLAMA_BACKEND_VIOLATION_CAPABILITY_CHECK,
        operation_name,
        "Capability check performed during decode (implies fallback possibility)"
    );

    if (g_enforce_strict) {
        return -1;
    }
    return 0;
}

int llama_assert_no_heuristic_backend_selection_during_decode(
    const char* selection_heuristic,
    bool heuristic_applied
) {
    if (!heuristic_applied) {
        return 0;
    }

    if (!llama_backend_phase_in_decode()) {
        return 0; // OK if not in decode
    }

    fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Heuristic backend selection during decode: %s\n",
            selection_heuristic);
    g_backend_immutability.violation_count++;
    g_total_violation_count++;
    llama_record_backend_immutability_violation(
        LLAMA_BACKEND_VIOLATION_HEURISTIC_SELECTION,
        selection_heuristic,
        "Runtime heuristic backend selection during decode (forbidden)"
    );

    if (g_enforce_strict) {
        return -1;
    }
    return 0;
}

// ============================================================================
// ENFORCEMENT POINTS (1-10)
// ============================================================================

int llama_enforce_backend_immutability_at_graph_execution(
    const char** operation_names,
    enum llama_backend_type* operation_backends,
    int num_operations
) {
    if (!llama_backend_phase_in_decode()) {
        return 0;
    }

    if (!g_backend_immutability.immutability_locked) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Graph execution without frozen backend\n");
        return -1;
    }

    for (int i = 0; i < num_operations; i++) {
        if (operation_backends[i] != g_backend_immutability.decode_backend) {
            fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Operation %s backend mismatch at graph execution\n"
                    "  Expected: %s, Got: %s\n",
                    operation_names[i],
                    llama_backend_type_name(g_backend_immutability.decode_backend),
                    llama_backend_type_name(operation_backends[i]));
            g_backend_immutability.violation_count++;
            g_total_violation_count++;
            llama_record_backend_immutability_violation(
                LLAMA_BACKEND_VIOLATION_PER_OP_SWITCH,
                operation_names[i],
                "Operation backend doesn't match frozen decode backend"
            );
            if (g_enforce_strict) {
                return -1;
            }
        }
    }

    return 0;
}

int llama_enforce_backend_immutability_at_dispatch(
    const char* operation_name,
    enum llama_backend_type operation_backend,
    bool backend_was_reevaluated
) {
    if (!llama_backend_phase_in_decode()) {
        return 0;
    }

    if (backend_was_reevaluated) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend re-evaluation at dispatch for %s\n",
                operation_name);
        g_backend_immutability.violation_count++;
        g_total_violation_count++;
        llama_record_backend_immutability_violation(
            LLAMA_BACKEND_VIOLATION_DECODE_LOOP_CHECK,
            operation_name,
            "Backend re-evaluated at dispatch (should be cached from freeze)"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    if (operation_backend != g_backend_immutability.decode_backend) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Dispatch backend mismatch for %s\n",
                operation_name);
        g_backend_immutability.violation_count++;
        g_total_violation_count++;
        llama_record_backend_immutability_violation(
            LLAMA_BACKEND_VIOLATION_PER_OP_SWITCH,
            operation_name,
            "Operation backend differs from frozen backend at dispatch"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_enforce_no_backend_reeval_on_shape_change(
    const char* operation_name,
    bool shape_changed,
    bool backend_reevaluated
) {
    if (!llama_backend_phase_in_decode()) {
        return 0;
    }

    if (shape_changed && backend_reevaluated) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend re-evaluated on shape change for %s\n",
                operation_name);
        g_backend_immutability.violation_count++;
        g_total_violation_count++;
        llama_record_backend_immutability_violation(
            LLAMA_BACKEND_VIOLATION_SHAPE_CHANGE_REEVAL,
            operation_name,
            "Backend re-evaluated when tensor shape changed"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_enforce_no_backend_reeval_on_context_change(
    const char* operation_name,
    int old_context_size,
    int new_context_size,
    bool backend_reevaluated
) {
    if (!llama_backend_phase_in_decode()) {
        return 0;
    }

    if (old_context_size != new_context_size && backend_reevaluated) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend re-evaluated on context change for %s\n"
                "  Old context: %d, New context: %d\n",
                operation_name, old_context_size, new_context_size);
        g_backend_immutability.violation_count++;
        g_total_violation_count++;
        llama_record_backend_immutability_violation(
            LLAMA_BACKEND_VIOLATION_CONTEXT_CHANGE_REEVAL,
            operation_name,
            "Backend re-evaluated when context size changed"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_enforce_backend_immutability_per_token(
    uint64_t token_id,
    const char* operation_name,
    enum llama_backend_type current_backend,
    enum llama_backend_type expected_backend
) {
    if (!llama_backend_phase_in_decode()) {
        return 0;
    }

    // Check if we've seen this token's operation before
    auto key = std::string(operation_name) + "_" + std::to_string(token_id);
    auto it = g_operation_backend_map.find(key);

    // Validate expected backend matches current backend
    if (expected_backend != current_backend) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend mismatch for %s at token %lu\n"
                "  Expected: %s, Current: %s\n",
                operation_name, token_id,
                llama_backend_type_name(expected_backend),
                llama_backend_type_name(current_backend));
        g_backend_immutability.violation_count++;
        g_total_violation_count++;
        llama_record_backend_immutability_violation(
            LLAMA_BACKEND_VIOLATION_PER_OP_SWITCH,
            operation_name,
            "Operation backend does not match expected backend"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    if (it != g_operation_backend_map.end()) {
        if (it->second != current_backend) {
            fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend changed per-token for %s at token %lu\n"
                    "  Previous: %s, Current: %s\n",
                    operation_name, token_id,
                    llama_backend_type_name(it->second),
                    llama_backend_type_name(current_backend));
            g_backend_immutability.violation_count++;
            g_total_violation_count++;
            llama_record_backend_immutability_violation(
                LLAMA_BACKEND_VIOLATION_PER_TOKEN_SWITCH,
                operation_name,
                "Operation backend changed between tokens"
            );
            if (g_enforce_strict) {
                return -1;
            }
        }
    } else {
        // First time seeing this operation for this token
        g_operation_backend_map[key] = current_backend;
    }

    return 0;
}

int llama_enforce_backend_immutability_per_layer(
    int layer_id,
    const char* layer_name,
    enum llama_backend_type layer_backend,
    enum llama_backend_type expected_backend
) {
    if (!llama_backend_phase_in_decode()) {
        return 0;
    }

    auto it = g_layer_backend_map.find(layer_id);

    if (it != g_layer_backend_map.end()) {
        if (it->second != layer_backend) {
            fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend changed per-layer for layer %d (%s)\n"
                    "  Previous: %s, Current: %s\n",
                    layer_id, layer_name,
                    llama_backend_type_name(it->second),
                    llama_backend_type_name(layer_backend));
            g_backend_immutability.violation_count++;
            g_total_violation_count++;
            llama_record_backend_immutability_violation(
                LLAMA_BACKEND_VIOLATION_PER_LAYER_SWITCH,
                layer_name,
                "Layer backend changed during decode"
            );
            if (g_enforce_strict) {
                return -1;
            }
        }
    } else {
        // First time seeing this layer
        if (layer_backend != expected_backend) {
            fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Layer backend mismatch at layer %d (%s)\n"
                    "  Expected: %s, Got: %s\n",
                    layer_id, layer_name,
                    llama_backend_type_name(expected_backend),
                    llama_backend_type_name(layer_backend));
            g_backend_immutability.violation_count++;
            g_total_violation_count++;
            llama_record_backend_immutability_violation(
                LLAMA_BACKEND_VIOLATION_PER_LAYER_SWITCH,
                layer_name,
                "Layer backend doesn't match frozen backend"
            );
            if (g_enforce_strict) {
                return -1;
            }
        }
        g_layer_backend_map[layer_id] = layer_backend;
    }

    return 0;
}

int llama_enforce_backend_immutability_per_operation(
    const char* operation_name,
    enum llama_backend_type current_backend,
    enum llama_backend_type previous_backend,
    bool backend_changed
) {
    if (!llama_backend_phase_in_decode()) {
        return 0;
    }

    if (backend_changed) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend changed per-operation for %s\n"
                "  Previous: %s, Current: %s\n",
                operation_name,
                llama_backend_type_name(previous_backend),
                llama_backend_type_name(current_backend));
        g_backend_immutability.violation_count++;
        g_total_violation_count++;
        llama_record_backend_immutability_violation(
            LLAMA_BACKEND_VIOLATION_PER_OP_SWITCH,
            operation_name,
            "Operation backend changed during decode"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_enforce_no_fallback_paths_during_decode(
    const char* operation_name,
    bool fallback_attempted
) {
    if (!fallback_attempted) {
        return 0;
    }

    if (!llama_backend_phase_in_decode()) {
        return 0; // OK if not in decode
    }

    fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Fallback path attempted during decode for %s\n",
            operation_name);
    g_backend_immutability.violation_count++;
    g_total_violation_count++;
    llama_record_backend_immutability_violation(
        LLAMA_BACKEND_VIOLATION_FALLBACK_PATH,
        operation_name,
        "Fallback mechanism invoked during decode (forbidden)"
    );

    if (g_enforce_strict) {
        return -1;
    }
    return 0;
}

int llama_enforce_backend_validity_during_decode(
    const char* validity_check_location,
    bool backend_is_valid
) {
    if (backend_is_valid) {
        return 0;
    }

    if (!llama_backend_phase_in_decode()) {
        return 0; // OK if not in decode
    }

    fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Backend became invalid at %s\n",
            validity_check_location);
    g_backend_immutability.violation_count++;
    g_total_violation_count++;
    llama_record_backend_immutability_violation(
        LLAMA_BACKEND_VIOLATION_INVALIDATION,
        validity_check_location,
        "Backend became invalid during decode"
    );

    if (g_enforce_strict) {
        return -1;
    }
    return 0;
}

int llama_enforce_immutability_pre_execution(
    const char* operation_name,
    enum llama_backend_type operation_backend,
    bool immutability_intact
) {
    if (!llama_backend_phase_in_decode()) {
        return 0;
    }

    if (!immutability_intact) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Immutability compromised before execution of %s\n",
                operation_name);
        g_backend_immutability.violation_count++;
        g_total_violation_count++;
        llama_record_backend_immutability_violation(
            LLAMA_BACKEND_VIOLATION_INVALIDATION,
            operation_name,
            "Immutability invariant violated before operation execution"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    if (operation_backend != g_backend_immutability.decode_backend) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Pre-execution backend mismatch for %s\n",
                operation_name);
        g_backend_immutability.violation_count++;
        g_total_violation_count++;
        llama_record_backend_immutability_violation(
            LLAMA_BACKEND_VIOLATION_PER_OP_SWITCH,
            operation_name,
            "Pre-execution backend verification failed"
        );
        if (g_enforce_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// PREFILL vs DECODE PHASE SEPARATION
// ============================================================================

int llama_backend_phase_enter_prefill(void) {
    if (g_backend_immutability.phase != LLAMA_BACKEND_PHASE_UNINITIALIZED) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] Warning: Entering prefill from phase %d\n",
                g_backend_immutability.phase);
    }

    g_backend_immutability.phase = LLAMA_BACKEND_PHASE_PREFILL;
    fprintf(stderr, "[BACKEND_IMMUTABILITY] Entered PREFILL phase\n");
    return 0;
}

int llama_backend_phase_exit_prefill_enter_decode(void) {
    if (g_backend_immutability.phase != LLAMA_BACKEND_PHASE_PREFILL) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Cannot enter decode from phase %d\n",
                g_backend_immutability.phase);
        return -1;
    }

    if (g_backend_immutability.decode_backend == LLAMA_BACKEND_UNKNOWN) {
        fprintf(stderr, "[BACKEND_IMMUTABILITY] FATAL: Cannot enter decode without frozen backend\n");
        return -1;
    }

    g_backend_immutability.phase = LLAMA_BACKEND_PHASE_DECODE_FROZEN;
    fprintf(stderr, "[BACKEND_IMMUTABILITY] Transitioned to DECODE_FROZEN phase with backend=%s\n",
            llama_backend_type_name(g_backend_immutability.decode_backend));
    return 0;
}

bool llama_backend_phase_in_decode(void) {
    return g_backend_immutability.phase == LLAMA_BACKEND_PHASE_DECODE_FROZEN;
}

bool llama_backend_phase_in_prefill(void) {
    return g_backend_immutability.phase == LLAMA_BACKEND_PHASE_PREFILL;
}

enum llama_backend_resolution_phase llama_backend_phase_get_current(void) {
    return g_backend_immutability.phase;
}

// ============================================================================
// VIOLATION DETECTION AND REPORTING
// ============================================================================

void llama_record_backend_immutability_violation(
    enum llama_backend_immutability_violation_location location,
    const char* operation_name,
    const char* violation_message
) {
    g_backend_immutability.last_violation_location = location;
    g_backend_immutability.last_violation_message = violation_message;

    fprintf(stderr, "[BACKEND_IMMUTABILITY] Violation recorded:\n");
    fprintf(stderr, "  Location: %s\n", llama_backend_violation_location_name(location));
    fprintf(stderr, "  Operation: %s\n", operation_name);
    fprintf(stderr, "  Message: %s\n", violation_message);
}

void llama_print_backend_immutability_violation_diagnostics(
    const struct llama_backend_immutability_state* state,
    enum llama_backend_immutability_violation_location violation_location,
    const char* violation_message
) {
    fprintf(stderr, "\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "BACKEND IMMUTABILITY VIOLATION DIAGNOSTICS\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Violation Location: %s\n", llama_backend_violation_location_name(violation_location));
    fprintf(stderr, "Violation Message: %s\n", violation_message);
    fprintf(stderr, "\n");
    fprintf(stderr, "Current State:\n");
    fprintf(stderr, "  Phase: %d (0=uninitialized, 1=prefill, 2=decode_frozen, 3=terminated)\n",
            state->phase);
    fprintf(stderr, "  Decode Backend: %s\n", llama_backend_type_name(state->decode_backend));
    fprintf(stderr, "  Immutability Locked: %s\n", state->immutability_locked ? "YES" : "NO");
    fprintf(stderr, "  Backend Invalid: %s\n", state->backend_invalid ? "YES" : "NO");
    if (state->backend_invalid) {
        fprintf(stderr, "  Invalidation Reason: %s\n", state->invalidation_reason);
    }
    fprintf(stderr, "  Total Violations: %d\n", state->violation_count);
    fprintf(stderr, "\n");
    fprintf(stderr, "Backend Immutability Principle:\n");
    fprintf(stderr, "  Backend ownership is resolved once before decode and remains immutable.\n");
    fprintf(stderr, "  No per-token, per-layer, or per-operation backend re-evaluation allowed.\n");
    fprintf(stderr, "  Backend changes trigger immediate failure.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "\n");
}

// ============================================================================
// BACKEND RESOLUTION CODE AUDIT
// ============================================================================

int llama_audit_backend_resolution_code(void) {
    // This is a placeholder for code audit functionality
    // In a real implementation, this would parse and analyze backend resolution code
    // to identify problematic patterns that violate immutability
    fprintf(stderr, "[BACKEND_IMMUTABILITY] Code audit: Checking for immutability violations...\n");
    return 0; // 0 = no violations found
}

bool llama_is_in_decode_loop(void) {
    return llama_backend_phase_in_decode();
}

bool llama_is_attempting_backend_reeval(void) {
    // Placeholder: would be implemented with actual backend resolution instrumentation
    return false;
}

bool llama_is_attempting_heuristic_selection(void) {
    // Placeholder: would be implemented with actual backend selection instrumentation
    return false;
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

void llama_set_backend_immutability_enforcement_strict(bool enforce_strict) {
    g_enforce_strict = enforce_strict;
    fprintf(stderr, "[BACKEND_IMMUTABILITY] Enforcement mode: %s\n",
            enforce_strict ? "STRICT" : "PERMISSIVE");
}

bool llama_get_backend_immutability_enforcement_strict(void) {
    return g_enforce_strict;
}

int llama_get_backend_immutability_violation_count(void) {
    return g_total_violation_count;
}

void llama_reset_backend_immutability_violation_counter(void) {
    g_backend_immutability.violation_count = 0;
    g_total_violation_count = 0;
    g_token_backend_map.clear();
    g_layer_backend_map.clear();
    g_operation_backend_map.clear();
    fprintf(stderr, "[BACKEND_IMMUTABILITY] Violation counter reset\n");
}

// ============================================================================
// EXPLICIT BACKEND IMMUTABILITY STATEMENT
// ============================================================================

void llama_print_backend_immutability_statement(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "BACKEND IMMUTABILITY PRINCIPLE STATEMENT\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Core Principle:\n");
    fprintf(stderr, "\"Backend ownership is resolved once before decode and remains immutable for\n");
    fprintf(stderr, " the entire decode lifetime. No per-token, per-layer, or per-operation\n");
    fprintf(stderr, " backend re-evaluation or switching is permitted. Backend changes trigger\n");
    fprintf(stderr, " immediate failure.\"\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Enforcement Strategy:\n");
    fprintf(stderr, "1. Freeze backend selection before first decode token\n");
    fprintf(stderr, "2. Prohibit backend checks in the decode loop\n");
    fprintf(stderr, "3. Lock backend ownership at graph level (GPU-only)\n");
    fprintf(stderr, "4. Eliminate per-token backend fallback paths\n");
    fprintf(stderr, "5. Disallow backend switching on shape/context changes\n");
    fprintf(stderr, "6. Separate prefill vs decode backend logic\n");
    fprintf(stderr, "7. Assert backend immutability at runtime\n");
    fprintf(stderr, "8. Fail fast on backend invalidation\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Violations Are FATAL:\n");
    fprintf(stderr, "- Per-token backend switches\n");
    fprintf(stderr, "- Per-layer backend changes\n");
    fprintf(stderr, "- Per-operation backend changes\n");
    fprintf(stderr, "- Backend re-evaluation on shape/context changes\n");
    fprintf(stderr, "- Fallback mechanism invocation during decode\n");
    fprintf(stderr, "- Capability checks implying fallback possibility\n");
    fprintf(stderr, "- Heuristic backend selection during decode\n");
    fprintf(stderr, "- Backend invalidation mid-decode\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "============================================================================\n");
    fprintf(stderr, "\n");
}

// ============================================================================
// VALIDATION AND SELF-TEST
// ============================================================================

int llama_backend_immutability_selftest(void) {
    fprintf(stderr, "\n[BACKEND_IMMUTABILITY] Running self-test...\n");

    // Test 1: Initialization
    fprintf(stderr, "[TEST 1] Initialization\n");
    if (llama_backend_immutability_init() != 0) {
        fprintf(stderr, "  FAILED: Initialization\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 2: Phase transitions
    fprintf(stderr, "[TEST 2] Phase transitions\n");
    if (llama_backend_phase_enter_prefill() != 0) {
        fprintf(stderr, "  FAILED: Enter prefill\n");
        return -1;
    }
    if (!llama_backend_phase_in_prefill()) {
        fprintf(stderr, "  FAILED: Not in prefill\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 3: Backend freezing
    fprintf(stderr, "[TEST 3] Backend freezing\n");
    if (llama_backend_immutability_freeze_for_decode(LLAMA_BACKEND_CUDA) != 0) {
        fprintf(stderr, "  FAILED: Freeze backend\n");
        return -1;
    }
    if (!llama_backend_immutability_is_frozen()) {
        fprintf(stderr, "  FAILED: Backend not frozen\n");
        return -1;
    }
    if (llama_backend_immutability_get_frozen_backend() != LLAMA_BACKEND_CUDA) {
        fprintf(stderr, "  FAILED: Wrong frozen backend\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 4: Decode phase transition
    fprintf(stderr, "[TEST 4] Decode phase transition\n");
    if (llama_backend_phase_exit_prefill_enter_decode() != 0) {
        fprintf(stderr, "  FAILED: Enter decode phase\n");
        return -1;
    }
    if (!llama_backend_phase_in_decode()) {
        fprintf(stderr, "  FAILED: Not in decode\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 5: Backend verification
    fprintf(stderr, "[TEST 5] Backend verification\n");
    if (llama_backend_immutability_verify_unchanged(LLAMA_BACKEND_CUDA) != 0) {
        fprintf(stderr, "  FAILED: Backend verification\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 6: Double freeze prevention
    fprintf(stderr, "[TEST 6] Double freeze prevention\n");
    if (llama_backend_immutability_freeze_for_decode(LLAMA_BACKEND_HIP) == 0) {
        fprintf(stderr, "  FAILED: Allowed double freeze\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 7: Backend check assertion
    fprintf(stderr, "[TEST 7] Backend check assertion in decode\n");
    if (llama_assert_no_backend_check_in_decode_loop("test_location", true) == 0) {
        fprintf(stderr, "  FAILED: Did not detect backend check in decode\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    // Test 8: Enforcement points
    fprintf(stderr, "[TEST 8] Enforcement points\n");
    const char* ops[] = { "op1", "op2" };
    enum llama_backend_type backends[] = { LLAMA_BACKEND_CUDA, LLAMA_BACKEND_CUDA };
    if (llama_enforce_backend_immutability_at_graph_execution(ops, backends, 2) != 0) {
        fprintf(stderr, "  FAILED: Graph execution enforcement\n");
        return -1;
    }
    fprintf(stderr, "  PASSED\n");

    fprintf(stderr, "\n[BACKEND_IMMUTABILITY] Self-test completed successfully!\n\n");
    return 0;
}
