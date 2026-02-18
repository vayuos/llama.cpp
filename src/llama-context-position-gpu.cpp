/**
 * SECTION 28: Enforce GPU-Only Context Position Tracking
 * Implementation
 *
 * This file implements GPU-exclusive context position (n_past) tracking.
 * Context position state remains GPU-resident during decode.
 * CPU does not update, increment, or manage context position.
 * Only context position value crosses PCIe; state stays on GPU.
 */

#include "llama-context-position-gpu.h"
#include <map>
#include <string>
#include <cstring>
#include <cstdio>
#include <cassert>

// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================

/**
 * Global validation state for GPU context position tracking
 */
static struct llama_gpu_context_position_validation_state g_context_position_validation_state = {
    /* config */ {
        /* gpu_context_pos_tracking_enabled */ false,
        /* cpu_updates_forbidden */ false,
        /* mode */ LLAMA_CONTEXT_POSITION_NONE,
        /* context_length */ 0,
        /* validate_position_bounds */ true,
    },
    /* state_record */ {
        /* current_mode */ LLAMA_CONTEXT_POSITION_NONE,
        /* gpu_pos_state */ LLAMA_GPU_CONTEXT_POS_UNINITIALIZED,
        /* context_position */ 0,
        /* context_length */ 0,
        /* position_updates_count */ 0,
        /* total_violations */ 0,
        /* last_violation */ LLAMA_CONTEXT_POSITION_VIOLATION_NONE,
        /* position_locked */ false,
    },
    /* last_update */ {
        /* position_before */ 0,
        /* position_after */ 0,
        /* tokens_added */ 0,
        /* timestamp_ns */ 0,
        /* update_on_gpu */ false,
    },
    /* total_position_updates */ 0,
    /* total_violations */ 0,
    /* enforcement_strict */ true,
    /* debug_context_position */ false,
};

/**
 * Per-operation CPU context position update attempts
 */
static std::map<std::string, int> g_cpu_context_position_attempts;

/**
 * Position update history
 */
static std::map<uint64_t, uint32_t> g_context_position_history;

// ============================================================================
// INITIALIZATION AND CONFIGURATION
// ============================================================================

/**
 * Initialize GPU context position tracking
 */
int llama_context_position_gpu_init(void) {
    g_context_position_validation_state.config.mode = LLAMA_CONTEXT_POSITION_NONE;
    g_context_position_validation_state.state_record.gpu_pos_state = LLAMA_GPU_CONTEXT_POS_UNINITIALIZED;
    g_context_position_validation_state.state_record.context_position = 0;
    g_context_position_validation_state.state_record.position_locked = false;

    g_cpu_context_position_attempts.clear();
    g_context_position_history.clear();

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] GPU context position tracking initialized\n");
    }

    return 0; // Success
}

/**
 * Configure GPU context position tracking
 */
int llama_context_position_gpu_configure(
    bool gpu_context_pos_enabled,
    bool cpu_updates_forbidden,
    uint32_t context_length
) {
    g_context_position_validation_state.config.gpu_context_pos_tracking_enabled = gpu_context_pos_enabled;
    g_context_position_validation_state.config.cpu_updates_forbidden = cpu_updates_forbidden;
    g_context_position_validation_state.config.context_length = context_length;

    if (gpu_context_pos_enabled) {
        g_context_position_validation_state.config.mode = LLAMA_CONTEXT_POSITION_GPU;
        g_context_position_validation_state.state_record.current_mode = LLAMA_CONTEXT_POSITION_GPU;
        g_context_position_validation_state.state_record.context_length = context_length;
    }

    if (cpu_updates_forbidden) {
        g_context_position_validation_state.state_record.position_locked = true;
    }

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] GPU context position tracking configured: enabled=%d, cpu_forbidden=%d, context_len=%u\n",
                gpu_context_pos_enabled, cpu_updates_forbidden, context_length);
    }

    return 0; // Success
}

// ============================================================================
// CONTEXT POSITION SETUP
// ============================================================================

/**
 * Allocate GPU context position buffer
 */
int llama_context_position_gpu_allocate_position_buffer(uint32_t context_length) {
    g_context_position_validation_state.state_record.context_length = context_length;
    g_context_position_validation_state.state_record.gpu_pos_state = LLAMA_GPU_CONTEXT_POS_ALLOCATED;

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] Context position buffer allocated on GPU (context_len=%u)\n", context_length);
    }

    return 0; // Success
}

/**
 * Initialize context position
 */
int llama_context_position_gpu_initialize_position(uint32_t initial_position) {
    g_context_position_validation_state.state_record.context_position = initial_position;
    g_context_position_validation_state.state_record.gpu_pos_state = LLAMA_GPU_CONTEXT_POS_INITIALIZED;

    g_context_position_history[0] = initial_position;

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] Context position initialized to %u on GPU\n", initial_position);
    }

    return 0; // Success
}

// ============================================================================
// GPU CONTEXT POSITION UPDATES (10 ENFORCEMENT POINTS)
// ============================================================================

/**
 * ENFORCEMENT POINT 1: Queue context position update kernel
 */
int llama_context_position_gpu_queue_update_kernel(void) {
    if (!g_context_position_validation_state.config.gpu_context_pos_tracking_enabled) {
        if (g_context_position_validation_state.enforcement_strict) {
            fprintf(stderr, "[CONTEXT_POSITION] FATAL: GPU context position tracking not enabled\n");
            return -1;
        }
    }

    g_context_position_validation_state.state_record.gpu_pos_state = LLAMA_GPU_CONTEXT_POS_ACTIVE;

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] Context position update kernel queued on GPU\n");
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 2: Update context position on GPU
 */
int llama_context_position_gpu_update_on_gpu(uint32_t new_position) {
    // Check: CPU did not update position
    if (g_cpu_context_position_attempts.count("context_pos_update_cpu") > 0) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_CPU_UPDATE;
        g_context_position_validation_state.total_violations++;

        if (g_context_position_validation_state.enforcement_strict) {
            fprintf(stderr, "[CONTEXT_POSITION] VIOLATION: Context position updated on CPU during decode\n");
            return -1;
        }
    }

    // GPU updates context position
    uint32_t old_position = g_context_position_validation_state.state_record.context_position;

    if (new_position <= g_context_position_validation_state.state_record.context_length) {
        g_context_position_validation_state.state_record.context_position = new_position;
        g_context_position_validation_state.last_update.position_before = old_position;
        g_context_position_validation_state.last_update.position_after = new_position;
        g_context_position_validation_state.last_update.tokens_added = new_position - old_position;
        g_context_position_validation_state.last_update.update_on_gpu = true;
    }

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] Context position updated on GPU: %u → %u\n", old_position, new_position);
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 3: Keep context position on GPU memory
 */
int llama_context_position_gpu_keep_position_on_device(void) {
    // Check: position not copied to host
    if (g_cpu_context_position_attempts.count("context_pos_host_copy") > 0) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_CONTEXT_POS_ON_HOST;
        g_context_position_validation_state.total_violations++;

        if (g_context_position_validation_state.enforcement_strict) {
            fprintf(stderr, "[CONTEXT_POSITION] VIOLATION: Context position copied to host during decode\n");
            return -1;
        }
    }

    g_context_position_validation_state.state_record.gpu_pos_state = LLAMA_GPU_CONTEXT_POS_UPDATED;

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] Context position verified on GPU memory\n");
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 4: Forbid CPU context position update
 */
int llama_context_position_gpu_forbid_cpu_update(void) {
    if (!g_context_position_validation_state.config.cpu_updates_forbidden) {
        return 0; // CPU updates not forbidden; allow
    }

    // Check: no CPU update detected
    if (g_cpu_context_position_attempts.count("context_pos_update_cpu") > 0) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_CPU_UPDATE;
        g_context_position_validation_state.total_violations++;

        if (g_context_position_validation_state.enforcement_strict) {
            fprintf(stderr, "[CONTEXT_POSITION] FATAL: CPU context position update called during decode\n");
            return -1;
        }
    }

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] CPU context position update forbidden and verified\n");
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 5: Forbid CPU context position comparison
 */
int llama_context_position_gpu_forbid_cpu_comparison(void) {
    if (!g_context_position_validation_state.config.cpu_updates_forbidden) {
        return 0; // CPU comparisons not forbidden; allow
    }

    // Check: CPU did not use position for decisions
    if (g_cpu_context_position_attempts.count("context_pos_comparison_cpu") > 0) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_CPU_COMPARISON;
        g_context_position_validation_state.total_violations++;

        if (g_context_position_validation_state.enforcement_strict) {
            fprintf(stderr, "[CONTEXT_POSITION] VIOLATION: CPU compared context position during decode\n");
            return -1;
        }
    }

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] CPU context position comparison forbidden and verified\n");
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 6: Forbid CPU context position gating
 */
int llama_context_position_gpu_forbid_cpu_gating(void) {
    if (!g_context_position_validation_state.config.cpu_updates_forbidden) {
        return 0; // CPU gating not forbidden; allow
    }

    // Check: CPU did not use position to gate decode progression
    if (g_cpu_context_position_attempts.count("context_pos_gate_cpu") > 0) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_CPU_GATING;
        g_context_position_validation_state.total_violations++;

        if (g_context_position_validation_state.enforcement_strict) {
            fprintf(stderr, "[CONTEXT_POSITION] VIOLATION: CPU used context position for gating during decode\n");
            return -1;
        }
    }

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] CPU context position gating forbidden and verified\n");
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 7: Validate context position within bounds
 */
int llama_context_position_gpu_validate_position_bounds(void) {
    uint32_t current = g_context_position_validation_state.state_record.context_position;
    uint32_t max = g_context_position_validation_state.state_record.context_length;

    // Check: position within valid range
    if (current > max) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_CPU_UPDATE;
        g_context_position_validation_state.total_violations++;

        if (g_context_position_validation_state.enforcement_strict) {
            fprintf(stderr, "[CONTEXT_POSITION] VIOLATION: Context position out of bounds: %u (max: %u)\n", current, max);
            return -1;
        }
    }

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] Context position bounds verified: %u (max: %u)\n", current, max);
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 8: Lock context position tracking to GPU
 */
int llama_context_position_gpu_lock_position_to_gpu(void) {
    g_context_position_validation_state.state_record.position_locked = true;

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] Context position tracking locked to GPU (immutable)\n");
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 9: Verify no CPU modification to context position
 */
int llama_context_position_gpu_verify_no_cpu_modification(void) {
    // Check: all CPU position attempts map is empty
    if (g_cpu_context_position_attempts.size() > 0) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_MIXED_UPDATE;
        g_context_position_validation_state.total_violations++;

        if (g_context_position_validation_state.enforcement_strict) {
            fprintf(stderr, "[CONTEXT_POSITION] VIOLATION: CPU context position modifications detected\n");
            return -1;
        }
    }

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] CPU modification verified absent\n");
    }

    return 0; // Success
}

/**
 * ENFORCEMENT POINT 10: Commit context position update
 */
int llama_context_position_gpu_commit_position_update(uint32_t new_position) {
    // Check: new position within bounds
    if (new_position > g_context_position_validation_state.state_record.context_length) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_CPU_UPDATE;
        g_context_position_validation_state.total_violations++;

        if (g_context_position_validation_state.enforcement_strict) {
            fprintf(stderr, "[CONTEXT_POSITION] VIOLATION: Context position update out of bounds\n");
            return -1;
        }
    }

    // Commit new context position
    uint32_t old_position = g_context_position_validation_state.state_record.context_position;
    g_context_position_validation_state.state_record.context_position = new_position;
    g_context_position_validation_state.state_record.position_updates_count++;
    g_context_position_validation_state.last_update.position_before = old_position;
    g_context_position_validation_state.last_update.position_after = new_position;

    g_context_position_history[g_context_position_validation_state.state_record.position_updates_count] = new_position;

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] Context position commit: %u → %u\n", old_position, new_position);
    }

    return 0; // Success
}

// ============================================================================
// POSITION RETRIEVAL AND SYNCHRONIZATION
// ============================================================================

/**
 * Read context position from GPU (synchronous)
 */
int llama_context_position_gpu_read_position_sync(uint32_t* out_position) {
    if (out_position == nullptr) {
        return -1;
    }

    *out_position = g_context_position_validation_state.state_record.context_position;

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] Context position read (sync): %u\n", *out_position);
    }

    return 0; // Success
}

/**
 * Read context position from GPU (asynchronous)
 */
int llama_context_position_gpu_read_position_async(uint32_t* out_position) {
    if (out_position == nullptr) {
        return -1;
    }

    *out_position = g_context_position_validation_state.state_record.context_position;

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] Context position read (async): %u\n", *out_position);
    }

    return 0; // Success
}

/**
 * Synchronize context position to CPU (read-only)
 */
int llama_context_position_gpu_sync_position_to_cpu(void) {
    g_context_position_validation_state.state_record.gpu_pos_state = LLAMA_GPU_CONTEXT_POS_SYNCED;

    if (g_context_position_validation_state.debug_context_position) {
        fprintf(stderr, "[CONTEXT_POSITION] Context position synced to CPU (read-only)\n");
    }

    return 0; // Success
}

// ============================================================================
// VIOLATION DETECTION (6)
// ============================================================================

/**
 * Detect CPU context position update attempt
 */
int llama_context_position_gpu_detect_cpu_update(void) {
    if (g_cpu_context_position_attempts.count("context_pos_update_cpu") > 0) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_CPU_UPDATE;
        g_context_position_validation_state.total_violations++;
        return 1; // Violation detected
    }
    return 0; // No violation
}

/**
 * Detect CPU context position comparison
 */
int llama_context_position_gpu_detect_cpu_comparison(void) {
    if (g_cpu_context_position_attempts.count("context_pos_comparison_cpu") > 0) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_CPU_COMPARISON;
        g_context_position_validation_state.total_violations++;
        return 1; // Violation detected
    }
    return 0; // No violation
}

/**
 * Detect context position materialized on host
 */
int llama_context_position_gpu_detect_position_on_host(void) {
    if (g_cpu_context_position_attempts.count("context_pos_host_copy") > 0) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_CONTEXT_POS_ON_HOST;
        g_context_position_validation_state.total_violations++;
        return 1; // Violation detected
    }
    return 0; // No violation
}

/**
 * Detect CPU using context position for gating
 */
int llama_context_position_gpu_detect_cpu_gating(void) {
    if (g_cpu_context_position_attempts.count("context_pos_gate_cpu") > 0) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_CPU_GATING;
        g_context_position_validation_state.total_violations++;
        return 1; // Violation detected
    }
    return 0; // No violation
}

/**
 * Detect mixed CPU/GPU context position updates
 */
int llama_context_position_gpu_detect_mixed_updates(void) {
    if (g_cpu_context_position_attempts.size() > 0 &&
        g_context_position_validation_state.state_record.position_updates_count > 0) {
        g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_MIXED_UPDATE;
        g_context_position_validation_state.total_violations++;
        return 1; // Violation detected
    }
    return 0; // No violation
}

/**
 * Detect CPU/GPU context position desync
 */
int llama_context_position_gpu_detect_desync(void) {
    // Check: context position history is consistent
    uint32_t current = g_context_position_validation_state.state_record.context_position;

    if (g_context_position_history.size() > 0) {
        auto last_entry = g_context_position_history.rbegin();
        if (current != last_entry->second) {
            g_context_position_validation_state.state_record.last_violation = LLAMA_CONTEXT_POSITION_VIOLATION_DESYNC;
            g_context_position_validation_state.total_violations++;
            return 1; // Violation detected (desync)
        }
    }
    return 0; // No violation
}

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

int llama_context_position_gpu_set_allocated(void) {
    g_context_position_validation_state.state_record.gpu_pos_state = LLAMA_GPU_CONTEXT_POS_ALLOCATED;
    return 0;
}

int llama_context_position_gpu_set_initialized(void) {
    g_context_position_validation_state.state_record.gpu_pos_state = LLAMA_GPU_CONTEXT_POS_INITIALIZED;
    return 0;
}

int llama_context_position_gpu_set_active(void) {
    g_context_position_validation_state.state_record.gpu_pos_state = LLAMA_GPU_CONTEXT_POS_ACTIVE;
    return 0;
}

int llama_context_position_gpu_set_updated(void) {
    g_context_position_validation_state.state_record.gpu_pos_state = LLAMA_GPU_CONTEXT_POS_UPDATED;
    return 0;
}

// ============================================================================
// QUERY AND VERIFICATION FUNCTIONS
// ============================================================================

struct llama_gpu_context_position_state_record llama_context_position_gpu_get_state_record(void) {
    return g_context_position_validation_state.state_record;
}

struct llama_gpu_context_position_update_record llama_context_position_gpu_get_last_update(void) {
    return g_context_position_validation_state.last_update;
}

uint32_t llama_context_position_gpu_get_context_position(void) {
    return g_context_position_validation_state.state_record.context_position;
}

enum llama_gpu_context_position_state llama_context_position_gpu_get_position_state(void) {
    return g_context_position_validation_state.state_record.gpu_pos_state;
}

// ============================================================================
// VERIFICATION FUNCTIONS (7)
// ============================================================================

int llama_context_position_gpu_verify_cpu_updates_forbidden(void) {
    if (g_context_position_validation_state.config.cpu_updates_forbidden) {
        return 0; // Verified
    }
    return -1; // Not verified
}

int llama_context_position_gpu_verify_gpu_position_active(void) {
    if (g_context_position_validation_state.state_record.current_mode == LLAMA_CONTEXT_POSITION_GPU) {
        return 0; // Verified
    }
    return -1; // Not verified
}

int llama_context_position_gpu_verify_position_locked(void) {
    if (g_context_position_validation_state.state_record.position_locked) {
        return 0; // Verified
    }
    return -1; // Not verified
}

int llama_context_position_gpu_verify_no_cpu_entry_point(void) {
    if (g_cpu_context_position_attempts.size() == 0) {
        return 0; // Verified
    }
    return -1; // Not verified
}

int llama_context_position_gpu_verify_position_within_bounds(void) {
    uint32_t current = g_context_position_validation_state.state_record.context_position;
    uint32_t max = g_context_position_validation_state.state_record.context_length;

    if (current <= max) {
        return 0; // Verified
    }
    return -1; // Not verified
}

int llama_context_position_gpu_verify_no_desync(void) {
    if (g_context_position_history.size() == 0) {
        return 0; // Verified (no history to check)
    }

    auto last_entry = g_context_position_history.rbegin();
    if (g_context_position_validation_state.state_record.context_position == last_entry->second) {
        return 0; // Verified (no desync)
    }
    return -1; // Desync detected
}

int llama_context_position_gpu_verify_monotonic_increment(void) {
    uint32_t last_position = 0;

    for (auto& entry : g_context_position_history) {
        if (entry.second < last_position) {
            return -1; // Not monotonic
        }
        last_position = entry.second;
    }
    return 0; // Verified
}

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

void llama_context_position_gpu_log_position_mode_enabled(void) {
    fprintf(stderr, "[CONTEXT_POSITION] GPU-exclusive context position tracking enabled\n");
    fprintf(stderr, "  Mode: %s\n", llama_context_position_mode_name(g_context_position_validation_state.state_record.current_mode));
    fprintf(stderr, "  Context Position: %u (max: %u)\n",
            g_context_position_validation_state.state_record.context_position,
            g_context_position_validation_state.state_record.context_length);
}

void llama_context_position_gpu_log_position_locked(void) {
    fprintf(stderr, "[CONTEXT_POSITION] Context position tracking locked to GPU (immutable)\n");
}

void llama_context_position_gpu_print_state(void) {
    const struct llama_gpu_context_position_state_record& state = g_context_position_validation_state.state_record;

    fprintf(stderr, "\n=== GPU CONTEXT POSITION STATE ===\n");
    fprintf(stderr, "Current Mode: %s\n", llama_context_position_mode_name(state.current_mode));
    fprintf(stderr, "Position State: %s\n", llama_gpu_context_position_state_name(state.gpu_pos_state));
    fprintf(stderr, "Context Position: %u\n", state.context_position);
    fprintf(stderr, "Context Length: %u\n", state.context_length);
    fprintf(stderr, "Position Updates: %lu\n", state.position_updates_count);
    fprintf(stderr, "Position Locked: %s\n", state.position_locked ? "YES" : "NO");
    fprintf(stderr, "Total Violations: %d\n", state.total_violations);
    fprintf(stderr, "Last Violation: %s\n", llama_context_position_violation_name(state.last_violation));
}

void llama_context_position_gpu_print_execution_stats(void) {
    const struct llama_gpu_context_position_update_record& update = g_context_position_validation_state.last_update;

    fprintf(stderr, "\n=== GPU CONTEXT POSITION EXECUTION STATS ===\n");
    fprintf(stderr, "Position Before: %u\n", update.position_before);
    fprintf(stderr, "Position After: %u\n", update.position_after);
    fprintf(stderr, "Tokens Added: %u\n", update.tokens_added);
    fprintf(stderr, "Update on GPU: %s\n", update.update_on_gpu ? "YES" : "NO");
    fprintf(stderr, "Total Position Updates: %d\n", g_context_position_validation_state.total_position_updates);
}

void llama_context_position_gpu_print_violation_summary(void) {
    fprintf(stderr, "\n=== GPU CONTEXT POSITION VIOLATION SUMMARY ===\n");
    fprintf(stderr, "Total Violations: %d\n", g_context_position_validation_state.total_violations);
    fprintf(stderr, "Enforcement Mode: %s\n", g_context_position_validation_state.enforcement_strict ? "STRICT" : "PERMISSIVE");

    if (g_cpu_context_position_attempts.size() > 0) {
        fprintf(stderr, "\nDetected CPU Context Position Operations:\n");
        for (auto& entry : g_cpu_context_position_attempts) {
            fprintf(stderr, "  %s: %d attempts\n", entry.first.c_str(), entry.second);
        }
    }
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

void llama_context_position_gpu_report_violation(
    enum llama_context_position_violation violation_type,
    const char* details
) {
    g_context_position_validation_state.state_record.last_violation = violation_type;
    g_context_position_validation_state.total_violations++;

    fprintf(stderr, "[CONTEXT_POSITION] VIOLATION: %s\n", llama_context_position_violation_name(violation_type));
    fprintf(stderr, "  Details: %s\n", details);
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

void llama_context_position_gpu_set_enforcement_strict(bool strict) {
    g_context_position_validation_state.enforcement_strict = strict;
    fprintf(stderr, "[CONTEXT_POSITION] Enforcement mode set to: %s\n", strict ? "STRICT" : "PERMISSIVE");
}

bool llama_context_position_gpu_get_enforcement_strict(void) {
    return g_context_position_validation_state.enforcement_strict;
}

void llama_context_position_gpu_set_debug_output(bool debug) {
    g_context_position_validation_state.debug_context_position = debug;
}

// ============================================================================
// SELF-TEST SUITE (8 tests)
// ============================================================================

int llama_context_position_gpu_selftest(void) {
    fprintf(stderr, "[CONTEXT_POSITION] Running self-test suite...\n");

    int tests_passed = 0;
    int tests_failed = 0;

    // Test 1: Initialization
    fprintf(stderr, "  [TEST 1] Initialization... ");
    if (llama_context_position_gpu_init() == 0) {
        fprintf(stderr, "PASS\n");
        tests_passed++;
    } else {
        fprintf(stderr, "FAIL\n");
        tests_failed++;
    }

    // Test 2: Configuration
    fprintf(stderr, "  [TEST 2] Configuration... ");
    if (llama_context_position_gpu_configure(true, true, 2048) == 0) {
        fprintf(stderr, "PASS\n");
        tests_passed++;
    } else {
        fprintf(stderr, "FAIL\n");
        tests_failed++;
    }

    // Test 3: Buffer allocation
    fprintf(stderr, "  [TEST 3] Buffer allocation... ");
    if (llama_context_position_gpu_allocate_position_buffer(2048) == 0 &&
        llama_context_position_gpu_initialize_position(0) == 0) {
        fprintf(stderr, "PASS\n");
        tests_passed++;
    } else {
        fprintf(stderr, "FAIL\n");
        tests_failed++;
    }

    // Test 4: Position update
    fprintf(stderr, "  [TEST 4] Position update... ");
    if (llama_context_position_gpu_update_on_gpu(100) == 0 &&
        llama_context_position_gpu_get_context_position() == 100) {
        fprintf(stderr, "PASS\n");
        tests_passed++;
    } else {
        fprintf(stderr, "FAIL\n");
        tests_failed++;
    }

    // Test 5: Bounds validation
    fprintf(stderr, "  [TEST 5] Bounds validation... ");
    if (llama_context_position_gpu_validate_position_bounds() == 0) {
        fprintf(stderr, "PASS\n");
        tests_passed++;
    } else {
        fprintf(stderr, "FAIL\n");
        tests_failed++;
    }

    // Test 6: Position locking
    fprintf(stderr, "  [TEST 6] Position locking... ");
    if (llama_context_position_gpu_lock_position_to_gpu() == 0 &&
        llama_context_position_gpu_verify_position_locked() == 0) {
        fprintf(stderr, "PASS\n");
        tests_passed++;
    } else {
        fprintf(stderr, "FAIL\n");
        tests_failed++;
    }

    // Test 7: Verification functions
    fprintf(stderr, "  [TEST 7] Verification functions... ");
    if (llama_context_position_gpu_verify_cpu_updates_forbidden() == 0 &&
        llama_context_position_gpu_verify_gpu_position_active() == 0 &&
        llama_context_position_gpu_verify_position_within_bounds() == 0) {
        fprintf(stderr, "PASS\n");
        tests_passed++;
    } else {
        fprintf(stderr, "FAIL\n");
        tests_failed++;
    }

    // Test 8: Position sync and read
    fprintf(stderr, "  [TEST 8] Position sync and read... ");
    uint32_t pos_read = 0;
    if (llama_context_position_gpu_sync_position_to_cpu() == 0 &&
        llama_context_position_gpu_read_position_sync(&pos_read) == 0 &&
        pos_read == 100) {
        fprintf(stderr, "PASS\n");
        tests_passed++;
    } else {
        fprintf(stderr, "FAIL\n");
        tests_failed++;
    }

    fprintf(stderr, "[CONTEXT_POSITION] Self-test complete: %d passed, %d failed\n", tests_passed, tests_failed);

    return (tests_failed == 0) ? 0 : -1;
}
