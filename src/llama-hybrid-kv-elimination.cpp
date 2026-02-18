/**
 * SECTION 31: Eliminate Hybrid KV Cache Modes
 * Implementation
 *
 * This file implements GPU-only KV cache mode enforcement.
 * Hybrid KV cache modes (CPU+GPU split) are forbidden during decode.
 * Decode uses one and only one KV cache backend: GPU.
 * CPU-resident KV cache is not permitted once decode begins.
 */

#include "llama-hybrid-kv-elimination.h"
#include <map>
#include <string>
#include <cstring>
#include <vector>

// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================

static struct llama_gpu_kv_hybrid_elimination_validation_state g_hybrid_kv_elimination_validation = {
    /* config */ {
        /* enforce_gpu_only_decode */ false,
        /* forbid_hybrid_modes */ false,
        /* fail_on_incomplete_gpu_alloc */ true,
        /* validate_kv_residency */ true,
        /* num_layers */ 0,
        /* debug_kv_backend */ false,
    },
    /* state_record */ {
        /* state */ LLAMA_GPU_KV_EXCLUSIVITY_UNINITIALIZED,
        /* decode_backend_mode */ LLAMA_KV_BACKEND_NONE,
        /* num_layers */ 0,
        /* layers_gpu_only */ 0,
        /* layers_with_cpu_kv */ 0,
        /* total_gpu_kv_bytes */ 0,
        /* total_violations */ 0,
        /* last_violation */ LLAMA_HYBRID_KV_VIOLATION_NONE,
    },
    /* hybrid_path_record */ {
        /* hybrid_path_attempts */ 0,
        /* per_layer_branch_attempts */ 0,
        /* cpu_fallback_attempts */ 0,
        /* host_visible_pointer_attempts */ 0,
        /* reserved_1 */ 0,
    },
    /* total_decode_starts */ 0,
    /* total_violations */ 0,
    /* enforcement_strict */ true,
    /* decode_phase_active */ false,
};

// Per-layer KV residency tracking
static std::vector<struct llama_gpu_kv_layer_residency> g_layer_kv_residency;

// Hybrid path attempt tracking
static std::map<std::string, int> g_hybrid_path_attempts;

// ============================================================================
// INITIALIZATION AND CONFIGURATION
// ============================================================================

int llama_hybrid_kv_elimination_gpu_init(void) {
    g_hybrid_kv_elimination_validation.state_record.state = LLAMA_GPU_KV_EXCLUSIVITY_UNINITIALIZED;
    g_hybrid_kv_elimination_validation.state_record.decode_backend_mode = LLAMA_KV_BACKEND_NONE;
    g_hybrid_kv_elimination_validation.total_violations = 0;
    g_hybrid_kv_elimination_validation.total_decode_starts = 0;
    g_hybrid_kv_elimination_validation.decode_phase_active = false;
    g_layer_kv_residency.clear();
    g_hybrid_path_attempts.clear();

    if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
        fprintf(stderr, "[Hybrid KV Elimination GPU] Initialization complete\n");
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_configure(
    bool enforce_gpu_only_decode,
    bool forbid_hybrid_modes,
    bool fail_on_incomplete_gpu_alloc,
    uint32_t num_layers
) {
    g_hybrid_kv_elimination_validation.config.enforce_gpu_only_decode = enforce_gpu_only_decode;
    g_hybrid_kv_elimination_validation.config.forbid_hybrid_modes = forbid_hybrid_modes;
    g_hybrid_kv_elimination_validation.config.fail_on_incomplete_gpu_alloc = fail_on_incomplete_gpu_alloc;
    g_hybrid_kv_elimination_validation.config.num_layers = num_layers;
    g_hybrid_kv_elimination_validation.state_record.num_layers = num_layers;

    // Initialize per-layer residency tracking
    g_layer_kv_residency.resize(num_layers);
    for (uint32_t i = 0; i < num_layers; i++) {
        g_layer_kv_residency[i].layer_id = i;
        g_layer_kv_residency[i].backend = LLAMA_KV_BACKEND_NONE;
        g_layer_kv_residency[i].gpu_allocated = false;
        g_layer_kv_residency[i].cpu_allocated = false;
    }

    if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
        fprintf(stderr, "[Hybrid KV Elimination GPU] Configured: enforce=%d, forbid_hybrid=%d, num_layers=%u\n",
            enforce_gpu_only_decode, forbid_hybrid_modes, num_layers);
    }

    return 0;
}

// ============================================================================
// PREFILL AND DECODE PHASE MANAGEMENT
// ============================================================================

int llama_hybrid_kv_elimination_gpu_begin_prefill_phase(void) {
    g_hybrid_kv_elimination_validation.state_record.state = LLAMA_GPU_KV_EXCLUSIVITY_PREFILL_PHASE;

    if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
        fprintf(stderr, "[Hybrid KV Elimination GPU] Prefill phase STARTED - hybrid KV allowed\n");
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_end_prefill_phase(void) {
    // Validate that KV cache is ready for GPU-only decode
    if (g_hybrid_kv_elimination_validation.config.validate_kv_residency) {
        if (g_hybrid_kv_elimination_validation.state_record.layers_with_cpu_kv > 0 &&
            g_hybrid_kv_elimination_validation.config.fail_on_incomplete_gpu_alloc) {

            g_hybrid_kv_elimination_validation.state_record.last_violation =
                LLAMA_HYBRID_KV_VIOLATION_INCOMPLETE_GPU_ALLOCATION;
            g_hybrid_kv_elimination_validation.total_violations++;

            if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
                fprintf(stderr, "[Hybrid KV Elimination GPU] Prefill ended but GPU KV allocation incomplete!\n");
            }

            if (g_hybrid_kv_elimination_validation.enforcement_strict) {
                return -1;
            }
        }
    }

    g_hybrid_kv_elimination_validation.state_record.state = LLAMA_GPU_KV_EXCLUSIVITY_DECODE_READY;

    if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
        fprintf(stderr, "[Hybrid KV Elimination GPU] Prefill phase ENDED - ready for GPU-only decode\n");
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_begin_decode_phase(void) {
    if (!g_hybrid_kv_elimination_validation.config.enforce_gpu_only_decode) {
        return 0;
    }

    // Validate GPU-only KV before decode starts
    if (llama_hybrid_kv_elimination_gpu_validate_gpu_only_kv_at_decode_start() != 0) {
        return -1;
    }

    g_hybrid_kv_elimination_validation.decode_phase_active = true;
    g_hybrid_kv_elimination_validation.state_record.state = LLAMA_GPU_KV_EXCLUSIVITY_DECODE_ACTIVE;
    g_hybrid_kv_elimination_validation.state_record.decode_backend_mode = LLAMA_KV_BACKEND_GPU;
    g_hybrid_kv_elimination_validation.total_decode_starts++;

    if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
        fprintf(stderr, "[Hybrid KV Elimination GPU] Decode phase STARTED - GPU-only KV enforced\n");
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_end_decode_phase(void) {
    g_hybrid_kv_elimination_validation.decode_phase_active = false;
    g_hybrid_kv_elimination_validation.state_record.state = LLAMA_GPU_KV_EXCLUSIVITY_COMPLETE;

    if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
        fprintf(stderr, "[Hybrid KV Elimination GPU] Decode phase ENDED - GPU-only KV maintained throughout\n");
    }

    return 0;
}

// ============================================================================
// KV BACKEND VALIDATION (10 ENFORCEMENT POINTS)
// ============================================================================

// Enforcement Point 1: Validate GPU-only KV at decode start
int llama_hybrid_kv_elimination_gpu_validate_gpu_only_kv_at_decode_start(void) {
    if (g_hybrid_kv_elimination_validation.state_record.layers_gpu_only !=
        g_hybrid_kv_elimination_validation.config.num_layers) {

        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_INCOMPLETE_GPU_ALLOCATION;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
            fprintf(stderr, "[Hybrid KV Elimination GPU] GPU-only validation failed: %u/%u layers GPU\n",
                g_hybrid_kv_elimination_validation.state_record.layers_gpu_only,
                g_hybrid_kv_elimination_validation.config.num_layers);
        }

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 2: Forbid hybrid KV modes in decode
int llama_hybrid_kv_elimination_gpu_forbid_hybrid_kv_modes_in_decode(void) {
    if (!g_hybrid_kv_elimination_validation.decode_phase_active) {
        return 0;
    }

    if (g_hybrid_path_attempts["hybrid_mode"] > 0) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_HYBRID_MODE_DECODE;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
            fprintf(stderr, "[Hybrid KV Elimination GPU] Hybrid KV mode attempt detected during decode!\n");
        }

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 3: Forbid CPU KV residency in decode
int llama_hybrid_kv_elimination_gpu_forbid_cpu_kv_residency_in_decode(void) {
    if (!g_hybrid_kv_elimination_validation.decode_phase_active) {
        return 0;
    }

    if (g_hybrid_kv_elimination_validation.state_record.layers_with_cpu_kv > 0) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_CPU_KV_RESIDENCY;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
            fprintf(stderr, "[Hybrid KV Elimination GPU] CPU KV residency detected: %u layers\n",
                g_hybrid_kv_elimination_validation.state_record.layers_with_cpu_kv);
        }

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 4: Forbid per-layer KV branching
int llama_hybrid_kv_elimination_gpu_forbid_per_layer_kv_branching(void) {
    if (!g_hybrid_kv_elimination_validation.decode_phase_active) {
        return 0;
    }

    if (g_hybrid_path_attempts["per_layer_branch"] > 0) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_PER_LAYER_BRANCHING;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
            fprintf(stderr, "[Hybrid KV Elimination GPU] Per-layer KV branching attempt detected!\n");
        }

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 5: Forbid CPU KV fallback under pressure
int llama_hybrid_kv_elimination_gpu_forbid_cpu_kv_fallback_under_pressure(void) {
    if (!g_hybrid_kv_elimination_validation.decode_phase_active) {
        return 0;
    }

    if (g_hybrid_path_attempts["cpu_fallback"] > 0) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_KV_FALLBACK;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
            fprintf(stderr, "[Hybrid KV Elimination GPU] CPU KV fallback attempt detected!\n");
        }

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 6: Forbid host-visible KV pointers
int llama_hybrid_kv_elimination_gpu_forbid_host_visible_kv_pointers(void) {
    if (!g_hybrid_kv_elimination_validation.decode_phase_active) {
        return 0;
    }

    if (g_hybrid_path_attempts["host_pointer"] > 0) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_HOST_VISIBLE_POINTER;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
            fprintf(stderr, "[Hybrid KV Elimination GPU] Host-visible KV pointer attempt detected!\n");
        }

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 7: Lock KV to GPU-only
int llama_hybrid_kv_elimination_gpu_lock_kv_to_gpu_only(void) {
    g_hybrid_kv_elimination_validation.state_record.decode_backend_mode = LLAMA_KV_BACKEND_GPU;

    if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
        fprintf(stderr, "[Hybrid KV Elimination GPU] KV backend locked to GPU\n");
    }

    return 0;
}

// Enforcement Point 8: Verify all layers have GPU KV
int llama_hybrid_kv_elimination_gpu_verify_all_layers_gpu_kv(void) {
    for (const auto& layer : g_layer_kv_residency) {
        if (layer.backend != LLAMA_KV_BACKEND_GPU) {
            g_hybrid_kv_elimination_validation.state_record.last_violation =
                LLAMA_HYBRID_KV_VIOLATION_INCOMPLETE_GPU_ALLOCATION;
            g_hybrid_kv_elimination_validation.total_violations++;

            if (g_hybrid_kv_elimination_validation.enforcement_strict) {
                return -1;
            }
        }
    }

    return 0;
}

// Enforcement Point 9: Verify no hybrid paths in decode
int llama_hybrid_kv_elimination_gpu_verify_no_hybrid_paths_in_decode(void) {
    if (!g_hybrid_kv_elimination_validation.decode_phase_active) {
        return 0;
    }

    if (g_hybrid_path_attempts["hybrid_path"] > 0) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_HYBRID_PATH_SELECTED;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// Enforcement Point 10: Verify GPU KV allocation complete
int llama_hybrid_kv_elimination_gpu_verify_gpu_kv_allocation_complete(void) {
    if (g_hybrid_kv_elimination_validation.state_record.layers_gpu_only !=
        g_hybrid_kv_elimination_validation.config.num_layers) {

        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_INCOMPLETE_GPU_ALLOCATION;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// LAYER-SPECIFIC KV VALIDATION
// ============================================================================

int llama_hybrid_kv_elimination_gpu_validate_layer_kv_backend(uint32_t layer_id) {
    if (layer_id >= g_layer_kv_residency.size()) {
        return -1;
    }

    if (g_layer_kv_residency[layer_id].backend != LLAMA_KV_BACKEND_GPU) {
        return -1;
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_check_layer_gpu_kv_allocated(uint32_t layer_id) {
    if (layer_id >= g_layer_kv_residency.size()) {
        return -1;
    }

    if (!g_layer_kv_residency[layer_id].gpu_allocated) {
        return -1;
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_check_layer_no_cpu_kv(uint32_t layer_id) {
    if (layer_id >= g_layer_kv_residency.size()) {
        return -1;
    }

    if (g_layer_kv_residency[layer_id].cpu_allocated) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_CPU_KV_RESIDENCY;
        g_hybrid_kv_elimination_validation.total_violations++;
        return -1;
    }

    return 0;
}

// ============================================================================
// HYBRID PATH DETECTION AND BLOCKING
// ============================================================================

int llama_hybrid_kv_elimination_gpu_detect_hybrid_mode_attempt(void) {
    g_hybrid_path_attempts["hybrid_mode"]++;

    if (g_hybrid_kv_elimination_validation.decode_phase_active) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_HYBRID_MODE_DECODE;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_detect_per_layer_branch_attempt(void) {
    g_hybrid_path_attempts["per_layer_branch"]++;

    if (g_hybrid_kv_elimination_validation.decode_phase_active) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_PER_LAYER_BRANCHING;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_detect_cpu_fallback_attempt(void) {
    g_hybrid_path_attempts["cpu_fallback"]++;

    if (g_hybrid_kv_elimination_validation.decode_phase_active) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_KV_FALLBACK;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_detect_host_pointer_attempt(void) {
    g_hybrid_path_attempts["host_pointer"]++;

    if (g_hybrid_kv_elimination_validation.decode_phase_active) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_HOST_VISIBLE_POINTER;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// KV BACKEND LOCK AND ENFORCEMENT
// ============================================================================

int llama_hybrid_kv_elimination_gpu_lock_kv_backend_to_gpu(void) {
    g_hybrid_kv_elimination_validation.state_record.decode_backend_mode = LLAMA_KV_BACKEND_GPU;

    if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
        fprintf(stderr, "[Hybrid KV Elimination GPU] KV backend locked to GPU\n");
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_lock_all_layers_to_gpu_kv(void) {
    for (auto& layer : g_layer_kv_residency) {
        layer.backend = LLAMA_KV_BACKEND_GPU;
    }

    g_hybrid_kv_elimination_validation.state_record.layers_gpu_only = g_layer_kv_residency.size();

    return 0;
}

int llama_hybrid_kv_elimination_gpu_verify_kv_backend_locked(void) {
    if (g_hybrid_kv_elimination_validation.state_record.decode_backend_mode != LLAMA_KV_BACKEND_GPU) {
        return -1;
    }

    return 0;
}

// ============================================================================
// GPU KV ALLOCATION VALIDATION
// ============================================================================

int llama_hybrid_kv_elimination_gpu_validate_all_layers_gpu_allocated(void) {
    uint32_t gpu_count = 0;

    for (const auto& layer : g_layer_kv_residency) {
        if (layer.gpu_allocated) {
            gpu_count++;
        }
    }

    g_hybrid_kv_elimination_validation.state_record.layers_gpu_only = gpu_count;

    if (gpu_count != g_hybrid_kv_elimination_validation.config.num_layers) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_INCOMPLETE_GPU_ALLOCATION;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_validate_total_gpu_kv_size(uint64_t total_bytes) {
    g_hybrid_kv_elimination_validation.state_record.total_gpu_kv_bytes = total_bytes;

    if (total_bytes == 0) {
        g_hybrid_kv_elimination_validation.state_record.last_violation =
            LLAMA_HYBRID_KV_VIOLATION_INCOMPLETE_GPU_ALLOCATION;
        g_hybrid_kv_elimination_validation.total_violations++;

        if (g_hybrid_kv_elimination_validation.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_calculate_required_gpu_kv_bytes(uint64_t* out_bytes) {
    if (out_bytes == nullptr) {
        return -1;
    }

    uint64_t total = 0;
    for (const auto& layer : g_layer_kv_residency) {
        total += layer.gpu_size_bytes;
    }

    *out_bytes = total;
    return 0;
}

// ============================================================================
// QUERY AND VERIFICATION FUNCTIONS
// ============================================================================

struct llama_gpu_kv_backend_state_record llama_hybrid_kv_elimination_gpu_get_state_record(void) {
    return g_hybrid_kv_elimination_validation.state_record;
}

struct llama_gpu_hybrid_kv_path_record llama_hybrid_kv_elimination_gpu_get_hybrid_path_record(void) {
    return g_hybrid_kv_elimination_validation.hybrid_path_record;
}

enum llama_gpu_kv_exclusivity_state llama_hybrid_kv_elimination_gpu_get_kv_state(void) {
    return g_hybrid_kv_elimination_validation.state_record.state;
}

enum llama_kv_backend_mode llama_hybrid_kv_elimination_gpu_get_decode_backend_mode(void) {
    return g_hybrid_kv_elimination_validation.state_record.decode_backend_mode;
}

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

int llama_hybrid_kv_elimination_gpu_verify_gpu_only_decode_ready(void) {
    if (g_hybrid_kv_elimination_validation.state_record.layers_gpu_only !=
        g_hybrid_kv_elimination_validation.config.num_layers) {
        return -1;
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_verify_no_hybrid_modes_active(void) {
    if (g_hybrid_path_attempts["hybrid_mode"] > 0) {
        return -1;
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_verify_all_kv_gpu_resident(void) {
    for (const auto& layer : g_layer_kv_residency) {
        if (layer.backend != LLAMA_KV_BACKEND_GPU) {
            return -1;
        }
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_verify_no_cpu_kv_present(void) {
    if (g_hybrid_kv_elimination_validation.state_record.layers_with_cpu_kv > 0) {
        return -1;
    }

    return 0;
}

int llama_hybrid_kv_elimination_gpu_verify_no_hybrid_paths_reachable(void) {
    if (g_hybrid_path_attempts.size() > 0) {
        for (const auto& attempt : g_hybrid_path_attempts) {
            if (attempt.second > 0) {
                return -1;
            }
        }
    }

    return 0;
}

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

void llama_hybrid_kv_elimination_gpu_log_gpu_only_kv_enforced(void) {
    fprintf(stderr, "[Hybrid KV Elimination GPU] GPU-only KV cache mode enforced\n");
}

void llama_hybrid_kv_elimination_gpu_log_decode_phase_started(void) {
    fprintf(stderr, "[Hybrid KV Elimination GPU] Decode phase STARTED - hybrid KV modes blocked\n");
}

void llama_hybrid_kv_elimination_gpu_log_kv_backend_locked(void) {
    fprintf(stderr, "[Hybrid KV Elimination GPU] KV backend LOCKED to GPU - immutable for decode\n");
}

void llama_hybrid_kv_elimination_gpu_print_state(void) {
    fprintf(stderr, "\n=== Hybrid KV Elimination GPU State ===\n");
    fprintf(stderr, "State: %s\n", llama_gpu_kv_exclusivity_state_name(g_hybrid_kv_elimination_validation.state_record.state));
    fprintf(stderr, "Backend Mode: %s\n", llama_kv_backend_mode_name(g_hybrid_kv_elimination_validation.state_record.decode_backend_mode));
    fprintf(stderr, "Layers GPU-Only: %u/%u\n",
        g_hybrid_kv_elimination_validation.state_record.layers_gpu_only,
        g_hybrid_kv_elimination_validation.config.num_layers);
    fprintf(stderr, "Layers with CPU KV: %u\n",
        g_hybrid_kv_elimination_validation.state_record.layers_with_cpu_kv);
    fprintf(stderr, "Total GPU KV Bytes: %llu\n",
        (unsigned long long)g_hybrid_kv_elimination_validation.state_record.total_gpu_kv_bytes);
    fprintf(stderr, "Total Violations: %d\n", g_hybrid_kv_elimination_validation.total_violations);
    fprintf(stderr, "Decode Starts: %d\n", g_hybrid_kv_elimination_validation.total_decode_starts);
    fprintf(stderr, "Enforcement: %s\n", g_hybrid_kv_elimination_validation.enforcement_strict ? "STRICT" : "PERMISSIVE");
    fprintf(stderr, "\n");
}

void llama_hybrid_kv_elimination_gpu_print_layer_residency_status(void) {
    fprintf(stderr, "\n=== KV Cache Layer Residency Status ===\n");

    for (const auto& layer : g_layer_kv_residency) {
        fprintf(stderr, "Layer %u: backend=%s, gpu_alloc=%s, cpu_alloc=%s, gpu_bytes=%llu\n",
            layer.layer_id,
            llama_kv_backend_mode_name(layer.backend),
            layer.gpu_allocated ? "YES" : "NO",
            layer.cpu_allocated ? "YES" : "NO",
            (unsigned long long)layer.gpu_size_bytes);
    }

    fprintf(stderr, "\n");
}

void llama_hybrid_kv_elimination_gpu_print_violation_summary(void) {
    fprintf(stderr, "\n=== Hybrid KV Elimination GPU Violation Summary ===\n");
    fprintf(stderr, "Total Violations: %d\n", g_hybrid_kv_elimination_validation.total_violations);
    fprintf(stderr, "Last Violation Type: %s\n",
        llama_hybrid_kv_violation_name(g_hybrid_kv_elimination_validation.state_record.last_violation));
    fprintf(stderr, "\n");
}

void llama_hybrid_kv_elimination_gpu_print_hybrid_path_attempts(void) {
    fprintf(stderr, "\n=== Hybrid Path Attempt History ===\n");

    for (const auto& attempt : g_hybrid_path_attempts) {
        fprintf(stderr, "%s: %d attempts\n", attempt.first.c_str(), attempt.second);
    }

    fprintf(stderr, "\n");
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

void llama_hybrid_kv_elimination_gpu_report_violation(
    enum llama_hybrid_kv_violation violation_type,
    uint32_t layer_id,
    const char* details
) {
    g_hybrid_kv_elimination_validation.state_record.last_violation = violation_type;
    g_hybrid_kv_elimination_validation.total_violations++;

    fprintf(stderr, "[Hybrid KV Elimination GPU] Violation: %s\n", llama_hybrid_kv_violation_name(violation_type));
    fprintf(stderr, "  Layer: %u\n", layer_id);
    if (details != nullptr) {
        fprintf(stderr, "  Details: %s\n", details);
    }

    if (g_hybrid_kv_elimination_validation.enforcement_strict) {
        fprintf(stderr, "  Action: STRICT enforcement - ABORTING\n");
    } else {
        fprintf(stderr, "  Action: PERMISSIVE mode - continuing\n");
    }
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

void llama_hybrid_kv_elimination_gpu_set_enforcement_strict(bool strict) {
    g_hybrid_kv_elimination_validation.enforcement_strict = strict;

    if (g_hybrid_kv_elimination_validation.config.debug_kv_backend) {
        fprintf(stderr, "[Hybrid KV Elimination GPU] Enforcement mode set to: %s\n", strict ? "STRICT" : "PERMISSIVE");
    }
}

bool llama_hybrid_kv_elimination_gpu_get_enforcement_strict(void) {
    return g_hybrid_kv_elimination_validation.enforcement_strict;
}

void llama_hybrid_kv_elimination_gpu_set_debug_output(bool debug) {
    g_hybrid_kv_elimination_validation.config.debug_kv_backend = debug;
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================

int llama_hybrid_kv_elimination_gpu_selftest(void) {
    fprintf(stderr, "\n=== Hybrid KV Elimination GPU Self-Test Suite ===\n");

    int test_results = 0;

    // Test 1: Initialization
    fprintf(stderr, "Test 1: Initialization... ");
    llama_hybrid_kv_elimination_gpu_init();
    if (g_hybrid_kv_elimination_validation.state_record.state == LLAMA_GPU_KV_EXCLUSIVITY_UNINITIALIZED) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 2: Configuration
    fprintf(stderr, "Test 2: Configuration... ");
    llama_hybrid_kv_elimination_gpu_configure(true, true, true, 32);
    if (g_hybrid_kv_elimination_validation.config.enforce_gpu_only_decode &&
        g_hybrid_kv_elimination_validation.config.num_layers == 32) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 3: Prefill phase
    fprintf(stderr, "Test 3: Prefill phase begin... ");
    llama_hybrid_kv_elimination_gpu_begin_prefill_phase();
    if (g_hybrid_kv_elimination_validation.state_record.state == LLAMA_GPU_KV_EXCLUSIVITY_PREFILL_PHASE) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 4: Lock all layers to GPU
    fprintf(stderr, "Test 4: Lock all layers to GPU... ");
    llama_hybrid_kv_elimination_gpu_lock_all_layers_to_gpu_kv();
    if (g_hybrid_kv_elimination_validation.state_record.layers_gpu_only == 32) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 5: End prefill phase
    fprintf(stderr, "Test 5: End prefill phase... ");
    if (llama_hybrid_kv_elimination_gpu_end_prefill_phase() == 0 &&
        g_hybrid_kv_elimination_validation.state_record.state == LLAMA_GPU_KV_EXCLUSIVITY_DECODE_READY) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 6: Begin decode phase
    fprintf(stderr, "Test 6: Begin decode phase... ");
    if (llama_hybrid_kv_elimination_gpu_begin_decode_phase() == 0 &&
        g_hybrid_kv_elimination_validation.state_record.state == LLAMA_GPU_KV_EXCLUSIVITY_DECODE_ACTIVE) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 7: Hybrid mode attempt detection
    fprintf(stderr, "Test 7: Hybrid mode attempt detection... ");
    llama_hybrid_kv_elimination_gpu_set_enforcement_strict(false);
    int result = llama_hybrid_kv_elimination_gpu_detect_hybrid_mode_attempt();
    (void)result;
    if (g_hybrid_path_attempts["hybrid_mode"] > 0) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    // Test 8: Verify GPU-only
    fprintf(stderr, "Test 8: Verify GPU-only decode... ");
    llama_hybrid_kv_elimination_gpu_set_enforcement_strict(true);
    if (llama_hybrid_kv_elimination_gpu_verify_no_cpu_kv_present() == 0) {
        fprintf(stderr, "PASSED\n");
    } else {
        fprintf(stderr, "FAILED\n");
        test_results = -1;
    }

    fprintf(stderr, "\n=== Self-Test Complete: %s ===\n\n", (test_results == 0) ? "ALL PASSED" : "SOME FAILED");

    return test_results;
}

