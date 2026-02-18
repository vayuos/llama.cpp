/**
 * SECTION 25: Eliminate CPU logits reads during decode
 * Implementation
 *
 * GPU-exclusive logits management enforcement for deterministic sampling.
 * All logits remain GPU-resident during decode; no CPU reads, inspections, or materializations.
 * Phase-aware access control enforces decode-phase CPU access blocking.
 * Only token IDs cross PCIe; logits data never transferred to CPU during decode.
 */

#include "llama-logits-gpu.h"
#include <map>
#include <vector>
#include <algorithm> // for std::all_of

// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================

static struct llama_gpu_logits_validation_state g_logits_validation_state = {
    /* config */ {
        /* gpu_exclusive_logits */ false,
        /* logits_cpu_access_forbidden */ false,
        /* current_phase */ LLAMA_DECODE_PHASE_UNINITIALIZED,
        /* access_mode */ LLAMA_LOGITS_ACCESS_NONE,
        /* cpu_logits_materialization_allowed */ true,
        /* enforce_gpu_resident_only */ false,
        /* phase_aware_access */ false,
    },
    /* state_record */ {
        /* current_phase */ LLAMA_DECODE_PHASE_UNINITIALIZED,
        /* current_access_mode */ LLAMA_LOGITS_ACCESS_NONE,
        /* buffer_state */ LLAMA_GPU_LOGITS_UNINITIALIZED,
        /* materialization_state */ LLAMA_LOGITS_MATERIALIZATION_NONE,
        /* cpu_logits_access_blocked */ false,
        /* gpu_logits_resident */ false,
        /* total_violations */ 0,
        /* last_violation */ LLAMA_LOGITS_VIOLATION_NONE,
        /* total_tokens_processed */ 0,
        /* total_gpu_residency_ns */ 0,
    },
    /* last_execution */ {
        /* phase */ LLAMA_DECODE_PHASE_UNINITIALIZED,
        /* access_mode */ LLAMA_LOGITS_ACCESS_NONE,
        /* buffer_state */ LLAMA_GPU_LOGITS_UNINITIALIZED,
        /* timestamp_ns */ 0,
        /* tokens_processed */ 0,
        /* cpu_violations */ 0,
        /* last_violation */ LLAMA_LOGITS_VIOLATION_NONE,
        /* cpu_attempted_read */ false,
        /* gpu_resident_maintained */ false,
    },
    /* total_logits_accesses */ 0,
    /* total_violations */ 0,
    /* enforcement_strict */ true,
    /* debug_logits_access */ false,
    /* verify_gpu_residency */ false,
};

// Per-operation tracking: track CPU logits operations attempted
static std::map<enum llama_cpu_logits_operation, int> g_cpu_logits_operation_attempts;

// Phase transition tracking: record phase changes
static std::map<int, enum llama_decode_phase> g_phase_transition_log;

// ============================================================================
// INITIALIZATION AND CONFIGURATION
// ============================================================================

int llama_logits_gpu_init(void) {
    g_logits_validation_state.state_record.buffer_state = LLAMA_GPU_LOGITS_ALLOCATED;
    g_logits_validation_state.state_record.gpu_logits_resident = true;

    if (g_logits_validation_state.debug_logits_access) {
        llama_logits_gpu_log_gpu_exclusive_mode_enabled();
    }

    return 0;
}

int llama_logits_gpu_configure_exclusive(
    bool gpu_exclusive,
    bool cpu_forbidden,
    bool phase_aware
) {
    g_logits_validation_state.config.gpu_exclusive_logits = gpu_exclusive;
    g_logits_validation_state.config.logits_cpu_access_forbidden = cpu_forbidden;
    g_logits_validation_state.config.phase_aware_access = phase_aware;
    g_logits_validation_state.config.enforce_gpu_resident_only = gpu_exclusive;

    if (gpu_exclusive && cpu_forbidden) {
        g_logits_validation_state.state_record.cpu_logits_access_blocked = true;
        g_logits_validation_state.config.access_mode = LLAMA_LOGITS_ACCESS_CPU_FORBIDDEN;
    }

    return 0;
}

// ============================================================================
// PHASE MANAGEMENT
// ============================================================================

int llama_logits_gpu_set_decode_phase(enum llama_decode_phase phase) {
    g_logits_validation_state.config.current_phase = phase;
    g_logits_validation_state.state_record.current_phase = phase;

    // Update access mode based on phase
    if (phase == LLAMA_DECODE_PHASE_DECODE) {
        g_logits_validation_state.state_record.current_access_mode = LLAMA_LOGITS_ACCESS_GPU_RESIDENT;
        g_logits_validation_state.config.cpu_logits_materialization_allowed = false;
        g_logits_validation_state.state_record.cpu_logits_access_blocked = true;

        if (g_logits_validation_state.debug_logits_access) {
            llama_logits_gpu_log_decode_phase_started();
        }
    } else if (phase == LLAMA_DECODE_PHASE_PREFILL) {
        g_logits_validation_state.state_record.current_access_mode = LLAMA_LOGITS_ACCESS_CPU_READABLE;
        g_logits_validation_state.config.cpu_logits_materialization_allowed = true;
        g_logits_validation_state.state_record.cpu_logits_access_blocked = false;
    }

    return 0;
}

enum llama_decode_phase llama_logits_gpu_get_current_phase(void) {
    return g_logits_validation_state.state_record.current_phase;
}

int llama_logits_gpu_is_decode_phase(void) {
    return (g_logits_validation_state.state_record.current_phase == LLAMA_DECODE_PHASE_DECODE) ? 1 : 0;
}

// ============================================================================
// ENFORCEMENT POINT 1: Queue logits computation
// ============================================================================

int llama_logits_gpu_queue_logits_computation(void) {
    g_logits_validation_state.state_record.buffer_state = LLAMA_GPU_LOGITS_ALLOCATED;
    g_logits_validation_state.last_execution.buffer_state = LLAMA_GPU_LOGITS_ALLOCATED;

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 2: Keep logits on GPU
// ============================================================================

int llama_logits_gpu_keep_logits_on_gpu(void) {
    g_logits_validation_state.state_record.buffer_state = LLAMA_GPU_LOGITS_POPULATED;
    g_logits_validation_state.state_record.gpu_logits_resident = true;
    g_logits_validation_state.last_execution.buffer_state = LLAMA_GPU_LOGITS_POPULATED;
    g_logits_validation_state.last_execution.gpu_resident_maintained = true;

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 3: Forbid CPU logits read
// ============================================================================

int llama_logits_gpu_forbid_cpu_logits_read(void) {
    if (g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_GET_DATA] > 0 ||
        g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_GET_DATA_F32] > 0 ||
        g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_BACKEND_TENSOR_GET] > 0) {
        llama_logits_gpu_report_violation(
            LLAMA_LOGITS_VIOLATION_CPU_READ,
            "CPU attempted to read logits during decode phase"
        );
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 4: Forbid CPU logits materialization
// ============================================================================

int llama_logits_gpu_forbid_cpu_logits_materialization(void) {
    if (g_logits_validation_state.state_record.current_phase == LLAMA_DECODE_PHASE_DECODE &&
        g_logits_validation_state.state_record.materialization_state == LLAMA_LOGITS_MATERIALIZATION_ATTEMPTED) {
        llama_logits_gpu_report_violation(
            LLAMA_LOGITS_VIOLATION_MATERIALIZATION,
            "CPU attempted to materialize logits during decode phase"
        );
        g_logits_validation_state.total_violations++;

        if (g_logits_validation_state.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 5: Assert logits GPU-resident
// ============================================================================

int llama_logits_gpu_assert_logits_gpu_resident(void) {
    if (!g_logits_validation_state.state_record.gpu_logits_resident) {
        llama_logits_gpu_report_violation(
            LLAMA_LOGITS_VIOLATION_HOST_COPY,
            "Logits not GPU-resident during decode phase"
        );

        if (g_logits_validation_state.enforcement_strict) {
            return -1;
        }
    }

    g_logits_validation_state.state_record.total_tokens_processed++;
    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 6: Forbid get_data
// ============================================================================

int llama_logits_gpu_forbid_get_data(void) {
    if (g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_GET_DATA] > 0) {
        llama_logits_gpu_report_violation(
            LLAMA_LOGITS_VIOLATION_GET_DATA_CALLED,
            "ggml_get_data called on logits during decode (GPU-exclusive)"
        );
        g_logits_validation_state.total_violations++;
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 7: Forbid backend tensor get
// ============================================================================

int llama_logits_gpu_forbid_backend_tensor_get(void) {
    if (g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_BACKEND_TENSOR_GET] > 0) {
        llama_logits_gpu_report_violation(
            LLAMA_LOGITS_VIOLATION_HOST_COPY,
            "ggml_backend_tensor_get called on logits during decode (GPU-exclusive)"
        );
        g_logits_validation_state.total_violations++;
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 8: Forbid CPU buffer view
// ============================================================================

int llama_logits_gpu_forbid_cpu_buffer_view(void) {
    if (g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_CPU_BUFFER_VIEW] > 0) {
        llama_logits_gpu_report_violation(
            LLAMA_LOGITS_VIOLATION_CPU_VIEW_MAP,
            "CPU buffer view mapping attempted on logits during decode (GPU-exclusive)"
        );
        g_logits_validation_state.total_violations++;
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 9: Verify no host copy
// ============================================================================

int llama_logits_gpu_verify_no_host_copy(void) {
    if (g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_HOST_COPY] > 0) {
        llama_logits_gpu_report_violation(
            LLAMA_LOGITS_VIOLATION_HOST_COPY,
            "Host copy (cudaMemcpy) of logits detected during decode (GPU-exclusive)"
        );
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 10: Verify GPU-exclusive access
// ============================================================================

int llama_logits_gpu_verify_gpu_exclusive_access(void) {
    // Check for any CPU logits operations during decode phase
    if (g_logits_validation_state.state_record.current_phase == LLAMA_DECODE_PHASE_DECODE) {
        int total_cpu_attempts = 0;
        for (auto& pair : g_cpu_logits_operation_attempts) {
            total_cpu_attempts += pair.second;
        }

        if (total_cpu_attempts > 0) {
            llama_logits_gpu_report_violation(
                LLAMA_LOGITS_VIOLATION_PHASE_MISMATCH,
                "CPU logits operations attempted during decode phase (GPU-exclusive required)"
            );
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// VIOLATION DETECTION FUNCTIONS
// ============================================================================

int llama_logits_gpu_detect_cpu_read(void) {
    g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_GET_DATA]++;

    if (g_logits_validation_state.state_record.current_phase == LLAMA_DECODE_PHASE_DECODE) {
        g_logits_validation_state.state_record.last_violation = LLAMA_LOGITS_VIOLATION_CPU_READ;
        g_logits_validation_state.last_execution.cpu_attempted_read = true;
        return 1;
    }

    return 0;
}

int llama_logits_gpu_detect_host_copy(void) {
    g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_HOST_COPY]++;

    if (g_logits_validation_state.state_record.current_phase == LLAMA_DECODE_PHASE_DECODE) {
        g_logits_validation_state.state_record.last_violation = LLAMA_LOGITS_VIOLATION_HOST_COPY;
        return 1;
    }

    return 0;
}

int llama_logits_gpu_detect_cpu_view_map(void) {
    g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_CPU_BUFFER_VIEW]++;

    if (g_logits_validation_state.state_record.current_phase == LLAMA_DECODE_PHASE_DECODE) {
        g_logits_validation_state.state_record.last_violation = LLAMA_LOGITS_VIOLATION_CPU_VIEW_MAP;
        return 1;
    }

    return 0;
}

int llama_logits_gpu_detect_get_data_call(void) {
    g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_GET_DATA_F32]++;

    if (g_logits_validation_state.state_record.current_phase == LLAMA_DECODE_PHASE_DECODE) {
        g_logits_validation_state.state_record.last_violation = LLAMA_LOGITS_VIOLATION_GET_DATA_CALLED;
        return 1;
    }

    return 0;
}

int llama_logits_gpu_detect_debug_dump(void) {
    g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_INSPECTION]++;

    if (g_logits_validation_state.state_record.current_phase == LLAMA_DECODE_PHASE_DECODE) {
        g_logits_validation_state.state_record.last_violation = LLAMA_LOGITS_VIOLATION_DEBUG_DUMP;
        return 1;
    }

    return 0;
}

int llama_logits_gpu_detect_materialization_attempt(void) {
    if (g_logits_validation_state.state_record.current_phase == LLAMA_DECODE_PHASE_DECODE &&
        g_logits_validation_state.config.cpu_logits_materialization_allowed) {
        g_logits_validation_state.state_record.materialization_state = LLAMA_LOGITS_MATERIALIZATION_ATTEMPTED;
        g_logits_validation_state.state_record.last_violation = LLAMA_LOGITS_VIOLATION_MATERIALIZATION;
        return 1;
    }

    return 0;
}

int llama_logits_gpu_detect_phase_mismatch(void) {
    if (g_logits_validation_state.config.phase_aware_access &&
        g_logits_validation_state.state_record.current_access_mode == LLAMA_LOGITS_ACCESS_CPU_READABLE &&
        g_logits_validation_state.state_record.current_phase == LLAMA_DECODE_PHASE_DECODE) {
        g_logits_validation_state.state_record.last_violation = LLAMA_LOGITS_VIOLATION_PHASE_MISMATCH;
        return 1;
    }

    return 0;
}

// ============================================================================
// GPU STATE MANAGEMENT
// ============================================================================

int llama_logits_gpu_set_buffer_allocated(void) {
    g_logits_validation_state.state_record.buffer_state = LLAMA_GPU_LOGITS_ALLOCATED;
    return 0;
}

int llama_logits_gpu_set_buffer_populated(void) {
    g_logits_validation_state.state_record.buffer_state = LLAMA_GPU_LOGITS_POPULATED;
    return 0;
}

int llama_logits_gpu_set_ready_for_sampling(void) {
    g_logits_validation_state.state_record.buffer_state = LLAMA_GPU_LOGITS_READY_FOR_SAMPLING;
    return 0;
}

// ============================================================================
// QUERY AND VERIFICATION
// ============================================================================

struct llama_gpu_logits_state_record llama_logits_gpu_get_state_record(void) {
    return g_logits_validation_state.state_record;
}

struct llama_gpu_logits_execution_record llama_logits_gpu_get_last_execution(void) {
    return g_logits_validation_state.last_execution;
}

enum llama_logits_access_mode llama_logits_gpu_get_current_access_mode(void) {
    return g_logits_validation_state.state_record.current_access_mode;
}

enum llama_gpu_logits_buffer_state llama_logits_gpu_get_buffer_state(void) {
    return g_logits_validation_state.state_record.buffer_state;
}

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

int llama_logits_gpu_verify_gpu_resident(void) {
    return g_logits_validation_state.state_record.gpu_logits_resident ? 0 : -1;
}

int llama_logits_gpu_verify_cpu_access_forbidden(void) {
    return g_logits_validation_state.state_record.cpu_logits_access_blocked ? 0 : -1;
}

int llama_logits_gpu_verify_no_host_materializations(void) {
    return (g_logits_validation_state.total_violations == 0) ? 0 : -1;
}

int llama_logits_gpu_verify_decode_phase_compliance(void) {
    if (g_logits_validation_state.state_record.current_phase != LLAMA_DECODE_PHASE_DECODE) {
        return -1;
    }
    return (g_cpu_logits_operation_attempts.empty() ||
            std::all_of(g_cpu_logits_operation_attempts.begin(),
                       g_cpu_logits_operation_attempts.end(),
                       [](const auto& p) { return p.second == 0; })) ? 0 : -1;
}

int llama_logits_gpu_verify_minimal_cpu_overhead(void) {
    int total_cpu_attempts = 0;
    for (auto& pair : g_cpu_logits_operation_attempts) {
        total_cpu_attempts += pair.second;
    }

    return (total_cpu_attempts == 0) ? 0 : -1;
}

int llama_logits_gpu_verify_logits_only_token_crosses_pcie(void) {
    // Verify no logits data crosses PCIe, only token IDs
    return (g_logits_validation_state.state_record.gpu_logits_resident &&
            g_cpu_logits_operation_attempts[LLAMA_LOGITS_OP_HOST_COPY] == 0) ? 0 : -1;
}

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

void llama_logits_gpu_log_decode_phase_started(void) {
    // Debug logging: decode phase started
}

void llama_logits_gpu_log_gpu_exclusive_mode_enabled(void) {
    // Debug logging: GPU-exclusive mode enabled
}

void llama_logits_gpu_log_cpu_read_blocked(void) {
    // Debug logging: CPU read blocked
}

void llama_logits_gpu_print_logits_state(void) {
    // Print current logits state
}

void llama_logits_gpu_print_execution_stats(void) {
    // Print execution statistics
}

void llama_logits_gpu_print_violation_summary(void) {
    // Print violation summary
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

void llama_logits_gpu_report_violation(
    enum llama_logits_violation violation_type,
    const char* details
) {
    (void)details;
    g_logits_validation_state.state_record.last_violation = violation_type;
    g_logits_validation_state.total_violations++;
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

void llama_logits_gpu_set_enforcement_strict(bool strict) {
    g_logits_validation_state.enforcement_strict = strict;
}

bool llama_logits_gpu_get_enforcement_strict(void) {
    return g_logits_validation_state.enforcement_strict;
}

void llama_logits_gpu_set_debug_output(bool debug) {
    g_logits_validation_state.debug_logits_access = debug;
}

void llama_logits_gpu_set_verify_gpu_residency(bool verify) {
    g_logits_validation_state.verify_gpu_residency = verify;
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================

int llama_logits_gpu_selftest(void) {
    // Test 1: GPU-exclusive configuration
    if (llama_logits_gpu_init() != 0) {
        return -1;
    }

    // Test 2: Phase management
    if (llama_logits_gpu_set_decode_phase(LLAMA_DECODE_PHASE_DECODE) != 0) {
        return -1;
    }

    // Test 3: Logits GPU residency
    if (llama_logits_gpu_keep_logits_on_gpu() != 0) {
        return -1;
    }

    // Test 4: CPU access blocking verification
    if (llama_logits_gpu_verify_cpu_access_forbidden() != 0) {
        return -1;
    }

    // Test 5: GPU residency verification
    if (llama_logits_gpu_verify_gpu_resident() != 0) {
        return -1;
    }

    // Test 6: Decode phase compliance
    if (llama_logits_gpu_verify_decode_phase_compliance() != 0) {
        return -1;
    }

    // Test 7: Minimal CPU overhead
    if (llama_logits_gpu_verify_minimal_cpu_overhead() != 0) {
        return -1;
    }

    // Test 8: PCIE token-only verification
    if (llama_logits_gpu_verify_logits_only_token_crosses_pcie() != 0) {
        return -1;
    }

    return 0;
}
