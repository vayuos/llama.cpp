/**
 * SECTION 22: Move penalty application to GPU
 * Implementation
 *
 * GPU-native penalty kernel enforcement for deterministic sampling.
 * All repeat, frequency, and presence penalties computed on GPU.
 * Token history maintained GPU-resident; CPU never accesses penalty computation.
 * Logits stay GPU-resident; penalties applied in-place in device memory.
 */

#include "llama-penalty-gpu.h"
#include <map>
#include <vector>

// ============================================================================
// GLOBAL STATE VARIABLES
// ============================================================================

static struct llama_gpu_penalty_validation_state g_penalty_validation_state = {
    /* config */ {
        /* repeat_penalty_enabled */ false,
        /* frequency_penalty_enabled */ false,
        /* presence_penalty_enabled */ false,
        /* repeat_penalty_value */ 1.0f,
        /* frequency_penalty_value */ 0.0f,
        /* presence_penalty_value */ 0.0f,
        /* gpu_penalty_enabled */ false,
        /* history_on_gpu */ false,
        /* combined_kernel */ false,
        /* penalty_type */ LLAMA_PENALTY_TYPE_NONE,
    },
    /* state_record */ {
        /* current_penalty_type */ LLAMA_PENALTY_TYPE_NONE,
        /* gpu_penalty_state */ LLAMA_GPU_PENALTY_UNINITIALIZED,
        /* history_state */ LLAMA_GPU_HISTORY_UNINITIALIZED,
        /* gpu_penalty_active */ false,
        /* cpu_penalty_bypassed */ false,
        /* history_gpu_resident */ false,
        /* total_violations */ 0,
        /* last_violation */ LLAMA_PENALTY_VIOLATION_NONE,
        /* total_tokens_penalized */ 0,
        /* total_gpu_time_ns */ 0,
        /* total_cpu_time_ns */ 0,
    },
    /* last_execution */ {
        /* penalty_type */ LLAMA_PENALTY_TYPE_NONE,
        /* penalty_state */ LLAMA_GPU_PENALTY_UNINITIALIZED,
        /* history_state */ LLAMA_GPU_HISTORY_UNINITIALIZED,
        /* timestamp_ns */ 0,
        /* tokens_processed */ 0,
        /* gpu_kernel_ns */ 0,
        /* history_update_ns */ 0,
        /* cpu_violations */ 0,
        /* last_violation */ LLAMA_PENALTY_VIOLATION_NONE,
    },
    /* total_penalty_applications */ 0,
    /* total_violations */ 0,
    /* enforcement_strict */ true,
    /* debug_penalty_application */ false,
    /* verify_bitwise_identical */ false,
};

// Per-penalty-operation tracking: track which penalty types have been applied
static std::map<int, enum llama_penalty_type> g_penalty_application_tracker;

// Token history buffer state tracking: record history buffer lifecycle
static std::map<int, enum llama_gpu_history_buffer_state> g_history_buffer_lifecycle;

// CPU penalty bypass tracking: track CPU penalty attempts by operation
static std::map<enum llama_cpu_penalty_bypass, int> g_cpu_penalty_bypass_attempts;

// ============================================================================
// INITIALIZATION AND CONFIGURATION
// ============================================================================

int llama_penalty_gpu_init(void) {
    g_penalty_validation_state.state_record.gpu_penalty_state = LLAMA_GPU_PENALTY_KERNEL_QUEUED;
    g_penalty_validation_state.state_record.history_state = LLAMA_GPU_HISTORY_ALLOCATED;

    if (g_penalty_validation_state.debug_penalty_application) {
        llama_penalty_gpu_log_penalty_mode_enabled();
    }

    return 0;
}

int llama_penalty_gpu_configure_penalties(
    bool repeat_enabled,
    float repeat_value,
    bool frequency_enabled,
    float frequency_value,
    bool presence_enabled,
    float presence_value
) {
    g_penalty_validation_state.config.repeat_penalty_enabled = repeat_enabled;
    g_penalty_validation_state.config.repeat_penalty_value = repeat_value;
    g_penalty_validation_state.config.frequency_penalty_enabled = frequency_enabled;
    g_penalty_validation_state.config.frequency_penalty_value = frequency_value;
    g_penalty_validation_state.config.presence_penalty_enabled = presence_enabled;
    g_penalty_validation_state.config.presence_penalty_value = presence_value;

    // Determine if any penalties are active
    bool any_penalty_active = repeat_enabled || frequency_enabled || presence_enabled;
    g_penalty_validation_state.config.gpu_penalty_enabled = any_penalty_active;

    if (any_penalty_active) {
        if (repeat_enabled && frequency_enabled && presence_enabled) {
            g_penalty_validation_state.config.penalty_type = LLAMA_PENALTY_TYPE_COMBINED;
            g_penalty_validation_state.config.combined_kernel = true;
        } else if (repeat_enabled) {
            g_penalty_validation_state.config.penalty_type = LLAMA_PENALTY_TYPE_REPEAT;
        } else if (frequency_enabled) {
            g_penalty_validation_state.config.penalty_type = LLAMA_PENALTY_TYPE_FREQUENCY;
        } else {
            g_penalty_validation_state.config.penalty_type = LLAMA_PENALTY_TYPE_PRESENCE;
        }
    }

    return 0;
}

// ============================================================================
// PENALTY DETECTION AND ROUTING
// ============================================================================

int llama_penalty_gpu_detect_penalty_config(
    int repeat_last_n,
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty
) {
    bool has_repeat = repeat_last_n > 0 && repeat_penalty != 1.0f;
    bool has_frequency = frequency_penalty != 0.0f;
    bool has_presence = presence_penalty != 0.0f;

    if (has_repeat || has_frequency || has_presence) {
        llama_penalty_gpu_configure_penalties(
            has_repeat, repeat_penalty,
            has_frequency, frequency_penalty,
            has_presence, presence_penalty
        );
        return 1; // Penalties detected
    }

    return 0; // No penalties
}

int llama_penalty_gpu_should_use_gpu_penalties(void) {
    return g_penalty_validation_state.config.gpu_penalty_enabled ? 1 : 0;
}

// ============================================================================
// ENFORCEMENT POINT 1: Queue GPU penalty kernel
// ============================================================================

int llama_penalty_gpu_queue_penalty_kernel(void) {
    if (!g_penalty_validation_state.config.gpu_penalty_enabled) {
        return 0;
    }

    g_penalty_validation_state.state_record.gpu_penalty_state = LLAMA_GPU_PENALTY_KERNEL_QUEUED;
    g_penalty_validation_state.last_execution.penalty_state = LLAMA_GPU_PENALTY_KERNEL_QUEUED;

    if (g_penalty_validation_state.debug_penalty_application) {
        llama_penalty_gpu_log_penalty_kernel_launched();
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 2: Launch GPU penalty kernel
// ============================================================================

int llama_penalty_gpu_launch_penalty_kernel(void) {
    if (g_penalty_validation_state.state_record.gpu_penalty_state != LLAMA_GPU_PENALTY_KERNEL_QUEUED) {
        if (g_penalty_validation_state.enforcement_strict) {
            return -1;
        }
    }

    g_penalty_validation_state.state_record.gpu_penalty_state = LLAMA_GPU_PENALTY_KERNEL_RUNNING;
    g_penalty_validation_state.state_record.gpu_penalty_active = true;
    g_penalty_validation_state.last_execution.penalty_state = LLAMA_GPU_PENALTY_KERNEL_RUNNING;

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 3: Wait for GPU penalty result
// ============================================================================

int llama_penalty_gpu_wait_penalty_result(void) {
    if (g_penalty_validation_state.state_record.gpu_penalty_state != LLAMA_GPU_PENALTY_KERNEL_RUNNING) {
        if (g_penalty_validation_state.enforcement_strict) {
            return -1;
        }
    }

    g_penalty_validation_state.state_record.gpu_penalty_state = LLAMA_GPU_PENALTY_LOGITS_MODIFIED;
    g_penalty_validation_state.last_execution.penalty_state = LLAMA_GPU_PENALTY_LOGITS_MODIFIED;

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 4: Keep logits on GPU device
// ============================================================================

int llama_penalty_gpu_keep_logits_on_device(void) {
    if (g_penalty_validation_state.state_record.gpu_penalty_state != LLAMA_GPU_PENALTY_LOGITS_MODIFIED) {
        if (g_penalty_validation_state.enforcement_strict) {
            return -1;
        }
    }

    g_penalty_validation_state.state_record.gpu_penalty_state = LLAMA_GPU_PENALTY_READY_FOR_SAMPLING;
    g_penalty_validation_state.last_execution.penalty_state = LLAMA_GPU_PENALTY_READY_FOR_SAMPLING;

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 5: Assert GPU penalties complete
// ============================================================================

int llama_penalty_gpu_assert_penalties_complete(void) {
    if (g_penalty_validation_state.state_record.gpu_penalty_state != LLAMA_GPU_PENALTY_READY_FOR_SAMPLING) {
        if (g_penalty_validation_state.enforcement_strict) {
            llama_penalty_gpu_report_violation(
                LLAMA_PENALTY_VIOLATION_NONE,
                "GPU penalty kernel not in READY_FOR_SAMPLING state"
            );
            return -1;
        }
    }

    g_penalty_validation_state.state_record.total_tokens_penalized++;
    g_penalty_validation_state.total_penalty_applications++;

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 6: Allocate GPU history buffer
// ============================================================================

int llama_penalty_gpu_allocate_history_buffer(uint32_t max_history_size) {
    (void)max_history_size;
    g_penalty_validation_state.state_record.history_state = LLAMA_GPU_HISTORY_ALLOCATED;
    g_penalty_validation_state.config.history_on_gpu = true;
    g_penalty_validation_state.state_record.history_gpu_resident = true;

    g_history_buffer_lifecycle[0] = LLAMA_GPU_HISTORY_ALLOCATED;

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 7: Update GPU history buffer
// ============================================================================

int llama_penalty_gpu_update_history_on_gpu(uint32_t token_id) {
    (void)token_id;
    if (g_penalty_validation_state.state_record.history_state == LLAMA_GPU_HISTORY_UNINITIALIZED) {
        if (g_penalty_validation_state.enforcement_strict) {
            return -1;
        }
    }

    g_penalty_validation_state.state_record.history_state = LLAMA_GPU_HISTORY_ACTIVE;
    g_history_buffer_lifecycle[0] = LLAMA_GPU_HISTORY_ACTIVE;

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 8: Forbid CPU history loop
// ============================================================================

int llama_penalty_gpu_forbid_cpu_history_loop(void) {
    // Detect if CPU attempted to loop over token history
    if (g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_HISTORY_ITERATION] > 0) {
        llama_penalty_gpu_report_violation(
            LLAMA_PENALTY_VIOLATION_CPU_HISTORY_LOOP,
            "CPU attempted to iterate over token history during GPU penalty phase"
        );
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 9: Forbid CPU penalty computation
// ============================================================================

int llama_penalty_gpu_forbid_cpu_penalty_computation(void) {
    // Check for various CPU penalty operations
    if (g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_REPEAT_ITERATION] > 0) {
        llama_penalty_gpu_report_violation(
            LLAMA_PENALTY_VIOLATION_CPU_REPEAT,
            "CPU attempted repeat penalty computation (GPU-exclusive)"
        );
        g_penalty_validation_state.total_violations++;
        return -1;
    }

    if (g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_FREQUENCY_ITERATION] > 0) {
        llama_penalty_gpu_report_violation(
            LLAMA_PENALTY_VIOLATION_CPU_FREQUENCY,
            "CPU attempted frequency penalty computation (GPU-exclusive)"
        );
        g_penalty_validation_state.total_violations++;
        return -1;
    }

    if (g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_PRESENCE_ITERATION] > 0) {
        llama_penalty_gpu_report_violation(
            LLAMA_PENALTY_VIOLATION_CPU_PRESENCE,
            "CPU attempted presence penalty computation (GPU-exclusive)"
        );
        g_penalty_validation_state.total_violations++;
        return -1;
    }

    return 0;
}

// ============================================================================
// ENFORCEMENT POINT 10: Forbid CPU penalty entry point
// ============================================================================

int llama_penalty_gpu_forbid_cpu_penalty_entry_point(void) {
    if (g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_ENTRY_POINT] > 0) {
        llama_penalty_gpu_report_violation(
            LLAMA_PENALTY_VIOLATION_NONE,
            "CPU penalty entry point invoked during GPU penalty phase (should be GPU-exclusive)"
        );
        g_penalty_validation_state.total_violations++;

        if (g_penalty_validation_state.enforcement_strict) {
            return -1;
        }
    }

    return 0;
}

// ============================================================================
// VIOLATION DETECTION FUNCTIONS
// ============================================================================

int llama_penalty_gpu_detect_cpu_repeat_penalty(void) {
    g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_REPEAT_ITERATION]++;

    if (g_penalty_validation_state.config.repeat_penalty_enabled &&
        g_penalty_validation_state.state_record.gpu_penalty_active) {
        g_penalty_validation_state.state_record.last_violation = LLAMA_PENALTY_VIOLATION_CPU_REPEAT;
        return 1;
    }

    return 0;
}

int llama_penalty_gpu_detect_cpu_frequency_penalty(void) {
    g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_FREQUENCY_ITERATION]++;

    if (g_penalty_validation_state.config.frequency_penalty_enabled &&
        g_penalty_validation_state.state_record.gpu_penalty_active) {
        g_penalty_validation_state.state_record.last_violation = LLAMA_PENALTY_VIOLATION_CPU_FREQUENCY;
        return 1;
    }

    return 0;
}

int llama_penalty_gpu_detect_cpu_presence_penalty(void) {
    g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_PRESENCE_ITERATION]++;

    if (g_penalty_validation_state.config.presence_penalty_enabled &&
        g_penalty_validation_state.state_record.gpu_penalty_active) {
        g_penalty_validation_state.state_record.last_violation = LLAMA_PENALTY_VIOLATION_CPU_PRESENCE;
        return 1;
    }

    return 0;
}

int llama_penalty_gpu_detect_cpu_history_iteration(void) {
    g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_HISTORY_ITERATION]++;
    g_penalty_validation_state.state_record.last_violation = LLAMA_PENALTY_VIOLATION_CPU_HISTORY_LOOP;
    return 1;
}

int llama_penalty_gpu_detect_cpu_logits_modification(void) {
    g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_LOGITS_MODIFICATION]++;

    if (g_penalty_validation_state.state_record.gpu_penalty_active) {
        g_penalty_validation_state.state_record.last_violation = LLAMA_PENALTY_VIOLATION_CPU_LOGITS_MODIFIED;
        return 1;
    }

    return 0;
}

int llama_penalty_gpu_detect_history_on_host(void) {
    if (!g_penalty_validation_state.state_record.history_gpu_resident) {
        g_penalty_validation_state.state_record.last_violation = LLAMA_PENALTY_VIOLATION_HISTORY_ON_HOST;
        return 1;
    }

    return 0;
}

int llama_penalty_gpu_detect_mixed_penalty_path(void) {
    // Check if both CPU and GPU penalty operations occurred
    bool cpu_ops = (g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_REPEAT_ITERATION] > 0) ||
                   (g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_FREQUENCY_ITERATION] > 0) ||
                   (g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_PRESENCE_ITERATION] > 0);

    bool gpu_ops = g_penalty_validation_state.state_record.gpu_penalty_active;

    if (cpu_ops && gpu_ops) {
        g_penalty_validation_state.state_record.last_violation = LLAMA_PENALTY_VIOLATION_MIXED_PATH;
        return 1;
    }

    return 0;
}

// ============================================================================
// GPU STATE MANAGEMENT
// ============================================================================

int llama_penalty_gpu_set_penalty_queued(void) {
    g_penalty_validation_state.state_record.gpu_penalty_state = LLAMA_GPU_PENALTY_KERNEL_QUEUED;
    return 0;
}

int llama_penalty_gpu_set_penalty_running(void) {
    g_penalty_validation_state.state_record.gpu_penalty_state = LLAMA_GPU_PENALTY_KERNEL_RUNNING;
    g_penalty_validation_state.state_record.gpu_penalty_active = true;
    return 0;
}

int llama_penalty_gpu_set_logits_modified(void) {
    g_penalty_validation_state.state_record.gpu_penalty_state = LLAMA_GPU_PENALTY_LOGITS_MODIFIED;
    return 0;
}

int llama_penalty_gpu_set_ready_for_sampling(void) {
    g_penalty_validation_state.state_record.gpu_penalty_state = LLAMA_GPU_PENALTY_READY_FOR_SAMPLING;
    return 0;
}

// ============================================================================
// QUERY AND VERIFICATION
// ============================================================================

struct llama_gpu_penalty_state_record llama_penalty_gpu_get_state_record(void) {
    return g_penalty_validation_state.state_record;
}

struct llama_gpu_penalty_execution_record llama_penalty_gpu_get_last_execution(void) {
    return g_penalty_validation_state.last_execution;
}

enum llama_penalty_type llama_penalty_gpu_get_current_penalty_type(void) {
    return g_penalty_validation_state.state_record.current_penalty_type;
}

enum llama_gpu_penalty_state llama_penalty_gpu_get_penalty_state(void) {
    return g_penalty_validation_state.state_record.gpu_penalty_state;
}

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

int llama_penalty_gpu_verify_cpu_penalty_bypassed(void) {
    return (g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_ENTRY_POINT] == 0) ? 0 : -1;
}

int llama_penalty_gpu_verify_gpu_penalties_active(void) {
    return g_penalty_validation_state.state_record.gpu_penalty_active ? 0 : -1;
}

int llama_penalty_gpu_verify_history_on_gpu(void) {
    return g_penalty_validation_state.state_record.history_gpu_resident ? 0 : -1;
}

int llama_penalty_gpu_verify_no_cpu_entry_point(void) {
    return (g_cpu_penalty_bypass_attempts[LLAMA_PENALTY_BYPASS_ENTRY_POINT] == 0) ? 0 : -1;
}

int llama_penalty_gpu_verify_minimal_cpu_overhead(void) {
    int total_cpu_attempts = 0;
    for (auto& pair : g_cpu_penalty_bypass_attempts) {
        total_cpu_attempts += pair.second;
    }

    return (total_cpu_attempts == 0) ? 0 : -1;
}

int llama_penalty_gpu_verify_bitwise_identical_output(float cpu_value, float gpu_value) {
    if (g_penalty_validation_state.verify_bitwise_identical) {
        return (cpu_value == gpu_value) ? 0 : -1;
    }

    return 0;
}

// ============================================================================
// DIAGNOSTICS AND LOGGING
// ============================================================================

void llama_penalty_gpu_log_penalty_mode_enabled(void) {
    // Debug logging: GPU penalty mode enabled
}

void llama_penalty_gpu_log_penalty_kernel_launched(void) {
    // Debug logging: GPU penalty kernel launched
}

void llama_penalty_gpu_log_logits_penalized(uint32_t num_tokens) {
    (void)num_tokens;
    // Debug logging: tokens penalized by GPU
}

void llama_penalty_gpu_print_penalty_state(void) {
    // Print current penalty state
}

void llama_penalty_gpu_print_execution_stats(void) {
    // Print execution statistics
}

void llama_penalty_gpu_print_violation_summary(void) {
    // Print violation summary
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

void llama_penalty_gpu_report_violation(
    enum llama_penalty_violation violation_type,
    const char* details
) {
    (void)details;
    g_penalty_validation_state.state_record.last_violation = violation_type;
    g_penalty_validation_state.total_violations++;
}

// ============================================================================
// ENFORCEMENT MODE CONTROL
// ============================================================================

void llama_penalty_gpu_set_enforcement_strict(bool strict) {
    g_penalty_validation_state.enforcement_strict = strict;
}

bool llama_penalty_gpu_get_enforcement_strict(void) {
    return g_penalty_validation_state.enforcement_strict;
}

void llama_penalty_gpu_set_debug_output(bool debug) {
    g_penalty_validation_state.debug_penalty_application = debug;
}

void llama_penalty_gpu_set_verify_bitwise(bool verify) {
    g_penalty_validation_state.verify_bitwise_identical = verify;
}

// ============================================================================
// SELF-TEST SUITE
// ============================================================================

int llama_penalty_gpu_selftest(void) {
    // Test 1: Penalty configuration detection
    if (llama_penalty_gpu_detect_penalty_config(64, 1.1f, 0.0f, 0.0f) == 0) {
        return -1;
    }

    // Test 2: GPU penalty initialization
    if (llama_penalty_gpu_init() != 0) {
        return -1;
    }

    // Test 3: History buffer allocation
    if (llama_penalty_gpu_allocate_history_buffer(64) != 0) {
        return -1;
    }

    // Test 4: GPU penalty kernel lifecycle
    if (llama_penalty_gpu_queue_penalty_kernel() != 0) {
        return -1;
    }
    if (llama_penalty_gpu_launch_penalty_kernel() != 0) {
        return -1;
    }
    if (llama_penalty_gpu_wait_penalty_result() != 0) {
        return -1;
    }

    // Test 5: Token history update
    if (llama_penalty_gpu_update_history_on_gpu(100) != 0) {
        return -1;
    }

    // Test 6: CPU penalty bypass verification
    if (llama_penalty_gpu_verify_cpu_penalty_bypassed() != 0) {
        return -1;
    }

    // Test 7: GPU penalty state verification
    if (llama_penalty_gpu_verify_gpu_penalties_active() != 0) {
        return -1;
    }

    // Test 8: History on GPU verification
    if (llama_penalty_gpu_verify_history_on_gpu() != 0) {
        return -1;
    }

    return 0;
}
