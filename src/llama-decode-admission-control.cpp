/**
 * SECTION 3 IMPLEMENTATION: Formalize Decode Admission Control
 *
 * Runtime enforcement of strict decode admission control mechanism
 */

#include "llama-decode-admission-control.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// GLOBAL ADMISSION CONTROL STATE
// ============================================================================

/**
 * Global decode admission control instance
 */
static struct llama_decode_admission_control g_decode_admission = {
    LLAMA_ADMISSION_STATE_UNINITIALIZED,
    LLAMA_ADMISSION_FAIL_UNKNOWN,
    { false, nullptr, false, 0, nullptr, false, nullptr, false, nullptr, false, nullptr, false },
    false,
    false,
    "",
    0,
    0
};

/**
 * Initialize decode admission control
 */
int llama_decode_admission_init(struct llama_decode_admission_control* admission) {
    if (!admission) return -1;

    admission->state = LLAMA_ADMISSION_STATE_UNINITIALIZED;
    admission->failure_reason = LLAMA_ADMISSION_FAIL_UNKNOWN;
    admission->admission_locked = false;
    admission->decode_has_started = false;
    admission->detailed_failure_message.clear();
    admission->eligibility_check_count = 0;
    admission->admission_time_us = 0;

    // Initialize eligibility criteria
    memset(&admission->eligibility, 0, sizeof(admission->eligibility));

    fprintf(stdout, "[DECODE ADMISSION] Initialized\n");
    return 0;
}

// ============================================================================
// ELIGIBILITY CRITERION 1: GPU BACKEND AVAILABILITY
// ============================================================================

int llama_admission_check_gpu_backend_available(
    struct llama_gpu_eligibility_criteria* criteria,
    const char** available_backends,
    int num_backends
) {
    if (!criteria || !available_backends || num_backends <= 0) {
        return -1;
    }

    criteria->has_valid_gpu_backend = false;
    criteria->available_gpu_backend = nullptr;

    for (int i = 0; i < num_backends; i++) {
        if (!available_backends[i]) continue;

        const char* backend = available_backends[i];

        // Check if this is a GPU backend
        bool is_gpu = (strcmp(backend, "CUDA") == 0 ||
                       strcmp(backend, "GPU") == 0 ||
                       strcmp(backend, "Metal") == 0 ||
                       strcmp(backend, "OpenCL") == 0 ||
                       strcmp(backend, "VULKAN") == 0);

        if (is_gpu) {
            criteria->has_valid_gpu_backend = true;
            criteria->available_gpu_backend = backend;
            fprintf(stdout, "[ADMISSION CRITERION 1] GPU backend available: %s\n", backend);
            return 0;
        }
    }

    fprintf(stderr, "[ADMISSION CRITERION 1] FAILED: No GPU backend found among %d backends\n", num_backends);
    criteria->has_valid_gpu_backend = false;
    return -1;
}

// ============================================================================
// ELIGIBILITY CRITERION 2: ALL DECODE-CRITICAL OPS ARE GPU-BOUND
// ============================================================================

int llama_admission_check_no_cpu_decode_ops(
    struct llama_gpu_eligibility_criteria* criteria,
    const char** decode_critical_ops,
    const char** op_backends,
    int num_ops
) {
    if (!criteria || !decode_critical_ops || !op_backends || num_ops <= 0) {
        return -1;
    }

    criteria->all_decode_critical_ops_gpu = true;
    criteria->decode_critical_ops_on_cpu = 0;
    criteria->first_cpu_decode_op = nullptr;

    for (int i = 0; i < num_ops; i++) {
        if (!decode_critical_ops[i] || !op_backends[i]) continue;

        bool is_cpu = (strcmp(op_backends[i], "CPU") == 0 ||
                       strcmp(op_backends[i], "CPP") == 0);

        if (is_cpu) {
            criteria->all_decode_critical_ops_gpu = false;
            criteria->decode_critical_ops_on_cpu++;
            if (!criteria->first_cpu_decode_op) {
                criteria->first_cpu_decode_op = decode_critical_ops[i];
            }
            fprintf(stderr, "[ADMISSION CRITERION 2] FAILED: Decode-critical op '%s' on CPU\n",
                    decode_critical_ops[i]);
        }
    }

    if (criteria->all_decode_critical_ops_gpu) {
        fprintf(stdout, "[ADMISSION CRITERION 2] All %d decode-critical ops are GPU-bound\n", num_ops);
        return 0;
    }

    return -1;
}

// ============================================================================
// ELIGIBILITY CRITERION 3: CUDA/GPU FEATURE VALIDATION
// ============================================================================

int llama_admission_check_cuda_features(
    struct llama_gpu_eligibility_criteria* criteria,
    const char** required_features,
    const char** available_features,
    int num_required,
    int num_available
) {
    if (!criteria || !required_features || !available_features) {
        return -1;
    }

    criteria->cuda_features_available = true;
    criteria->missing_cuda_feature = nullptr;

    for (int i = 0; i < num_required; i++) {
        if (!required_features[i]) continue;

        bool found = false;
        for (int j = 0; j < num_available; j++) {
            if (!available_features[j]) continue;
            if (strcmp(required_features[i], available_features[j]) == 0) {
                found = true;
                break;
            }
        }

        if (!found) {
            criteria->cuda_features_available = false;
            criteria->missing_cuda_feature = required_features[i];
            fprintf(stderr, "[ADMISSION CRITERION 3] FAILED: Missing CUDA feature '%s'\n",
                    required_features[i]);
            return -1;
        }
    }

    fprintf(stdout, "[ADMISSION CRITERION 3] All %d required CUDA features available\n", num_required);
    return 0;
}

// ============================================================================
// ELIGIBILITY CRITERION 4: KV CACHE GPU RESIDENCY
// ============================================================================

int llama_admission_check_kv_cache_gpu_resident(
    struct llama_gpu_eligibility_criteria* criteria,
    const char* kv_cache_location
) {
    if (!criteria || !kv_cache_location) {
        return -1;
    }

    criteria->kv_cache_location = kv_cache_location;

    // KV cache is GPU-resident if on GPU backend
    bool is_gpu = (strcmp(kv_cache_location, "GPU") == 0 ||
                   strcmp(kv_cache_location, "CUDA") == 0 ||
                   strcmp(kv_cache_location, "Metal") == 0 ||
                   strcmp(kv_cache_location, "OpenCL") == 0 ||
                   strcmp(kv_cache_location, "VULKAN") == 0);

    if (is_gpu) {
        criteria->kv_cache_gpu_resident = true;
        fprintf(stdout, "[ADMISSION CRITERION 4] KV cache GPU-resident: %s\n", kv_cache_location);
        return 0;
    }

    criteria->kv_cache_gpu_resident = false;
    fprintf(stderr, "[ADMISSION CRITERION 4] FAILED: KV cache not GPU-resident (location: %s)\n",
            kv_cache_location);
    return -1;
}

// ============================================================================
// ELIGIBILITY CRITERION 5: BACKEND SELECTION FROZEN
// ============================================================================

int llama_admission_check_backend_frozen(
    struct llama_gpu_eligibility_criteria* criteria,
    bool backend_is_frozen,
    const char* freeze_reason
) {
    if (!criteria || !freeze_reason) {
        return -1;
    }

    criteria->backend_selection_frozen = backend_is_frozen;
    criteria->backend_freeze_reason = freeze_reason;

    if (backend_is_frozen) {
        fprintf(stdout, "[ADMISSION CRITERION 5] Backend selection frozen: %s\n", freeze_reason);
        return 0;
    }

    fprintf(stderr, "[ADMISSION CRITERION 5] FAILED: Backend selection not frozen (%s)\n", freeze_reason);
    return -1;
}

// ============================================================================
// EXHAUSTIVE ELIGIBILITY CHECK
// ============================================================================

int llama_admission_check_gpu_eligibility(struct llama_gpu_eligibility_criteria* criteria) {
    if (!criteria) {
        return -1;
    }

    // All five criteria must pass for eligibility
    criteria->all_criteria_satisfied = (
        criteria->has_valid_gpu_backend &&
        criteria->all_decode_critical_ops_gpu &&
        criteria->cuda_features_available &&
        criteria->kv_cache_gpu_resident &&
        criteria->backend_selection_frozen
    );

    if (criteria->all_criteria_satisfied) {
        fprintf(stdout, "[ADMISSION ELIGIBILITY] ALL CRITERIA SATISFIED - GPU-exclusive execution guaranteed\n");
        return 0;
    }

    fprintf(stderr, "[ADMISSION ELIGIBILITY] FAILED - At least one criterion not satisfied\n");
    return -1;
}

// ============================================================================
// DECODE ADMISSION GATE
// ============================================================================

int llama_decode_admission_check_and_gate(
    struct llama_decode_admission_control* admission,
    struct llama_gpu_eligibility_criteria* criteria
) {
    if (!admission || !criteria) {
        return -1;
    }

    admission->eligibility_check_count++;

    // Check if admission has already been locked
    if (admission->admission_locked && admission->state == LLAMA_ADMISSION_STATE_ADMITTED) {
        fprintf(stderr, "ERROR: Attempt to re-check eligibility after decode admitted (invariant violation)\n");
        return -1;
    }

    fprintf(stdout, "[DECODE ADMISSION GATE] Performing GPU-exclusive eligibility check...\n");

    // Perform exhaustive eligibility check (all 5 criteria)
    if (llama_admission_check_gpu_eligibility(criteria) != 0) {
        admission->state = LLAMA_ADMISSION_STATE_INELIGIBLE;

        // Determine specific failure reason
        if (!criteria->has_valid_gpu_backend) {
            admission->failure_reason = LLAMA_ADMISSION_FAIL_NO_GPU_BACKEND;
        } else if (!criteria->all_decode_critical_ops_gpu) {
            admission->failure_reason = LLAMA_ADMISSION_FAIL_DECODE_OP_CPU;
        } else if (!criteria->cuda_features_available) {
            admission->failure_reason = LLAMA_ADMISSION_FAIL_INVALID_CUDA_FEATURES;
        } else if (!criteria->kv_cache_gpu_resident) {
            admission->failure_reason = LLAMA_ADMISSION_FAIL_KV_CACHE_NOT_GPU;
        } else if (!criteria->backend_selection_frozen) {
            admission->failure_reason = LLAMA_ADMISSION_FAIL_BACKEND_NOT_FROZEN;
        }

        fprintf(stderr, "FATAL: Decode admission REJECTED - %s\n",
                llama_admission_failure_name(admission->failure_reason));
        return -1;
    }

    // All criteria passed - decode is admitted
    admission->state = LLAMA_ADMISSION_STATE_ELIGIBLE;
    fprintf(stdout, "[DECODE ADMISSION GATE] PASSED - Decode is eligible for GPU-exclusive execution\n");
    return 0;
}

// ============================================================================
// ADMISSION LOCKING
// ============================================================================

int llama_decode_admission_lock(struct llama_decode_admission_control* admission) {
    if (!admission) {
        return -1;
    }

    if (admission->state != LLAMA_ADMISSION_STATE_ELIGIBLE) {
        fprintf(stderr, "ERROR: Cannot lock admission in state %s (must be ELIGIBLE)\n",
                llama_admission_state_name(admission->state));
        return -1;
    }

    if (admission->admission_locked) {
        fprintf(stderr, "ERROR: Admission already locked\n");
        return -1;
    }

    admission->admission_locked = true;
    admission->decode_has_started = true;
    admission->state = LLAMA_ADMISSION_STATE_ADMITTED;

    fprintf(stdout, "[DECODE ADMISSION] Locked - Backend selection frozen, decode proceeding GPU-only\n");
    return 0;
}

// ============================================================================
// ADMISSION VERIFICATION
// ============================================================================

int llama_decode_admission_verify_locked(const struct llama_decode_admission_control* admission) {
    if (!admission) {
        return -1;
    }

    if (!admission->admission_locked || admission->state != LLAMA_ADMISSION_STATE_ADMITTED) {
        fprintf(stderr, "FATAL: Decode not admitted (invariant violation)\n");
        return -1;
    }

    return 0;
}

// ============================================================================
// SESSION TERMINATION
// ============================================================================

int llama_decode_admission_terminate_session(
    struct llama_decode_admission_control* admission,
    const char* termination_reason
) {
    if (!admission) {
        return -1;
    }

    if (!termination_reason) {
        termination_reason = "unknown";
    }

    admission->state = LLAMA_ADMISSION_STATE_TERMINATED;
    admission->detailed_failure_message = termination_reason;

    fprintf(stderr, "FATAL: Decode session terminated - GPU-exclusive conditions violated\n");
    fprintf(stderr, "       Reason: %s\n", termination_reason);
    fprintf(stderr, "       No fallback to CPU execution. Decode terminated immediately.\n");

    return -1;
}

// ============================================================================
// DIAGNOSTICS AND REPORTING
// ============================================================================

void llama_admission_print_failure_diagnostics(
    const struct llama_decode_admission_control* admission,
    const struct llama_gpu_eligibility_criteria* criteria
) {
    if (!admission || !criteria) return;

    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "DECODE ADMISSION FAILURE DIAGNOSTICS\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "FAILURE REASON: %s\n\n", llama_admission_failure_name(admission->failure_reason));

    fprintf(stdout, "CRITERION STATUS:\n");
    fprintf(stdout, "  1. GPU Backend Available:        %s\n",
            criteria->has_valid_gpu_backend ? "✓ PASS" : "✗ FAIL");
    if (!criteria->has_valid_gpu_backend) {
        fprintf(stdout, "     → No GPU backend found\n");
    }

    fprintf(stdout, "  2. All Decode-Critical Ops GPU:  %s\n",
            criteria->all_decode_critical_ops_gpu ? "✓ PASS" : "✗ FAIL");
    if (!criteria->all_decode_critical_ops_gpu) {
        fprintf(stdout, "     → %d decode-critical ops on CPU\n", criteria->decode_critical_ops_on_cpu);
        if (criteria->first_cpu_decode_op) {
            fprintf(stdout, "     → First: '%s'\n", criteria->first_cpu_decode_op);
        }
    }

    fprintf(stdout, "  3. CUDA Features Available:      %s\n",
            criteria->cuda_features_available ? "✓ PASS" : "✗ FAIL");
    if (!criteria->cuda_features_available && criteria->missing_cuda_feature) {
        fprintf(stdout, "     → Missing: '%s'\n", criteria->missing_cuda_feature);
    }

    fprintf(stdout, "  4. KV Cache GPU-Resident:       %s\n",
            criteria->kv_cache_gpu_resident ? "✓ PASS" : "✗ FAIL");
    if (!criteria->kv_cache_gpu_resident && criteria->kv_cache_location) {
        fprintf(stdout, "     → Location: %s\n", criteria->kv_cache_location);
    }

    fprintf(stdout, "  5. Backend Selection Frozen:     %s\n",
            criteria->backend_selection_frozen ? "✓ PASS" : "✗ FAIL");
    if (!criteria->backend_selection_frozen && criteria->backend_freeze_reason) {
        fprintf(stdout, "     → Reason: %s\n", criteria->backend_freeze_reason);
    }

    fprintf(stdout, "\n");
    fprintf(stdout, "CONSEQUENCE:\n");
    fprintf(stdout, "  Decode was NOT admitted. GPU-exclusive execution cannot be guaranteed.\n");
    fprintf(stdout, "  Decode will not start. No hybrid or degraded execution.\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

void llama_admission_print_status_summary(const struct llama_decode_admission_control* admission) {
    if (!admission) return;

    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "DECODE ADMISSION STATUS\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "Admission State: %s\n", llama_admission_state_name(admission->state));
    fprintf(stdout, "Decode Admitted: %s\n", (admission->state == LLAMA_ADMISSION_STATE_ADMITTED) ? "YES" : "NO");
    fprintf(stdout, "Admission Locked: %s\n", admission->admission_locked ? "YES (irreversible)" : "NO");
    fprintf(stdout, "GPU-Exclusive Path: %s\n",
            (admission->state == LLAMA_ADMISSION_STATE_ADMITTED) ? "CONFIRMED" : "NOT CONFIRMED");

    fprintf(stdout, "\n");
    fprintf(stdout, "Eligibility Checks: %d\n", admission->eligibility_check_count);

    if (admission->state == LLAMA_ADMISSION_STATE_INELIGIBLE) {
        fprintf(stdout, "Failure Reason: %s\n", llama_admission_failure_name(admission->failure_reason));
    }

    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

// ============================================================================
// EXPLICIT ADMISSION STATEMENT
// ============================================================================

void llama_print_decode_admission_statement(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "DECODE ADMISSION CONTROL (Section 3)\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "PRINCIPLE:\n");
    fprintf(stdout, "  Decode execution is admitted ONLY when GPU-exclusive execution is guaranteed.\n");
    fprintf(stdout, "  Decode never starts in hybrid or degraded mode.\n");
    fprintf(stdout, "  Failure is immediate and final.\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "GPU-ONLY ELIGIBILITY CRITERIA (all must be satisfied):\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  1. GPU Backend Availability\n");
    fprintf(stdout, "     At least one valid GPU backend must be available.\n");
    fprintf(stdout, "     Supported: CUDA, Metal, OpenCL, Vulkan\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  2. All Decode-Critical Ops GPU-Bound\n");
    fprintf(stdout, "     Every decode-critical operation must have GPU backend.\n");
    fprintf(stdout, "     No decode-critical op shall resolve to CPU.\n");
    fprintf(stdout, "     Operations: forward pass, attention, MLP, KV cache, logits, sampling\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  3. CUDA/GPU Features Available\n");
    fprintf(stdout, "     All required CUDA features must be present.\n");
    fprintf(stdout, "     Verified at admission time.\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  4. KV Cache GPU-Resident\n");
    fprintf(stdout, "     KV cache for decode-critical layers must be on GPU.\n");
    fprintf(stdout, "     Cannot be on CPU or hybrid.\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  5. Backend Selection Frozen\n");
    fprintf(stdout, "     Backend choices cannot change after admission.\n");
    fprintf(stdout, "     Selection is locked before first token decode.\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "ADMISSION GATE:\n");
    fprintf(stdout, "  - Performs exhaustive check of all 5 criteria\n");
    fprintf(stdout, "  - Called exactly once, before first decode token\n");
    fprintf(stdout, "  - Passes: Decode is ADMITTED (GPU-exclusive guaranteed)\n");
    fprintf(stdout, "  - Fails: Decode is REJECTED (fails fast, no fallback)\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "AFTER ADMISSION:\n");
    fprintf(stdout, "  - Admission is LOCKED (irreversible)\n");
    fprintf(stdout, "  - Backend selection is FROZEN\n");
    fprintf(stdout, "  - No re-checking of eligibility\n");
    fprintf(stdout, "  - Mid-run condition changes → immediate termination (no CPU fallback)\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "FAILURE MODES:\n");
    fprintf(stdout, "  - No GPU backend: Decode rejected\n");
    fprintf(stdout, "  - CPU decode-critical op: Decode rejected\n");
    fprintf(stdout, "  - Missing CUDA features: Decode rejected\n");
    fprintf(stdout, "  - KV cache not GPU: Decode rejected\n");
    fprintf(stdout, "  - Backend not frozen: Decode rejected\n");
    fprintf(stdout, "  - Mid-run violation: Decode terminated (no fallback)\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

// ============================================================================
// SELF-TEST
// ============================================================================

int llama_decode_admission_selftest(void) {
    fprintf(stdout, "[DECODE ADMISSION SELFTEST] Running...\n");

    struct llama_decode_admission_control test_admission = {};
    struct llama_gpu_eligibility_criteria test_criteria = {};

    // Test 1: Initialization
    if (llama_decode_admission_init(&test_admission) != 0) {
        fprintf(stderr, "SELFTEST FAIL: Initialization failed\n");
        return -1;
    }

    if (test_admission.state != LLAMA_ADMISSION_STATE_UNINITIALIZED) {
        fprintf(stderr, "SELFTEST FAIL: Initial state should be UNINITIALIZED\n");
        return -1;
    }

    // Test 2: Criterion 1 - GPU backend
    const char* backends[] = {"CPU", "CUDA"};
    if (llama_admission_check_gpu_backend_available(&test_criteria, (const char**)backends, 2) != 0) {
        fprintf(stderr, "SELFTEST FAIL: GPU backend check should pass with CUDA\n");
        return -1;
    }

    // Test 3: Criterion 2 - Decode ops GPU
    const char* ops[] = {"attention", "mlp"};
    const char* op_backends[] = {"CUDA", "CUDA"};
    if (llama_admission_check_no_cpu_decode_ops(&test_criteria, (const char**)ops, (const char**)op_backends, 2) != 0) {
        fprintf(stderr, "SELFTEST FAIL: Decode ops GPU check should pass\n");
        return -1;
    }

    // Test 4: Criterion 4 - KV cache
    if (llama_admission_check_kv_cache_gpu_resident(&test_criteria, "CUDA") != 0) {
        fprintf(stderr, "SELFTEST FAIL: KV cache GPU check should pass\n");
        return -1;
    }

    // Test 5: Criterion 5 - Backend frozen
    if (llama_admission_check_backend_frozen(&test_criteria, true, "frozen at admission") != 0) {
        fprintf(stderr, "SELFTEST FAIL: Backend frozen check should pass\n");
        return -1;
    }

    // Test 6: Admission gate (all criteria pass)
    if (llama_decode_admission_check_and_gate(&test_admission, &test_criteria) != 0) {
        fprintf(stderr, "SELFTEST FAIL: Admission gate should pass when all criteria satisfied\n");
        return -1;
    }

    if (test_admission.state != LLAMA_ADMISSION_STATE_ELIGIBLE) {
        fprintf(stderr, "SELFTEST FAIL: State should be ELIGIBLE after passing gate\n");
        return -1;
    }

    // Test 7: Admission lock
    if (llama_decode_admission_lock(&test_admission) != 0) {
        fprintf(stderr, "SELFTEST FAIL: Admission lock should succeed\n");
        return -1;
    }

    if (test_admission.state != LLAMA_ADMISSION_STATE_ADMITTED) {
        fprintf(stderr, "SELFTEST FAIL: State should be ADMITTED after lock\n");
        return -1;
    }

    // Test 8: Verify locked
    if (llama_decode_admission_verify_locked(&test_admission) != 0) {
        fprintf(stderr, "SELFTEST FAIL: Verify locked should pass\n");
        return -1;
    }

    fprintf(stdout, "[DECODE ADMISSION SELFTEST] PASSED\n");
    return 0;
}
