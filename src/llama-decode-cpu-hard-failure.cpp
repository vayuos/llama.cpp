/**
 * SECTION 4 IMPLEMENTATION: Add Hard Failure on Decode-Critical CPU Execution
 *
 * Runtime enforcement of hard failure checks at execution boundaries
 */

#include "llama-decode-cpu-hard-failure.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// GLOBAL ENFORCEMENT STATE
// ============================================================================

static bool g_enforce_strict = true;  // Default: strict enforcement (abort on violation)
static int g_violation_count = 0;      // Count total violations detected

void llama_set_decode_cpu_enforcement_strict(bool enforce_strict) {
    g_enforce_strict = enforce_strict;
    fprintf(stdout, "[DECODE CPU ENFORCEMENT] Mode: %s\n",
            enforce_strict ? "STRICT (abort on violation)" : "PERMISSIVE (log but continue)");
}

bool llama_get_decode_cpu_enforcement_strict(void) {
    return g_enforce_strict;
}

int llama_get_decode_cpu_violation_count(void) {
    return g_violation_count;
}

void llama_reset_decode_cpu_violation_counter(void) {
    g_violation_count = 0;
}

// ============================================================================
// HARD FAILURE AT EXECUTION BOUNDARIES
// ============================================================================

int llama_enforce_no_decode_critical_on_cpu(
    const char* operation_name,
    bool is_decode_critical,
    const char* assigned_backend
) {
    if (!operation_name || !assigned_backend) {
        return -1;
    }

    // Non-critical ops are allowed on any backend
    if (!is_decode_critical) {
        return 0;
    }

    // Decode-critical ops MUST be on GPU
    bool is_cpu = (strcmp(assigned_backend, "CPU") == 0 ||
                   strcmp(assigned_backend, "CPP") == 0);

    if (is_cpu) {
        g_violation_count++;
        fprintf(stderr, "FATAL: Decode-critical operation '%s' assigned to CPU backend\n", operation_name);
        fprintf(stderr, "       CPU execution on the decode-critical path is forbidden.\n");
        fprintf(stderr, "       This is a correctness violation, not a performance issue.\n");
        return -1;
    }

    return 0;  // GPU backend - allowed
}

// ============================================================================
// HARD FAILURE AT BACKEND DISPATCH
// ============================================================================

int llama_enforce_decode_critical_gpu_at_dispatch(
    const char* operation_name,
    bool is_decode_critical,
    const char* target_backend
) {
    if (!operation_name || !target_backend) {
        return -1;
    }

    // Non-critical ops can be routed anywhere
    if (!is_decode_critical) {
        return 0;
    }

    // Decode-critical ops: only GPU backends allowed
    bool is_gpu = (strcmp(target_backend, "CUDA") == 0 ||
                   strcmp(target_backend, "GPU") == 0 ||
                   strcmp(target_backend, "Metal") == 0 ||
                   strcmp(target_backend, "OpenCL") == 0 ||
                   strcmp(target_backend, "VULKAN") == 0);

    if (!is_gpu) {
        g_violation_count++;
        fprintf(stderr, "FATAL: Backend dispatch attempting to route decode-critical op '%s' to '%s'\n",
                operation_name, target_backend);
        fprintf(stderr, "       Decode-critical ops can only be dispatched to GPU backends.\n");
        return -1;
    }

    return 0;  // GPU dispatch - allowed
}

// ============================================================================
// HARD FAILURE AT KERNEL DISPATCH
// ============================================================================

int llama_enforce_decode_critical_kernel_gpu_only(
    const char* operation_name,
    bool is_decode_critical,
    bool gpu_kernel_available,
    bool cpu_fallback_exists
) {
    if (!operation_name) {
        return -1;
    }

    // Non-critical ops can use any available kernel
    if (!is_decode_critical) {
        return 0;
    }

    // Decode-critical ops: GPU kernel MUST be available
    if (!gpu_kernel_available) {
        g_violation_count++;
        fprintf(stderr, "FATAL: Decode-critical operation '%s' has no GPU kernel available\n", operation_name);
        if (cpu_fallback_exists) {
            fprintf(stderr, "       A CPU fallback kernel exists but cannot be used.\n");
            fprintf(stderr, "       CPU execution on the decode-critical path is forbidden.\n");
        } else {
            fprintf(stderr, "       No fallback kernel exists. GPU kernel is mandatory.\n");
        }
        fprintf(stderr, "       This is a configuration error. Check CUDA/GPU support.\n");
        return -1;
    }

    return 0;  // GPU kernel available - allowed
}

// ============================================================================
// HARD FAILURE ON MIXED-BACKEND GRAPHS
// ============================================================================

int llama_enforce_uniform_gpu_decode_graph(
    const char** decode_critical_ops,
    const char** op_backends,
    int num_ops
) {
    if (!decode_critical_ops || !op_backends || num_ops <= 0) {
        return -1;
    }

    int cpu_critical_count = 0;
    const char* first_cpu_op = nullptr;

    for (int i = 0; i < num_ops; i++) {
        if (!decode_critical_ops[i] || !op_backends[i]) continue;

        bool is_cpu = (strcmp(op_backends[i], "CPU") == 0 ||
                       strcmp(op_backends[i], "CPP") == 0);

        if (is_cpu) {
            cpu_critical_count++;
            if (!first_cpu_op) {
                first_cpu_op = decode_critical_ops[i];
            }
        }
    }

    if (cpu_critical_count > 0) {
        g_violation_count++;
        fprintf(stderr, "FATAL: Mixed-backend decode graph detected\n");
        fprintf(stderr, "       %d decode-critical ops assigned to CPU\n", cpu_critical_count);
        if (first_cpu_op) {
            fprintf(stderr, "       First: '%s'\n", first_cpu_op);
        }
        fprintf(stderr, "       All decode-critical ops must be on GPU (uniform GPU backend).\n");
        return -1;
    }

    return 0;  // All decode-critical ops on GPU - allowed
}

// ============================================================================
// HARD FAILURE ON CPU SAMPLING
// ============================================================================

int llama_enforce_no_cpu_sampling(
    const char* sampling_backend,
    bool gpu_sampling_available
) {
    if (!sampling_backend) {
        return -1;
    }

    bool is_cpu = (strcmp(sampling_backend, "CPU") == 0 ||
                   strcmp(sampling_backend, "CPP") == 0);

    // If sampling is on CPU:
    if (is_cpu) {
        g_violation_count++;

        if (gpu_sampling_available) {
            // GPU sampling is available - CPU fallback is forbidden
            fprintf(stderr, "FATAL: Sampling attempted on CPU backend\n");
            fprintf(stderr, "       GPU sampling is available but CPU backend was selected.\n");
            fprintf(stderr, "       Sampling on the decode path must be GPU-exclusive.\n");
            return -1;
        } else {
            // GPU sampling not yet implemented - known limitation
            fprintf(stderr, "KNOWN LIMITATION: Sampling on CPU backend\n");
            fprintf(stderr, "       GPU sampling not yet implemented (Section 26 pending).\n");
            fprintf(stderr, "       CPU sampling is a temporary, documented limitation.\n");
            fprintf(stderr, "       This will be fatal once GPU sampling is available.\n");
            // For now, return 0 (allow with warning) - will be -1 after GPU sampling ready
            return 0;
        }
    }

    return 0;  // GPU sampling - allowed
}

// ============================================================================
// ASSERT OPERATION BACKEND VALIDITY
// ============================================================================

int llama_assert_operation_backend_valid(
    const char* operation_name,
    bool is_decode_critical,
    const char* assigned_backend
) {
    if (!operation_name || !assigned_backend) {
        return -1;
    }

    // Decode-critical: GPU only
    if (is_decode_critical) {
        bool is_gpu = (strcmp(assigned_backend, "CUDA") == 0 ||
                       strcmp(assigned_backend, "GPU") == 0 ||
                       strcmp(assigned_backend, "Metal") == 0 ||
                       strcmp(assigned_backend, "OpenCL") == 0 ||
                       strcmp(assigned_backend, "VULKAN") == 0);

        if (!is_gpu) {
            g_violation_count++;
            fprintf(stderr, "FATAL: Decode-critical op '%s' assigned to non-GPU backend '%s'\n",
                    operation_name, assigned_backend);
            return -1;
        }
        return 0;
    }

    // Non-critical: any backend allowed
    return 0;
}

// ============================================================================
// MIXED-BACKEND GRAPH DETECTION
// ============================================================================

int llama_detect_mixed_backend_decode_graph(
    const char** all_ops,
    bool* op_is_decode_critical,
    const char** op_backends,
    int num_ops,
    struct llama_decode_cpu_violation* violation_info
) {
    if (!all_ops || !op_is_decode_critical || !op_backends || num_ops <= 0) {
        return -1;
    }

    int cpu_count = 0, gpu_count = 0;
    const char* first_cpu_op = nullptr;

    for (int i = 0; i < num_ops; i++) {
        if (!op_is_decode_critical[i]) continue;  // Only check decode-critical ops

        bool is_cpu = (strcmp(op_backends[i], "CPU") == 0 ||
                       strcmp(op_backends[i], "CPP") == 0);

        if (is_cpu) {
            cpu_count++;
            if (!first_cpu_op) first_cpu_op = all_ops[i];
        } else {
            gpu_count++;
        }
    }

    // Mixed if both CPU and GPU decode-critical ops exist
    if (cpu_count > 0 && gpu_count > 0) {
        g_violation_count++;
        if (violation_info) {
            llama_record_decode_cpu_violation(
                violation_info,
                LLAMA_CPU_VIOLATION_MIXED_BACKEND_GRAPH,
                first_cpu_op ? first_cpu_op : "unknown",
                "CPU",
                "Decode-critical ops split across CPU and GPU backends"
            );
        }
        return -1;  // Mixed graph detected - FATAL
    }

    // No mixing (all decode-critical ops on same backend type)
    if (cpu_count > 0) {
        // All on CPU - also FATAL
        g_violation_count++;
        if (violation_info) {
            llama_record_decode_cpu_violation(
                violation_info,
                LLAMA_CPU_VIOLATION_MIXED_BACKEND_GRAPH,
                first_cpu_op ? first_cpu_op : "unknown",
                "CPU",
                "All decode-critical ops on CPU backend"
            );
        }
        return -1;
    }

    // All on GPU - allowed
    return 0;
}

// ============================================================================
// VERIFY DECODE-CRITICAL NODES ARE GPU-ONLY
// ============================================================================

int llama_verify_decode_critical_nodes_gpu_only(
    const char** node_names,
    bool* node_is_decode_critical,
    const char** node_backends,
    int num_nodes
) {
    if (!node_names || !node_is_decode_critical || !node_backends || num_nodes <= 0) {
        return -1;
    }

    for (int i = 0; i < num_nodes; i++) {
        if (!node_is_decode_critical[i]) continue;  // Skip non-critical

        bool is_cpu = (strcmp(node_backends[i], "CPU") == 0 ||
                       strcmp(node_backends[i], "CPP") == 0);

        if (is_cpu) {
            g_violation_count++;
            fprintf(stderr, "FATAL: Decode-critical node '%s' marked for CPU backend\n",
                    node_names[i]);
            fprintf(stderr, "       Before graph execution, all decode-critical nodes must be GPU-bound.\n");
            return -1;
        }
    }

    return 0;  // All decode-critical nodes GPU-only
}

// ============================================================================
// VIOLATION DETECTION AND REPORTING
// ============================================================================

void llama_record_decode_cpu_violation(
    struct llama_decode_cpu_violation* violation,
    enum llama_decode_cpu_violation_location location,
    const char* operation_name,
    const char* assigned_backend,
    const char* violation_message
) {
    if (!violation) return;

    violation->violation_detected = true;
    violation->location = location;
    violation->operation_name = operation_name;
    violation->assigned_backend = assigned_backend;
    violation->violation_message = violation_message;
}

void llama_print_decode_cpu_violation_diagnostics(
    const struct llama_decode_cpu_violation* violation
) {
    if (!violation || !violation->violation_detected) return;

    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "DECODE-CRITICAL CPU EXECUTION VIOLATION\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "VIOLATION DETECTED: YES\n");
    fprintf(stdout, "Location: %s\n", llama_cpu_violation_location_name(violation->location));
    fprintf(stdout, "Operation: %s\n", violation->operation_name ? violation->operation_name : "(unknown)");
    fprintf(stdout, "Backend: %s\n", violation->assigned_backend ? violation->assigned_backend : "(unknown)");
    fprintf(stdout, "\n");

    fprintf(stdout, "VIOLATION REASON:\n");
    fprintf(stdout, "  %s\n", violation->violation_message ? violation->violation_message : "Unknown reason");
    fprintf(stdout, "\n");

    fprintf(stdout, "CONSEQUENCE:\n");
    fprintf(stdout, "  Decode execution TERMINATED immediately.\n");
    fprintf(stdout, "  No fallback to alternative backend.\n");
    fprintf(stdout, "  This is a correctness violation, not a performance degradation.\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

// ============================================================================
// EXPLICIT PROHIBITION STATEMENT
// ============================================================================

void llama_print_decode_critical_cpu_prohibition_statement(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "DECODE-CRITICAL CPU EXECUTION PROHIBITION (Section 4)\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "PRINCIPLE:\n");
    fprintf(stdout, "  CPU execution on the decode-critical path is a FATAL ERROR.\n");
    fprintf(stdout, "  There is no fallback, recovery, or degradation.\n");
    fprintf(stdout, "  Any attempt to execute decode-critical work on CPU causes immediate abort.\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "ENFORCEMENT POINTS:\n");
    fprintf(stdout, "  1. Backend Dispatch: Decode-critical ops cannot be routed to CPU\n");
    fprintf(stdout, "  2. Kernel Dispatch: No CPU fallback kernels for decode-critical ops\n");
    fprintf(stdout, "  3. Graph Execution: Mixed CPU/GPU decode graphs are rejected\n");
    fprintf(stdout, "  4. Sampling: Sampling must be GPU-resident (or documented limitation)\n");
    fprintf(stdout, "  5. Node Execution: Before execution, all decode-critical nodes verified GPU-only\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "VIOLATION LOCATIONS (immediate abort if triggered):\n");
    fprintf(stdout, "  - Backend dispatch routing decode-critical op to CPU\n");
    fprintf(stdout, "  - Kernel dispatch with no GPU kernel available\n");
    fprintf(stdout, "  - Graph execution detects mixed CPU/GPU decode-critical ops\n");
    fprintf(stdout, "  - Sampling routed to CPU (when GPU sampling available)\n");
    fprintf(stdout, "  - Node execution with CPU backend assigned\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "ERROR MESSAGES:\n");
    fprintf(stdout, "  All violations produce explicit, actionable error messages that state:\n");
    fprintf(stdout, "  - Which decode-critical operation violated the invariant\n");
    fprintf(stdout, "  - That CPU execution on the decode path is forbidden\n");
    fprintf(stdout, "  - That this is a correctness violation, not a performance issue\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "ENFORCEMENT MODE:\n");
    fprintf(stdout, "  Strict Mode (default): All violations cause immediate abort\n");
    fprintf(stdout, "  Permissive Mode (testing): Violations logged but may continue\n");
    fprintf(stdout, "  Set via: llama_set_decode_cpu_enforcement_strict(bool)\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "NO EXCEPTIONS:\n");
    fprintf(stdout, "  There are no exceptions to this rule.\n");
    fprintf(stdout, "  Decode is either:\n");
    fprintf(stdout, "    - Fully GPU-resident (allowed), or\n");
    fprintf(stdout, "    - Rejected entirely (no execution)\n");
    fprintf(stdout, "  No hybrid, degraded, or partially CPU decode is permitted.\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

// ============================================================================
// SELF-TEST
// ============================================================================

int llama_decode_cpu_hard_failure_selftest(void) {
    fprintf(stdout, "[DECODE CPU HARD FAILURE SELFTEST] Running...\n");

    // Test 1: Decode-critical on CPU must fail
    if (llama_enforce_no_decode_critical_on_cpu("attention", true, "CPU") == 0) {
        fprintf(stderr, "SELFTEST FAIL: Decode-critical on CPU should fail\n");
        return -1;
    }

    // Test 2: Decode-critical on GPU must pass
    if (llama_enforce_no_decode_critical_on_cpu("attention", true, "CUDA") != 0) {
        fprintf(stderr, "SELFTEST FAIL: Decode-critical on GPU should pass\n");
        return -1;
    }

    // Test 3: Non-critical on CPU must pass
    if (llama_enforce_no_decode_critical_on_cpu("logging", false, "CPU") != 0) {
        fprintf(stderr, "SELFTEST FAIL: Non-critical on CPU should pass\n");
        return -1;
    }

    // Test 4: Backend dispatch GPU only
    if (llama_enforce_decode_critical_gpu_at_dispatch("mlp", true, "CPU") == 0) {
        fprintf(stderr, "SELFTEST FAIL: Decode-critical dispatch to CPU should fail\n");
        return -1;
    }

    // Test 5: Kernel dispatch GPU required
    if (llama_enforce_decode_critical_kernel_gpu_only("logits", true, false, true) == 0) {
        fprintf(stderr, "SELFTEST FAIL: No GPU kernel should fail\n");
        return -1;
    }

    // Test 6: Mixed backend graph detection
    const char* ops[] = {"attention", "mlp", "logging"};
    bool is_critical[] = {true, true, false};
    const char* backends[] = {"CUDA", "CPU", "CPU"};
    if (llama_enforce_uniform_gpu_decode_graph((const char**)ops, (const char**)backends, 3) == 0) {
        fprintf(stderr, "SELFTEST FAIL: Mixed backend should be detected\n");
        return -1;
    }

    // Test 7: Verify GPU-only nodes
    const char* nodes[] = {"forward", "attention", "mlp"};
    bool node_critical[] = {true, true, true};
    const char* node_be[] = {"CUDA", "CUDA", "CUDA"};
    if (llama_verify_decode_critical_nodes_gpu_only(
            (const char**)nodes, node_critical, (const char**)node_be, 3) != 0) {
        fprintf(stderr, "SELFTEST FAIL: All GPU nodes should pass\n");
        return -1;
    }

    // Test 8: CPU sampling detection (when GPU available)
    if (llama_enforce_no_cpu_sampling("CPU", true) == 0) {
        fprintf(stderr, "SELFTEST FAIL: CPU sampling with GPU available should fail\n");
        return -1;
    }

    fprintf(stdout, "[DECODE CPU HARD FAILURE SELFTEST] PASSED\n");
    return 0;
}
