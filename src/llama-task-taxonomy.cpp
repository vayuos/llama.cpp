/**
 * SECTION 2 IMPLEMENTATION: Introduce Decode-Critical vs Non-Critical Task Taxonomy
 *
 * Runtime validation and enforcement of the two-class task taxonomy system
 */

#include "llama-task-taxonomy.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// GLOBAL TASK TAXONOMY STATE
// ============================================================================

/**
 * Global task taxonomy instance
 * Shared across all contexts during a session
 */
static struct llama_task_taxonomy_state g_task_taxonomy_state = {
    false, // taxonomy_initialized
    0,     // total_tasks_classified
    0,     // decode_critical_count
    0,     // non_critical_count
    true   // enforce_irreversibility
};

/**
 * Initialize the global task taxonomy
 */
void llama_task_taxonomy_init(void) {
    g_task_taxonomy_state.taxonomy_initialized = true;
    g_task_taxonomy_state.total_tasks_classified = 0;
    g_task_taxonomy_state.decode_critical_count = 0;
    g_task_taxonomy_state.non_critical_count = 0;
    g_task_taxonomy_state.enforce_irreversibility = true;

    fprintf(stdout, "[TASK TAXONOMY] Initialized (two-class: DECODE_CRITICAL + NON_CRITICAL)\n");
}

/**
 * Get the global task taxonomy state
 */
struct llama_task_taxonomy_state llama_get_task_taxonomy_state(void) {
    return g_task_taxonomy_state;
}

// ============================================================================
// TASK CLASSIFICATION TRACKING
// ============================================================================

/**
 * Record a task classification decision
 * Tracks statistics for logging and validation
 */
int llama_task_record_classification(
    const char* task_name,
    enum llama_task_class task_class
) {
    if (!task_name) {
        fprintf(stderr, "ERROR: Cannot classify unnamed task\\n");
        return -1;
    }

    if (!g_task_taxonomy_state.taxonomy_initialized) {
        fprintf(stderr, "ERROR: Task taxonomy not initialized\\n");
        return -1;
    }

    g_task_taxonomy_state.total_tasks_classified++;

    switch (task_class) {
        case LLAMA_TASK_CLASS_DECODE_CRITICAL:
            g_task_taxonomy_state.decode_critical_count++;
            fprintf(stdout, "[TASK CLASSIFY] '%s' → DECODE_CRITICAL (GPU-only)\\n", task_name);
            break;

        case LLAMA_TASK_CLASS_NON_CRITICAL:
            g_task_taxonomy_state.non_critical_count++;
            fprintf(stdout, "[TASK CLASSIFY] '%s' → NON_CRITICAL (CPU-only)\\n", task_name);
            break;

        case LLAMA_TASK_CLASS_UNKNOWN:
            fprintf(stderr, "WARNING: Task '%s' classified as UNKNOWN\\n", task_name);
            return -1;

        default:
            fprintf(stderr, "ERROR: Invalid task class for '%s'\\n", task_name);
            return -1;
    }

    return 0;
}

// ============================================================================
// CLASSIFICATION VERIFICATION
// ============================================================================

/**
 * Verify that a task's classification matches its actual work
 * Returns 0 if classification is correct, -1 if incorrect
 */
int llama_task_verify_classification(
    const char* task_name,
    enum llama_task_class assigned_class
) {
    if (!task_name) return -1;

    // Determine what the classification SHOULD be
    enum llama_task_class inferred_class = llama_classify_task(task_name);

    if (inferred_class == LLAMA_TASK_CLASS_UNKNOWN) {
        // Cannot verify unknown tasks
        fprintf(stderr, "WARNING: Cannot verify classification of unknown task '%s'\\n", task_name);
        return -1;
    }

    if (inferred_class != assigned_class) {
        fprintf(stderr,
                "ERROR: Task '%s' classification mismatch: assigned %s but should be %s\\n",
                task_name,
                llama_task_class_name(assigned_class),
                llama_task_class_name(inferred_class));
        return -1;
    }

    return 0;  // Classification matches
}

// ============================================================================
// IRREVERSIBILITY ENFORCEMENT
// ============================================================================

/**
 * Strict reclassification prevention
 * Once locked, a task's class cannot change
 */
int llama_task_strict_lock_classification(struct llama_task_metadata* meta) {
    if (!meta) {
        fprintf(stderr, "ERROR: Cannot lock null metadata\\n");
        return -1;
    }

    if (meta->class_is_locked) {
        fprintf(stderr, "ERROR: Task '%s' already locked (reclassification forbidden)\\n",
                meta->task_name ? meta->task_name : "(unnamed)");
        return -1;
    }

    meta->class_is_locked = true;
    fprintf(stdout, "[TASK LOCK] Task '%s' classification locked (%s)\\n",
            meta->task_name ? meta->task_name : "(unnamed)",
            llama_task_class_name(meta->task_class));

    return 0;
}

// ============================================================================
// BACKEND BINDING VALIDATION
// ============================================================================

/**
 * Validate a task against its assigned backend
 * DECODE_CRITICAL must be GPU; NON_CRITICAL must be CPU
 */
int llama_task_validate_backend_assignment(
    const char* task_name,
    enum llama_task_class task_class,
    const char* backend_name
) {
    if (!task_name || !backend_name) {
        return -1;
    }

    int result = llama_validate_task_backend_binding(task_class, backend_name);

    if (result != 0) {
        fprintf(stderr,
                "ERROR: Task '%s' (class=%s) cannot execute on backend '%s'\\n",
                task_name,
                llama_task_class_name(task_class),
                backend_name);
    }

    return result;
}

// ============================================================================
// EXHAUSTIVE CLASSIFICATION VERIFICATION
// ============================================================================

/**
 * Verify that all tasks in a system are classified exhaustively
 * No task should remain UNKNOWN
 */
int llama_task_verify_exhaustive_classification(
    const char* task_names[],
    int num_tasks
) {
    if (!task_names || num_tasks <= 0) {
        return -1;
    }

    int unknown_count = 0;

    fprintf(stdout, "[TASK TAXONOMY] Verifying exhaustive classification of %d tasks\\n", num_tasks);

    for (int i = 0; i < num_tasks; i++) {
        enum llama_task_class cls = llama_classify_task(task_names[i]);

        if (cls == LLAMA_TASK_CLASS_UNKNOWN) {
            fprintf(stderr, "  UNCLASSIFIED: '%s'\\n", task_names[i]);
            unknown_count++;
        }
    }

    if (unknown_count > 0) {
        fprintf(stderr,
                "ERROR: %d/%d tasks remain unclassified (exhaustivity violated)\\n",
                unknown_count, num_tasks);
        return -1;
    }

    fprintf(stdout, "[TASK TAXONOMY] All %d tasks exhaustively classified ✓\\n", num_tasks);
    return 0;
}

// ============================================================================
// QUEUE ASSIGNMENT AND ROUTING
// ============================================================================

/**
 * Get the correct queue for a task based on its class
 * Used for task routing and scheduling
 */
const char* llama_task_get_assigned_queue(enum llama_task_class task_class) {
    return llama_get_task_queue(task_class);
}

/**
 * Validate that a task is routed to the correct queue
 */
int llama_task_validate_queue_routing(
    const char* task_name,
    enum llama_task_class task_class,
    const char* assigned_queue
) {
    if (!task_name || !assigned_queue) {
        return -1;
    }

    const char* correct_queue = llama_get_task_queue(task_class);

    if (strcmp(assigned_queue, correct_queue) != 0) {
        fprintf(stderr,
                "ERROR: Task '%s' (class=%s) routed to '%s' queue but should be '%s'\\n",
                task_name,
                llama_task_class_name(task_class),
                assigned_queue,
                correct_queue);
        return -1;
    }

    return 0;
}

// ============================================================================
// VALIDATION AND ASSERTIONS
// ============================================================================

/**
 * Assert that decode-critical tasks never execute on CPU
 */
int llama_task_assert_decode_critical_gpu_only(
    enum llama_task_class task_class,
    const char* actual_backend
) {
    if (task_class != LLAMA_TASK_CLASS_DECODE_CRITICAL) {
        return 0;  // Not a decode-critical task
    }

    if (!actual_backend) {
        return -1;
    }

    const bool is_cpu = (strcmp(actual_backend, "CPU") == 0 ||
                         strcmp(actual_backend, "CPP") == 0);

    if (is_cpu) {
        fprintf(stderr,
                "FATAL: DECODE_CRITICAL task executed on CPU backend (invariant violation)\\n");
        return -1;
    }

    return 0;
}

/**
 * Assert that non-critical tasks do not block decode progression
 */
int llama_task_assert_non_critical_non_blocking(
    enum llama_task_class task_class,
    bool is_blocking_decode
) {
    if (task_class != LLAMA_TASK_CLASS_NON_CRITICAL) {
        return 0;  // Not a non-critical task
    }

    if (is_blocking_decode) {
        fprintf(stderr,
                "WARNING: NON_CRITICAL task is blocking decode progression\\n");
        return -1;
    }

    return 0;
}

// ============================================================================
// STATISTICS AND REPORTING
// ============================================================================

/**
 * Print task taxonomy statistics
 */
void llama_task_print_statistics(void) {
    struct llama_task_taxonomy_state state = llama_get_task_taxonomy_state();

    fprintf(stdout, "\\n");
    fprintf(stdout, "================================================================================\\n");
    fprintf(stdout, "TASK TAXONOMY STATISTICS\\n");
    fprintf(stdout, "================================================================================\\n");
    fprintf(stdout, "\\n");
    fprintf(stdout, "Total tasks classified: %d\\n", state.total_tasks_classified);
    fprintf(stdout, "  - DECODE_CRITICAL (GPU-only): %d\\n", state.decode_critical_count);
    fprintf(stdout, "  - NON_CRITICAL (CPU-only): %d\\n", state.non_critical_count);
    fprintf(stdout, "\\n");
    fprintf(stdout, "================================================================================\\n");
    fprintf(stdout, "\\n");
}

// ============================================================================
// SELF-TEST
// ============================================================================

/**
 * Self-test: verify task taxonomy enforcement
 */
int llama_task_taxonomy_selftest(void) {
    fprintf(stdout, "[TASK TAXONOMY SELFTEST] Running...\\n");

    // Test 1: Initialization
    if (!g_task_taxonomy_state.taxonomy_initialized) {
        fprintf(stderr, "SELFTEST FAIL: Taxonomy should be initialized\\n");
        return -1;
    }

    // Test 2: Classification of known tasks
    enum llama_task_class cls_attn = llama_classify_task("attention_layer_5");
    if (cls_attn != LLAMA_TASK_CLASS_DECODE_CRITICAL) {
        fprintf(stderr, "SELFTEST FAIL: Attention should be DECODE_CRITICAL\\n");
        return -1;
    }

    enum llama_task_class cls_log = llama_classify_task("log_metric");
    if (cls_log != LLAMA_TASK_CLASS_NON_CRITICAL) {
        fprintf(stderr, "SELFTEST FAIL: Logging should be NON_CRITICAL\\n");
        return -1;
    }

    // Test 3: Exhaustive classification
    const char* test_tasks[] = {
        "forward_pass",
        "mlp_compute",
        "tokenize_input",
        "cache_write"
    };
    if (llama_task_verify_exhaustive_classification((const char**)test_tasks, 4) != 0) {
        fprintf(stderr, "SELFTEST FAIL: Exhaustive classification check failed\\n");
        return -1;
    }

    // Test 4: Backend binding validation
    int backend_check = llama_validate_task_backend_binding(
        LLAMA_TASK_CLASS_DECODE_CRITICAL,
        "CUDA");
    if (backend_check != 0) {
        fprintf(stderr, "SELFTEST FAIL: DECODE_CRITICAL+CUDA should be valid\\n");
        return -1;
    }

    backend_check = llama_validate_task_backend_binding(
        LLAMA_TASK_CLASS_DECODE_CRITICAL,
        "CPU");
    if (backend_check == 0) {
        fprintf(stderr, "SELFTEST FAIL: DECODE_CRITICAL+CPU should be invalid\\n");
        return -1;
    }

    fprintf(stdout, "[TASK TAXONOMY SELFTEST] PASSED\\n");
    return 0;
}
