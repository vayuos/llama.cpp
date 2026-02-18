/**
 * SECTION 2: Introduce Decode-Critical vs Non-Critical Task Taxonomy
 *
 * This file implements the foundational task classification system for GPU-exclusive decode.
 * All work in the system must be classified as either DECODE_CRITICAL or NON_CRITICAL.
 * This classification is STATIC, EXPLICIT, and IRREVERSIBLE.
 *
 * CORE PRINCIPLE:
 *   DECODE_CRITICAL tasks gate next-token emission → GPU only, latency-critical
 *   NON_CRITICAL tasks do not affect tokens/sec → CPU only, can scale independently
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

// ============================================================================
// TASK CLASS DEFINITION
// ============================================================================

/**
 * Enum defining the two fundamental task classes in the system
 *
 * This is the central taxonomy. All work must be classified into one of these two classes.
 */
enum llama_task_class {
    LLAMA_TASK_CLASS_UNKNOWN = 0,       // Not yet classified (invalid)
    LLAMA_TASK_CLASS_DECODE_CRITICAL = 1,  // Latency-critical, gates token emission → GPU only
    LLAMA_TASK_CLASS_NON_CRITICAL = 2,     // Does not affect tokens/sec → CPU only
};

/**
 * Convert task class enum to human-readable string
 */
static inline const char* llama_task_class_name(enum llama_task_class cls) {
    switch (cls) {
        case LLAMA_TASK_CLASS_UNKNOWN:
            return "UNKNOWN";
        case LLAMA_TASK_CLASS_DECODE_CRITICAL:
            return "DECODE_CRITICAL";
        case LLAMA_TASK_CLASS_NON_CRITICAL:
            return "NON_CRITICAL";
        default:
            return "(invalid)";
    }
}

// ============================================================================
// EXHAUSTIVE DECODE-CRITICAL TASK CLASSIFICATION
// ============================================================================

/**
 * Static classifier for decode-critical tasks
 *
 * DECODE_CRITICAL tasks are those that:
 * - Gate next-token emission
 * - Must complete before the next token can be generated
 * - Directly affect tokens/sec
 * - Must execute on GPU
 */
struct llama_decode_critical_classifier {
    // Transformer forward pass (all layers)
    static bool is_forward_pass(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "forward") != nullptr ||
               strstr(task_name, "layer") != nullptr ||
               strstr(task_name, "transformer") != nullptr;
    }

    // Attention computation
    static bool is_attention(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "attention") != nullptr ||
               strstr(task_name, "attn") != nullptr ||
               strstr(task_name, "qkv") != nullptr ||
               strstr(task_name, "multi_head") != nullptr;
    }

    // MLP computation
    static bool is_mlp(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "mlp") != nullptr ||
               strstr(task_name, "feed_forward") != nullptr ||
               strstr(task_name, "ffn") != nullptr;
    }

    // KV cache reads and writes
    static bool is_kv_cache_op(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "kv_cache") != nullptr ||
               strstr(task_name, "cache_read") != nullptr ||
               strstr(task_name, "cache_write") != nullptr ||
               strstr(task_name, "cache_append") != nullptr;
    }

    // Logits computation
    static bool is_logits(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "logits") != nullptr ||
               strstr(task_name, "lm_head") != nullptr;
    }

    // Sampling / argmax / token selection
    static bool is_sampling(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "sample") != nullptr ||
               strstr(task_name, "argmax") != nullptr ||
               strstr(task_name, "top_k") != nullptr ||
               strstr(task_name, "top_p") != nullptr ||
               strstr(task_name, "token_select") != nullptr;
    }

    // Termination checks dependent on token output
    static bool is_termination_check(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "termination") != nullptr ||
               strstr(task_name, "eos_check") != nullptr ||
               strstr(task_name, "stop_check") != nullptr;
    }

    // COMPLETE DECODE-CRITICAL CLASSIFICATION
    static bool is_decode_critical(const char* task_name) {
        return is_forward_pass(task_name) ||
               is_attention(task_name) ||
               is_mlp(task_name) ||
               is_kv_cache_op(task_name) ||
               is_logits(task_name) ||
               is_sampling(task_name) ||
               is_termination_check(task_name);
    }
};

// ============================================================================
// EXHAUSTIVE NON-CRITICAL TASK CLASSIFICATION
// ============================================================================

/**
 * Static classifier for non-critical tasks
 *
 * NON_CRITICAL tasks are those that:
 * - Do NOT gate next-token emission
 * - Do NOT affect tokens/sec
 * - Can execute on CPU
 * - Must never block decode progression
 */
struct llama_non_critical_classifier {
    // Tokenization and prompt preprocessing
    static bool is_tokenization(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "tokenize") != nullptr ||
               strstr(task_name, "encode") != nullptr ||
               strstr(task_name, "preprocess") != nullptr;
    }

    // Request parsing and routing
    static bool is_request_handling(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "parse_request") != nullptr ||
               strstr(task_name, "route_request") != nullptr ||
               strstr(task_name, "request_queue") != nullptr;
    }

    // Logging and metrics
    static bool is_logging(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "log") != nullptr ||
               strstr(task_name, "metric") != nullptr ||
               strstr(task_name, "stats") != nullptr ||
               strstr(task_name, "telemetry") != nullptr;
    }

    // I/O operations (disk, network)
    static bool is_io(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "io_") != nullptr ||
               strstr(task_name, "read_disk") != nullptr ||
               strstr(task_name, "write_disk") != nullptr ||
               strstr(task_name, "network") != nullptr ||
               strstr(task_name, "http_request") != nullptr;
    }

    // RAG retrieval
    static bool is_rag(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "rag_retrieve") != nullptr ||
               strstr(task_name, "vector_search") != nullptr ||
               strstr(task_name, "retrieval") != nullptr;
    }

    // Embedding database lookups
    static bool is_embedding_lookup(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "embedding_lookup") != nullptr ||
               strstr(task_name, "vector_db") != nullptr ||
               strstr(task_name, "index_query") != nullptr;
    }

    // Background housekeeping (KV eviction, cache cleanup)
    static bool is_housekeeping(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "kv_evict") != nullptr ||
               strstr(task_name, "cache_cleanup") != nullptr ||
               strstr(task_name, "housekeeping") != nullptr ||
               strstr(task_name, "garbage_collect") != nullptr;
    }

    // Admission control and scheduling
    static bool is_admission_control(const char* task_name) {
        if (!task_name) return false;
        return strstr(task_name, "admission") != nullptr ||
               strstr(task_name, "schedule") != nullptr ||
               strstr(task_name, "queue_manage") != nullptr;
    }

    // COMPLETE NON-CRITICAL CLASSIFICATION
    static bool is_non_critical(const char* task_name) {
        return is_tokenization(task_name) ||
               is_request_handling(task_name) ||
               is_logging(task_name) ||
               is_io(task_name) ||
               is_rag(task_name) ||
               is_embedding_lookup(task_name) ||
               is_housekeeping(task_name) ||
               is_admission_control(task_name);
    }
};

// ============================================================================
// TASK TAXONOMY ENFORCEMENT
// ============================================================================

/**
 * Classify a task exhaustively
 * Returns the task class, or UNKNOWN if classification fails
 */
static inline enum llama_task_class llama_classify_task(const char* task_name) {
    if (!task_name) {
        return LLAMA_TASK_CLASS_UNKNOWN;
    }

    // Check decode-critical first (stricter, higher priority)
    if (llama_decode_critical_classifier::is_decode_critical(task_name)) {
        return LLAMA_TASK_CLASS_DECODE_CRITICAL;
    }

    // Then check non-critical
    if (llama_non_critical_classifier::is_non_critical(task_name)) {
        return LLAMA_TASK_CLASS_NON_CRITICAL;
    }

    // No match in either category
    return LLAMA_TASK_CLASS_UNKNOWN;
}

// ============================================================================
// STATIC TASK METADATA
// ============================================================================

/**
 * Task metadata: class is static, explicit, irreversible
 */
struct llama_task_metadata {
    const char* task_name;              // Task identifier
    enum llama_task_class task_class;   // Classification (static, assigned at creation time)
    bool class_is_locked;               // True if classification is locked (irreversible)
};

/**
 * Lock task classification (make it irreversible)
 */
static inline void llama_task_lock_classification(struct llama_task_metadata* meta) {
    if (meta) {
        meta->class_is_locked = true;
    }
}

/**
 * Attempt to reclassify a task (should fail if locked)
 */
static inline int llama_task_attempt_reclassify(
    struct llama_task_metadata* meta,
    enum llama_task_class new_class
) {
    if (!meta) {
        return -1;
    }

    if (meta->class_is_locked) {
        fprintf(stderr, "ERROR: Task '%s' classification is locked (irreversible)\n",
                meta->task_name ? meta->task_name : "(unnamed)");
        return -1;
    }

    if (new_class == meta->task_class) {
        return 0;  // No change
    }

    fprintf(stderr, "ERROR: Attempting to reclassify task '%s' from %s to %s (forbidden)\n",
            meta->task_name ? meta->task_name : "(unnamed)",
            llama_task_class_name(meta->task_class),
            llama_task_class_name(new_class));
    return -1;
}

// ============================================================================
// BACKEND BINDING ENFORCEMENT
// ============================================================================

/**
 * Enforce the binding between task class and backend
 * DECODE_CRITICAL → GPU only
 * NON_CRITICAL → CPU only
 */
static inline int llama_validate_task_backend_binding(
    enum llama_task_class task_class,
    const char* backend_name
) {
    if (!backend_name) {
        return -1;
    }

    const bool is_gpu = (strcmp(backend_name, "CUDA") == 0 ||
                         strcmp(backend_name, "GPU") == 0 ||
                         strcmp(backend_name, "Metal") == 0 ||
                         strcmp(backend_name, "OpenCL") == 0);

    const bool is_cpu = (strcmp(backend_name, "CPU") == 0 ||
                         strcmp(backend_name, "CPP") == 0);

    switch (task_class) {
        case LLAMA_TASK_CLASS_DECODE_CRITICAL:
            if (!is_gpu) {
                fprintf(stderr, "ERROR: DECODE_CRITICAL task assigned to CPU backend '%s' (must be GPU)\n",
                        backend_name);
                return -1;
            }
            return 0;

        case LLAMA_TASK_CLASS_NON_CRITICAL:
            if (!is_cpu) {
                fprintf(stderr, "WARNING: NON_CRITICAL task assigned to GPU backend '%s' (should be CPU)\n",
                        backend_name);
                // Non-critical on GPU is allowed (not optimal but not forbidden)
            }
            return 0;

        case LLAMA_TASK_CLASS_UNKNOWN:
            fprintf(stderr, "ERROR: Task has UNKNOWN class (must be explicitly classified)\n");
            return -1;

        default:
            return -1;
    }
}

// ============================================================================
// QUEUE MANAGEMENT
// ============================================================================

/**
 * Queue assignment based on task class
 */
static inline const char* llama_get_task_queue(enum llama_task_class task_class) {
    switch (task_class) {
        case LLAMA_TASK_CLASS_DECODE_CRITICAL:
            return "GPU_QUEUE";  // Decode-critical tasks go to GPU queue
        case LLAMA_TASK_CLASS_NON_CRITICAL:
            return "CPU_QUEUE";  // Non-critical tasks go to CPU queue
        default:
            return "UNKNOWN_QUEUE";
    }
}

// ============================================================================
// VALIDATION AND ASSERTIONS
// ============================================================================

/**
 * Assert that a decode-critical task never executes on CPU
 */
static inline int llama_assert_decode_critical_not_cpu(
    enum llama_task_class task_class,
    const char* backend_name
) {
    if (task_class == LLAMA_TASK_CLASS_DECODE_CRITICAL) {
        const bool is_cpu = (strcmp(backend_name, "CPU") == 0 ||
                             strcmp(backend_name, "CPP") == 0);
        if (is_cpu) {
            fprintf(stderr, "FATAL: DECODE_CRITICAL task executed on CPU backend (invariant violation)\n");
            return -1;
        }
    }
    return 0;
}

/**
 * Assert that a non-critical task does not block decode
 * (Validation hook: can be disabled in production)
 */
static inline int llama_assert_non_critical_does_not_block(
    enum llama_task_class task_class,
    bool is_blocking_decode
) {
    if (task_class == LLAMA_TASK_CLASS_NON_CRITICAL && is_blocking_decode) {
        fprintf(stderr, "WARNING: NON_CRITICAL task is blocking decode progression\n");
        return -1;
    }
    return 0;
}

// ============================================================================
// EXPLICIT TAXONOMY STATEMENT
// ============================================================================

/**
 * Print the task taxonomy statement explicitly
 */
static inline void llama_print_task_taxonomy_statement(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "TASK TAXONOMY (Section 2)\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "ALL WORK IN THE SYSTEM IS CLASSIFIED INTO TWO CATEGORIES:\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "1. DECODE_CRITICAL (GPU-only, latency-critical)\n");
    fprintf(stdout, "   - Gates next-token emission\n");
    fprintf(stdout, "   - Must complete before next token generated\n");
    fprintf(stdout, "   - Directly affects tokens/sec\n");
    fprintf(stdout, "   - Exhaustive list:\n");
    fprintf(stdout, "     * Transformer forward pass (all layers)\n");
    fprintf(stdout, "     * Attention computation (QKV, scores, softmax, accumulation)\n");
    fprintf(stdout, "     * MLP computation\n");
    fprintf(stdout, "     * KV cache reads and writes\n");
    fprintf(stdout, "     * Logits computation\n");
    fprintf(stdout, "     * Sampling / argmax / token selection\n");
    fprintf(stdout, "     * Termination checks dependent on token output\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "2. NON_CRITICAL (CPU-only, can scale independently)\n");
    fprintf(stdout, "   - Does NOT gate next-token emission\n");
    fprintf(stdout, "   - Does NOT affect tokens/sec\n");
    fprintf(stdout, "   - Must never block decode\n");
    fprintf(stdout, "   - Exhaustive list:\n");
    fprintf(stdout, "     * Tokenization and prompt preprocessing\n");
    fprintf(stdout, "     * Request parsing and routing\n");
    fprintf(stdout, "     * Logging and metrics\n");
    fprintf(stdout, "     * I/O (disk, network)\n");
    fprintf(stdout, "     * RAG retrieval\n");
    fprintf(stdout, "     * Embedding database lookups\n");
    fprintf(stdout, "     * Background housekeeping (KV eviction, cache cleanup)\n");
    fprintf(stdout, "     * Admission control and scheduling\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "CLASSIFICATION RULES:\n");
    fprintf(stdout, "  - Classification is STATIC (assigned at task creation time)\n");
    fprintf(stdout, "  - Classification is EXPLICIT (must be stated before execution)\n");
    fprintf(stdout, "  - Classification is IRREVERSIBLE (cannot be reclassified)\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "BACKEND BINDING:\n");
    fprintf(stdout, "  - DECODE_CRITICAL → GPU queue only\n");
    fprintf(stdout, "  - NON_CRITICAL → CPU queue only\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "ENFORCEMENT:\n");
    fprintf(stdout, "  - Tasks must be classified before submission\n");
    fprintf(stdout, "  - Tasks must execute in their assigned queue\n");
    fprintf(stdout, "  - Reclassification attempts fail with hard error\n");
    fprintf(stdout, "  - Migration between queues is forbidden\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}



// ============================================================================
// STATE MANAGEMENT STRUCTS
// ============================================================================

/**
 * Global task taxonomy state structure
 */
struct llama_task_taxonomy_state {
    bool taxonomy_initialized;
    int total_tasks_classified;
    int decode_critical_count;
    int non_critical_count;
    bool enforce_irreversibility;
};

// ============================================================================
// API FUNCTION DECLARATIONS
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Initialization
void llama_task_taxonomy_init(void);
struct llama_task_taxonomy_state llama_get_task_taxonomy_state(void);

// Task Classification Tracking
int llama_task_record_classification(
    const char* task_name,
    enum llama_task_class task_class
);

// Classification Verification
int llama_task_verify_classification(
    const char* task_name,
    enum llama_task_class assigned_class
);

// Irreversibility Enforcement
int llama_task_strict_lock_classification(struct llama_task_metadata* meta);

// Backend Binding Validation
int llama_task_validate_backend_assignment(
    const char* task_name,
    enum llama_task_class task_class,
    const char* backend_name
);

// Exhaustive Verification
int llama_task_verify_exhaustive_classification(
    const char* task_names[],
    int num_tasks
);

// Queue Routing
const char* llama_task_get_assigned_queue(enum llama_task_class task_class);
int llama_task_validate_queue_routing(
    const char* task_name,
    enum llama_task_class task_class,
    const char* assigned_queue
);

// Assertions (Non-static versions used by other modules)
int llama_task_assert_decode_critical_gpu_only(
    enum llama_task_class task_class,
    const char* actual_backend
);

int llama_task_assert_non_critical_non_blocking(
    enum llama_task_class task_class,
    bool is_blocking_decode
);

// Statistics and Reporting
void llama_task_print_statistics(void);

// Self-Test
int llama_task_taxonomy_selftest(void);

#ifdef __cplusplus
}
#endif
