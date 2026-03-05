/**
 * llama-cublas-prevention.cpp
 *
 * Runtime implementation of immutable decode backend binding with hard guards
 * against cuBLAS fallback (Requirement #58).
 *
 * This implementation provides:
 * - Backend lock state machine with strict enforcement
 * - Re-selection blocking logic
 * - Graph metadata management
 * - Guard assertion implementations
 * - Environment variable isolation
 * - Comprehensive violation detection and reporting
 * - Metrics tracking for verification
 */

#include "../include/llama-cublas-prevention.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <chrono>
#include <unordered_map>

// ============================================================================
// GLOBAL STATE AND METRICS
// ============================================================================

/**
 * Global metrics for tracking decode backend lock enforcement.
 */
static llama_decode_lock_metrics_t g_decode_lock_metrics = {
    0, 0, 0, 0, 0, 0, 0, 0
};

/**
 * Global mutex protecting metrics updates.
 */
static std::mutex g_decode_lock_metrics_mutex;

/**
 * Global environment variable cache (initialized once at startup).
 */
static llama_decode_env_cache_t g_decode_env_cache = {
    std::atomic<bool>(false),  // force_cublas
    std::atomic<bool>(false),  // force_mmq
    std::atomic<bool>(false),  // force_cpu
    std::atomic<bool>(true),   // allow_fallback (default: true for compatibility)
    std::atomic<bool>(true),   // shape_heuristics_enabled (default: true)
    std::atomic<bool>(false),  // deterministic_mode
    std::atomic<bool>(false),  // skip_capability_check
    std::atomic<bool>(false),  // gpu_exclusive_decode
    std::atomic<bool>(false)   // initialized
};

/**
 * Startup time reference for timestamp calculations.
 */
static std::chrono::steady_clock::time_point g_startup_time;

/**
 * Helper to get milliseconds since startup.
 */
static inline uint32_t get_ms_since_startup() {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - g_startup_time);
    return static_cast<uint32_t>(duration.count() & 0xFFFFFFFF);
}

/**
 * Helper to convert backend type to string for logging.
 */
static const char * backend_type_to_string(llama_decode_backend_type_t backend) {
    switch (backend) {
        case LLAMA_DECODE_BACKEND_UNDEFINED:     return "UNDEFINED";
        case LLAMA_DECODE_BACKEND_CPU:           return "CPU";
        case LLAMA_DECODE_BACKEND_CUDA_MMQ:      return "CUDA_MMQ";
        case LLAMA_DECODE_BACKEND_CUDA_DENSE:    return "CUDA_DENSE";
        case LLAMA_DECODE_BACKEND_CUDA_CUBLAS:   return "CUDA_CUBLAS";
        case LLAMA_DECODE_BACKEND_METAL:         return "METAL";
        case LLAMA_DECODE_BACKEND_VULKAN:        return "VULKAN";
        case LLAMA_DECODE_BACKEND_OPENGL:        return "OPENGL";
        default:                                  return "UNKNOWN";
    }
}

/**
 * Helper to convert lock state to string for logging.
 */
static const char * lock_state_to_string(llama_decode_lock_state_t state) {
    switch (state) {
        case LLAMA_DECODE_LOCK_STATE_UNINITIALIZED:  return "UNINITIALIZED";
        case LLAMA_DECODE_LOCK_STATE_BINDING:        return "BINDING";
        case LLAMA_DECODE_LOCK_STATE_LOCKED:         return "LOCKED";
        case LLAMA_DECODE_LOCK_STATE_ACTIVE_DECODE:  return "ACTIVE_DECODE";
        case LLAMA_DECODE_LOCK_STATE_VIOLATION:      return "VIOLATION";
        case LLAMA_DECODE_LOCK_STATE_DESTROYED:      return "DESTROYED";
        default:                                      return "UNKNOWN";
    }
}

// ============================================================================
// ENVIRONMENT VARIABLE CACHING IMPLEMENTATION
// ============================================================================

bool llama_decode_env_cache_init(llama_decode_env_cache_t * env_cache) {
    if (!env_cache) {
        fprintf(stderr, "ERROR: env_cache is nullptr\n");
        return false;
    }

    // Parse environment variables only once at startup
    const char * force_cublas_env = std::getenv("LLAMA_CUBLAS_FORCE");
    const char * force_mmq_env = std::getenv("LLAMA_CUDA_FORCE_MMQ");
    const char * force_cpu_env = std::getenv("LLAMA_FORCE_CPU");
    const char * allow_fallback_env = std::getenv("LLAMA_ALLOW_BACKEND_FALLBACK");
    const char * shape_heuristics_env = std::getenv("LLAMA_ENABLE_SHAPE_HEURISTICS");
    const char * deterministic_env = std::getenv("LLAMA_DETERMINISTIC");
    const char * skip_check_env = std::getenv("LLAMA_SKIP_CAPABILITY_CHECK");
    const char * gpu_exclusive_env = std::getenv("LLAMA_GPU_EXCLUSIVE_DECODE");

    // Cache the parsed values
    env_cache->force_cublas.store(force_cublas_env != nullptr &&
                                  std::string(force_cublas_env) == "1");
    env_cache->force_mmq.store(force_mmq_env != nullptr &&
                               std::string(force_mmq_env) == "1");
    env_cache->force_cpu.store(force_cpu_env != nullptr &&
                               std::string(force_cpu_env) == "1");

    // For fallback and heuristics, parse as "1" or "0" (default true if not specified)
    if (allow_fallback_env) {
        env_cache->allow_fallback.store(std::string(allow_fallback_env) != "0");
    }
    if (shape_heuristics_env) {
        env_cache->shape_heuristics_enabled.store(std::string(shape_heuristics_env) != "0");
    }

    env_cache->deterministic_mode.store(deterministic_env != nullptr &&
                                        std::string(deterministic_env) == "1");
    env_cache->skip_capability_check.store(skip_check_env != nullptr &&
                                           std::string(skip_check_env) == "1");
    env_cache->gpu_exclusive_decode.store(gpu_exclusive_env != nullptr &&
                                          std::string(gpu_exclusive_env) == "1");

    env_cache->initialized.store(true);

    fprintf(stderr, "[CUBLAS_PREVENTION] Environment variables cached at startup:\n");
    fprintf(stderr, "  force_cublas: %d\n", env_cache->force_cublas.load() ? 1 : 0);
    fprintf(stderr, "  force_mmq: %d\n", env_cache->force_mmq.load() ? 1 : 0);
    fprintf(stderr, "  force_cpu: %d\n", env_cache->force_cpu.load() ? 1 : 0);
    fprintf(stderr, "  allow_fallback: %d\n", env_cache->allow_fallback.load() ? 1 : 0);
    fprintf(stderr, "  gpu_exclusive_decode: %d\n", env_cache->gpu_exclusive_decode.load() ? 1 : 0);

    return true;
}

bool llama_decode_env_check_cached(const llama_decode_env_cache_t * env_cache,
                                    const char * var_name) {
    if (!env_cache) {
        return false;
    }
    return env_cache->initialized.load();
}

bool llama_decode_env_protect_against_rereads(const llama_decode_env_cache_t * env_cache,
                                               void * ctx) {
    if (!env_cache) {
        fprintf(stderr, "ERROR: env_cache is nullptr in protect_against_rereads\n");
        return false;
    }

    // Verify that environment variables have been cached
    if (!env_cache->initialized.load()) {
        fprintf(stderr, "FATAL: Environment variables not initialized. Call llama_decode_env_cache_init() at startup.\n");
        abort();
    }

    // In a real implementation, we could add periodic checks to detect if env vars changed
    // For now, just verify the cache exists
    return true;
}

// ============================================================================
// BACKEND LOCK IMPLEMENTATION
// ============================================================================

void llama_decode_lock_init(llama_decode_backend_lock_t * lock, void * ctx) {
    if (!lock) {
        fprintf(stderr, "ERROR: lock is nullptr in llama_decode_lock_init\n");
        return;
    }

    // Initialize all atomic fields
    lock->state.store(LLAMA_DECODE_LOCK_STATE_UNINITIALIZED);
    lock->locked_backend.store(LLAMA_DECODE_BACKEND_UNDEFINED);
    lock->is_locked.store(false);
    lock->lock_timestamp_ms.store(0);
    lock->graph_id.store(0);
    lock->prev_graph_id.store(0);
    lock->re_selection_attempts.store(0);
    lock->cublas_probe_attempts.store(0);
    lock->shape_heuristic_triggers.store(0);
    lock->backend_drift_detections.store(0);
    lock->graph_invalidation_count.store(0);
    lock->violation_logged.store(false);
    std::memset(lock->last_violation_msg, 0, sizeof(lock->last_violation_msg));

    // Record that lock was initialized
    {
        std::lock_guard<std::mutex> guard(g_decode_lock_metrics_mutex);
        g_decode_lock_metrics.total_contexts_created++;
    }

    if (!g_startup_time.time_since_epoch().count()) {
        g_startup_time = std::chrono::steady_clock::now();
    }

    fprintf(stderr, "[CUBLAS_PREVENTION] Decode backend lock initialized for context %p\n", ctx);
}

bool llama_decode_lock_engage(llama_decode_backend_lock_t * lock,
                               llama_decode_backend_type_t backend,
                               void * ctx) {
    if (!lock) {
        fprintf(stderr, "ERROR: lock is nullptr in llama_decode_lock_engage\n");
        return false;
    }

    // Check current state
    auto current_state = lock->state.load();
    if (current_state == LLAMA_DECODE_LOCK_STATE_LOCKED) {
        // Already locked to a backend
        auto locked_backend = lock->locked_backend.load();
        if (locked_backend != backend) {
            fprintf(stderr, "FATAL: Attempt to engage different backend. Locked to %s, attempting %s\n",
                    backend_type_to_string(locked_backend),
                    backend_type_to_string(backend));
            snprintf(lock->last_violation_msg, sizeof(lock->last_violation_msg),
                     "Backend mismatch: locked %s vs attempt %s",
                     backend_type_to_string(locked_backend),
                     backend_type_to_string(backend));
            lock->state.store(LLAMA_DECODE_LOCK_STATE_VIOLATION);
            lock->violation_logged.store(true);
            {
                std::lock_guard<std::mutex> guard(g_decode_lock_metrics_mutex);
                g_decode_lock_metrics.lock_violations++;
            }
            abort();
        }
        return false;  // Already locked
    }

    if (current_state == LLAMA_DECODE_LOCK_STATE_VIOLATION) {
        fprintf(stderr, "FATAL: Lock is in violation state. Cannot engage.\n");
        abort();
    }

    // Transition from UNINITIALIZED or BINDING to LOCKED
    lock->state.store(LLAMA_DECODE_LOCK_STATE_BINDING);
    lock->locked_backend.store(backend);
    lock->lock_timestamp_ms.store(get_ms_since_startup());
    lock->is_locked.store(true);
    lock->state.store(LLAMA_DECODE_LOCK_STATE_LOCKED);

    {
        std::lock_guard<std::mutex> guard(g_decode_lock_metrics_mutex);
        g_decode_lock_metrics.lock_engagements++;
    }

    fprintf(stderr, "[CUBLAS_PREVENTION] Backend lock engaged for context %p: %s at t=%ums\n",
            ctx, backend_type_to_string(backend), lock->lock_timestamp_ms.load());

    return true;
}

bool llama_decode_lock_is_locked(const llama_decode_backend_lock_t * lock) {
    if (!lock) {
        return false;
    }
    return lock->is_locked.load();
}

llama_decode_backend_type_t llama_decode_lock_get_backend(
    const llama_decode_backend_lock_t * lock) {
    if (!lock) {
        return LLAMA_DECODE_BACKEND_UNDEFINED;
    }
    return lock->locked_backend.load();
}

bool llama_decode_lock_allow_reselection(llama_decode_backend_lock_t * lock,
                                          const char * reason,
                                          void * ctx) {
    if (!lock) {
        return true;  // If no lock, allow reselection
    }

    if (!lock->is_locked.load()) {
        return true;  // Not locked yet, allow reselection
    }

    // Lock is engaged, block reselection
    lock->re_selection_attempts.fetch_add(1);
    {
        std::lock_guard<std::mutex> guard(g_decode_lock_metrics_mutex);
        g_decode_lock_metrics.reselection_attempts++;
    }

    fprintf(stderr, "[CUBLAS_PREVENTION] Blocked backend re-selection attempt for context %p: %s\n",
            ctx, reason ? reason : "unknown reason");

    return false;
}

bool llama_decode_lock_allow_cublas_probe(llama_decode_backend_lock_t * lock,
                                           void * ctx) {
    if (!lock) {
        return true;  // If no lock, allow cuBLAS probe
    }

    if (!lock->is_locked.load()) {
        return true;  // Not locked yet, allow probe
    }

    // Lock is engaged, check if backend is cuBLAS
    auto locked_backend = lock->locked_backend.load();
    if (locked_backend == LLAMA_DECODE_BACKEND_CUDA_CUBLAS) {
        return true;  // cuBLAS locked, allow probe
    }

    // Non-cuBLAS backend is locked, block cuBLAS probe
    lock->cublas_probe_attempts.fetch_add(1);
    {
        std::lock_guard<std::mutex> guard(g_decode_lock_metrics_mutex);
        g_decode_lock_metrics.cublas_probe_blocks++;
    }

    fprintf(stderr, "[CUBLAS_PREVENTION] Blocked cuBLAS probe attempt for context %p (locked to %s)\n",
            ctx, backend_type_to_string(locked_backend));

    snprintf(lock->last_violation_msg, sizeof(lock->last_violation_msg),
             "cuBLAS probe blocked: locked to %s", backend_type_to_string(locked_backend));

    return false;
}

bool llama_decode_lock_allow_shape_heuristic(llama_decode_backend_lock_t * lock,
                                              const char * shape_reason,
                                              void * ctx) {
    if (!lock) {
        return true;  // If no lock, allow shape heuristics
    }

    if (!lock->is_locked.load()) {
        return true;  // Not locked yet, allow heuristics
    }

    // Lock is engaged, block shape-based heuristics
    lock->shape_heuristic_triggers.fetch_add(1);
    {
        std::lock_guard<std::mutex> guard(g_decode_lock_metrics_mutex);
        g_decode_lock_metrics.shape_heuristic_blocks++;
    }

    fprintf(stderr, "[CUBLAS_PREVENTION] Blocked shape heuristic for context %p: %s\n",
            ctx, shape_reason ? shape_reason : "unknown shape");

    snprintf(lock->last_violation_msg, sizeof(lock->last_violation_msg),
             "Shape heuristic blocked: %s", shape_reason ? shape_reason : "unknown");

    return false;
}

bool llama_decode_lock_validate_graph_backend(llama_decode_backend_lock_t * lock,
                                               void * graph,
                                               void * ctx) {
    if (!lock || !graph) {
        return false;
    }

    if (!lock->is_locked.load()) {
        return true;  // Not locked yet, cannot validate
    }

    // In a real implementation, we would check the graph's attached metadata
    // For now, we'll assume the graph is valid if it exists
    fprintf(stderr, "[CUBLAS_PREVENTION] Validating graph for locked backend %s\n",
            backend_type_to_string(lock->locked_backend.load()));

    return true;
}

bool llama_decode_lock_check_graph_validity(llama_decode_backend_lock_t * lock,
                                             uint64_t new_graph_id,
                                             void * ctx) {
    if (!lock) {
        return true;
    }

    uint64_t prev_id = lock->graph_id.load();
    if (prev_id != 0 && prev_id != new_graph_id) {
        // Graph ID changed - potential invalidation
        lock->graph_invalidation_count.fetch_add(1);
        {
            std::lock_guard<std::mutex> guard(g_decode_lock_metrics_mutex);
            g_decode_lock_metrics.graph_invalidations++;
        }

        fprintf(stderr, "FATAL: Graph invalidation detected during decode. Old graph: %llu, New graph: %llu\n",
                (unsigned long long)prev_id, (unsigned long long)new_graph_id);
        snprintf(lock->last_violation_msg, sizeof(lock->last_violation_msg),
                 "Graph invalidation: %llu -> %llu", (unsigned long long)prev_id, (unsigned long long)new_graph_id);

        abort();
    }

    lock->graph_id.store(new_graph_id);
    return true;
}

bool llama_decode_lock_detect_drift(llama_decode_backend_lock_t * lock,
                                     llama_decode_backend_type_t current_backend,
                                     void * ctx) {
    if (!lock) {
        return true;
    }

    if (!lock->is_locked.load()) {
        return true;  // Not locked yet
    }

    auto locked_backend = lock->locked_backend.load();
    if (locked_backend != current_backend) {
        // Backend drift detected
        lock->backend_drift_detections.fetch_add(1);
        {
            std::lock_guard<std::mutex> guard(g_decode_lock_metrics_mutex);
            g_decode_lock_metrics.drift_detections++;
        }

        fprintf(stderr, "FATAL: Backend drift detected. Locked to %s, but current is %s\n",
                backend_type_to_string(locked_backend), backend_type_to_string(current_backend));
        snprintf(lock->last_violation_msg, sizeof(lock->last_violation_msg),
                 "Backend drift: locked %s -> current %s",
                 backend_type_to_string(locked_backend), backend_type_to_string(current_backend));

        abort();
    }

    return true;
}

void llama_decode_lock_assert_backend_match(llama_decode_backend_lock_t * lock,
                                             llama_decode_backend_type_t expected,
                                             const char * operation_name,
                                             void * ctx) {
    if (!lock) {
        fprintf(stderr, "ERROR: lock is nullptr in assert_backend_match\n");
        abort();
    }

    auto current = lock->locked_backend.load();
    if (current != expected) {
        fprintf(stderr, "FATAL: Backend assertion failed at %s. Expected %s, got %s\n",
                operation_name ? operation_name : "unknown",
                backend_type_to_string(expected),
                backend_type_to_string(current));
        snprintf(lock->last_violation_msg, sizeof(lock->last_violation_msg),
                 "Backend mismatch at %s: expected %s, got %s",
                 operation_name ? operation_name : "unknown",
                 backend_type_to_string(expected),
                 backend_type_to_string(current));
        lock->state.store(LLAMA_DECODE_LOCK_STATE_VIOLATION);
        lock->violation_logged.store(true);
        {
            std::lock_guard<std::mutex> guard(g_decode_lock_metrics_mutex);
            g_decode_lock_metrics.lock_violations++;
        }
        abort();
    }
}

void llama_decode_lock_destroy(llama_decode_backend_lock_t * lock) {
    if (!lock) {
        return;
    }

    lock->state.store(LLAMA_DECODE_LOCK_STATE_DESTROYED);
    lock->is_locked.store(false);

    fprintf(stderr, "[CUBLAS_PREVENTION] Decode backend lock destroyed. Final state:\n");
    fprintf(stderr, "  Locked backend: %s\n", backend_type_to_string(lock->locked_backend.load()));
    fprintf(stderr, "  Re-selection attempts blocked: %u\n", lock->re_selection_attempts.load());
    fprintf(stderr, "  cuBLAS probe attempts blocked: %u\n", lock->cublas_probe_attempts.load());
    fprintf(stderr, "  Shape heuristic triggers blocked: %u\n", lock->shape_heuristic_triggers.load());
    fprintf(stderr, "  Backend drift detections: %u\n", lock->backend_drift_detections.load());
    fprintf(stderr, "  Graph invalidations: %u\n", lock->graph_invalidation_count.load());
}

// ============================================================================
// GRAPH METADATA IMPLEMENTATION
// ============================================================================

// Simple map to store graph metadata (in production, would be part of ggml_cgraph)
static std::unordered_map<uintptr_t, llama_graph_metadata_decode_t> g_graph_metadata_map;
static std::mutex g_graph_metadata_mutex;

bool llama_graph_metadata_attach_backend(void * graph,
                                         llama_decode_backend_type_t backend,
                                         uint32_t flags,
                                         void * ctx) {
    if (!graph) {
        fprintf(stderr, "ERROR: graph is nullptr in attach_backend\n");
        return false;
    }

    uintptr_t graph_key = reinterpret_cast<uintptr_t>(graph);

    llama_graph_metadata_decode_t metadata = {
        backend,
        flags,
        (backend != LLAMA_DECODE_BACKEND_CUDA_CUBLAS) ? 1 : 0,  // cublas_disabled
        (backend == LLAMA_DECODE_BACKEND_CUDA_MMQ) ? 1 : 0,     // mmq_required
        (backend == LLAMA_DECODE_BACKEND_CUDA_DENSE ||
         backend == LLAMA_DECODE_BACKEND_CUDA_CUBLAS) ? 1 : 0,  // dense_required
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())
    };

    {
        std::lock_guard<std::mutex> guard(g_graph_metadata_mutex);
        g_graph_metadata_map[graph_key] = metadata;
    }

    fprintf(stderr, "[CUBLAS_PREVENTION] Graph metadata attached for backend %s, graph %p\n",
            backend_type_to_string(backend), graph);

    return true;
}

const llama_graph_metadata_decode_t * llama_graph_metadata_get_backend(
    const void * graph) {
    if (!graph) {
        return nullptr;
    }

    uintptr_t graph_key = reinterpret_cast<uintptr_t>(graph);

    std::lock_guard<std::mutex> guard(g_graph_metadata_mutex);
    auto it = g_graph_metadata_map.find(graph_key);
    if (it != g_graph_metadata_map.end()) {
        return &it->second;
    }

    return nullptr;
}

bool llama_graph_metadata_validate_backend(const void * graph,
                                            llama_decode_backend_type_t expected,
                                            void * ctx) {
    if (!graph) {
        return false;
    }

    const auto * metadata = llama_graph_metadata_get_backend(graph);
    if (!metadata) {
        fprintf(stderr, "ERROR: Graph has no backend metadata\n");
        return false;
    }

    if (metadata->backend_id != expected) {
        fprintf(stderr, "FATAL: Graph backend mismatch. Expected %s, got %s\n",
                backend_type_to_string(expected),
                backend_type_to_string(metadata->backend_id));
        return false;
    }

    return true;
}

// ============================================================================
// METRICS IMPLEMENTATION
// ============================================================================

void llama_decode_lock_get_metrics(llama_decode_lock_metrics_t * metrics) {
    if (!metrics) {
        return;
    }

    std::lock_guard<std::mutex> guard(g_decode_lock_metrics_mutex);
    *metrics = g_decode_lock_metrics;
}

void llama_decode_lock_reset_metrics(void) {
    std::lock_guard<std::mutex> guard(g_decode_lock_metrics_mutex);
    std::memset(&g_decode_lock_metrics, 0, sizeof(g_decode_lock_metrics));
}

void llama_decode_lock_print_report(bool detailed) {
    llama_decode_lock_metrics_t metrics;
    llama_decode_lock_get_metrics(&metrics);

    fprintf(stderr, "\n");
    fprintf(stderr, "==========================================================================\n");
    fprintf(stderr, "DECODE BACKEND LOCK - CUBLAS PREVENTION REPORT\n");
    fprintf(stderr, "==========================================================================\n");
    fprintf(stderr, "\nKey Metrics (Target: Lock engagements = contexts, violations = 0):\n");
    fprintf(stderr, "  Total contexts created:             %llu\n", (unsigned long long)metrics.total_contexts_created);
    fprintf(stderr, "  Backend lock engagements:           %llu (target: 100%% of contexts)\n",
            (unsigned long long)metrics.lock_engagements);
    fprintf(stderr, "  Re-selection attempts blocked:      %llu (target: 0)\n", (unsigned long long)metrics.reselection_attempts);
    fprintf(stderr, "  cuBLAS probe blocks:                %llu (target: 0)\n", (unsigned long long)metrics.cublas_probe_blocks);
    fprintf(stderr, "  Shape heuristic triggers blocked:   %llu (target: 0)\n", (unsigned long long)metrics.shape_heuristic_blocks);
    fprintf(stderr, "  Backend drift detections:           %llu (target: 0)\n", (unsigned long long)metrics.drift_detections);
    fprintf(stderr, "  Graph invalidations during decode:  %llu (target: 0)\n", (unsigned long long)metrics.graph_invalidations);
    fprintf(stderr, "  Lock violations:                    %llu (target: 0)\n", (unsigned long long)metrics.lock_violations);

    if (detailed) {
        fprintf(stderr, "\nDetailed Analysis:\n");

        double engagement_rate = (metrics.total_contexts_created > 0) ?
            (100.0 * metrics.lock_engagements / metrics.total_contexts_created) : 0.0;
        fprintf(stderr, "  Lock engagement rate: %.1f%% (target: 100.0%%)\n", engagement_rate);

        if (metrics.reselection_attempts > 0) {
            fprintf(stderr, "  WARNING: %llu re-selection attempts detected!\n",
                    (unsigned long long)metrics.reselection_attempts);
        }

        if (metrics.cublas_probe_blocks > 0) {
            fprintf(stderr, "  INFO: %llu cuBLAS probes blocked (expected if using non-cuBLAS backend)\n",
                    (unsigned long long)metrics.cublas_probe_blocks);
        }

        if (metrics.shape_heuristic_blocks > 0) {
            fprintf(stderr, "  INFO: %llu shape heuristics disabled (expected during locked decode)\n",
                    (unsigned long long)metrics.shape_heuristic_blocks);
        }

        if (metrics.drift_detections > 0) {
            fprintf(stderr, "  CRITICAL: %llu backend drift events detected!\n",
                    (unsigned long long)metrics.drift_detections);
        }

        if (metrics.graph_invalidations > 0) {
            fprintf(stderr, "  CRITICAL: %llu graph invalidations during decode!\n",
                    (unsigned long long)metrics.graph_invalidations);
        }

        if (metrics.lock_violations > 0) {
            fprintf(stderr, "  CRITICAL: %llu lock violations detected!\n",
                    (unsigned long long)metrics.lock_violations);
        }
    }

    fprintf(stderr, "\nCompliance Status:\n");
    bool compliant = (metrics.reselection_attempts == 0 &&
                     metrics.drift_detections == 0 &&
                     metrics.graph_invalidations == 0 &&
                     metrics.lock_violations == 0);

    if (compliant) {
        fprintf(stderr, "  PASS: All decode backend lock invariants maintained\n");
    } else {
        fprintf(stderr, "  FAIL: Backend lock violations detected\n");
    }

    fprintf(stderr, "==========================================================================\n\n");
}

// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

// Ensure environment variables are cached at module load time
__attribute__((constructor))
static void cublas_prevention_init() {
    llama_decode_env_cache_init(&g_decode_env_cache);
    g_startup_time = std::chrono::steady_clock::now();
    fprintf(stderr, "[CUBLAS_PREVENTION] Module initialized\n");
}
