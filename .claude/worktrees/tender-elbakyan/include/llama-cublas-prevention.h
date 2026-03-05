/**
 * llama-cublas-prevention.h
 *
 * Immutable Decode Backend Binding with Hard Guards Against cuBLAS Fallback
 *
 * This header implements Requirement #58: Backend Lock Enforcement.
 * Provides a state machine-based mechanism to bind the decode backend at the
 * moment of first decode step and prevent any runtime re-selection, specifically
 * blocking cuBLAS from becoming active if a different backend (MMQ, dense CUDA)
 * was initially selected.
 *
 * Key Features:
 * - Decode-Backend Lock Flag: Set immutable after first decode step
 * - Backend Re-Selection Blocking: Skip capability probing during locked decode
 * - cuBLAS Path Disabling: Guard against cuBLAS activation when locked to other backend
 * - Shape-Based Fallback Removal: Disable GEMM heuristics during decode
 * - Graph Invalidation Prevention: Abort decode on graph changes
 * - MatMul Dispatch Assertions: Assert backend ownership at compute time
 * - Quantized Path Protection: Fail hard when MMQ unavailable, no FP16 fallback
 * - Environment Variable Isolation: Parse at startup, never during decode
 * - Graph Metadata Binding: Attach backend ID to graph for validation
 * - Backend Drift Detection: Detect and report unauthorized backend changes
 *
 * Enforcement Rules (11 total):
 * 1. Introduce Decode-Backend Lock Flag
 * 2. Block Backend Re-Selection in ggml-backend
 * 3. Disable cuBLAS Path When Decode Backend != cuBLAS
 * 4. Remove Shape-Based cuBLAS Rebinding
 * 5. Prevent cuBLAS Rebinding After Graph Invalidation
 * 6. Add Decode-Time Assertion in MatMul Dispatch
 * 7. Remove cuBLAS Fallback for Quantized Paths
 * 8. Audit Implicit cuBLAS Usage
 * 9. Lock Backend in Graph Metadata
 * 10. Remove Environment-Driven Switching During Decode
 * 11. Expected Result: Zero cuBLAS fallback capability
 *
 * Performance Targets:
 * - Backend lock engagement rate: 100% (zero misses on first decode)
 * - cuBLAS re-selection attempts: 0 per context lifetime
 * - Shape-triggered fallback attempts: 0
 * - Backend drift detections: 0
 * - Graph invalidation during decode: 0
 * - Environment variable re-reads during decode: 0
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <mutex>
#include <cstring>
#include <stdexcept>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BACKEND TYPES AND CONSTANTS
// ============================================================================

/**
 * Enumeration of supported backends for decode operations.
 * Used to identify which backend is locked for the decode operation.
 */
typedef enum {
    LLAMA_DECODE_BACKEND_UNDEFINED = 0,    ///< Not yet determined
    LLAMA_DECODE_BACKEND_CPU       = 1,    ///< CPU-only backend
    LLAMA_DECODE_BACKEND_CUDA_MMQ  = 2,    ///< CUDA with MMQ kernels (quantized)
    LLAMA_DECODE_BACKEND_CUDA_DENSE = 3,   ///< CUDA with dense BLAS (FP16/FP32)
    LLAMA_DECODE_BACKEND_CUDA_CUBLAS = 4,  ///< CUDA with cuBLAS (dense only, high fallback risk)
    LLAMA_DECODE_BACKEND_METAL     = 5,    ///< Metal backend (Apple devices)
    LLAMA_DECODE_BACKEND_VULKAN    = 6,    ///< Vulkan backend (cross-platform GPU)
    LLAMA_DECODE_BACKEND_OPENGL    = 7,    ///< OpenGL backend (fallback GPU)
} llama_decode_backend_type_t;

/**
 * State machine for decode backend lock enforcement.
 * Tracks the lifecycle of backend binding from initialization through
 * graph computation and eventual context destruction.
 */
typedef enum {
    LLAMA_DECODE_LOCK_STATE_UNINITIALIZED = 0,  ///< Context created, no decode yet
    LLAMA_DECODE_LOCK_STATE_BINDING       = 1,  ///< First decode: probing backend options
    LLAMA_DECODE_LOCK_STATE_LOCKED        = 2,  ///< Backend selected and locked
    LLAMA_DECODE_LOCK_STATE_ACTIVE_DECODE = 3,  ///< Currently executing decode loop
    LLAMA_DECODE_LOCK_STATE_VIOLATION     = 4,  ///< Lock violation detected, fatal
    LLAMA_DECODE_LOCK_STATE_DESTROYED     = 5,  ///< Context destroyed, lock released
} llama_decode_lock_state_t;

// ============================================================================
// BACKEND LOCK STRUCTURE
// ============================================================================

/**
 * Core structure representing the immutable backend lock for a decode context.
 * This structure is embedded in llama_context and enforces the invariant that
 * once a backend is selected for decode, it cannot be changed.
 */
typedef struct {
    // Lock state tracking
    std::atomic<llama_decode_lock_state_t> state;      ///< Current state in lock lifecycle
    std::atomic<llama_decode_backend_type_t> locked_backend;  ///< Locked backend type (immutable after binding)

    // Lock protection
    std::atomic<bool> is_locked;                        ///< True once backend is bound to context
    std::atomic<uint32_t> lock_timestamp_ms;            ///< When lock was engaged (milliseconds from startup)

    // Graph ownership tracking
    std::atomic<uint64_t> graph_id;                     ///< ID of current compute graph
    std::atomic<uint64_t> prev_graph_id;                ///< ID of previous graph (for invalidation detection)

    // Violation detection
    std::atomic<uint32_t> re_selection_attempts;        ///< Count of attempts to change backend
    std::atomic<uint32_t> cublas_probe_attempts;        ///< Count of cuBLAS capability probes during locked state
    std::atomic<uint32_t> shape_heuristic_triggers;     ///< Count of shape-based fallback attempts
    std::atomic<uint32_t> backend_drift_detections;     ///< Count of unauthorized backend changes detected
    std::atomic<uint32_t> graph_invalidation_count;     ///< Count of graph invalidations during decode

    // Lock violation message (for diagnostics)
    char last_violation_msg[256];                       ///< Last violation message (for debugging)
    std::atomic<bool> violation_logged;                 ///< Whether violation was logged
} llama_decode_backend_lock_t;

// ============================================================================
// GRAPH METADATA STRUCTURES
// ============================================================================

/**
 * Metadata attached to each compute graph to enforce backend immutability.
 * Allows validation at execution time that the graph was built for the
 * currently-locked backend.
 */
typedef struct {
    llama_decode_backend_type_t backend_id;             ///< Backend this graph was built for
    uint32_t backend_flags;                             ///< Backend capability flags at build time
    uint32_t cublas_disabled;                           ///< 1 if cuBLAS explicitly disabled for this graph
    uint32_t mmq_required;                              ///< 1 if MMQ is required (quantized decode)
    uint32_t dense_required;                            ///< 1 if dense ops are required (embedding/attention)
    uint64_t graph_id;                                  ///< Unique graph identifier for invalidation tracking
} llama_graph_metadata_decode_t;

// ============================================================================
// ENVIRONMENT VARIABLE CACHING (Startup-only parsing)
// ============================================================================

/**
 * Cached environment variables parsed at startup.
 * These must never be re-read during decode execution.
 */
typedef struct {
    // Backend selection flags (parsed once at startup)
    std::atomic<bool> force_cublas;                     ///< LLAMA_CUBLAS_FORCE (parse once)
    std::atomic<bool> force_mmq;                        ///< LLAMA_CUDA_FORCE_MMQ (parse once)
    std::atomic<bool> force_cpu;                        ///< LLAMA_FORCE_CPU (parse once)

    // Feature flags (parsed once at startup)
    std::atomic<bool> allow_fallback;                   ///< LLAMA_ALLOW_BACKEND_FALLBACK (parse once)
    std::atomic<bool> shape_heuristics_enabled;         ///< LLAMA_ENABLE_SHAPE_HEURISTICS (parse once)
    std::atomic<bool> deterministic_mode;               ///< LLAMA_DETERMINISTIC (parse once)

    // Capability probing flags
    std::atomic<bool> skip_capability_check;            ///< LLAMA_SKIP_CAPABILITY_CHECK (parse once)
    std::atomic<bool> gpu_exclusive_decode;             ///< LLAMA_GPU_EXCLUSIVE_DECODE (parse once)

    // Validation flag: whether env vars were already parsed
    std::atomic<bool> initialized;                      ///< True once environment variables are cached
} llama_decode_env_cache_t;

// ============================================================================
// ASSERTION AND GUARD MACROS
// ============================================================================

/**
 * Macro: Assert that decode backend is locked before performing operations
 * that must have an immutable backend.
 *
 * Usage:
 *   LLAMA_DECODE_ASSERT_LOCKED(ctx, "Attempting GEMM dispatch");
 */
#define LLAMA_DECODE_ASSERT_LOCKED(ctx, msg) \
    do { \
        if (!(ctx)->decode_backend_lock.is_locked.load()) { \
            fprintf(stderr, "FATAL: Decode backend not locked. %s\n", (msg)); \
            abort(); \
        } \
    } while (0)

/**
 * Macro: Assert that the specified backend matches the locked backend.
 * Used in hot paths to ensure no backend drift.
 *
 * Usage:
 *   LLAMA_DECODE_ASSERT_BACKEND_MATCH(ctx, LLAMA_DECODE_BACKEND_CUDA_MMQ, "MatMul dispatch");
 */
#define LLAMA_DECODE_ASSERT_BACKEND_MATCH(ctx, expected_backend, msg) \
    do { \
        auto locked = (ctx)->decode_backend_lock.locked_backend.load(); \
        if (locked != (expected_backend)) { \
            fprintf(stderr, "FATAL: Backend mismatch at %s. Expected %d, got %d\n", \
                    (msg), (int)(expected_backend), (int)locked); \
            abort(); \
        } \
    } while (0)

/**
 * Macro: Guard against re-selection attempts during locked state.
 * Should be checked before entering backend selection logic.
 *
 * Usage:
 *   if (LLAMA_DECODE_SHOULD_SKIP_RESELECTION(ctx)) return;
 */
#define LLAMA_DECODE_SHOULD_SKIP_RESELECTION(ctx) \
    ((ctx)->decode_backend_lock.is_locked.load())

/**
 * Macro: Guard against cuBLAS probing when locked to non-cuBLAS backend.
 *
 * Usage:
 *   if (LLAMA_DECODE_BLOCK_CUBLAS_PROBE(ctx)) return;
 */
#define LLAMA_DECODE_BLOCK_CUBLAS_PROBE(ctx) \
    ((ctx)->decode_backend_lock.is_locked.load() && \
     (ctx)->decode_backend_lock.locked_backend.load() != LLAMA_DECODE_BACKEND_CUDA_CUBLAS)

/**
 * Macro: Guard against shape-based heuristic fallbacks during decode.
 * Shape heuristics must be disabled when backend is locked.
 *
 * Usage:
 *   if (LLAMA_DECODE_BLOCK_SHAPE_HEURISTICS(ctx)) return;
 */
#define LLAMA_DECODE_BLOCK_SHAPE_HEURISTICS(ctx) \
    ((ctx)->decode_backend_lock.is_locked.load())

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

/**
 * Initialize the decode backend lock for a new context.
 * Called during llama_context construction.
 *
 * @param lock  Pointer to llama_decode_backend_lock_t to initialize
 * @param ctx   Pointer to llama_context for logging purposes
 */
void llama_decode_lock_init(llama_decode_backend_lock_t * lock, void * ctx);

/**
 * Engage the backend lock after successful backend selection and binding.
 * Called after the first decode step when backend capabilities are confirmed.
 * This is the point of no return: after this call, backend cannot change.
 *
 * @param lock     Pointer to llama_decode_backend_lock_t
 * @param backend  The backend type to lock (from llama_decode_backend_type_t)
 * @param ctx      Pointer to llama_context for logging
 * @return         True if lock was successfully engaged, false if already locked
 */
bool llama_decode_lock_engage(llama_decode_backend_lock_t * lock,
                               llama_decode_backend_type_t backend,
                               void * ctx);

/**
 * Check if the decode backend is currently locked.
 * Lightweight check used in hot paths.
 *
 * @param lock  Pointer to llama_decode_backend_lock_t
 * @return      True if backend is locked and immutable
 */
bool llama_decode_lock_is_locked(const llama_decode_backend_lock_t * lock);

/**
 * Get the currently locked backend type.
 * Only valid after llama_decode_lock_engage() has been called.
 *
 * @param lock  Pointer to llama_decode_backend_lock_t
 * @return      The locked backend type (or LLAMA_DECODE_BACKEND_UNDEFINED if not locked)
 */
llama_decode_backend_type_t llama_decode_lock_get_backend(
    const llama_decode_backend_lock_t * lock);

/**
 * Attempt backend re-selection during locked state.
 * Records violation and returns false to prevent re-selection.
 * Called before entering backend selection logic.
 *
 * @param lock    Pointer to llama_decode_backend_lock_t
 * @param reason  String describing why re-selection was attempted
 * @param ctx     Pointer to llama_context for logging
 * @return        False if locked (re-selection blocked), true if unlocked (reselection allowed)
 */
bool llama_decode_lock_allow_reselection(llama_decode_backend_lock_t * lock,
                                          const char * reason,
                                          void * ctx);

/**
 * Record a cuBLAS probing attempt during locked state.
 * Used to detect violations of the "no cuBLAS when locked to MMQ" rule.
 *
 * @param lock    Pointer to llama_decode_backend_lock_t
 * @param ctx     Pointer to llama_context for logging
 * @return        False if cuBLAS probing should be blocked, true if allowed
 */
bool llama_decode_lock_allow_cublas_probe(llama_decode_backend_lock_t * lock,
                                           void * ctx);

/**
 * Record a shape-based heuristic fallback attempt during locked state.
 * Shape heuristics must be disabled after backend lock engagement.
 *
 * @param lock           Pointer to llama_decode_backend_lock_t
 * @param shape_reason   String describing the shape condition
 * @param ctx            Pointer to llama_context for logging
 * @return               False if shape heuristics should be blocked, true if allowed
 */
bool llama_decode_lock_allow_shape_heuristic(llama_decode_backend_lock_t * lock,
                                              const char * shape_reason,
                                              void * ctx);

/**
 * Validate that graph was built for the locked backend.
 * Called before graph execution to ensure no graph drift.
 *
 * @param lock     Pointer to llama_decode_backend_lock_t
 * @param graph    Pointer to ggml_cgraph with attached metadata
 * @param ctx      Pointer to llama_context for logging
 * @return         True if graph matches locked backend, false if mismatch
 */
bool llama_decode_lock_validate_graph_backend(llama_decode_backend_lock_t * lock,
                                               void * graph,
                                               void * ctx);

/**
 * Check for graph invalidation and prevent decode continuation.
 * Graph invalidation during decode is a critical error.
 *
 * @param lock      Pointer to llama_decode_backend_lock_t
 * @param new_graph_id  The new graph ID after potential invalidation
 * @param ctx       Pointer to llama_context for logging
 * @return          True if graph is valid for continued decode, false if invalidated
 */
bool llama_decode_lock_check_graph_validity(llama_decode_backend_lock_t * lock,
                                             uint64_t new_graph_id,
                                             void * ctx);

/**
 * Detect backend drift: check if backend has changed without authorization.
 * Used at critical junctures to ensure backend selection is immutable.
 *
 * @param lock          Pointer to llama_decode_backend_lock_t
 * @param current_backend   The backend currently in use
 * @param ctx           Pointer to llama_context for logging
 * @return              True if backend matches lock, false if drift detected
 */
bool llama_decode_lock_detect_drift(llama_decode_backend_lock_t * lock,
                                     llama_decode_backend_type_t current_backend,
                                     void * ctx);

/**
 * Assert that backend matches expected value and abort if not.
 * Strong guard for critical operations like MatMul dispatch.
 *
 * @param lock            Pointer to llama_decode_backend_lock_t
 * @param expected        The backend that should be active
 * @param operation_name  String describing the operation for error messages
 * @param ctx             Pointer to llama_context for logging
 */
void llama_decode_lock_assert_backend_match(llama_decode_backend_lock_t * lock,
                                             llama_decode_backend_type_t expected,
                                             const char * operation_name,
                                             void * ctx);

/**
 * Destroy the decode backend lock and release resources.
 * Called during llama_context destruction.
 *
 * @param lock  Pointer to llama_decode_backend_lock_t
 */
void llama_decode_lock_destroy(llama_decode_backend_lock_t * lock);

// ============================================================================
// GRAPH METADATA MANAGEMENT
// ============================================================================

/**
 * Attach backend metadata to a compute graph.
 * Called when graph is built to record which backend it targets.
 *
 * @param graph     Pointer to ggml_cgraph to attach metadata to
 * @param backend   The backend this graph was built for
 * @param flags     Backend capability flags at build time
 * @param ctx       Pointer to llama_context for logging
 * @return          True if metadata was attached successfully
 */
bool llama_graph_metadata_attach_backend(void * graph,
                                         llama_decode_backend_type_t backend,
                                         uint32_t flags,
                                         void * ctx);

/**
 * Retrieve backend metadata from a compute graph.
 * Called before graph execution to validate backend compatibility.
 *
 * @param graph  Pointer to ggml_cgraph
 * @return       Pointer to llama_graph_metadata_decode_t, or nullptr if no metadata
 */
const llama_graph_metadata_decode_t * llama_graph_metadata_get_backend(
    const void * graph);

/**
 * Validate that graph metadata matches the locked backend.
 * Called during graph execution to ensure immutability invariant.
 *
 * @param graph     Pointer to ggml_cgraph
 * @param expected  The backend we expect this graph to target
 * @param ctx       Pointer to llama_context for logging
 * @return          True if metadata matches expected backend, false if mismatch
 */
bool llama_graph_metadata_validate_backend(const void * graph,
                                            llama_decode_backend_type_t expected,
                                            void * ctx);

// ============================================================================
// ENVIRONMENT VARIABLE CACHING
// ============================================================================

/**
 * Initialize environment variable cache at startup.
 * Called once during program initialization, never during decode.
 *
 * @param env_cache  Pointer to llama_decode_env_cache_t to initialize
 * @return           True if cache was successfully initialized
 */
bool llama_decode_env_cache_init(llama_decode_env_cache_t * env_cache);

/**
 * Check if an environment variable was cached (preventing re-reads).
 * If not cached, returns false and subsequent calls should cache the value.
 *
 * @param env_cache   Pointer to llama_decode_env_cache_t
 * @param var_name    Name of the environment variable (for diagnostics)
 * @return            True if variable was already parsed at startup
 */
bool llama_decode_env_check_cached(const llama_decode_env_cache_t * env_cache,
                                    const char * var_name);

/**
 * Prevent environment variable re-reads during decode.
 * Called at decode entry point to ensure env vars stay constant.
 *
 * @param env_cache  Pointer to llama_decode_env_cache_t
 * @param ctx        Pointer to llama_context for logging
 * @return           True if environment is stable (no re-reads detected)
 */
bool llama_decode_env_protect_against_rereads(const llama_decode_env_cache_t * env_cache,
                                               void * ctx);

// ============================================================================
// METRICS AND DIAGNOSTICS
// ============================================================================

/**
 * Structure for tracking decode backend lock metrics.
 */
typedef struct {
    uint64_t lock_engagements;              ///< Number of times backend lock was engaged
    uint64_t reselection_attempts;          ///< Blocked re-selection attempts
    uint64_t cublas_probe_blocks;           ///< Blocked cuBLAS probes during lock
    uint64_t shape_heuristic_blocks;        ///< Blocked shape heuristics during lock
    uint64_t drift_detections;              ///< Backend drift events detected
    uint64_t graph_invalidations;           ///< Graph invalidations during decode
    uint64_t total_contexts_created;        ///< Total contexts created with lock
    uint64_t lock_violations;               ///< Total lock violations detected
} llama_decode_lock_metrics_t;

/**
 * Get current metrics for decode backend lock enforcement.
 *
 * @param metrics  Pointer to llama_decode_lock_metrics_t to fill with current metrics
 */
void llama_decode_lock_get_metrics(llama_decode_lock_metrics_t * metrics);

/**
 * Reset all decode backend lock metrics to zero.
 * Useful for benchmarking and testing.
 */
void llama_decode_lock_reset_metrics(void);

/**
 * Print a human-readable report of decode backend lock metrics and status.
 * Used for diagnostics and verification.
 *
 * @param detailed  If true, print detailed information; if false, summary only
 */
void llama_decode_lock_print_report(bool detailed);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_CUBLAS_PREVENTION_H
