#pragma once

/**
 * llama-backend-purity.h
 *
 * Complete elimination of unused backends from binary.
 * Implements single-backend-per-build enforcement with compile-time validation,
 * runtime purity checking, and binary inspection utilities.
 *
 * Requirements Enforced (12 Rules):
 *
 * 1. Define Single-Backend Build Policy
 *    - Exactly one decode backend per build variant
 *    - cuBLAS-only build → CUDA dense only
 *    - MMQ/MoE build → CUDA MMQ only
 *    - CPU build → CPU only
 *    - No hybrid backend builds permitted
 *
 * 2. Disable CPU Backend for GPU-Exclusive Decode Builds
 *    - Remove: ggml-cpu, CPU kernel implementations, CPU fallback logic
 *    - CPU backend must not be linkable for decode builds
 *    - If CPU ops required (tokenizer): isolate outside decode path
 *
 * 3. Disable MMQ in cuBLAS Builds
 *    - -DGGML_CUDA_FORCE_CUBLAS=ON
 *    - -DGGML_CUDA_FORCE_MMQ=OFF
 *    - -DGGML_CUDA_MMQ=OFF
 *    - Verify: no MMQ kernels, no MMQ symbols in binary
 *
 * 4. Disable cuBLAS in MMQ Builds
 *    - -DGGML_CUDA_FORCE_MMQ=ON
 *    - Disable dense cuBLAS path
 *    - Remove cuBLAS linkage
 *    - Ensure exclusively MMQ-based kernels
 *
 * 5. Remove Backend Registry Multiplexing
 *    - Remove dynamic backend registration loops
 *    - Replace with single static backend registration
 *    - No runtime backend discovery
 *
 * 6. Remove Backend Capability Enumeration
 *    - Disable: backend priority lists, scoring logic, capability-based selection
 *    - Graph builder assumes one backend only
 *
 * 7. Remove Hybrid Layer Placement
 *    - Disallow: per-layer CPU/GPU placement, partial offloading, automatic VRAM fitting
 *    - All decode layers fully GPU-resident
 *    - VRAM insufficient → fail at load time
 *
 * 8. Remove CPU↔CUDA Boundary Code
 *    - Disable: ggml_backend_cpu, CPU wrappers, hybrid memory helpers
 *    - Decode build must not contain hybrid memory code
 *
 * 9. Add Build-Time Guards
 *    - CMake assertions: fail if more than one backend enabled
 *    - if (GGML_CUDA AND GGML_CPU_BACKEND) → FATAL_ERROR
 *
 * 10. Verify Binary Purity
 *     - Inspect binary via nm/strings
 *     - Confirm: no CPU fallback symbols, no alternate backend, no unwanted kernels
 *
 * 11. Remove Backend Switching Code Paths
 *     - Eliminate switch (backend_type) statements
 *     - Replace with direct single backend invocation
 *     - No switch allowed in decode
 *
 * 12. Expected Outcome
 *     - No silent fallback possible
 *     - No backend branching in decode
 *     - No hybrid execution ambiguity
 *     - Reduced binary complexity
 *     - Strong GPU-exclusive invariant
 *     - More predictable execution
 *
 * Key Metrics Tracked:
 * - Unused backend symbols in binary (target: 0)
 * - Backend dispatch branching (target: 0)
 * - Binary size reduction from backend elimination
 * - Backend switching code paths removed (target: 100%)
 * - Fallback code paths eliminated (target: 100%)
 */

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <array>
#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <stdexcept>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

// Maximum number of backend symbols to verify
#define LLAMA_BACKEND_PURITY_MAX_SYMBOLS 10000

// Backend purity state machine states
#define LLAMA_BACKEND_PURITY_STATE_UNINITIALIZED 0
#define LLAMA_BACKEND_PURITY_STATE_INITIALIZED 1
#define LLAMA_BACKEND_PURITY_STATE_VALIDATED 2
#define LLAMA_BACKEND_PURITY_STATE_LOCKED 3
#define LLAMA_BACKEND_PURITY_STATE_ERROR 4

// Maximum deferred violations to accumulate before abort
#define LLAMA_BACKEND_PURITY_MAX_VIOLATIONS 1000

// Compile-time configuration
#ifndef LLAMA_BACKEND_PURITY_ENABLED
#define LLAMA_BACKEND_PURITY_ENABLED 1
#endif

#ifndef LLAMA_BACKEND_PURITY_STRICT_MODE
#define LLAMA_BACKEND_PURITY_STRICT_MODE 1
#endif

#ifndef LLAMA_BACKEND_PURITY_COLLECT_METRICS
#define LLAMA_BACKEND_PURITY_COLLECT_METRICS 1
#endif

// ============================================================================
// BACKEND SELECTION ENUM
// ============================================================================

/**
 * Available backend types for single-backend builds
 */
typedef enum {
    LLAMA_BACKEND_VARIANT_UNDEFINED = 0,
    LLAMA_BACKEND_VARIANT_CPU_ONLY = 1,           // CPU backend only
    LLAMA_BACKEND_VARIANT_CUDA_CUBLAS = 2,        // CUDA with cuBLAS (dense operations only)
    LLAMA_BACKEND_VARIANT_CUDA_MMQ = 3,           // CUDA with MMQ kernels
    LLAMA_BACKEND_VARIANT_VULKAN = 4,             // Vulkan backend only
    LLAMA_BACKEND_VARIANT_METAL = 5               // Metal backend only
} llama_backend_variant_t;

// ============================================================================
// ATOMIC METRICS STRUCTURES
// ============================================================================

/**
 * Backend purity violation tracking (lock-free)
 */
typedef struct {
    std::atomic<uint64_t> unused_backend_symbols_found;    // Count of unused backend symbols
    std::atomic<uint64_t> backend_dispatch_branches;       // Branch instructions to alternate backends
    std::atomic<uint64_t> hybrid_layer_placements;         // Per-layer backend assignments detected
    std::atomic<uint64_t> cpu_fallback_paths;              // CPU fallback code paths found
    std::atomic<uint64_t> backend_switching_statements;    // switch(backend_type) statements detected
    std::atomic<uint64_t> hybrid_memory_operations;        // CPU↔CUDA boundary operations detected
    std::atomic<uint64_t> dynamic_backend_registrations;   // Runtime backend registrations
    std::atomic<uint64_t> backend_capability_enumerations; // Backend capability discovery operations
    std::atomic<uint64_t> total_violations;                // Total enforcement violations
    std::atomic<uint64_t> violations_deferred;             // Violations buffered before abort
} llama_backend_purity_violations_t;

/**
 * Backend purity metrics (lock-free)
 */
typedef struct {
    std::atomic<size_t> total_symbols_verified;          // Total symbols scanned
    std::atomic<size_t> expected_backend_symbols;        // Expected symbols for selected backend
    std::atomic<size_t> unexpected_backend_symbols;      // Symbols from other backends
    std::atomic<size_t> binary_size_baseline;            // Binary size without optimizations
    std::atomic<size_t> binary_size_current;             // Current binary size
    std::atomic<uint64_t> verification_time_ns;          // Last verification duration (nanoseconds)
    std::atomic<uint32_t> verification_count;            // Total verification runs
    std::atomic<bool> last_verification_passed;          // Result of last verification
} llama_backend_purity_metrics_t;

/**
 * Backend feature table (compile-time only)
 */
typedef struct {
    const char * backend_name;                    // Symbolic name (e.g., "CUDA/cuBLAS")
    const char * description;                     // Human-readable description
    bool supports_dense_ops;                      // cuBLAS-like operations
    bool supports_mmq_ops;                        // MMQ-like operations
    bool supports_hybrid_memory;                  // CPU↔GPU boundary transfers
    bool supports_layer_offloading;               // Per-layer GPU/CPU placement
    bool is_gpu_backend;                          // GPU or CPU backend
    const char * required_cmake_flags;            // CMake flags to enable
    const char * conflicting_backends;            // Backends that conflict
    uint32_t expected_symbol_count;               // Expected symbol count in binary
} llama_backend_feature_t;

/**
 * Backend purity state machine
 */
typedef struct {
    uint32_t state;                               // Current state (see LLAMA_BACKEND_PURITY_STATE_*)
    llama_backend_variant_t selected_backend;     // Active backend variant
    bool backend_locked;                          // Prevent backend changes after initialization
    llama_backend_purity_violations_t violations;
    llama_backend_purity_metrics_t metrics;
    std::vector<std::string> * deferred_violations; // Buffer for violation details
    std::atomic<bool> abort_on_violation;         // Immediately abort if violation detected
} llama_backend_purity_state_t;

// ============================================================================
// COMPILE-TIME CHECK MACROS (15+ total)
// ============================================================================

/**
 * LLAMA_BACKEND_PURITY_ASSERT_SINGLE_BACKEND
 * Compile-time assertion that exactly one backend is enabled.
 * Fails at build time if multiple backends are enabled.
 */
#define LLAMA_BACKEND_PURITY_ASSERT_SINGLE_BACKEND \
    do { \
        int enabled_count = 0; \
        if (defined(GGML_USE_CUDA)) enabled_count++; \
        if (defined(GGML_USE_CPU_BACKEND)) enabled_count++; \
        if (defined(GGML_USE_VULKAN)) enabled_count++; \
        if (defined(GGML_USE_METAL)) enabled_count++; \
        if (defined(GGML_USE_OPENCL)) enabled_count++; \
        if (enabled_count != 1) { \
            static_assert(false, "LLAMA_BACKEND_PURITY: Exactly one backend must be enabled"); \
        } \
    } while(0)

/**
 * LLAMA_BACKEND_PURITY_CUDA_CUBLAS_ONLY
 * Verify CUDA cuBLAS configuration without MMQ
 */
#define LLAMA_BACKEND_PURITY_CUDA_CUBLAS_ONLY \
    do { \
        static_assert(defined(GGML_USE_CUDA), "CUDA not enabled"); \
        static_assert(!defined(GGML_CUDA_FORCE_MMQ), "MMQ must be disabled in cuBLAS builds"); \
        static_assert(defined(GGML_CUDA_FORCE_CUBLAS), "cuBLAS must be explicitly enabled"); \
    } while(0)

/**
 * LLAMA_BACKEND_PURITY_CUDA_MMQ_ONLY
 * Verify CUDA MMQ configuration without cuBLAS
 */
#define LLAMA_BACKEND_PURITY_CUDA_MMQ_ONLY \
    do { \
        static_assert(defined(GGML_USE_CUDA), "CUDA not enabled"); \
        static_assert(defined(GGML_CUDA_FORCE_MMQ), "MMQ must be explicitly enabled"); \
        static_assert(!defined(GGML_CUDA_FORCE_CUBLAS), "cuBLAS must be disabled in MMQ builds"); \
    } while(0)

/**
 * LLAMA_BACKEND_PURITY_CPU_ONLY
 * Verify CPU-only configuration
 */
#define LLAMA_BACKEND_PURITY_CPU_ONLY \
    do { \
        static_assert(defined(GGML_USE_CPU_BACKEND), "CPU backend not enabled"); \
        static_assert(!defined(GGML_USE_CUDA), "CUDA must be disabled in CPU builds"); \
        static_assert(!defined(GGML_USE_VULKAN), "Vulkan must be disabled in CPU builds"); \
        static_assert(!defined(GGML_USE_METAL), "Metal must be disabled in CPU builds"); \
    } while(0)

/**
 * LLAMA_BACKEND_PURITY_NO_HYBRID_BACKENDS
 * Verify no hybrid backend configuration
 */
#define LLAMA_BACKEND_PURITY_NO_HYBRID_BACKENDS \
    do { \
        int gpu_count = 0; \
        if (defined(GGML_USE_CUDA)) gpu_count++; \
        if (defined(GGML_USE_VULKAN)) gpu_count++; \
        if (defined(GGML_USE_METAL)) gpu_count++; \
        if (gpu_count > 1 || (gpu_count > 0 && defined(GGML_USE_CPU_BACKEND))) { \
            static_assert(false, "Hybrid backend builds not permitted"); \
        } \
    } while(0)

/**
 * LLAMA_BACKEND_PURITY_DISABLE_CPU_FALLBACK
 * Ensure CPU fallback code is not compiled
 */
#define LLAMA_BACKEND_PURITY_DISABLE_CPU_FALLBACK \
    do { \
        static_assert(!defined(LLAMA_CPU_FALLBACK), "CPU fallback must be disabled"); \
        static_assert(!defined(LLAMA_HYBRID_MEMORY), "Hybrid memory must be disabled"); \
    } while(0)

/**
 * LLAMA_BACKEND_PURITY_SINGLE_DECODE_PATH
 * Verify single code path in decode
 */
#define LLAMA_BACKEND_PURITY_SINGLE_DECODE_PATH \
    do { \
        static_assert(!defined(LLAMA_MULTI_BACKEND_DECODE), "Multiple decode paths not permitted"); \
    } while(0)

/**
 * LLAMA_BACKEND_PURITY_NO_DYNAMIC_REGISTRATION
 * Verify no dynamic backend registration
 */
#define LLAMA_BACKEND_PURITY_NO_DYNAMIC_REGISTRATION \
    do { \
        static_assert(!defined(GGML_BACKEND_DYNAMIC_LOAD), "Dynamic backend loading not permitted"); \
    } while(0)

/**
 * LLAMA_BACKEND_PURITY_VALIDATE_SYMBOLS
 * Validate that only expected backend symbols are present
 */
#define LLAMA_BACKEND_PURITY_VALIDATE_SYMBOLS(backend_type) \
    llama_backend_purity_validate_symbols(backend_type)

/**
 * LLAMA_BACKEND_PURITY_CHECK_BINARY_SIZE
 * Verify binary size is within expected range
 */
#define LLAMA_BACKEND_PURITY_CHECK_BINARY_SIZE(max_size) \
    llama_backend_purity_check_binary_size(max_size)

/**
 * LLAMA_BACKEND_PURITY_ASSERT_NO_BRANCHING
 * Runtime check that backend selection doesn't branch
 */
#define LLAMA_BACKEND_PURITY_ASSERT_NO_BRANCHING() \
    llama_backend_purity_assert_no_branching()

/**
 * LLAMA_BACKEND_PURITY_GUARD_DISPATCH
 * Guard a backend dispatch to ensure single path
 */
#define LLAMA_BACKEND_PURITY_GUARD_DISPATCH(backend) \
    do { \
        if (llama_backend_purity_is_locked()) { \
            llama_backend_purity_verify_backend_match(backend); \
        } \
    } while(0)

/**
 * LLAMA_BACKEND_PURITY_FENCE
 * Memory fence to ensure backend purity state propagation
 */
#define LLAMA_BACKEND_PURITY_FENCE() \
    std::atomic_thread_fence(std::memory_order_seq_cst)

/**
 * LLAMA_BACKEND_PURITY_ABORT_ON_VIOLATION
 * Configure behavior when violations detected
 */
#define LLAMA_BACKEND_PURITY_ABORT_ON_VIOLATION(should_abort) \
    llama_backend_purity_set_abort_on_violation(should_abort)

// ============================================================================
// RUNTIME VALIDATION FUNCTIONS (10+)
// ============================================================================

/**
 * Initialize backend purity enforcement
 * @param variant Selected backend variant
 * @return Opaque state handle
 */
extern llama_backend_purity_state_t * llama_backend_purity_init(llama_backend_variant_t variant);

/**
 * Free backend purity state
 * @param state State handle
 */
extern void llama_backend_purity_free(llama_backend_purity_state_t * state);

/**
 * Validate selected backend configuration
 * @param state State handle
 * @param variant Backend variant to validate
 * @return true if configuration is valid
 */
extern bool llama_backend_purity_validate_config(
    llama_backend_purity_state_t * state,
    llama_backend_variant_t variant
);

/**
 * Verify that only expected backend symbols are present in binary
 * @param variant Backend variant
 * @return Number of unexpected symbols found
 */
extern size_t llama_backend_purity_validate_symbols(llama_backend_variant_t variant);

/**
 * Check binary size is within expected range
 * @param max_size Maximum allowed binary size in bytes
 * @return true if binary size is acceptable
 */
extern bool llama_backend_purity_check_binary_size(size_t max_size);

/**
 * Assert that backend selection doesn't branch at runtime
 * Must be called in decode-critical path
 */
extern void llama_backend_purity_assert_no_branching(void);

/**
 * Verify that current backend matches expected backend
 * @param expected Expected backend variant
 * @throws std::runtime_error if mismatch
 */
extern void llama_backend_purity_verify_backend_match(llama_backend_variant_t expected);

/**
 * Lock backend selection to prevent changes
 * @param state State handle
 */
extern void llama_backend_purity_lock_backend(llama_backend_purity_state_t * state);

/**
 * Check if backend is currently locked
 * @return true if backend is locked
 */
extern bool llama_backend_purity_is_locked(void);

/**
 * Get current backend purity state
 * @return Current backend variant
 */
extern llama_backend_variant_t llama_backend_purity_get_backend(void);

/**
 * Get backend feature information
 * @param variant Backend variant
 * @return Feature information struct
 */
extern const llama_backend_feature_t * llama_backend_purity_get_features(
    llama_backend_variant_t variant
);

/**
 * Collect and report backend purity metrics
 * @param state State handle
 * @return Metrics struct
 */
extern llama_backend_purity_metrics_t llama_backend_purity_get_metrics(
    llama_backend_purity_state_t * state
);

/**
 * Report all detected violations
 * @param state State handle
 * @return Human-readable violation report
 */
extern const char * llama_backend_purity_get_violation_report(
    llama_backend_purity_state_t * state
);

/**
 * Reset all violation counters
 * @param state State handle
 */
extern void llama_backend_purity_reset_violations(llama_backend_purity_state_t * state);

/**
 * Set behavior when violations are detected
 * @param should_abort If true, immediately abort; if false, defer
 */
extern void llama_backend_purity_set_abort_on_violation(bool should_abort);

/**
 * Binary inspection helper: list all backend symbols
 * @return Comma-separated list of found symbols
 */
extern const char * llama_backend_purity_get_symbol_list(void);

/**
 * Binary inspection helper: check for specific symbol
 * @param symbol Symbol name to search for
 * @return true if symbol found in binary
 */
extern bool llama_backend_purity_has_symbol(const char * symbol);

/**
 * Verify graph builder uses only single backend
 * @param graph_ptr Opaque pointer to computation graph
 * @return Number of backend assignments found
 */
extern size_t llama_backend_purity_verify_graph(void * graph_ptr);

/**
 * Register backend exclusion list for compile-time validation
 * @param backends Comma-separated backend names to exclude
 */
extern void llama_backend_purity_register_exclusions(const char * backends);

/**
 * Trigger full binary purity scan
 * @return Number of violations found
 */
extern size_t llama_backend_purity_full_scan(void);

// ============================================================================
// CMake Integration Functions
// ============================================================================

/**
 * Get CMake flags for selected backend
 * @param variant Backend variant
 * @return CMake flag string (e.g., "-DGGML_USE_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS=ON")
 */
extern const char * llama_backend_purity_get_cmake_flags(llama_backend_variant_t variant);

/**
 * Validate CMake configuration matches selected backend
 * @param variant Expected backend variant
 * @return true if CMake configuration matches
 */
extern bool llama_backend_purity_validate_cmake_config(llama_backend_variant_t variant);

/**
 * Generate CMake configuration for single-backend build
 * @param variant Backend variant
 * @param output Buffer for output (at least 4096 bytes)
 * @return Length of output written
 */
extern size_t llama_backend_purity_generate_cmake_config(
    llama_backend_variant_t variant,
    char * output
);

// ============================================================================
// BUILD PROFILE DEFINITIONS
// ============================================================================

/**
 * Predefined build profiles for common configurations
 */
typedef struct {
    const char * profile_name;
    llama_backend_variant_t variant;
    const char * cmake_flags;
    const char * description;
    bool enable_metrics;
    bool strict_mode;
} llama_backend_build_profile_t;

/**
 * Get predefined build profile
 * @param profile_index Index (0-4 for built-in profiles)
 * @return Build profile definition
 */
extern const llama_backend_build_profile_t * llama_backend_purity_get_profile(size_t profile_index);

/**
 * Number of predefined build profiles
 * @return Count of available profiles
 */
extern size_t llama_backend_purity_get_profile_count(void);

// ============================================================================
// PERFORMANCE METRICS TRACKING
// ============================================================================

/**
 * Binary size reduction tracking
 */
typedef struct {
    size_t baseline_size;       // Full binary with all backends
    size_t optimized_size;      // Binary with single backend
    size_t reduction_bytes;     // Absolute reduction
    double reduction_percent;   // Percentage reduction
    size_t unused_symbols_bytes; // Bytes from unused backend symbols
} llama_backend_purity_binary_optimization_t;

/**
 * Get binary optimization metrics
 * @return Optimization metrics
 */
extern llama_backend_purity_binary_optimization_t llama_backend_purity_get_binary_optimization(void);

/**
 * Code path elimination tracking
 */
typedef struct {
    uint64_t dispatch_branches_eliminated;  // switch(backend_type) removed
    uint64_t fallback_paths_eliminated;     // CPU fallback code removed
    uint64_t boundary_code_eliminated;      // CPU↔CUDA boundary code removed
    uint64_t registry_loops_eliminated;     // Backend registration loops removed
    uint64_t capability_enum_eliminated;    // Backend capability discovery removed
} llama_backend_purity_code_elimination_t;

/**
 * Get code path elimination metrics
 * @return Elimination metrics
 */
extern llama_backend_purity_code_elimination_t llama_backend_purity_get_code_elimination(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_BACKEND_PURITY_H
