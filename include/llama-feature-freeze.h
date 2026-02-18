#pragma once

/**
 * llama-feature-freeze.h
 *
 * Complete compile-time resolution of all decode-affecting feature flags.
 * Implements comprehensive feature freezing with build-time profiles,
 * compile-time validation macros, and startup compatibility enforcement.
 *
 * Requirements Enforced (11 Rules):
 * 1. Identify All Decode-Affecting Feature Flags
 *    - CUDA vs CPU backend selection
 *    - cuBLAS vs MMQ kernel mode
 *    - Flash attention availability
 *    - CUDA graphs support
 *    - Hybrid memory mode
 *    - Determinism mode
 *    - Speculative decoding
 *    - Server-specific behaviors
 *    - Debug logging flags
 *    - Verbose metrics output
 *    - Experimental kernels
 *    - OpenMP vs pthreads threading
 *    - Fallback logic paths
 *
 * 2. Convert Runtime Feature Checks to Compile-Time Guards
 *    - Replace: if (GGML_CUDA) with #if defined(GGML_CUDA)
 *    - Replace: if (enable_flash_attention) with #if defined(LLAMA_ENABLE_FLASH_ATTENTION)
 *    - Decode binary must not evaluate feature availability dynamically
 *
 * 3. Eliminate Feature Capability Probing During Decode
 *    - GPU capability detection occurs at build time only
 *    - Kernel support probing occurs at build time only
 *    - Tensor-core checks resolved at build time only
 *    - MMQ support checks resolved at build time only
 *
 * 4. Remove Optional Decode Paths from Binary
 *    - GPU-exclusive decode build: compile out CPU backend
 *    - Disable hybrid memory mode
 *    - Disable speculative decoding paths
 *    - Disable CUDA graph optional toggles
 *    - Disable server verbose toggles
 *
 * 5. Produce Dedicated Build Profiles
 *    - build_cuda_cublas_dense: CUDA + cuBLAS + dense computation
 *    - build_cuda_mmq_moe: CUDA + MMQ + MoE support
 *    - build_cuda_flash_attention: CUDA + Flash Attention
 *    - build_cpu_only: CPU-only fallback
 *    - Each profile: hardcoded backend, removed unused backends
 *
 * 6. Disable Runtime Backend Fallback Code
 *    - Replace: if (!cuda_supported) fallback_to_cpu()
 *    - With: #error "CUDA backend required" or fatal startup error
 *    - Never fallback inside decode loop
 *
 * 7. Remove Debug and Logging Flags from Hot Path
 *    - Compile-time elimination: #ifdef DEBUG_ENABLED
 *    - Remove profiling hooks, trace instrumentation
 *    - No debug branches in decode loop
 *
 * 8. Freeze Determinism Mode at Build
 *    - If deterministic required: compile with deterministic reductions
 *    - If relaxed mode desired: compile separate relaxed build
 *    - No runtime switching between modes
 *
 * 9. Validate Zero Runtime Feature Branching
 *    - Inspect decode loop assembly
 *    - Confirm no conditional branches on feature flags
 *    - Confirm no feature symbol lookups in decode path
 *
 * 10. Enforce Feature Immutability
 *     - At startup: validate hardware compatibility
 *     - Abort if mismatch detected
 *     - Do not attempt auto-adjustment
 *     - Hardware mismatch triggers startup error, not decode-path changes
 *
 * 11. Expected Result
 *     - Decode loop structurally fixed
 *     - Backend selection immutable
 *     - No feature toggles during execution
 *     - Branch misprediction reduced
 *     - Control flow deterministic and minimal
 *     - GPU-exclusive invariant strengthened
 *
 * Key Metrics Tracked:
 * - Build-time feature resolution count (target: 100%)
 * - Runtime feature checks (target: 0)
 * - Conditional branches in decode (target: 0)
 * - Feature symbol lookups (target: 0)
 * - Binary size reduction from feature compilation
 * - Assembly instruction count reduction in decode
 */

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <mutex>
#include <array>
#include <memory>
#include <functional>
#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BUILD-TIME FEATURE FLAG DEFINITIONS
// ============================================================================

/**
 * LLAMA_FEATURE_FREEZE_PROFILE: Defines the build profile for feature resolution.
 * Must be set at compile time via -D flag or CMake configuration.
 *
 * Valid values:
 * - LLAMA_BUILD_CUDA_CUBLAS_DENSE   (1): CUDA + cuBLAS + Dense compute
 * - LLAMA_BUILD_CUDA_MMQ_MOE        (2): CUDA + MMQ + MoE support
 * - LLAMA_BUILD_CUDA_FLASH_ATTENTION(3): CUDA + Flash Attention optimized
 * - LLAMA_BUILD_CPU_ONLY            (4): CPU-only fallback build
 * - LLAMA_BUILD_METAL_OPTIMIZED     (5): Metal + Apple GPU optimized
 */
#ifndef LLAMA_FEATURE_FREEZE_PROFILE
#define LLAMA_FEATURE_FREEZE_PROFILE LLAMA_BUILD_CUDA_CUBLAS_DENSE
#endif

// Profile enumeration constants
#define LLAMA_BUILD_CUDA_CUBLAS_DENSE    1
#define LLAMA_BUILD_CUDA_MMQ_MOE         2
#define LLAMA_BUILD_CUDA_FLASH_ATTENTION 3
#define LLAMA_BUILD_CPU_ONLY             4
#define LLAMA_BUILD_METAL_OPTIMIZED      5

/**
 * Feature freeze compilation mode.
 * Enables compile-time elimination of unused feature paths.
 */
#ifndef LLAMA_FEATURE_FREEZE_ENABLED
#define LLAMA_FEATURE_FREEZE_ENABLED 1
#endif

/**
 * Compile-time metric collection for feature freeze validation.
 */
#ifndef LLAMA_FEATURE_FREEZE_COLLECT_METRICS
#define LLAMA_FEATURE_FREEZE_COLLECT_METRICS 1
#endif

/**
 * Runtime validation of feature compatibility at startup.
 */
#ifndef LLAMA_FEATURE_FREEZE_VALIDATE_STARTUP
#define LLAMA_FEATURE_FREEZE_VALIDATE_STARTUP 1
#endif

// ============================================================================
// BUILD PROFILE FEATURE MATRIX (Compile-time resolved)
// ============================================================================

/**
 * Profile-specific feature enablement.
 * These are resolved at compile time and immutable at runtime.
 */

#if LLAMA_FEATURE_FREEZE_PROFILE == LLAMA_BUILD_CUDA_CUBLAS_DENSE
    #define LLAMA_FEATURE_CUDA_ENABLED              1
    #define LLAMA_FEATURE_CPU_ENABLED               0
    #define LLAMA_FEATURE_CUBLAS_ENABLED            1
    #define LLAMA_FEATURE_MMQ_ENABLED               0
    #define LLAMA_FEATURE_FLASH_ATTENTION_ENABLED   0
    #define LLAMA_FEATURE_CUDA_GRAPHS_ENABLED       1
    #define LLAMA_FEATURE_HYBRID_MEMORY_ENABLED     0
    #define LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED 0
    #define LLAMA_FEATURE_DETERMINISM_STRICT        0
    #define LLAMA_FEATURE_EXPERIMENTAL_KERNELS      0
    #define LLAMA_FEATURE_PROFILE_NAME "CUDA_cuBLAS_Dense"

#elif LLAMA_FEATURE_FREEZE_PROFILE == LLAMA_BUILD_CUDA_MMQ_MOE
    #define LLAMA_FEATURE_CUDA_ENABLED              1
    #define LLAMA_FEATURE_CPU_ENABLED               0
    #define LLAMA_FEATURE_CUBLAS_ENABLED            0
    #define LLAMA_FEATURE_MMQ_ENABLED               1
    #define LLAMA_FEATURE_FLASH_ATTENTION_ENABLED   0
    #define LLAMA_FEATURE_CUDA_GRAPHS_ENABLED       1
    #define LLAMA_FEATURE_HYBRID_MEMORY_ENABLED     0
    #define LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED 1
    #define LLAMA_FEATURE_DETERMINISM_STRICT        0
    #define LLAMA_FEATURE_EXPERIMENTAL_KERNELS      1
    #define LLAMA_FEATURE_PROFILE_NAME "CUDA_MMQ_MoE"

#elif LLAMA_FEATURE_FREEZE_PROFILE == LLAMA_BUILD_CUDA_FLASH_ATTENTION
    #define LLAMA_FEATURE_CUDA_ENABLED              1
    #define LLAMA_FEATURE_CPU_ENABLED               0
    #define LLAMA_FEATURE_CUBLAS_ENABLED            0
    #define LLAMA_FEATURE_MMQ_ENABLED               1
    #define LLAMA_FEATURE_FLASH_ATTENTION_ENABLED   1
    #define LLAMA_FEATURE_CUDA_GRAPHS_ENABLED       1
    #define LLAMA_FEATURE_HYBRID_MEMORY_ENABLED     0
    #define LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED 1
    #define LLAMA_FEATURE_DETERMINISM_STRICT        0
    #define LLAMA_FEATURE_EXPERIMENTAL_KERNELS      1
    #define LLAMA_FEATURE_PROFILE_NAME "CUDA_FlashAttention"

#elif LLAMA_FEATURE_FREEZE_PROFILE == LLAMA_BUILD_CPU_ONLY
    #define LLAMA_FEATURE_CUDA_ENABLED              0
    #define LLAMA_FEATURE_CPU_ENABLED               1
    #define LLAMA_FEATURE_CUBLAS_ENABLED            0
    #define LLAMA_FEATURE_MMQ_ENABLED               0
    #define LLAMA_FEATURE_FLASH_ATTENTION_ENABLED   0
    #define LLAMA_FEATURE_CUDA_GRAPHS_ENABLED       0
    #define LLAMA_FEATURE_HYBRID_MEMORY_ENABLED     0
    #define LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED 0
    #define LLAMA_FEATURE_DETERMINISM_STRICT        0
    #define LLAMA_FEATURE_EXPERIMENTAL_KERNELS      0
    #define LLAMA_FEATURE_PROFILE_NAME "CPU_Only"

#elif LLAMA_FEATURE_FREEZE_PROFILE == LLAMA_BUILD_METAL_OPTIMIZED
    #define LLAMA_FEATURE_CUDA_ENABLED              0
    #define LLAMA_FEATURE_CPU_ENABLED               0
    #define LLAMA_FEATURE_CUBLAS_ENABLED            0
    #define LLAMA_FEATURE_MMQ_ENABLED               0
    #define LLAMA_FEATURE_FLASH_ATTENTION_ENABLED   1
    #define LLAMA_FEATURE_CUDA_GRAPHS_ENABLED       0
    #define LLAMA_FEATURE_HYBRID_MEMORY_ENABLED     1
    #define LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED 0
    #define LLAMA_FEATURE_DETERMINISM_STRICT        0
    #define LLAMA_FEATURE_EXPERIMENTAL_KERNELS      0
    #define LLAMA_FEATURE_PROFILE_NAME "Metal_Optimized"

#else
    #error "Invalid LLAMA_FEATURE_FREEZE_PROFILE: must be 1-5"
#endif

// ============================================================================
// COMPILE-TIME VALIDATION MACROS
// ============================================================================

/**
 * LLAMA_FEATURE_REQUIRES_CUDA: Assert CUDA is enabled at compile time
 * Usage: #include "llama-feature-freeze.h"
 *        LLAMA_FEATURE_REQUIRES_CUDA("my_function requires CUDA backend")
 */
#if LLAMA_FEATURE_FREEZE_ENABLED
    #define LLAMA_FEATURE_REQUIRES_CUDA(reason) \
        do { \
            static_assert(LLAMA_FEATURE_CUDA_ENABLED == 1, \
                "Feature freeze error: " reason " requires CUDA, but CUDA is disabled. " \
                "Rebuild with LLAMA_FEATURE_FREEZE_PROFILE=1 (CUDA_cuBLAS) or 2 (CUDA_MMQ)"); \
        } while(0)

    /**
     * LLAMA_FEATURE_REQUIRES_CUBLAS: Assert cuBLAS is enabled at compile time
     */
    #define LLAMA_FEATURE_REQUIRES_CUBLAS(reason) \
        do { \
            static_assert(LLAMA_FEATURE_CUBLAS_ENABLED == 1, \
                "Feature freeze error: " reason " requires cuBLAS, but cuBLAS is disabled. " \
                "Rebuild with LLAMA_FEATURE_FREEZE_PROFILE=1"); \
        } while(0)

    /**
     * LLAMA_FEATURE_REQUIRES_MMQ: Assert MMQ kernels are enabled at compile time
     */
    #define LLAMA_FEATURE_REQUIRES_MMQ(reason) \
        do { \
            static_assert(LLAMA_FEATURE_MMQ_ENABLED == 1, \
                "Feature freeze error: " reason " requires MMQ kernels, but MMQ is disabled. " \
                "Rebuild with LLAMA_FEATURE_FREEZE_PROFILE=2 (CUDA_MMQ) or 3 (Flash Attention)"); \
        } while(0)

    /**
     * LLAMA_FEATURE_REQUIRES_FLASH_ATTENTION: Assert Flash Attention is enabled
     */
    #define LLAMA_FEATURE_REQUIRES_FLASH_ATTENTION(reason) \
        do { \
            static_assert(LLAMA_FEATURE_FLASH_ATTENTION_ENABLED == 1, \
                "Feature freeze error: " reason " requires Flash Attention, but it is disabled. " \
                "Rebuild with LLAMA_FEATURE_FREEZE_PROFILE=3"); \
        } while(0)

    /**
     * LLAMA_FEATURE_REQUIRES_CPU: Assert CPU backend is enabled at compile time
     */
    #define LLAMA_FEATURE_REQUIRES_CPU(reason) \
        do { \
            static_assert(LLAMA_FEATURE_CPU_ENABLED == 1, \
                "Feature freeze error: " reason " requires CPU backend, but CPU backend is disabled. " \
                "Rebuild with LLAMA_FEATURE_FREEZE_PROFILE=4"); \
        } while(0)
#else
    #define LLAMA_FEATURE_REQUIRES_CUDA(reason)
    #define LLAMA_FEATURE_REQUIRES_CUBLAS(reason)
    #define LLAMA_FEATURE_REQUIRES_MMQ(reason)
    #define LLAMA_FEATURE_REQUIRES_FLASH_ATTENTION(reason)
    #define LLAMA_FEATURE_REQUIRES_CPU(reason)
#endif

// ============================================================================
// COMPILE-TIME FEATURE CHECKS (No runtime overhead)
// ============================================================================

/**
 * LLAMA_FEATURE_CUDA_ONLY: Compile-time assertion for CUDA-only code sections
 * Usage: #if LLAMA_FEATURE_CUDA_ONLY
 */
#define LLAMA_FEATURE_CUDA_ONLY \
    (LLAMA_FEATURE_FREEZE_ENABLED && LLAMA_FEATURE_CUDA_ENABLED == 1)

/**
 * LLAMA_FEATURE_CPU_ONLY: Compile-time assertion for CPU-only code sections
 */
#define LLAMA_FEATURE_CPU_ONLY \
    (LLAMA_FEATURE_FREEZE_ENABLED && LLAMA_FEATURE_CPU_ENABLED == 1)

/**
 * LLAMA_FEATURE_BACKEND_EXCLUSIVE: Validates exactly one backend is enabled
 */
#define LLAMA_FEATURE_BACKEND_EXCLUSIVE \
    ((LLAMA_FEATURE_CUDA_ENABLED + LLAMA_FEATURE_CPU_ENABLED) == 1)

/**
 * LLAMA_FEATURE_IMMUTABLE_AT_RUNTIME: Mark functions that cannot change behavior
 */
#define LLAMA_FEATURE_IMMUTABLE_AT_RUNTIME \
    __attribute__((const)) __attribute__((always_inline))

// ============================================================================
// COMPILE-TIME ELIMINATION MACROS
// ============================================================================

/**
 * LLAMA_FEATURE_DISPATCH_CUDA: Dispatch to CUDA code at compile time
 * Usage: LLAMA_FEATURE_DISPATCH_CUDA( cuda_function() ) else { cpu_function(); }
 */
#if LLAMA_FEATURE_CUDA_ONLY
    #define LLAMA_FEATURE_DISPATCH_CUDA(cuda_code) \
        do { cuda_code; } while(0)
#else
    #define LLAMA_FEATURE_DISPATCH_CUDA(cuda_code) \
        do { } while(0)
#endif

/**
 * LLAMA_FEATURE_DISPATCH_CPU: Dispatch to CPU code at compile time
 */
#if LLAMA_FEATURE_CPU_ONLY
    #define LLAMA_FEATURE_DISPATCH_CPU(cpu_code) \
        do { cpu_code; } while(0)
#else
    #define LLAMA_FEATURE_DISPATCH_CPU(cpu_code) \
        do { } while(0)
#endif

/**
 * LLAMA_FEATURE_COMPILE_OUT: Compile out code entirely if feature disabled
 * Usage: #if LLAMA_FEATURE_COMPILE_OUT(SPECULATIVE_DECODE_ENABLED)
 */
#define LLAMA_FEATURE_COMPILE_OUT(feature) \
    (LLAMA_FEATURE_FREEZE_ENABLED && LLAMA_FEATURE_##feature == 1)

// ============================================================================
// FEATURE DISPATCH TABLE STRUCTURE
// ============================================================================

/**
 * Feature capabilities resolved at build time.
 * This structure is immutable and set at compile time only.
 */
typedef struct {
    // Backend selection
    uint32_t cuda_enabled : 1;
    uint32_t cpu_enabled : 1;

    // CUDA-specific features
    uint32_t cublas_enabled : 1;
    uint32_t mmq_enabled : 1;
    uint32_t flash_attention_enabled : 1;
    uint32_t cuda_graphs_enabled : 1;

    // Memory and computation modes
    uint32_t hybrid_memory_enabled : 1;
    uint32_t speculative_decode_enabled : 1;
    uint32_t determinism_strict : 1;
    uint32_t experimental_kernels : 1;

    // Reserved for future features
    uint32_t reserved : 22;
} llama_feature_freeze_capabilities;

/**
 * Build-time feature dispatch table.
 * Provides compile-time resolution of feature availability.
 */
typedef struct {
    // Immutable build profile identifier
    const char* profile_name;
    uint32_t profile_id;

    // Feature capability flags (compile-time resolved)
    llama_feature_freeze_capabilities features;

    // Build-time resolved feature vectors
    void (*dispatch_compute)(void);  // Backend-specific compute dispatch
    void (*dispatch_memory)(void);   // Backend-specific memory management
    void (*dispatch_sync)(void);     // Backend-specific synchronization

    // Validation function pointers
    int (*validate_hardware)(void);  // Validate hardware compatibility at startup
    int (*validate_features)(void);  // Validate feature consistency
} llama_feature_freeze_dispatch_table;

// ============================================================================
// METRICS AND VALIDATION STRUCTURES
// ============================================================================

/**
 * Feature freeze metrics (compile-time collection)
 */
typedef struct {
    std::atomic<uint64_t> build_time_features_resolved;
    std::atomic<uint64_t> runtime_feature_checks_blocked;
    std::atomic<uint64_t> compile_out_paths_eliminated;
    std::atomic<uint64_t> decode_branches_removed;
    std::atomic<uint64_t> feature_symbol_lookups_eliminated;
} llama_feature_freeze_metrics;

/**
 * Feature freeze state machine states
 */
#define LLAMA_FEATURE_FREEZE_STATE_UNINITIALIZED    0
#define LLAMA_FEATURE_FREEZE_STATE_VALIDATED        1
#define LLAMA_FEATURE_FREEZE_STATE_IMMUTABLE        2
#define LLAMA_FEATURE_FREEZE_STATE_HARDWARE_MISMATCH 3
#define LLAMA_FEATURE_FREEZE_STATE_ERROR            4

/**
 * Feature freeze startup validation state
 */
typedef struct {
    uint32_t state;  // Current validation state
    int hardware_compatible;  // 1 = compatible, 0 = incompatible
    int validation_error_code;  // Error code if validation failed
    const char* validation_error_message;  // Error message if validation failed
} llama_feature_freeze_validation_state;

// ============================================================================
// C API FUNCTION DECLARATIONS
// ============================================================================

/**
 * Initialize feature freeze system at startup.
 * Validates build profile against hardware capabilities.
 * Must be called exactly once during application initialization.
 *
 * Returns:
 * - 0 on success (features are frozen and immutable)
 * - Non-zero error code if hardware mismatch or validation failed
 * - Aborts process if CUDA required but unavailable and fallback disabled
 */
int llama_feature_freeze_init(void);

/**
 * Get the current feature freeze validation state.
 * Thread-safe, constant-time lookup.
 *
 * Returns: Pointer to immutable validation state structure
 */
const llama_feature_freeze_validation_state*
llama_feature_freeze_get_validation_state(void);

/**
 * Get the compiled build profile identifier.
 * Resolved entirely at compile time.
 *
 * Returns: Profile ID (1-5) matching compile-time configuration
 */
uint32_t llama_feature_freeze_get_profile(void);

/**
 * Get human-readable build profile name.
 * Resolved entirely at compile time.
 *
 * Returns: Static string describing the build profile
 */
const char* llama_feature_freeze_get_profile_name(void);

/**
 * Get feature capability flags.
 * All flags resolved at compile time, immutable at runtime.
 *
 * Returns: Pointer to immutable feature capability structure
 */
const llama_feature_freeze_capabilities*
llama_feature_freeze_get_features(void);

/**
 * Validate feature freeze integrity.
 * Checks that all feature flags match compile-time expectations.
 * Should be called during startup verification.
 *
 * Returns:
 * - 0 if all features match build configuration
 * - Non-zero if any mismatch detected (indicates binary corruption)
 */
int llama_feature_freeze_validate_integrity(void);

/**
 * Get feature freeze metrics.
 * Tracks compile-time and runtime feature resolution counts.
 * Thread-safe, uses lock-free atomics.
 *
 * Returns: Pointer to metrics structure
 */
const llama_feature_freeze_metrics*
llama_feature_freeze_get_metrics(void);

/**
 * Log feature freeze configuration to stderr.
 * Useful for debugging and validation.
 * Call after llama_feature_freeze_init() for complete information.
 */
void llama_feature_freeze_log_config(void);

/**
 * Check if a specific feature is enabled.
 * This is primarily for runtime assertion purposes.
 * Most code should use compile-time checks (#if) instead.
 *
 * Parameters:
 * - feature_id: Feature identifier (platform-specific)
 *
 * Returns:
 * - 1 if feature is enabled
 * - 0 if feature is disabled
 */
int llama_feature_freeze_is_feature_enabled(uint32_t feature_id);

/**
 * Hardware compatibility validation callback.
 * Called during initialization to verify GPU/hardware is compatible.
 * Must be implemented per-backend.
 *
 * Returns:
 * - 0 if hardware is compatible
 * - Non-zero error code if incompatible
 */
typedef int (*llama_feature_freeze_validate_hardware_fn)(void);

/**
 * Register custom hardware validation callback.
 * Allows backends to provide custom validation logic.
 * Must be called before llama_feature_freeze_init().
 */
void llama_feature_freeze_register_validator(
    llama_feature_freeze_validate_hardware_fn validator);

// ============================================================================
// STATIC ASSERTION COMPILE-TIME CHECKS
// ============================================================================

// Verify that exactly one backend is enabled
#if LLAMA_FEATURE_FREEZE_ENABLED
#if !LLAMA_FEATURE_BACKEND_EXCLUSIVE
#error "Feature freeze error: Exactly one backend must be enabled at compile time. " \
        "Check LLAMA_FEATURE_FREEZE_PROFILE setting in CMakeLists.txt"
#endif
#endif

// Verify profile is valid
#if LLAMA_FEATURE_FREEZE_ENABLED
#if (LLAMA_FEATURE_FREEZE_PROFILE < 1 || LLAMA_FEATURE_FREEZE_PROFILE > 5)
#error "Feature freeze error: LLAMA_FEATURE_FREEZE_PROFILE must be 1-5"
#endif
#endif

// Verify CUDA-specific constraints
#if LLAMA_FEATURE_CUDA_ENABLED == 0
    // CPU-only builds must not have CUDA features enabled
    #if LLAMA_FEATURE_CUBLAS_ENABLED == 1 || LLAMA_FEATURE_MMQ_ENABLED == 1 || \
        LLAMA_FEATURE_CUDA_GRAPHS_ENABLED == 1
    #error "Feature freeze error: CUDA features enabled in CPU-only build"
    #endif
#endif

#ifdef __cplusplus
}
#endif
