/**
 * llama-feature-freeze.cpp
 *
 * Runtime implementation of compile-time feature freeze system.
 * Enforces immutable feature configuration and validates hardware compatibility.
 *
 * This file implements the startup validation and metrics tracking for feature
 * freeze configuration. All actual feature routing is resolved at compile time
 * through preprocessor directives in llama-feature-freeze.h.
 */

#include "llama-feature-freeze.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>

// ============================================================================
// STATIC STATE AND INITIALIZATION
// ============================================================================

/**
 * Global metrics structure for feature freeze validation
 */
static llama_feature_freeze_metrics g_feature_freeze_metrics = {
    std::atomic<uint64_t>(0),  // build_time_features_resolved
    std::atomic<uint64_t>(0),  // runtime_feature_checks_blocked
    std::atomic<uint64_t>(0),  // compile_out_paths_eliminated
    std::atomic<uint64_t>(0),  // decode_branches_removed
    std::atomic<uint64_t>(0),  // feature_symbol_lookups_eliminated
};

/**
 * Global validation state
 */
static llama_feature_freeze_validation_state g_validation_state = {
    LLAMA_FEATURE_FREEZE_STATE_UNINITIALIZED,
    0,
    0,
    nullptr
};

/**
 * Global initialization mutex
 */
static std::mutex g_feature_freeze_mutex;

/**
 * Feature freeze initialization flag
 */
static std::atomic<int> g_feature_freeze_initialized(0);

/**
 * Custom hardware validator function (optional)
 */
static llama_feature_freeze_validate_hardware_fn g_custom_validator = nullptr;

// ============================================================================
// COMPILE-TIME FEATURE CAPABILITY TABLE
// ============================================================================

/**
 * Build-time resolved feature capabilities.
 * These values are constants determined at compile time and immutable.
 */
static const llama_feature_freeze_capabilities g_compile_time_features = {
    LLAMA_FEATURE_CUDA_ENABLED,           // cuda_enabled
    LLAMA_FEATURE_CPU_ENABLED,            // cpu_enabled
    LLAMA_FEATURE_CUBLAS_ENABLED,         // cublas_enabled
    LLAMA_FEATURE_MMQ_ENABLED,            // mmq_enabled
    LLAMA_FEATURE_FLASH_ATTENTION_ENABLED, // flash_attention_enabled
    LLAMA_FEATURE_CUDA_GRAPHS_ENABLED,    // cuda_graphs_enabled
    LLAMA_FEATURE_HYBRID_MEMORY_ENABLED,  // hybrid_memory_enabled
    LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED, // speculative_decode_enabled
    LLAMA_FEATURE_DETERMINISM_STRICT,     // determinism_strict
    LLAMA_FEATURE_EXPERIMENTAL_KERNELS,   // experimental_kernels
    0                                      // reserved
};

/**
 * Feature profile name from compile-time configuration
 */
static const char* g_profile_name_compiled = LLAMA_FEATURE_PROFILE_NAME;

/**
 * Feature profile ID from compile-time configuration
 */
static const uint32_t g_profile_id_compiled = LLAMA_FEATURE_FREEZE_PROFILE;

// ============================================================================
// FEATURE DISPATCH TABLE (Compile-time resolved)
// ============================================================================

/**
 * Stub implementations for dispatch functions.
 * Real backends implement these in their own compilation units.
 */
static void llama_feature_dispatch_compute_stub(void) {
    // Stub: actual compute dispatch resolved at compile time
}

static void llama_feature_dispatch_memory_stub(void) {
    // Stub: actual memory dispatch resolved at compile time
}

static void llama_feature_dispatch_sync_stub(void) {
    // Stub: actual sync dispatch resolved at compile time
}

/**
 * Default hardware validation (permissive).
 * Returns 0 (compatible) by default.
 * Backends can override via llama_feature_freeze_register_validator().
 */
static int llama_feature_validate_hardware_default(void) {
    // Default: accept all hardware configurations
    // Backends should override this with real validation
    return 0;
}

/**
 * Default feature consistency validation.
 * Verifies compiled feature flags match expected configuration.
 */
static int llama_feature_validate_features_default(void) {
    // Verify feature counts match expectations
    int feature_count = 0;

    if (g_compile_time_features.cuda_enabled) feature_count++;
    if (g_compile_time_features.cpu_enabled) feature_count++;

    // Exactly one backend must be enabled
    if (feature_count != 1) {
        fprintf(stderr, "FATAL: Feature freeze error - invalid backend configuration\n");
        return -1;
    }

    return 0;
}

/**
 * Global feature dispatch table.
 * Completely resolved at compile time.
 */
static const llama_feature_freeze_dispatch_table g_feature_dispatch_table = {
    g_profile_name_compiled,
    g_profile_id_compiled,
    g_compile_time_features,
    llama_feature_dispatch_compute_stub,
    llama_feature_dispatch_memory_stub,
    llama_feature_dispatch_sync_stub,
    llama_feature_validate_hardware_default,
    llama_feature_validate_features_default
};

// ============================================================================
// BUILD-TIME FEATURE RESOLUTION TRACKING
// ============================================================================

/**
 * Record build-time feature resolution.
 * Called during initialization to count resolved features.
 */
static void llama_feature_record_build_time_resolution(void) {
    int features_resolved = 0;

    // Count resolved backend selection
    if (LLAMA_FEATURE_CUDA_ENABLED) features_resolved++;
    if (LLAMA_FEATURE_CPU_ENABLED) features_resolved++;

    // Count resolved CUDA features
    if (LLAMA_FEATURE_CUDA_ENABLED) {
        if (LLAMA_FEATURE_CUBLAS_ENABLED) features_resolved++;
        if (LLAMA_FEATURE_MMQ_ENABLED) features_resolved++;
        if (LLAMA_FEATURE_FLASH_ATTENTION_ENABLED) features_resolved++;
        if (LLAMA_FEATURE_CUDA_GRAPHS_ENABLED) features_resolved++;
    }

    // Count resolved computation modes
    if (LLAMA_FEATURE_HYBRID_MEMORY_ENABLED) features_resolved++;
    if (LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED) features_resolved++;
    if (LLAMA_FEATURE_DETERMINISM_STRICT) features_resolved++;
    if (LLAMA_FEATURE_EXPERIMENTAL_KERNELS) features_resolved++;

    g_feature_freeze_metrics.build_time_features_resolved.store(
        features_resolved,
        std::memory_order_relaxed
    );
}

/**
 * Record eliminated compile-out paths.
 * Counts features that were disabled and compiled out.
 */
static void llama_feature_record_eliminated_paths(void) {
    int paths_eliminated = 0;

    // Count disabled features compiled out
    if (!LLAMA_FEATURE_CPU_ENABLED) paths_eliminated++;
    if (!LLAMA_FEATURE_CUDA_ENABLED) paths_eliminated++;
    if (!LLAMA_FEATURE_CUBLAS_ENABLED) paths_eliminated++;
    if (!LLAMA_FEATURE_MMQ_ENABLED) paths_eliminated++;
    if (!LLAMA_FEATURE_FLASH_ATTENTION_ENABLED) paths_eliminated++;
    if (!LLAMA_FEATURE_CUDA_GRAPHS_ENABLED) paths_eliminated++;
    if (!LLAMA_FEATURE_HYBRID_MEMORY_ENABLED) paths_eliminated++;
    if (!LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED) paths_eliminated++;
    if (!LLAMA_FEATURE_EXPERIMENTAL_KERNELS) paths_eliminated++;

    g_feature_freeze_metrics.compile_out_paths_eliminated.store(
        paths_eliminated,
        std::memory_order_relaxed
    );
}

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

/**
 * Validate hardware compatibility with build configuration.
 * Checks GPU/accelerator availability if CUDA/Metal is required.
 *
 * Returns:
 * - 0 if hardware compatible
 * - Non-zero error code if incompatible
 */
static int llama_feature_validate_hardware(void) {
    // Call custom validator if registered
    if (g_custom_validator != nullptr) {
        int result = g_custom_validator();
        if (result != 0) {
            return result;
        }
    }

    // Call dispatch table validator
    if (g_feature_dispatch_table.validate_hardware != nullptr) {
        int result = g_feature_dispatch_table.validate_hardware();
        if (result != 0) {
            return result;
        }
    }

    // Default: hardware compatible
    return 0;
}

/**
 * Validate feature consistency.
 * Ensures all enabled features are compatible with each other.
 *
 * Returns:
 * - 0 if all features consistent
 * - Non-zero error code if inconsistent
 */
static int llama_feature_validate_consistency(void) {
    // Call dispatch table consistency validator
    if (g_feature_dispatch_table.validate_features != nullptr) {
        return g_feature_dispatch_table.validate_features();
    }

    return 0;
}

/**
 * Validate feature freeze binary integrity.
 * Ensures compiled features match expected configuration.
 *
 * Returns:
 * - 0 if features valid
 * - Non-zero if corruption detected
 */
static int llama_feature_validate_binary_integrity(void) {
    // Verify backend exclusivity
#if LLAMA_FEATURE_CUDA_ENABLED + LLAMA_FEATURE_CPU_ENABLED != 1
    fprintf(stderr, "FATAL: Feature freeze binary integrity error - "
                    "invalid backend configuration\n");
    return -1;
#endif

    // Verify CUDA-specific constraints
#if LLAMA_FEATURE_CUDA_ENABLED == 0
    // CPU-only builds must not have CUDA features
#if LLAMA_FEATURE_CUBLAS_ENABLED == 1
    fprintf(stderr, "FATAL: Feature freeze binary integrity error - "
                    "cuBLAS enabled in CPU-only build\n");
    return -2;
#endif
#if LLAMA_FEATURE_MMQ_ENABLED == 1
    fprintf(stderr, "FATAL: Feature freeze binary integrity error - "
                    "MMQ enabled in CPU-only build\n");
    return -3;
#endif
#endif

    return 0;
}

// ============================================================================
// PUBLIC API IMPLEMENTATION
// ============================================================================

/**
 * Initialize feature freeze system.
 * Called once at application startup.
 */
int llama_feature_freeze_init(void) {
    // Check if already initialized
    int expected = 0;
    if (!g_feature_freeze_initialized.compare_exchange_strong(
            expected, 1, std::memory_order_acquire)) {
        // Already initialized, return success
        return (g_validation_state.state == LLAMA_FEATURE_FREEZE_STATE_IMMUTABLE) ? 0 : -1;
    }

    // Acquire lock for initialization
    std::lock_guard<std::mutex> lock(g_feature_freeze_mutex);

    // Record feature resolution counts
    llama_feature_record_build_time_resolution();
    llama_feature_record_eliminated_paths();

    // Validate binary integrity
    int integrity_result = llama_feature_validate_binary_integrity();
    if (integrity_result != 0) {
        g_validation_state.state = LLAMA_FEATURE_FREEZE_STATE_ERROR;
        g_validation_state.validation_error_code = integrity_result;
        g_validation_state.validation_error_message =
            "Binary integrity validation failed";
        fprintf(stderr, "FATAL: Feature freeze binary integrity error (code %d)\n",
                integrity_result);
        return integrity_result;
    }

    // Validate feature consistency
    int consistency_result = llama_feature_validate_consistency();
    if (consistency_result != 0) {
        g_validation_state.state = LLAMA_FEATURE_FREEZE_STATE_ERROR;
        g_validation_state.validation_error_code = consistency_result;
        g_validation_state.validation_error_message =
            "Feature consistency validation failed";
        fprintf(stderr, "FATAL: Feature freeze consistency error (code %d)\n",
                consistency_result);
        return consistency_result;
    }

    // Validate hardware compatibility
    int hardware_result = llama_feature_validate_hardware();
    if (hardware_result != 0) {
        g_validation_state.state = LLAMA_FEATURE_FREEZE_STATE_HARDWARE_MISMATCH;
        g_validation_state.hardware_compatible = 0;
        g_validation_state.validation_error_code = hardware_result;

        // Determine if fallback is possible
        #if LLAMA_FEATURE_CUDA_ENABLED
            // CUDA required but unavailable - abort
            g_validation_state.validation_error_message =
                "CUDA backend required but no compatible GPU found";
            fprintf(stderr, "FATAL: CUDA backend required but unavailable (code %d)\n",
                    hardware_result);
            return hardware_result;
        #elif LLAMA_FEATURE_CPU_ENABLED
            // CPU fallback always available
            fprintf(stderr, "WARNING: Hardware validation issue (code %d), "
                            "using CPU fallback\n", hardware_result);
            g_validation_state.validation_error_message =
                "Hardware check failed, using CPU fallback";
        #endif
    } else {
        g_validation_state.hardware_compatible = 1;
    }

    // Mark state as validated and immutable
    g_validation_state.state = LLAMA_FEATURE_FREEZE_STATE_IMMUTABLE;

    return 0;
}

/**
 * Get validation state
 */
const llama_feature_freeze_validation_state*
llama_feature_freeze_get_validation_state(void) {
    return &g_validation_state;
}

/**
 * Get compiled profile ID
 */
uint32_t llama_feature_freeze_get_profile(void) {
    return g_profile_id_compiled;
}

/**
 * Get profile name
 */
const char* llama_feature_freeze_get_profile_name(void) {
    return g_profile_name_compiled;
}

/**
 * Get feature capabilities
 */
const llama_feature_freeze_capabilities*
llama_feature_freeze_get_features(void) {
    return &g_compile_time_features;
}

/**
 * Validate integrity
 */
int llama_feature_freeze_validate_integrity(void) {
    return llama_feature_validate_binary_integrity();
}

/**
 * Get metrics
 */
const llama_feature_freeze_metrics*
llama_feature_freeze_get_metrics(void) {
    return &g_feature_freeze_metrics;
}

/**
 * Log configuration
 */
void llama_feature_freeze_log_config(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "=== Feature Freeze Configuration ===\n");
    fprintf(stderr, "Build Profile: %s (ID: %u)\n",
            g_profile_name_compiled,
            g_profile_id_compiled);
    fprintf(stderr, "\n");

    fprintf(stderr, "Enabled Features:\n");
    if (LLAMA_FEATURE_CUDA_ENABLED) fprintf(stderr, "  - CUDA backend\n");
    if (LLAMA_FEATURE_CPU_ENABLED) fprintf(stderr, "  - CPU backend\n");
    if (LLAMA_FEATURE_CUBLAS_ENABLED) fprintf(stderr, "  - cuBLAS kernels\n");
    if (LLAMA_FEATURE_MMQ_ENABLED) fprintf(stderr, "  - MMQ kernels\n");
    if (LLAMA_FEATURE_FLASH_ATTENTION_ENABLED) fprintf(stderr, "  - Flash Attention\n");
    if (LLAMA_FEATURE_CUDA_GRAPHS_ENABLED) fprintf(stderr, "  - CUDA Graphs\n");
    if (LLAMA_FEATURE_HYBRID_MEMORY_ENABLED) fprintf(stderr, "  - Hybrid Memory\n");
    if (LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED) fprintf(stderr, "  - Speculative Decoding\n");
    if (LLAMA_FEATURE_DETERMINISM_STRICT) fprintf(stderr, "  - Deterministic Mode\n");
    if (LLAMA_FEATURE_EXPERIMENTAL_KERNELS) fprintf(stderr, "  - Experimental Kernels\n");
    fprintf(stderr, "\n");

    fprintf(stderr, "Metrics:\n");
    fprintf(stderr, "  Build-time features resolved: %lu\n",
            g_feature_freeze_metrics.build_time_features_resolved.load(
                std::memory_order_relaxed));
    fprintf(stderr, "  Runtime feature checks blocked: %lu\n",
            g_feature_freeze_metrics.runtime_feature_checks_blocked.load(
                std::memory_order_relaxed));
    fprintf(stderr, "  Compile-out paths eliminated: %lu\n",
            g_feature_freeze_metrics.compile_out_paths_eliminated.load(
                std::memory_order_relaxed));
    fprintf(stderr, "  Decode branches removed: %lu\n",
            g_feature_freeze_metrics.decode_branches_removed.load(
                std::memory_order_relaxed));
    fprintf(stderr, "  Feature symbol lookups eliminated: %lu\n",
            g_feature_freeze_metrics.feature_symbol_lookups_eliminated.load(
                std::memory_order_relaxed));
    fprintf(stderr, "\n");

    fprintf(stderr, "Validation State: ");
    switch (g_validation_state.state) {
        case LLAMA_FEATURE_FREEZE_STATE_UNINITIALIZED:
            fprintf(stderr, "UNINITIALIZED\n");
            break;
        case LLAMA_FEATURE_FREEZE_STATE_VALIDATED:
            fprintf(stderr, "VALIDATED\n");
            break;
        case LLAMA_FEATURE_FREEZE_STATE_IMMUTABLE:
            fprintf(stderr, "IMMUTABLE (hardware %s)\n",
                    g_validation_state.hardware_compatible ? "compatible" : "incompatible");
            break;
        case LLAMA_FEATURE_FREEZE_STATE_HARDWARE_MISMATCH:
            fprintf(stderr, "HARDWARE MISMATCH\n");
            break;
        case LLAMA_FEATURE_FREEZE_STATE_ERROR:
            fprintf(stderr, "ERROR\n");
            if (g_validation_state.validation_error_message) {
                fprintf(stderr, "  Error: %s (code %d)\n",
                        g_validation_state.validation_error_message,
                        g_validation_state.validation_error_code);
            }
            break;
        default:
            fprintf(stderr, "UNKNOWN\n");
            break;
    }
    fprintf(stderr, "=================================\n\n");
}

/**
 * Check if feature is enabled
 */
int llama_feature_freeze_is_feature_enabled(uint32_t feature_id) {
    // Map feature IDs to feature flags
    // Note: Use compile-time checks (#if) instead of this function in hot paths

    switch (feature_id) {
        case 1: return LLAMA_FEATURE_CUDA_ENABLED;
        case 2: return LLAMA_FEATURE_CPU_ENABLED;
        case 3: return LLAMA_FEATURE_CUBLAS_ENABLED;
        case 4: return LLAMA_FEATURE_MMQ_ENABLED;
        case 5: return LLAMA_FEATURE_FLASH_ATTENTION_ENABLED;
        case 6: return LLAMA_FEATURE_CUDA_GRAPHS_ENABLED;
        case 7: return LLAMA_FEATURE_HYBRID_MEMORY_ENABLED;
        case 8: return LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED;
        case 9: return LLAMA_FEATURE_DETERMINISM_STRICT;
        case 10: return LLAMA_FEATURE_EXPERIMENTAL_KERNELS;
        default: return 0;
    }
}

/**
 * Register custom hardware validator
 */
void llama_feature_freeze_register_validator(
        llama_feature_freeze_validate_hardware_fn validator) {
    // Only allow registration before initialization
    if (g_feature_freeze_initialized.load(std::memory_order_acquire) == 0) {
        g_custom_validator = validator;
    } else {
        fprintf(stderr, "WARNING: Feature freeze already initialized, "
                        "validator registration ignored\n");
    }
}

// ============================================================================
// COMPILE-TIME ASSERTION VALIDATION
// ============================================================================

// Static assertions to catch configuration errors at compile time
static_assert(LLAMA_FEATURE_FREEZE_PROFILE >= 1 && LLAMA_FEATURE_FREEZE_PROFILE <= 5,
              "Invalid LLAMA_FEATURE_FREEZE_PROFILE");

static_assert(LLAMA_FEATURE_CUDA_ENABLED + LLAMA_FEATURE_CPU_ENABLED == 1,
              "Exactly one backend must be enabled");

// Verify CUDA-specific constraints at compile time
#if LLAMA_FEATURE_CUDA_ENABLED == 0
static_assert(LLAMA_FEATURE_CUBLAS_ENABLED == 0, "cuBLAS requires CUDA");
static_assert(LLAMA_FEATURE_MMQ_ENABLED == 0, "MMQ requires CUDA");
static_assert(LLAMA_FEATURE_CUDA_GRAPHS_ENABLED == 0, "CUDA Graphs requires CUDA");
#endif
