#pragma once

/**
 * llama-debug-stripping.h
 *
 * Complete elimination of diagnostic instrumentation from decode-critical path.
 * Implements compile-time guards for all debug/tracing code with fallback runtime
 * configuration, preventing branch mispredictions and instruction-cache pollution.
 *
 * ENFORCEMENT RULES (12 Total):
 *
 * 1. Define Decode Hot Path Scope
 *    - Decode loop progression
 *    - Graph execution dispatch
 *    - CUDA kernel launch wrappers
 *    - Sampling pipeline
 *    - KV-cache updates
 *    - Backend dispatch logic
 *    - All logging, assertions, tracing, timing must be eliminated
 *
 * 2. Compile-Out Logging in Decode
 *    - Replace runtime checks with compile-time guards
 *    - Instead of: if (g_log_level >= LOG_DEBUG) log_debug(...)
 *    - Use: #if defined(LLAMA_ENABLE_DEBUG) log_debug(...) #endif
 *    - Production builds: -DLLAMA_ENABLE_DEBUG=0
 *    - No log-level branching in decode
 *
 * 3. Remove Per-Token Timing Instrumentation
 *    - Eliminate: per-token timing, debug timers, profiling markers, token-step logging
 *    - No: auto t0 = now(); ... auto dt = now() - t0;
 *    - Timing must be optional, fully disabled in release builds
 *
 * 4. Remove GGML Debug Macros from CUDA Path
 *    - Disable: GGML_CUDA_DEBUG, kernel debug prints, device-side assertions
 *    - Remove printf, debug sync from ggml-cuda.cu
 *    - CUDA debug paths completely compiled out
 *
 * 5. Remove Conditional Branching for Debug Modes
 *    - Eliminate: if (ctx->debug_mode)
 *    - Replace with compile-time flags only
 *    - Hot path must not branch on debug config
 *
 * 6. Remove Verbose Sampling Traces
 *    - In sampling.cpp, sampling.cu
 *    - Remove: probability dumps, token ranking traces, penalty debug output
 *    - Sampling executes without trace overhead
 *
 * 7. Remove Server Decode Logging
 *    - Disable: request progress logging, per-token printouts, verbose streaming diagnostics
 *    - All decode logs off in production
 *
 * 8. Remove Assertions from Inner Loops
 *    - Replace: assert(condition) with pre-validation at graph build
 *    - Assertions never execute per token
 *
 * 9. Disable Runtime Feature Probes in Hot Path
 *    - Remove: backend capability checks, tensor-type validation per token, shape validation
 *    - Validate once at startup, never during decode
 *
 * 10. Strip Unused Symbols
 *     - Use -s (strip symbols) in build
 *     - Remove debug sections
 *     - Avoid profiling library linkage
 *     - Reduce I-cache pressure
 *
 * 11. Enforce Clean Release Build
 *     - Release config: NDEBUG defined, debug macros disabled
 *     - No trace instrumentation compiled
 *     - No profiling hooks enabled
 *     - Separate debug build variant
 *
 * 12. Expected Outcome
 *     - Fewer branch mispredictions (target: 0 debug branches in hot path)
 *     - Lower CPU overhead per token (target: <0.5% per-token overhead)
 *     - Reduced instruction-cache pressure
 *     - Cleaner decode loop
 *     - More stable tokens/sec
 *     - Minimal control-flow pipeline
 *
 * Key Metrics Tracked:
 * - Debug branches in hot path (target: 0)
 * - Logging statements in decode (target: 0)
 * - Per-token assertions (target: 0)
 * - Timing instrumentation (target: 0)
 * - Binary size reduction from stripping
 * - Instruction cache efficiency improvement
 */

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <array>
#include <functional>
#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// COMPILE-TIME CONFIGURATION MACROS
// ============================================================================

/**
 * Primary compile-time control: enable/disable debug instrumentation
 * Production builds: define as 0
 * Debug builds: define as 1
 * Default: auto-detect from NDEBUG
 */
#ifndef LLAMA_ENABLE_DEBUG
  #if defined(NDEBUG)
    #define LLAMA_ENABLE_DEBUG 0
  #else
    #define LLAMA_ENABLE_DEBUG 1
  #endif
#endif

/**
 * Enable CUDA debug instrumentation (kernel launch traces, device assertions)
 * Production: 0
 * Debug: 1
 */
#ifndef LLAMA_ENABLE_CUDA_DEBUG
  #define LLAMA_ENABLE_CUDA_DEBUG LLAMA_ENABLE_DEBUG
#endif

/**
 * Enable per-token timing instrumentation
 * This includes all LLAMA_TIMING_START/STOP macros
 * Production: 0
 * Debug: 1
 */
#ifndef LLAMA_ENABLE_TIMING_INSTRUMENTATION
  #define LLAMA_ENABLE_TIMING_INSTRUMENTATION LLAMA_ENABLE_DEBUG
#endif

/**
 * Enable per-token assertions in hot paths
 * Replace with compile-time validation only
 * Production: 0
 * Debug: 1
 */
#ifndef LLAMA_ENABLE_HOT_PATH_ASSERTIONS
  #define LLAMA_ENABLE_HOT_PATH_ASSERTIONS LLAMA_ENABLE_DEBUG
#endif

/**
 * Enable verbose sampling traces (probability dumps, penalty calculations)
 * Production: 0
 * Debug: 1
 */
#ifndef LLAMA_ENABLE_SAMPLING_TRACES
  #define LLAMA_ENABLE_SAMPLING_TRACES LLAMA_ENABLE_DEBUG
#endif

/**
 * Enable server decode progress logging
 * Production: 0
 * Debug: 1
 */
#ifndef LLAMA_ENABLE_SERVER_DECODE_LOGGING
  #define LLAMA_ENABLE_SERVER_DECODE_LOGGING LLAMA_ENABLE_DEBUG
#endif

/**
 * Enable runtime feature probes (backend checks, tensor validation)
 * Should be 0 in hot path - validate at startup only
 * Production: 0
 * Debug: 1
 */
#ifndef LLAMA_ENABLE_HOT_PATH_PROBES
  #define LLAMA_ENABLE_HOT_PATH_PROBES LLAMA_ENABLE_DEBUG
#endif

/**
 * Strip symbols from binary (reduce I-cache pressure)
 * Production: 1
 * Debug: 0
 */
#ifndef LLAMA_STRIP_SYMBOLS
  #define LLAMA_STRIP_SYMBOLS (1 - LLAMA_ENABLE_DEBUG)
#endif

/**
 * Enable collection of debug metrics (lock-free counters)
 * Can be enabled even in release builds for diagnostics
 * Production: 1 (for performance monitoring)
 * Debug: 1
 */
#ifndef LLAMA_ENABLE_DEBUG_METRICS
  #define LLAMA_ENABLE_DEBUG_METRICS 1
#endif

/**
 * Runtime fallback: if compile-time guard is not used, check this flag
 * Only used when no compile-time guard wraps the debug code
 * Allows graceful handling of old code that hasn't been annotated
 */
#ifndef LLAMA_RUNTIME_DEBUG_GUARD_FALLBACK
  #define LLAMA_RUNTIME_DEBUG_GUARD_FALLBACK LLAMA_ENABLE_DEBUG
#endif

// ============================================================================
// HOT PATH INSTRUMENTATION MACROS (No-op in release builds)
// ============================================================================

/**
 * Mark a function as part of decode hot path
 * In release builds: no-op
 * In debug builds: can collect metrics
 */
#if LLAMA_ENABLE_DEBUG
  #define LLAMA_HOT_PATH_MARKER() \
    do { \
      extern std::atomic<uint64_t> g_llama_debug_hot_path_entries; \
      g_llama_debug_hot_path_entries.fetch_add(1, std::memory_order_relaxed); \
    } while(0)
#else
  #define LLAMA_HOT_PATH_MARKER() do { } while(0)
#endif

/**
 * Start timing a hot path operation
 * Must be paired with LLAMA_TIMING_STOP
 * Completely compiled out in release builds
 */
#if LLAMA_ENABLE_TIMING_INSTRUMENTATION
  #define LLAMA_TIMING_START(name) \
    auto _timing_start_##name = std::chrono::high_resolution_clock::now()
#else
  #define LLAMA_TIMING_START(name) do { } while(0)
#endif

/**
 * Stop timing and record elapsed time
 * Completely compiled out in release builds
 */
#if LLAMA_ENABLE_TIMING_INSTRUMENTATION
  #define LLAMA_TIMING_STOP(name) \
    do { \
      auto _timing_stop_##name = std::chrono::high_resolution_clock::now(); \
      auto _elapsed = std::chrono::duration_cast<std::chrono::microseconds>( \
          _timing_stop_##name - _timing_start_##name).count(); \
      extern std::atomic<uint64_t> g_llama_debug_timing_##name; \
      g_llama_debug_timing_##name.fetch_add(_elapsed, std::memory_order_relaxed); \
    } while(0)
#else
  #define LLAMA_TIMING_STOP(name) do { } while(0)
#endif

/**
 * Emit debug log only in debug mode, completely compiled out in release
 */
#if LLAMA_ENABLE_DEBUG
  #define LLAMA_DEBUG_LOG(fmt, ...) \
    do { \
      fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__); \
    } while(0)
#else
  #define LLAMA_DEBUG_LOG(fmt, ...) do { } while(0)
#endif

/**
 * Assert that is completely compiled out in release builds
 * Use for hot path validation - should never fail
 */
#if LLAMA_ENABLE_HOT_PATH_ASSERTIONS
  #define LLAMA_HOT_PATH_ASSERT(cond) \
    do { \
      if (!(cond)) { \
        fprintf(stderr, "[FATAL] Hot path assertion failed: %s:%d\n", __FILE__, __LINE__); \
        abort(); \
      } \
    } while(0)
#else
  #define LLAMA_HOT_PATH_ASSERT(cond) do { } while(0)
#endif

/**
 * CUDA debug macro - completely compiled out in release
 */
#if LLAMA_ENABLE_CUDA_DEBUG
  #define LLAMA_CUDA_DEBUG_SYNC() cudaDeviceSynchronize()
  #define LLAMA_CUDA_DEBUG_PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
  #define LLAMA_CUDA_DEBUG_SYNC() do { } while(0)
  #define LLAMA_CUDA_DEBUG_PRINTF(fmt, ...) do { } while(0)
#endif

/**
 * Sampling debug trace - completely compiled out in release
 */
#if LLAMA_ENABLE_SAMPLING_TRACES
  #define LLAMA_SAMPLING_TRACE(fmt, ...) \
    do { \
      fprintf(stderr, "[SAMPLING] " fmt "\n", ##__VA_ARGS__); \
    } while(0)
#else
  #define LLAMA_SAMPLING_TRACE(fmt, ...) do { } while(0)
#endif

/**
 * Server decode logging - completely compiled out in release
 */
#if LLAMA_ENABLE_SERVER_DECODE_LOGGING
  #define LLAMA_SERVER_DECODE_LOG(fmt, ...) \
    do { \
      fprintf(stdout, "[SERVER_DECODE] " fmt "\n", ##__VA_ARGS__); \
    } while(0)
#else
  #define LLAMA_SERVER_DECODE_LOG(fmt, ...) do { } while(0)
#endif

/**
 * Hot path probe - runtime feature check
 * Compiled out in release builds to prevent branches
 */
#if LLAMA_ENABLE_HOT_PATH_PROBES
  #define LLAMA_HOT_PATH_PROBE(condition) (condition)
#else
  #define LLAMA_HOT_PATH_PROBE(condition) true
#endif

/**
 * Mark code as debug-only (completely eliminated in release)
 */
#if LLAMA_ENABLE_DEBUG
  #define LLAMA_DEBUG_ONLY(stmt) do { stmt } while(0)
#else
  #define LLAMA_DEBUG_ONLY(stmt) do { } while(0)
#endif

// ============================================================================
// RELEASE BUILD VALIDATION MACROS
// ============================================================================

/**
 * In release builds, validate condition at startup only (not per-token)
 * Used to replace assertions in hot paths
 */
#define LLAMA_RELEASE_BUILD_VALIDATE(cond, msg) \
  do { \
    if (!(cond)) { \
      fprintf(stderr, "[VALIDATION ERROR] %s\n", msg); \
      return -1; \
    } \
  } while(0)

/**
 * Startup validation - called once at initialization
 * Never called during decode
 */
#define LLAMA_STARTUP_VALIDATION(cond, msg) \
  do { \
    if (!(cond)) { \
      fprintf(stderr, "[STARTUP VALIDATION FAILED] %s:%d - %s\n", __FILE__, __LINE__, msg); \
      return -1; \
    } \
  } while(0)

// ============================================================================
// DEBUG BUILD INSTRUMENTATION STRUCTURES
// ============================================================================

/**
 * Per-token debug metrics collected in debug builds
 * Lock-free atomic updates
 */
typedef struct {
  // Hot path entry points
  std::atomic<uint64_t> decode_loop_entries;
  std::atomic<uint64_t> graph_execute_entries;
  std::atomic<uint64_t> cuda_kernel_launches;
  std::atomic<uint64_t> sampling_entries;

  // Debug operations prevented in release
  std::atomic<uint64_t> debug_logs_suppressed;
  std::atomic<uint64_t> timing_operations_skipped;
  std::atomic<uint64_t> assertions_skipped;
  std::atomic<uint64_t> feature_probes_skipped;

  // Timing statistics (debug only)
  std::atomic<uint64_t> decode_loop_total_ns;
  std::atomic<uint64_t> graph_execute_total_ns;
  std::atomic<uint64_t> sampling_total_ns;

  // Error tracking
  std::atomic<uint64_t> compile_time_guard_bypasses;
  std::atomic<uint64_t> runtime_guard_invocations;
} llama_debug_stripping_metrics;

// ============================================================================
// DEBUG CONFIGURATION STRUCTURE
// ============================================================================

/**
 * Runtime debug configuration
 * Can be modified at runtime for diagnostics, but doesn't affect compiled-out code
 */
typedef struct {
  // Flags for runtime fallback (only used if code wasn't wrapped with compile-time guard)
  bool enable_debug_logging;
  bool enable_timing_instrumentation;
  bool enable_hot_path_assertions;
  bool enable_sampling_traces;
  bool enable_server_decode_logging;

  // Behavior options
  bool abort_on_assertion_failure;
  bool collect_metrics;
  bool verbose_metrics;

  // Thresholds for warnings
  uint64_t max_timing_ns_per_token;
  uint64_t max_debug_operations_per_token;
} llama_debug_stripping_config;

/**
 * Debug stripping state
 */
typedef struct {
  llama_debug_stripping_config config;
  llama_debug_stripping_metrics metrics;
  std::atomic<bool> initialized;
  std::atomic<int32_t> initialization_status;
} llama_debug_stripping_state;

// ============================================================================
// GLOBAL STATE DECLARATIONS
// ============================================================================

/**
 * Global debug metrics (lock-free)
 */
extern std::atomic<uint64_t> g_llama_debug_hot_path_entries;
extern std::atomic<uint64_t> g_llama_debug_compile_time_guards_active;
extern std::atomic<uint64_t> g_llama_debug_runtime_guard_invocations;

/**
 * Timing metric counters (per-operation)
 * Only updated when LLAMA_ENABLE_TIMING_INSTRUMENTATION is 1
 */
extern std::atomic<uint64_t> g_llama_debug_timing_decode_loop;
extern std::atomic<uint64_t> g_llama_debug_timing_graph_execute;
extern std::atomic<uint64_t> g_llama_debug_timing_cuda_kernel;
extern std::atomic<uint64_t> g_llama_debug_timing_sampling;

/**
 * Global debug stripping state
 */
extern llama_debug_stripping_state g_llama_debug_stripping;

/**
 * Initialize debug stripping state with default configuration
 */
void init_debug_stripping_state(llama_debug_stripping_state & state);

// ============================================================================
// PUBLIC API
// ============================================================================

/**
 * Initialize debug stripping system
 * Called once at startup
 * Returns: 0 on success, -1 on error
 */
int llama_debug_stripping_init(void);

/**
 * Cleanup debug stripping system
 */
void llama_debug_stripping_fini(void);

/**
 * Set runtime debug configuration
 * Note: Only affects code that uses runtime fallback
 * Compile-time guarded code is not affected by this
 */
int llama_debug_stripping_set_config(const llama_debug_stripping_config * config);

/**
 * Get current debug configuration
 */
int llama_debug_stripping_get_config(llama_debug_stripping_config * config);

/**
 * Get collected metrics
 */
int llama_debug_stripping_get_metrics(llama_debug_stripping_metrics * metrics);

/**
 * Reset all metrics to zero
 */
int llama_debug_stripping_reset_metrics(void);

/**
 * Print summary of debug stripping effectiveness
 * Shows which instrumentation was compiled out
 */
int llama_debug_stripping_print_summary(FILE * fp);

/**
 * Validate that hot path has no debug instrumentation
 * Runs various compile-time checks
 * Returns: 0 if clean, >0 if issues found
 */
int llama_debug_stripping_validate_hot_path(void);

/**
 * Verify symbol stripping in binary
 * Checks for unnecessary debug symbols
 * Returns: 0 if clean, >0 if unnecessary symbols found
 */
int llama_debug_stripping_verify_symbol_stripping(void);

/**
 * Get string description of build configuration
 */
const char * llama_debug_stripping_get_build_config(void);

/**
 * Get compile-time configuration as string
 */
const char * llama_debug_stripping_get_compile_config(void);

/**
 * Hot path marker - called to indicate hot path entry
 * No-op in release builds
 */
void llama_hot_path_marker(const char * path_name);

/**
 * Record timing for hot path operation
 * Only records if timing instrumentation enabled
 */
void llama_hot_path_timing_record(const char * op_name, uint64_t elapsed_ns);

/**
 * Runtime debug guard (fallback for un-annotated code)
 * Checks runtime configuration
 * Returns: 1 if debug should proceed, 0 if should be skipped
 */
int llama_runtime_debug_guard_check(const char * component, const char * operation);

// ============================================================================
// HELPER MACROS FOR COMPILE-TIME CONFIGURATION VERIFICATION
// ============================================================================

/**
 * Compile-time check: verify debug configuration is consistent
 * Included in headers to fail at compile time if misconfigured
 */
#if LLAMA_ENABLE_DEBUG && defined(NDEBUG)
  #error "LLAMA_ENABLE_DEBUG=1 but NDEBUG is defined - conflicting build configuration"
#endif

#if !LLAMA_ENABLE_DEBUG && !defined(NDEBUG)
  #error "LLAMA_ENABLE_DEBUG=0 but NDEBUG is not defined - release build not properly configured"
#endif

/**
 * Provide helpful error message for common misconfiguration
 */
#if LLAMA_ENABLE_HOT_PATH_ASSERTIONS && LLAMA_ENABLE_TIMING_INSTRUMENTATION
  /* This is OK - debug builds have both assertions and timing */
#endif

#ifdef __cplusplus
}
#endif
