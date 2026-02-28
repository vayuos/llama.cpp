/**
 * ASYNC PIPELINING VALIDATION
 * Header: GPU-CPU parallelism correctness and performance validation
 *
 * Provides:
 * - Output correctness verification (byte-for-byte matching)
 * - Performance measurement (throughput, GPU/CPU utilization)
 * - Stall detection (GPU/CPU waiting times)
 * - Safety checks (deadlock detection, timeouts)
 */

#pragma once

#include <cstdint>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Pipeline validation metrics
 */
struct llama_pipeline_validation {
    // Correctness
    bool outputs_match;              // Output byte-for-byte match with baseline
    int num_tokens_tested;           // Tokens generated during validation
    int mismatched_tokens;           // Tokens with incorrect output

    // Performance
    double baseline_tokens_sec;      // Single-stream throughput (baseline)
    double pipelined_tokens_sec;     // Multi-stream throughput (with pipelining)
    double improvement_percent;      // % improvement ((pipelined - baseline) / baseline * 100)

    // GPU/CPU Coordination
    uint64_t gpu_stall_count;        // Times GPU had to wait for CPU
    uint64_t cpu_stall_count;        // Times CPU had to wait for GPU
    uint64_t gpu_stall_time_ns;      // Total GPU wait time
    uint64_t cpu_stall_time_ns;      // Total CPU wait time

    // Safety
    bool deadlock_detected;          // True if deadlock timeout triggered
    bool memory_error_detected;      // True if memory corruption detected
    int timeout_count;               // Number of timeout events
};

/**
 * Validate async pipelining is working correctly.
 *
 * @param num_tokens_to_generate Number of tokens to test (recommend 100-1000)
 * @param timeout_seconds Timeout for hanging detection (recommend 60)
 * @return Validation results with correctness and performance metrics
 */
struct llama_pipeline_validation llama_validate_async_pipeline(
    int num_tokens_to_generate,
    int timeout_seconds);

/**
 * Print validation results in human-readable format.
 *
 * @param validation Validation results to print
 */
void llama_print_pipeline_validation(
    const struct llama_pipeline_validation * validation);

/**
 * Check if validation passed (outputs correct, no deadlocks, improved throughput).
 *
 * @param validation Validation results
 * @return true if validation passed (outputs_match && !deadlock && improvement > 0)
 */
bool llama_pipeline_validation_passed(
    const struct llama_pipeline_validation * validation);

#ifdef __cplusplus
}
#endif
