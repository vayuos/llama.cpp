#pragma once

/**
 * Decode Kernel Fusion Enforcement
 *
 * Minimizes CUDA kernel launches per token through aggressive fusion.
 * The goal is execution density, not kernel speed:
 * Fewer launches → lower CPU overhead → higher GPU utilization.
 */

#include "llama-graph.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Kernel execution metrics
 */
typedef struct {
    uint64_t total_launches;           ///< Total kernels launched
    uint64_t launches_per_token;       ///< Launches per token average
    uint64_t launches_per_layer;       ///< Launches per layer average
    uint64_t attention_launches;       ///< Launches in attention blocks
    uint64_t ffn_launches;             ///< Launches in FFN blocks
    uint64_t norm_launches;            ///< Normalization kernel count
    uint64_t sampler_launches;         ///< Sampling kernel count
    uint32_t qkv_fusion_state;         ///< QKV fusion status (0=split, 1=fused)
    uint32_t norm_matmul_fusion;       ///< RMSNorm+MatMul fusion status
    uint32_t bias_activation_fusion;   ///< Bias+Activation fusion status
    uint32_t attention_kernel_type;    ///< 0=flash, 1=unfused, 2=invalid
    uint32_t sampling_kernel_type;     ///< 0=fused, 1=split, 2=invalid
} llama_kernel_metrics;

/**
 * Decode kernel fusion enforcement state
 */
typedef struct {
    bool       enforce_active;         ///< Whether fusion enforcement is active
    uint64_t   baseline_launches;      ///< Baseline kernel count (measured first)
    uint32_t   target_max_launches;    ///< Target max launches per token
    uint32_t   layer_count;            ///< Number of layers
    uint32_t   max_launches_per_layer; ///< Max allowed launches per layer

    // Fusion enforcement flags
    bool enforce_qkv_fusion;           ///< QKV must be fused
    bool enforce_norm_matmul_fusion;   ///< RMSNorm+MatMul must be fused
    bool enforce_bias_activation;      ///< Bias+Activation must be fused
    bool enforce_flash_attention;      ///< Must use flash attention
    bool enforce_single_stream;        ///< Single CUDA stream required
    bool enforce_persistent_kernels;   ///< Persistent kernel model preferred

    // Metrics
    llama_kernel_metrics metrics;
} llama_kernel_fusion_state;

/**
 * Initialize kernel fusion enforcement
 */
void llama_kernel_fusion_init(llama_kernel_fusion_state * state);

/**
 * Activate kernel fusion enforcement for decode
 *
 * Enables all fusion requirements and metric tracking.
 *
 * @param state Fusion enforcement state
 * @param n_layers Number of model layers
 * @param target_launches Target max launches per token
 */
void llama_kernel_fusion_activate(
    llama_kernel_fusion_state * state,
    uint32_t n_layers,
    uint32_t target_launches);

/**
 * [CRITICAL] Enforce QKV projection fusion
 *
 * Q, K, V projections MUST be fused into single kernel.
 * Three separate kernels → one fused kernel.
 *
 * @param state Fusion state
 * @param graph Computation graph
 * @return true if QKV properly fused, false if split kernels detected
 */
bool llama_kernel_fusion_enforce_qkv(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph);

/**
 * [CRITICAL] Enforce RMSNorm + MatMul fusion
 *
 * RMSNorm must not execute separately from subsequent MatMul.
 * Pattern: RMSNorm → sync → MatMul (FORBIDDEN)
 * Must become: FusedNormMatMul (REQUIRED)
 *
 * @param state Fusion state
 * @param graph Computation graph
 * @return true if fusion enforced, false if split detected
 */
bool llama_kernel_fusion_enforce_norm_matmul(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph);

/**
 * [CRITICAL] Enforce Bias + Activation fusion
 *
 * BiasAdd and Activation must be fused.
 * Standalone bias kernels forbidden in decode path.
 *
 * @param state Fusion state
 * @param graph Computation graph
 * @return true if fused, false if split kernels found
 */
bool llama_kernel_fusion_enforce_bias_activation(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph);

/**
 * [CRITICAL] Enforce Flash Attention usage
 *
 * If attention is used during decode, must use flash-attention.
 * Multi-stage attention launches forbidden.
 * Single fused attention kernel required.
 *
 * @param state Fusion state
 * @param graph Computation graph
 * @return true if flash attention detected, false if multi-stage
 */
bool llama_kernel_fusion_enforce_flash_attention(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph);

/**
 * [CRITICAL] Eliminate micro-kernels
 *
 * Small kernels (element-wise, scalar, row-wise ops) must be inlined.
 * Standalone:
 * - scale kernels
 * - unary ops
 * - small reductions
 * Are forbidden in decode path.
 *
 * @param state Fusion state
 * @param graph Computation graph
 * @return true if no micro-kernels, false if found
 */
bool llama_kernel_fusion_eliminate_micro_kernels(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph);

/**
 * [CRITICAL] Eliminate redundant memory ops
 *
 * Prohibit during decode:
 * - Device-to-device copies
 * - Temporary reshapes per token
 * - Format conversions per token
 *
 * @param state Fusion state
 * @param graph Computation graph
 * @return true if no redundant memory ops, false if found
 */
bool llama_kernel_fusion_eliminate_memory_ops(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph);

/**
 * [CRITICAL] Enforce single CUDA stream
 *
 * Decode must use single stream to minimize launch overhead.
 * Multiple streams require additional synchronization.
 *
 * @param state Fusion state
 * @param n_streams Number of CUDA streams in scheduler
 * @return true if single stream, false if multiple
 */
bool llama_kernel_fusion_enforce_single_stream(
    llama_kernel_fusion_state * state,
    int n_streams);

/**
 * [CRITICAL] Enforce persistent kernel model (if available)
 *
 * Persistent kernels loop internally over tokens.
 * Eliminates launch overhead entirely.
 *
 * Optional: if not feasible, minimize to 2-3 launches per layer.
 *
 * @param state Fusion state
 * @param use_persistent Whether persistent kernels should be used
 * @return true if persistent model used or minimized launches
 */
bool llama_kernel_fusion_enforce_persistent_model(
    llama_kernel_fusion_state * state,
    bool use_persistent);

/**
 * [CRITICAL] Collapse KV update into attention kernel
 *
 * KV cache update must not be separate kernel.
 * Merged into attention kernel tail.
 *
 * @param state Fusion state
 * @param graph Computation graph
 * @return true if fused, false if separate kernels
 */
bool llama_kernel_fusion_collapse_kv_update(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph);

/**
 * [CRITICAL] Collapse sampling sub-kernels
 *
 * Sampling must be single kernel:
 * NOT: logits_copy → penalty → top_k → top_p → argmax → sample
 * MUST: single_sampling_kernel
 *
 * @param state Fusion state
 * @param graph Computation graph
 * @return true if single kernel, false if split
 */
bool llama_kernel_fusion_collapse_sampling(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph);

/**
 * [CRITICAL] Validate kernel launch count
 *
 * Count launches in graph and verify against target.
 * Aborts if exceeds threshold.
 *
 * @param state Fusion state (contains target threshold)
 * @param actual_launches Actual kernel count measured
 * @return true if within target, false if exceeds
 */
bool llama_kernel_fusion_validate_launch_count(
    llama_kernel_fusion_state * state,
    uint64_t actual_launches);

/**
 * Count actual kernel launches in a graph
 *
 * Scans graph nodes and counts CUDA kernel operations.
 *
 * @param graph Computation graph
 * @return Estimated kernel count
 */
uint64_t llama_kernel_fusion_count_launches(struct ggml_cgraph * graph);

/**
 * Measure baseline kernel launches per token
 *
 * Called once to establish baseline before fusion enforcement.
 * Becomes tracked metric.
 *
 * @param state Fusion state (baseline stored here)
 * @param graph Computation graph
 */
void llama_kernel_fusion_measure_baseline(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph);

/**
 * Record kernel metrics
 *
 * Updates metrics with current measurements.
 *
 * @param state Fusion state
 * @param total_launches Total launches measured
 * @param launches_per_token Calculated per-token average
 */
void llama_kernel_fusion_record_metrics(
    llama_kernel_fusion_state * state,
    uint64_t total_launches,
    uint64_t launches_per_token);

/**
 * Get current kernel metrics
 *
 * @param state Fusion state
 * @return Kernel execution metrics
 */
llama_kernel_metrics llama_kernel_fusion_get_metrics(
    const llama_kernel_fusion_state * state);

/**
 * Audit entire decode graph for fusion compliance
 *
 * Runs all fusion enforcement checks.
 * Single function to validate complete fusion requirements.
 *
 * @param state Fusion state
 * @param graph Computation graph
 * @return true if all requirements met, false if any violation
 */
bool llama_kernel_fusion_audit_graph(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph);

/**
 * Dump kernel fusion metrics and status
 *
 * Logs detailed breakdown of kernel usage.
 *
 * @param state Fusion state (contains metrics)
 */
void llama_kernel_fusion_dump_metrics(const llama_kernel_fusion_state * state);

/**
 * Deactivate kernel fusion enforcement
 *
 * @param state Fusion state
 */
void llama_kernel_fusion_deactivate(llama_kernel_fusion_state * state);

#ifdef __cplusplus
}
#endif
