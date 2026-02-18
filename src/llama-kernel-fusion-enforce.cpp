/**
 * Decode Kernel Fusion Enforcement Implementation
 *
 * Minimizes CUDA kernel launches per token through aggressive fusion.
 * Goal: fewer launches → lower CPU dispatch overhead → higher GPU utilization.
 */

#include "llama-kernel-fusion-enforce.h"

// Define logging and abort macros if not present
#ifndef LLAMA_LOG_INFO
#define LLAMA_LOG_INFO(...) fprintf(stdout, __VA_ARGS__)
#endif

#ifndef LLAMA_LOG_WARN
#define LLAMA_LOG_WARN(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef LLAMA_LOG_ERROR
#define LLAMA_LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef LLAMA_LOG_DEBUG
#define LLAMA_LOG_DEBUG(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef LLAMA_ABORT
#define LLAMA_ABORT(msg) do { fprintf(stderr, "LLAMA_ABORT: %s\n", msg); abort(); } while(0)
#endif

#include "llama-impl.h"

#include "../ggml/src/ggml-impl.h"
#include "../ggml/include/ggml-backend.h"

#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

/**
 * Initialize kernel fusion enforcement
 */
void llama_kernel_fusion_init(llama_kernel_fusion_state * state) {
    if (!state) return;

    state->enforce_active = false;
    state->baseline_launches = 0;
    state->target_max_launches = 8;  // Default target: 8 launches per token
    state->layer_count = 0;
    state->max_launches_per_layer = 2;

    state->enforce_qkv_fusion = false;
    state->enforce_norm_matmul_fusion = false;
    state->enforce_bias_activation = false;
    state->enforce_flash_attention = false;
    state->enforce_single_stream = false;
    state->enforce_persistent_kernels = false;

    // Initialize metrics
    state->metrics.total_launches = 0;
    state->metrics.launches_per_token = 0;
    state->metrics.launches_per_layer = 0;
    state->metrics.attention_launches = 0;
    state->metrics.ffn_launches = 0;
    state->metrics.norm_launches = 0;
    state->metrics.sampler_launches = 0;
    state->metrics.qkv_fusion_state = 0;
    state->metrics.norm_matmul_fusion = 0;
    state->metrics.bias_activation_fusion = 0;
    state->metrics.attention_kernel_type = 2;  // invalid
    state->metrics.sampling_kernel_type = 2;   // invalid
}

/**
 * Activate kernel fusion enforcement
 */
void llama_kernel_fusion_activate(
    llama_kernel_fusion_state * state,
    uint32_t n_layers,
    uint32_t target_launches) {

    if (!state) return;

    state->enforce_active = true;
    state->layer_count = n_layers;
    state->target_max_launches = target_launches;
    state->max_launches_per_layer = std::max(2u, target_launches / n_layers);

    // Activate all fusion enforcement rules
    state->enforce_qkv_fusion = true;
    state->enforce_norm_matmul_fusion = true;
    state->enforce_bias_activation = true;
    state->enforce_flash_attention = true;
    state->enforce_single_stream = true;

    LLAMA_LOG_INFO(
        "KERNEL FUSION: Activated (%u layers, %u target launches/token, %u/layer)\n",
        n_layers, target_launches, state->max_launches_per_layer);
}

/**
 * [CRITICAL] Enforce QKV fusion
 */
bool llama_kernel_fusion_enforce_qkv(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph) {

    if (!state || !graph || !state->enforce_qkv_fusion) {
        return true;
    }

    int qkv_count = 0;
    int fused_qkv_count = 0;

    // Scan graph for Q, K, V projections
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const char * name = node->name ? node->name : "";

        // Count individual Q/K/V projections
        if (strstr(name, "_q_proj") || strstr(name, "_q_in") ||
            strstr(name, "_k_proj") || strstr(name, "_k_in") ||
            strstr(name, "_v_proj") || strstr(name, "_v_in")) {
            qkv_count++;
        }

        // Count fused QKV projections
        if (strstr(name, "_qkv_") || strstr(name, "qkv_proj")) {
            fused_qkv_count++;
        }
    }

    // If we have individual Q, K, V but no fused versions → violation
    if (qkv_count > 0 && fused_qkv_count == 0) {
        LLAMA_LOG_ERROR(
            "KERNEL FUSION: QKV projections not fused!\n"
            "  Found %d individual Q/K/V kernels\n"
            "  Must use single fused QKV kernel\n",
            qkv_count);
        LLAMA_ABORT("QKV fusion enforcement failed");
        return false;
    }

    if (fused_qkv_count > 0) {
        state->metrics.qkv_fusion_state = 1;  // fused
        LLAMA_LOG_DEBUG("KERNEL FUSION: QKV fusion verified (%d fused kernels)\n", fused_qkv_count);
    }

    return true;
}

/**
 * [CRITICAL] Enforce RMSNorm + MatMul fusion
 */
bool llama_kernel_fusion_enforce_norm_matmul(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph) {

    if (!state || !graph || !state->enforce_norm_matmul_fusion) {
        return true;
    }

    int norm_only = 0;
    int fused_norm_matmul = 0;

    // Scan for standalone norm and fused norm+matmul patterns
    for (int i = 0; i < graph->n_nodes - 1; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * next = graph->nodes[i + 1];

        const char * op_name = ggml_op_name(node->op);
        const char * next_op = ggml_op_name(next->op);

        // Check for standalone RMSNorm not fused with MatMul
        if ((strcmp(op_name, "rms_norm") == 0 || strcmp(op_name, "norm") == 0) &&
            (strcmp(next_op, "mul_mat") != 0 && strcmp(next_op, "mul_mat_q") != 0)) {
            norm_only++;
        }

        // Check for fused norm+matmul
        if (strstr(node->name ? node->name : "", "norm_mul") ||
            strstr(node->name ? node->name : "", "norm_mat")) {
            fused_norm_matmul++;
        }
    }

    // If standalone norms found with many of them → likely violation
    if (norm_only > state->layer_count / 2) {
        LLAMA_LOG_ERROR(
            "KERNEL FUSION: RMSNorm not fused with MatMul!\n"
            "  Found %d standalone norm kernels\n"
            "  Must fuse norm into subsequent matmul\n",
            norm_only);
        LLAMA_ABORT("RMSNorm+MatMul fusion enforcement failed");
        return false;
    }

    if (fused_norm_matmul > 0) {
        state->metrics.norm_matmul_fusion = 1;  // fused
    }

    return true;
}

/**
 * [CRITICAL] Enforce Bias + Activation fusion
 */
bool llama_kernel_fusion_enforce_bias_activation(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph) {

    if (!state || !graph || !state->enforce_bias_activation) {
        return true;
    }

    int standalone_bias = 0;
    int standalone_activation = 0;

    // Scan for standalone bias and activation kernels
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const char * op_name = ggml_op_name(node->op);
        const char * name = node->name ? node->name : "";

        // Standalone bias add
        if ((strcmp(op_name, "add") == 0) && strstr(name, "bias")) {
            standalone_bias++;
        }

        // Standalone activation
        if ((strcmp(op_name, "gelu") == 0 || strcmp(op_name, "silu") == 0 ||
             strcmp(op_name, "relu") == 0) && !strstr(name, "fused")) {
            standalone_activation++;
        }
    }

    // If standalone kernels found, they should be fused
    if (standalone_bias > state->layer_count / 2) {
        LLAMA_LOG_ERROR(
            "KERNEL FUSION: Bias add kernels not fused!\n"
            "  Found %d standalone bias kernels\n"
            "  Bias must be fused with matmul\n",
            standalone_bias);
        LLAMA_ABORT("Bias fusion enforcement failed");
        return false;
    }

    if (standalone_activation > state->layer_count / 2) {
        LLAMA_LOG_ERROR(
            "KERNEL FUSION: Activation kernels not fused!\n"
            "  Found %d standalone activation kernels\n"
            "  Activation must be fused with bias/matmul\n",
            standalone_activation);
        LLAMA_ABORT("Activation fusion enforcement failed");
        return false;
    }

    if (standalone_bias == 0 && standalone_activation == 0) {
        state->metrics.bias_activation_fusion = 1;  // fused
    }

    return true;
}

/**
 * [CRITICAL] Enforce Flash Attention
 */
bool llama_kernel_fusion_enforce_flash_attention(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph) {

    if (!state || !graph || !state->enforce_flash_attention) {
        return true;
    }

    int flash_attention = 0;
    int multi_stage_attention = 0;
    bool has_attention = false;

    // Scan for attention patterns
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const char * op_name = ggml_op_name(node->op);
        const char * name = node->name ? node->name : "";

        // Look for attention operations
        if (strstr(name, "attn") || strstr(name, "attention")) {
            has_attention = true;

            // Flash attention (single kernel)
            if (strstr(name, "flash")) {
                flash_attention++;
            }

            // Multi-stage attention (multiple kernels)
            if (strcmp(op_name, "mul_mat") == 0 && strstr(name, "attn")) {
                multi_stage_attention++;
            }
        }
    }

    // If attention is used but not flash attention → error
    if (has_attention && flash_attention == 0 && multi_stage_attention > 2) {
        LLAMA_LOG_ERROR(
            "KERNEL FUSION: Multi-stage attention detected!\n"
            "  Found %d separate attention kernels\n"
            "  Must use flash attention (single fused kernel)\n",
            multi_stage_attention);
        LLAMA_ABORT("Flash attention enforcement failed");
        return false;
    }

    if (flash_attention > 0) {
        state->metrics.attention_kernel_type = 0;  // flash
    } else if (has_attention) {
        state->metrics.attention_kernel_type = 1;  // unfused
    }

    return true;
}

/**
 * [CRITICAL] Eliminate micro-kernels
 */
bool llama_kernel_fusion_eliminate_micro_kernels(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph) {

    if (!state || !graph) {
        return true;
    }

    int micro_kernel_count = 0;

    // Scan for small kernels that should be inlined
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const char * op_name = ggml_op_name(node->op);
        const char * name = node->name ? node->name : "";

        // Element-wise ops that should be inlined
        if (strcmp(op_name, "scale") == 0 ||
            strcmp(op_name, "mul") == 0 ||
            strcmp(op_name, "div") == 0) {
            micro_kernel_count++;
        }

        // Unary ops (unless part of fused operation)
        if ((strcmp(op_name, "gelu_new") == 0 ||
             strcmp(op_name, "hardswish") == 0 ||
             strcmp(op_name, "hardsigmoid") == 0) &&
            !strstr(name, "fused")) {
            micro_kernel_count++;
        }
    }

    // If many micro-kernels found, warn (but don't abort if some are acceptable)
    if (micro_kernel_count > state->layer_count) {
        LLAMA_LOG_WARN(
            "KERNEL FUSION: Many micro-kernels found (%d)\n"
            "  Consider inlining element-wise and unary ops\n",
            micro_kernel_count);
    }

    return true;
}

/**
 * [CRITICAL] Eliminate redundant memory ops
 */
bool llama_kernel_fusion_eliminate_memory_ops(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph) {

    if (!state || !graph) {
        return true;
    }

    int redundant_copies = 0;
    int reshape_count = 0;

    // Scan for memory operations
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const char * op_name = ggml_op_name(node->op);

        // Device-to-device copies (unnecessary in optimized path)
        if (strcmp(op_name, "cpy") == 0) {
            redundant_copies++;
        }

        // Reshape operations (layout should be frozen)
        if (strcmp(op_name, "reshape") == 0 ||
            strcmp(op_name, "view") == 0 ||
            strcmp(op_name, "transpose") == 0) {
            reshape_count++;
        }
    }

    // If many redundant operations, warn
    if (redundant_copies > state->layer_count / 4) {
        LLAMA_LOG_WARN(
            "KERNEL FUSION: Redundant copy kernels detected (%d)\n"
            "  Should be eliminated in decode path\n",
            redundant_copies);
    }

    if (reshape_count > state->layer_count / 2) {
        LLAMA_LOG_WARN(
            "KERNEL FUSION: Reshape kernels in decode path (%d)\n"
            "  Layout should be frozen pre-decode\n",
            reshape_count);
    }

    return true;
}

/**
 * [CRITICAL] Enforce single CUDA stream
 */
bool llama_kernel_fusion_enforce_single_stream(
    llama_kernel_fusion_state * state,
    int n_streams) {

    if (!state || !state->enforce_single_stream) {
        return true;
    }

    if (n_streams > 1) {
        LLAMA_LOG_ERROR(
            "KERNEL FUSION: Multiple CUDA streams detected!\n"
            "  Decode requires single stream to minimize launch overhead\n"
            "  Streams: %d (expected 1)\n",
            n_streams);
        LLAMA_ABORT("Single stream enforcement failed");
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Enforce persistent kernel model
 */
bool llama_kernel_fusion_enforce_persistent_model(
    llama_kernel_fusion_state * state,
    bool use_persistent) {

    if (!state) {
        return false;
    }

    if (use_persistent) {
        state->enforce_persistent_kernels = true;
        LLAMA_LOG_INFO("KERNEL FUSION: Persistent kernel model enabled\n");
        return true;
    }

    // If not using persistent kernels, ensure minimal launches per layer
    if (state->max_launches_per_layer > 3) {
        LLAMA_LOG_WARN(
            "KERNEL FUSION: Persistent kernels not available\n"
            "  Minimizing launches per layer to 2-3\n");
        state->max_launches_per_layer = 3;
    }

    return true;
}

/**
 * [CRITICAL] Collapse KV update into attention
 */
bool llama_kernel_fusion_collapse_kv_update(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph) {

    if (!state || !graph) {
        return true;
    }

    int separate_kv_writes = 0;

    // Scan for separate KV write operations
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const char * name = node->name ? node->name : "";

        // Separate KV write kernel
        if (strstr(name, "kv_write") || strstr(name, "cache_append") ||
            (strstr(name, "set_rows") && strstr(name, "kv"))) {
            separate_kv_writes++;
        }
    }

    if (separate_kv_writes > 0) {
        LLAMA_LOG_WARN(
            "KERNEL FUSION: Separate KV write kernels detected (%d)\n"
            "  Should be merged into attention kernel\n",
            separate_kv_writes);
    }

    return true;
}

/**
 * [CRITICAL] Collapse sampling sub-kernels
 */
bool llama_kernel_fusion_collapse_sampling(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph) {

    if (!state || !graph) {
        return true;
    }

    int sampling_sub_kernels = 0;
    int fused_sampling = 0;

    // Scan for sampling kernels
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const char * name = node->name ? node->name : "";

        // Fused sampling kernel
        if (strstr(name, "sample_") && strstr(name, "fused")) {
            fused_sampling++;
        }

        // Individual sampling sub-kernels
        if (strstr(name, "penalty") || strstr(name, "top_k") ||
            strstr(name, "top_p") || strstr(name, "sample_argmax")) {
            sampling_sub_kernels++;
        }
    }

    // If sampling sub-kernels found but no fused version → warning
    if (sampling_sub_kernels > 1 && fused_sampling == 0) {
        LLAMA_LOG_WARN(
            "KERNEL FUSION: Sampling split into %d sub-kernels\n"
            "  Should use single fused sampling kernel\n",
            sampling_sub_kernels);
        state->metrics.sampling_kernel_type = 1;  // split
    } else if (fused_sampling > 0) {
        state->metrics.sampling_kernel_type = 0;  // fused
    }

    return true;
}

/**
 * [CRITICAL] Validate kernel launch count
 */
bool llama_kernel_fusion_validate_launch_count(
    llama_kernel_fusion_state * state,
    uint64_t actual_launches) {

    if (!state) {
        return false;
    }

    // Check against target
    if (actual_launches > state->target_max_launches) {
        LLAMA_LOG_ERROR(
            "KERNEL FUSION: Launch count exceeds target!\n"
            "  Actual: %lu\n"
            "  Target: %u\n"
            "  Decrease launches to meet target\n",
            actual_launches, state->target_max_launches);
        LLAMA_ABORT("Kernel launch count validation failed");
        return false;
    }

    return true;
}

/**
 * Count kernel launches in graph
 */
uint64_t llama_kernel_fusion_count_launches(struct ggml_cgraph * graph) {
    if (!graph) {
        return 0;
    }

    uint64_t launch_count = 0;

    // Rough estimate: count compute-intensive ops
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        const char * op_name = ggml_op_name(node->op);

        // Operations that require kernel launches
        if (strcmp(op_name, "mul_mat") == 0 ||
            strcmp(op_name, "mul_mat_q") == 0 ||
            strcmp(op_name, "mul_mat_id") == 0 ||
            strcmp(op_name, "mul_mat_id_q") == 0 ||
            strcmp(op_name, "rms_norm") == 0 ||
            strcmp(op_name, "norm") == 0 ||
            strcmp(op_name, "soft_max") == 0 ||
            strcmp(op_name, "gelu") == 0 ||
            strcmp(op_name, "silu") == 0) {
            launch_count++;
        }
    }

    return launch_count;
}

/**
 * Measure baseline launches
 */
void llama_kernel_fusion_measure_baseline(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph) {

    if (!state || !graph) {
        return;
    }

    state->baseline_launches = llama_kernel_fusion_count_launches(graph);
    LLAMA_LOG_INFO("KERNEL FUSION: Baseline launches = %lu\n", state->baseline_launches);
}

/**
 * Record metrics
 */
void llama_kernel_fusion_record_metrics(
    llama_kernel_fusion_state * state,
    uint64_t total_launches,
    uint64_t launches_per_token) {

    if (!state) {
        return;
    }

    state->metrics.total_launches = total_launches;
    state->metrics.launches_per_token = launches_per_token;
    if (state->layer_count > 0) {
        state->metrics.launches_per_layer = total_launches / state->layer_count;
    }
}

/**
 * Get metrics
 */
llama_kernel_metrics llama_kernel_fusion_get_metrics(
    const llama_kernel_fusion_state * state) {

    if (!state) {
        return {};
    }

    return state->metrics;
}

/**
 * Audit entire graph
 */
bool llama_kernel_fusion_audit_graph(
    llama_kernel_fusion_state * state,
    struct ggml_cgraph * graph) {

    if (!state || !graph) {
        return false;
    }

    // Run all fusion enforcement checks
    if (!llama_kernel_fusion_enforce_qkv(state, graph)) {
        return false;
    }

    if (!llama_kernel_fusion_enforce_norm_matmul(state, graph)) {
        return false;
    }

    if (!llama_kernel_fusion_enforce_bias_activation(state, graph)) {
        return false;
    }

    if (!llama_kernel_fusion_enforce_flash_attention(state, graph)) {
        return false;
    }

    llama_kernel_fusion_eliminate_micro_kernels(state, graph);
    llama_kernel_fusion_eliminate_memory_ops(state, graph);
    llama_kernel_fusion_collapse_kv_update(state, graph);
    llama_kernel_fusion_collapse_sampling(state, graph);

    // Measure and validate launch count
    uint64_t launches = llama_kernel_fusion_count_launches(graph);
    if (!llama_kernel_fusion_validate_launch_count(state, launches)) {
        return false;
    }

    return true;
}

/**
 * Dump metrics
 */
void llama_kernel_fusion_dump_metrics(const llama_kernel_fusion_state * state) {
    if (!state) {
        return;
    }

    LLAMA_LOG_INFO("KERNEL FUSION METRICS:\n");
    LLAMA_LOG_INFO("  Total launches: %lu\n", state->metrics.total_launches);
    LLAMA_LOG_INFO("  Per-token: %lu\n", state->metrics.launches_per_token);
    LLAMA_LOG_INFO("  Per-layer: %lu\n", state->metrics.launches_per_layer);
    LLAMA_LOG_INFO("  Attention launches: %lu\n", state->metrics.attention_launches);
    LLAMA_LOG_INFO("  FFN launches: %lu\n", state->metrics.ffn_launches);
    LLAMA_LOG_INFO("  Norm launches: %lu\n", state->metrics.norm_launches);
    LLAMA_LOG_INFO("  Sampling launches: %lu\n", state->metrics.sampler_launches);

    const char * qkv_status = state->metrics.qkv_fusion_state ? "fused" : "split";
    const char * norm_status = state->metrics.norm_matmul_fusion ? "fused" : "split";
    const char * bias_status = state->metrics.bias_activation_fusion ? "fused" : "split";
    const char * attn_type = state->metrics.attention_kernel_type == 0 ? "flash" :
                             state->metrics.attention_kernel_type == 1 ? "unfused" : "unknown";
    const char * samp_type = state->metrics.sampling_kernel_type == 0 ? "fused" :
                             state->metrics.sampling_kernel_type == 1 ? "split" : "unknown";

    LLAMA_LOG_INFO("FUSION STATUS:\n");
    LLAMA_LOG_INFO("  QKV: %s\n", qkv_status);
    LLAMA_LOG_INFO("  Norm+MatMul: %s\n", norm_status);
    LLAMA_LOG_INFO("  Bias+Activation: %s\n", bias_status);
    LLAMA_LOG_INFO("  Attention type: %s\n", attn_type);
    LLAMA_LOG_INFO("  Sampling type: %s\n", samp_type);
}

/**
 * Deactivate kernel fusion enforcement
 */
void llama_kernel_fusion_deactivate(llama_kernel_fusion_state * state) {
    if (!state) {
        return;
    }

    state->enforce_active = false;
    LLAMA_LOG_INFO("KERNEL FUSION: Deactivated\n");
}
