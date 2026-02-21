#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <memory>

/**
 * [Hard Invariant] llama_decode_engine
 * 
 * This structure encapsulates all state and logic required for GPU-exclusive decode.
 * Once decode begins, the CPU must never execute any function inside this engine.
 * The CPU participates only in control-plane and asynchronous I/O.
 */

struct llama_gpu_context {
    ggml_backend_t backend = nullptr;
    // Dedicated stream for decode to eliminate host-side sync churn
    void * decode_stream = nullptr; 
};

struct llama_decode_graph {
    struct ggml_cgraph * gf = nullptr;
    class llm_graph_result * res = nullptr;
    // CUDA Graph handles
    void * cuda_graph = nullptr;
    void * cuda_graph_exec = nullptr;
    bool is_captured = false;

    // GPU-resident decode state tensors
    struct ggml_tensor * t_pos    = nullptr;
    struct ggml_tensor * t_n_past = nullptr;
    struct ggml_tensor * t_token  = nullptr;
    struct ggml_tensor * t_stop   = nullptr;
    struct ggml_tensor * t_history = nullptr;
    struct ggml_tensor * t_seed    = nullptr;
};

struct llama_gpu_kv_cache {
    struct llama_kv_cache * kv = nullptr;
    bool is_frozen = false;
};

struct llama_gpu_moe_cache {
    // [STRICT] MoE Expert Streaming Control
    // Manages dynamic expert weights for massive MoE models
    bool   is_streaming = false;
    size_t n_expert_gpu = 0;     // Max experts in GPU VRAM
    std::vector<int32_t> lru_list; // LRU for expert swapping
};

struct llama_gpu_sampler {
    // Current GPU sampling pipeline state
    void * cuda_ctx = nullptr; // Opaque pointer to cuda_sampling_context_t
};

struct llama_decode_engine {
    // GPU-resident context and stream management
    std::unique_ptr<llama_gpu_context> gpu_ctx;

    // Persistent computation graph (CUDA Graph)
    std::unique_ptr<llama_decode_graph> graph;

    // Fully GPU-resident KV cache (FP8/Q8 compressed)
    std::unique_ptr<llama_gpu_kv_cache> kv;

    // GPU-resident sampler pipeline
    std::unique_ptr<llama_gpu_sampler> sampler;

    // MoE expert streaming cache
    std::unique_ptr<llama_gpu_moe_cache> moe;

    // Invariant enforcement state
    bool is_running = false;
    bool is_locked = false;

    llama_decode_engine() {
        gpu_ctx = std::make_unique<llama_gpu_context>();
        graph   = std::make_unique<llama_decode_graph>();
        kv      = std::make_unique<llama_gpu_kv_cache>();
        sampler = std::make_unique<llama_gpu_sampler>();
        moe     = std::make_unique<llama_gpu_moe_cache>();
    }
};

// Data plane entry points (Execution only on GPU/Data Plane)
void llama_decode_engine_init(struct llama_decode_engine * engine);
void llama_decode_engine_run(struct llama_decode_engine * engine);
void llama_decode_engine_stop(struct llama_decode_engine * engine);

// Invariant verification
void llama_decode_engine_verify_gpu_resident(const struct llama_decode_engine * engine);
