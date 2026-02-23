#include "llama-decode-engine.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include <cstdio>

void llama_decode_engine_init(struct llama_decode_engine * engine, const struct llama_model * model, const struct llama_context_params * params) {
    if (!engine || !model) return;

    engine->gpu_ctx = std::make_unique<llama_gpu_context>();
    engine->graph   = std::make_unique<llama_decode_graph>();
    engine->kv      = std::make_unique<llama_gpu_kv_cache>();
    engine->sampler = std::make_unique<llama_gpu_sampler>();
    engine->moe     = std::make_unique<llama_gpu_moe_cache>();

    engine->is_running = false;
    engine->is_locked  = false;

    // Initialize MoE Expert Cache if model is MoE
    if (model->hparams.n_expert > 0) {
        const auto & layer = model->layers[0];
        size_t expert_size = 0;
        
        if (layer.ffn_gate_exps) expert_size += ggml_nbytes(layer.ffn_gate_exps) / model->hparams.n_expert;
        if (layer.ffn_down_exps) expert_size += ggml_nbytes(layer.ffn_down_exps) / model->hparams.n_expert;
        if (layer.ffn_up_exps)   expert_size += ggml_nbytes(layer.ffn_up_exps)   / model->hparams.n_expert;

        if (expert_size > 0) {
            engine->moe->is_streaming = true;
            // Default to 1/4 of experts in GPU if we don't have enough VRAM
            size_t n_slots = std::max((size_t)1, model->hparams.n_expert / 4);
            
            ggml_backend_t backend = model->devices.empty() ? nullptr : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, 0);

            engine->moe->cache = std::make_unique<llama_expert_cache>(
                backend,
                model->hparams.n_expert,
                n_slots,
                expert_size,
                layer.ffn_gate_exps ? layer.ffn_gate_exps->ne[0] : 0, layer.ffn_gate_exps ? layer.ffn_gate_exps->ne[1] : 0,
                layer.ffn_up_exps   ? layer.ffn_up_exps->ne[0]   : 0, layer.ffn_up_exps   ? layer.ffn_up_exps->ne[1]   : 0,
                layer.ffn_down_exps ? layer.ffn_down_exps->ne[0] : 0, layer.ffn_down_exps ? layer.ffn_down_exps->ne[1] : 0
            );
            
            LLAMA_LOG_INFO("%s: initialized MoE expert streaming cache with %zu slots (%zu MB total VRAM)\n",
                __func__, n_slots, (n_slots * expert_size) / (1024 * 1024));
        }
    }
}

void llama_decode_engine_run(struct llama_decode_engine * engine) {
    GGML_ASSERT(engine && "Engine must be initialized");
    GGML_ASSERT(engine->is_locked && "Engine must be locked before running");
    
    engine->is_running = true;
    // Execution loop would be triggered here, but for now we expect 
    // it to be driven by ggml_backend_sched_graph_compute_autonomous
}

void llama_decode_engine_stop(struct llama_decode_engine * engine) {
    if (!engine) return;
    engine->is_running = false;
}

void llama_decode_engine_verify_gpu_resident(const struct llama_decode_engine * engine) {
    GGML_ASSERT(engine);
    // Invariant: engine components must exist and be valid
    GGML_ASSERT(engine->gpu_ctx);
    GGML_ASSERT(engine->graph);
    GGML_ASSERT(engine->kv);
    GGML_ASSERT(engine->sampler);
    
    // Future: Add checks to verify tensors are actually in GPU buffers
}
