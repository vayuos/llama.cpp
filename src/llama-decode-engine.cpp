#include "llama-decode-engine.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-backend.h"
#include <cstdio>

void llama_decode_engine_init(struct llama_decode_engine * engine, const struct llama_model * model, const struct llama_context_params * params) {
    (void)params;
    if (!engine || !model) return;

    engine->gpu_ctx = std::make_unique<llama_gpu_context>();
    engine->graph   = std::make_unique<llama_decode_graph>();
    engine->kv      = std::make_unique<llama_gpu_kv_cache>();
    engine->sampler = std::make_unique<llama_gpu_sampler>();
    engine->moe     = std::make_unique<llama_gpu_moe_cache>();

    engine->is_running = false;
    engine->is_locked  = false;

    // NOTE: MoE expert streaming cache is disabled.
    // The original code called ggml_backend_init_by_type() which creates a NEW independent
    // CUDA backend with its own stream/context, separate from the main inference backend.
    // GPU memory allocated on this foreign backend is inaccessible from the main CUDA context,
    // causing cudaErrorIllegalAddress. The slot-remapping path in build_moe_ffn is also
    // disabled (shape mismatch), so this cache is unused. Leaving as no-op for now.
    if (model->hparams.n_expert > 0) {
        LLAMA_LOG_INFO("%s: MoE model detected (%zu experts) — expert streaming cache disabled (slot-remapping path not active)\n",
            __func__, (size_t)model->hparams.n_expert);
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
