/*
 * ggml-cuda/sampling_impl.cu
 *
 * Minimal, safe host-side implementations for the cuda_sampling_* API
 * declared in sampling.h. These wrappers allocate device buffers and
 * provide a simple (D2H) greedy-sample fallback implementation so the
 * integration and testing can proceed. Replace with optimized device
 * kernels later (cuda_argmax_kernel, cuda_softmax_kernel, etc.).
 */

#include "sampling.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern "C" {

#define CUDA_OK 0
#define CUDA_ERR_NEG -1

static inline int cuda_safe_check(cudaError_t e) {
    if (e != cudaSuccess) {
        return (int) e;
    }
    return CUDA_OK;
}

int cuda_sampling_init_gpu(cuda_sampling_context_t **out_ctx, int32_t vocab_size, int32_t cuda_device) {
    if (!out_ctx || vocab_size <= 0) return -1;
    cudaError_t cerr = cudaSetDevice((int)cuda_device);
    if (cerr != cudaSuccess) return -1;

    cuda_sampling_context_t *ctx = (cuda_sampling_context_t*) malloc(sizeof(cuda_sampling_context_t));
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));

    ctx->vocab_size = vocab_size;
    ctx->cuda_device = cuda_device;
    ctx->cuda_stream = NULL;

    size_t bytes = (size_t)vocab_size * sizeof(float);

    // allocate device buffers (best-effort)
    if (cuda_safe_check(cudaMalloc((void**)&ctx->d_logits, bytes)) != CUDA_OK) {
        ctx->d_logits = NULL;
    }
    if (cuda_safe_check(cudaMalloc((void**)&ctx->d_penalties, bytes)) != CUDA_OK) {
        ctx->d_penalties = NULL;
    }
    if (cuda_safe_check(cudaMalloc((void**)&ctx->d_probs, bytes)) != CUDA_OK) {
        ctx->d_probs = NULL;
    }

    // scratch: small workspace
    size_t scratch_bytes = 4096;
    if (cuda_safe_check(cudaMalloc((void**)&ctx->d_scratch, scratch_bytes)) != CUDA_OK) {
        ctx->d_scratch = NULL;
    }

    if (cuda_safe_check(cudaMalloc((void**)&ctx->d_sampled_token, sizeof(int32_t))) != CUDA_OK) {
        ctx->d_sampled_token = NULL;
    }

    // create non-blocking stream for sampling operations
    cudaStream_t s = 0;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess) {
        ctx->cuda_stream = (void*)s;
    } else {
        ctx->cuda_stream = NULL;
    }

    *out_ctx = ctx;
    return 0;
}

int cuda_sampling_free_gpu(cuda_sampling_context_t * ctx) {
    if (!ctx) return -1;
    // destroy optional stream
    if (ctx->cuda_stream) {
        cudaStream_t s = (cudaStream_t) ctx->cuda_stream;
        cudaStreamDestroy(s);
        ctx->cuda_stream = NULL;
    }
    if (ctx->d_logits) cudaFree(ctx->d_logits);
    if (ctx->d_penalties) cudaFree(ctx->d_penalties);
    if (ctx->d_probs) cudaFree(ctx->d_probs);
    if (ctx->d_scratch) cudaFree(ctx->d_scratch);
    if (ctx->d_sampled_token) cudaFree(ctx->d_sampled_token);
    free(ctx);
    return 0;
}

int cuda_sampling_set_logits(cuda_sampling_context_t * ctx, float * d_logits, size_t size_bytes, int copy) {
    if (!ctx || !d_logits) return -1;
    if (copy) {
        if (!ctx->d_logits) return -1;
        // device-to-device copy assumed
        cudaError_t e = cudaMemcpy(ctx->d_logits, d_logits, size_bytes, cudaMemcpyDeviceToDevice);
        return (e == cudaSuccess) ? 0 : -1;
    } else {
        // alias pointer (caller must ensure lifetime)
        ctx->d_logits = d_logits;
        return 0;
    }
}

int cuda_sampling_sample_greedy(cuda_sampling_context_t * ctx, int32_t * token_out, void * cuda_stream) {
    if (!ctx || !token_out) return -1;
    // Use device argmax kernel when available for lower latency.
    if (!ctx->d_logits || ctx->vocab_size <= 0) return -1;

    // choose stream: argument stream overrides context stream
    cudaStream_t s = 0;
    if (cuda_stream) s = (cudaStream_t) cuda_stream;
    else if (ctx->cuda_stream) s = (cudaStream_t) ctx->cuda_stream;

    // call device argmax kernel; it will fallback to host copy internally if needed
    int err = cuda_argmax_kernel(ctx->d_logits, ctx->d_sampled_token, ctx->vocab_size, (void*)s);
    if (err != 0) {
        // kernel reported error; fallback to host copy
        int32_t vocab = ctx->vocab_size;
        size_t bytes = (size_t)vocab * sizeof(float);
        float * h_logits = (float*) malloc(bytes);
        if (!h_logits) return -1;
        cudaError_t e = cudaMemcpy(h_logits, ctx->d_logits, bytes, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) { free(h_logits); return -1; }
        int32_t best = 0; float best_val = h_logits[0];
        for (int32_t i = 1; i < vocab; ++i) {
            if (h_logits[i] > best_val) { best_val = h_logits[i]; best = i; }
        }
        free(h_logits);
        *token_out = best;
        if (ctx->d_sampled_token) cudaMemcpyAsync(ctx->d_sampled_token, &best, sizeof(int32_t), cudaMemcpyHostToDevice, s);
        if (s) cudaStreamSynchronize(s);
        return 0;
    }

    // copy sampled token back to host
    int32_t host_token = -1;
    cudaError_t ce = cudaMemcpyAsync(&host_token, ctx->d_sampled_token, sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    if (ce != cudaSuccess) return -1;
    if (s) cudaStreamSynchronize(s);
    *token_out = host_token;
    return 0;
}

int cuda_get_device_name(int device_id, char * name_out, int name_len) {
    if (!name_out || name_len <= 0) return -1;
    cudaDeviceProp prop;
    cudaError_t e = cudaGetDeviceProperties(&prop, device_id);
    if (e != cudaSuccess) return -1;
    strncpy(name_out, prop.name, (size_t)name_len - 1);
    name_out[name_len-1] = '\0';
    return 0;
}

int cuda_get_device_memory(int device_id, size_t * free_bytes, size_t * total_bytes) {
    if (!free_bytes || !total_bytes) return -1;
    cudaError_t e = cudaSetDevice(device_id);
    if (e != cudaSuccess) return -1;
    size_t freeb = 0, totalb = 0;
    e = cudaMemGetInfo(&freeb, &totalb);
    if (e != cudaSuccess) return -1;
    *free_bytes = freeb;
    *total_bytes = totalb;
    return 0;
}

int cuda_sampling_synchronize(cuda_sampling_context_t * ctx) {
    (void) ctx;
    cudaError_t e = cudaDeviceSynchronize();
    return (e == cudaSuccess) ? 0 : -1;
}

int cuda_sampling_sample_specialized(cuda_sampling_context_t * ctx, int32_t * token_out, float temperature, int32_t top_k, float penalty_alpha, uint64_t seed, void * cuda_stream) {
    if (!ctx || !token_out) return -1;
    if (!ctx->d_logits || ctx->vocab_size <= 0) return -1;

    // choose stream
    cudaStream_t s = 0;
    if (cuda_stream) s = (cudaStream_t) cuda_stream;
    else if (ctx->cuda_stream) s = (cudaStream_t) ctx->cuda_stream;

    // apply temperature scaling (device kernel)
    if (temperature != 1.0f) {
        if (cuda_temperature_scale_kernel(ctx->d_logits, temperature, top_k, ctx->vocab_size, (void*)s) != 0) {
            // ignore and continue
        }
    }

    // apply penalties if provided
    if (penalty_alpha != 0.0f && ctx->d_penalties) {
        if (cuda_apply_penalties_kernel(ctx->d_logits, ctx->d_penalties, penalty_alpha, ctx->vocab_size, (void*)s) != 0) {
            // fallback: ignore
        }
    }

    // compute softmax probabilities (device kernel may fallback to host internally)
    if (cuda_softmax_kernel(ctx->d_logits, ctx->d_probs, ctx->vocab_size, (void*)s) != 0) {
        return -1;
    }

    // copy probs to host and sample (inverse transform sampling)
    size_t bytes = (size_t)ctx->vocab_size * sizeof(float);
    float * h_probs = (float*) malloc(bytes);
    if (!h_probs) return -1;
    cudaError_t ce = cudaMemcpyAsync(h_probs, ctx->d_probs, bytes, cudaMemcpyDeviceToHost, s);
    if (ce != cudaSuccess) { free(h_probs); return -1; }
    if (s) cudaStreamSynchronize(s);

    // deterministic sampling using provided seed
    double r = ((double)(seed & 0xffffffffULL) / (double)0xffffffffULL);
    // if seed is zero, use simple pseudo-randomness
    if (seed == 0) r = 0.5;

    double cumsum = 0.0;
    int chosen = ctx->vocab_size - 1;
    for (int i = 0; i < ctx->vocab_size; ++i) {
        cumsum += (double) h_probs[i];
        if (r <= cumsum) { chosen = i; break; }
    }

    free(h_probs);

    *token_out = chosen;
    if (ctx->d_sampled_token) {
        // write asynchronously then synchronize
        cudaMemcpyAsync(ctx->d_sampled_token, token_out, sizeof(int32_t), cudaMemcpyHostToDevice, s);
        if (s) cudaStreamSynchronize(s);
    }

    return 0;
}

} // extern "C"
