#include "ggml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Try to include CUDA runtime. If this fails build, we'll know.
#include <cuda_runtime.h>

// Include GPU sampling interface
extern "C" {
typedef struct {
    float *   d_logits;
    float *   d_penalties;
    float *   d_probs;
    float *   d_scratch;
    int32_t * d_sampled_token;
    int32_t   vocab_size;
    int32_t   cuda_device;
    void *    cuda_stream;
} cuda_sampling_context_t;

int  cuda_sampling_init_gpu(cuda_sampling_context_t ** ctx, int vocab_size, int device);
void cuda_sampling_free_gpu(cuda_sampling_context_t * ctx);
int  cuda_sampling_set_logits(cuda_sampling_context_t * ctx, float * d_logits, size_t size_bytes, int copy);
int  cuda_sampling_sample_greedy(cuda_sampling_context_t * ctx, int32_t * token_out, void * cuda_stream);
int  cuda_sampling_sample_specialized(cuda_sampling_context_t * ctx,
                                      int32_t *                 token_out,
                                      float                     temperature,
                                      int32_t                   top_k,
                                      float                     penalty_alpha,
                                      uint64_t                  seed,
                                      void *                    cuda_stream);
}

int main() {
    printf("Running GPU Sampling Test...\n");

    // simple argmax correctness test
    const int vocab = 10;
    float     host_logits[vocab];
    for (int i = 0; i < vocab; ++i) {
        host_logits[i] = (float) i;
    }
    host_logits[5] = 100.0f;  // Argmax should be 5

    cuda_sampling_context_t * ctx = NULL;
    if (cuda_sampling_init_gpu(&ctx, vocab, 0) != 0) {
        printf("SKIP: cuda_sampling_init_gpu failed (no device?)\n");
        return 0;
    }

    if (!ctx->d_logits) {
        printf("FAIL: ctx->d_logits not allocated\n");
        cuda_sampling_free_gpu(ctx);
        return 1;
    }

    // Copy logits to device
    if (cudaMemcpy(ctx->d_logits, host_logits, vocab * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("FAIL: cudaMemcpy failed\n");
        cuda_sampling_free_gpu(ctx);
        return 1;
    }

    // Test 1: Greedy Sampling (Argmax)
    int32_t token = -1;
    if (cuda_sampling_sample_greedy(ctx, &token, NULL) != 0) {
        printf("FAIL: cuda_sampling_sample_greedy failed\n");
        cuda_sampling_free_gpu(ctx);
        return 1;
    }

    printf("Argmax Result: %d (Expected: 5)\n", token);
    if (token != 5) {
        printf("FAIL: Argmax incorrect\n");
        cuda_sampling_free_gpu(ctx);
        return 1;
    }

    // Test 2: Softmax / Sampling
    // Use temp=1.0, top_k=0 (full distribution)
    // Logits: [0, 1, 2, 3, 4, 100, 6, 7, 8, 9]
    // Softmax will be ~1.0 for index 5 and ~0.0 for others.
    // Deterministic sampling check is hard without fixed seed logic or extreme values.
    // But since 100 is >> others, it should almost always pick 5.

    if (cuda_sampling_sample_specialized(ctx, &token, 1.0f, 0, 0.0f, 1234, NULL) != 0) {
        printf("FAIL: cuda_sampling_sample_specialized failed\n");
        cuda_sampling_free_gpu(ctx);
        return 1;
    }

    printf("Softmax Sample Result: %d (Expected: 5)\n", token);
    if (token != 5) {
        printf("FAIL: Softmax Sample incorrect (could be random, but unlikely with delta=100)\n");
        cuda_sampling_free_gpu(ctx);
        return 1;
    }

    printf("PASS: GPU Sampling tests passed\n");
    cuda_sampling_free_gpu(ctx);
    return 0;
}
