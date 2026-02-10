#include "sampling.cuh"

#include <device_launch_parameters.h>

__global__ void apply_penalties_optimized_kernel(float *         logits,
                                                 const int32_t * last_tokens,
                                                 int32_t         n_vocab,
                                                 int32_t         n_last,
                                                 float           repeat_penalty,
                                                 float           alpha_frequency,
                                                 float           alpha_presence) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_last) {
        return;
    }

    int32_t token = last_tokens[tid];
    if (token < 0 || token >= n_vocab) {
        return;
    }

    // Only the first occurrence maps to an update
    for (int i = 0; i < tid; ++i) {
        if (last_tokens[i] == token) {
            return;
        }
    }

    int count = 0;
    for (int i = 0; i < n_last; ++i) {
        if (last_tokens[i] == token) {
            count++;
        }
    }

    float logit = logits[token];
    if (logit <= 0.0f) {
        logit *= repeat_penalty;
    } else {
        logit /= repeat_penalty;
    }

    logit -= (float) count * alpha_frequency + (float) (count > 0) * alpha_presence;
    logits[token] = logit;
}

void ggml_cuda_penalties(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    const int64_t n_vocab = src0->ne[0];
    const int64_t nrows   = ggml_nrows(src0);
    const int64_t n_last  = src1->ne[0];

    const float repeat_penalty  = ggml_get_op_params_f32(dst, 0);
    const float alpha_frequency = ggml_get_op_params_f32(dst, 1);
    const float alpha_presence  = ggml_get_op_params_f32(dst, 2);

    float *         logits_d      = (float *) src0->data;
    const int32_t * last_tokens_d = (const int32_t *) src1->data;

    cudaStream_t stream = ctx.stream();

    if (n_last > 0) {
        int block_size = 128;
        int num_blocks = (n_last + block_size - 1) / block_size;

        for (int64_t i = 0; i < nrows; ++i) {
            apply_penalties_optimized_kernel<<<num_blocks, block_size, 0, stream>>>(
                logits_d + i * n_vocab, last_tokens_d, n_vocab, n_last, repeat_penalty, alpha_frequency,
                alpha_presence);
        }
    }
}
