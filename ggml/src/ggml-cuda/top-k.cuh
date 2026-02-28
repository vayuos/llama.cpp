#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GPU-accelerated top-k values and indices computation
void ggml_cuda_top_k_values_indices(ggml_backend_cuda_context & ctx,
                                    const float * d_logits,
                                    float * d_values,
                                    int * d_indices,
                                    int n_vocab,
                                    int k);
