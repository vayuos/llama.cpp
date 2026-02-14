#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_top_k_values_indices(ggml_backend_cuda_context & ctx,
                                    const float * src,
                                    float * dst_values,
                                    int * dst_indices,
                                    int n_probs,
                                    int k);
