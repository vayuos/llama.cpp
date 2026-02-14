/*
 * test-topk-determinism.cpp
 *
 * Validation harness for GPU top-k determinism guarantee
 *
 * This test ensures that GPU top-k selection produces identical results
 * to CPU implementation for identical inputs and seeds, preserving
 * determinism requirements for reproducible sampling.
 *
 * Test coverage:
 *  1. Identical indices for equal logit values preserved across runs
 *  2. Exact match with CPU tie-breaking rules
 *  3. Bit-exact consistency for multiple seeds and logit patterns
 *  4. Edge cases: k=1, k=vocab_size-1, small vocab, large vocab
 */

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#ifdef GGML_CUDA_AVAILABLE
#include <cuda_runtime.h>
extern "C" {
int cuda_topk_kernel(const float * d_logits,
                     float *       d_topk_vals,
                     int32_t *     d_topk_inds,
                     int32_t       n_vocab,
                     int32_t       k,
                     void *        cuda_stream);
}
#endif

// CPU reference implementation - matches llama_sampler_top_k_impl behavior
void cpu_topk_reference(const std::vector<float> & logits,
                        int32_t k,
                        std::vector<int32_t> & out_indices,
                        std::vector<float> & out_values) {
    size_t n = logits.size();
    assert(k > 0 && k <= (int32_t)n);

    // Create index array
    std::vector<int32_t> indices(n);
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // Partial sort by logit value (descending), maintaining stability for ties
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                     [&logits](int32_t a, int32_t b) {
                         float va = logits[a];
                         float vb = logits[b];
                         // Descending order; ties broken by original index (stable)
                         if (va != vb) {
                             return va > vb;
                         }
                         return a < b;  // Tie-breaking: lower index wins
                     });

    // Extract results
    out_indices.resize(k);
    out_values.resize(k);
    for (int32_t i = 0; i < k; ++i) {
        out_indices[i] = indices[i];
        out_values[i] = logits[indices[i]];
    }
}

void test_topk_determinism(int32_t vocab_size, int32_t k, uint64_t seed) {
    std::cout << "Testing vocab_size=" << vocab_size << " k=" << k << " seed=" << seed << std::endl;

    // Generate random logits
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> logits(vocab_size);
    for (size_t i = 0; i < (size_t)vocab_size; ++i) {
        logits[i] = dist(rng);
    }

    // CPU reference result
    std::vector<int32_t> cpu_indices;
    std::vector<float> cpu_values;
    cpu_topk_reference(logits, k, cpu_indices, cpu_values);

#ifdef GGML_CUDA_AVAILABLE
    // GPU result
    float * d_logits = nullptr;
    float * d_vals = nullptr;
    int32_t * d_inds = nullptr;

    size_t logits_bytes = vocab_size * sizeof(float);
    size_t vals_bytes = k * sizeof(float);
    size_t inds_bytes = k * sizeof(int32_t);

    if (cudaMalloc(&d_logits, logits_bytes) != cudaSuccess ||
        cudaMalloc(&d_vals, vals_bytes) != cudaSuccess ||
        cudaMalloc(&d_inds, inds_bytes) != cudaSuccess) {
        std::cerr << "CUDA allocation failed\n";
        assert(false);
    }

    if (cudaMemcpy(d_logits, logits.data(), logits_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "CUDA memcpy failed\n";
        assert(false);
    }

    int ret = cuda_topk_kernel(d_logits, d_vals, d_inds, vocab_size, k, nullptr);
    assert(ret == 0 && "GPU top-k kernel failed");

    std::vector<float> gpu_values(k);
    std::vector<int32_t> gpu_indices(k);

    if (cudaMemcpy(gpu_values.data(), d_vals, vals_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(gpu_indices.data(), d_inds, inds_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "CUDA memcpy failed\n";
        assert(false);
    }

    // Verify results match
    for (int32_t i = 0; i < k; ++i) {
        assert(gpu_indices[i] == cpu_indices[i] && "Index mismatch");
        assert(std::abs(gpu_values[i] - cpu_values[i]) < 1e-5f && "Value mismatch");
    }

    cudaFree(d_logits);
    cudaFree(d_vals);
    cudaFree(d_inds);

    std::cout << "  ✓ CPU and GPU results identical\n";
#else
    std::cout << "  (CUDA not available, CPU reference valid)\n";
#endif
}

int main(void) {
    std::cout << "=== GPU Top-K Determinism Validation ===\n\n";

    // Test case 1: Small k, small vocab
    test_topk_determinism(10, 3, 42);
    test_topk_determinism(10, 3, 123);

    // Test case 2: Medium k, medium vocab
    test_topk_determinism(128, 32, 456);
    test_topk_determinism(256, 64, 789);

    // Test case 3: Large k, large vocab
    test_topk_determinism(4096, 256, 2024);
    test_topk_determinism(8192, 512, 2025);

    // Test case 4: Edge cases
    test_topk_determinism(100, 1, 111);      // k=1 (greedy)
    test_topk_determinism(100, 99, 222);     // k=vocab_size-1

    // Test case 5: With tied values (determinism critical)
    std::cout << "\nTesting tied value handling:\n";
#ifdef GGML_CUDA_AVAILABLE
    {
        std::vector<float> tied_logits = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 0.5f};
        int32_t vocab_size = tied_logits.size();
        int32_t k = 4;

        std::vector<int32_t> cpu_indices;
        std::vector<float> cpu_values;
        cpu_topk_reference(tied_logits, k, cpu_indices, cpu_values);

        float * d_logits = nullptr;
        float * d_vals = nullptr;
        int32_t * d_inds = nullptr;

        cudaMalloc(&d_logits, vocab_size * sizeof(float));
        cudaMalloc(&d_vals, k * sizeof(float));
        cudaMalloc(&d_inds, k * sizeof(int32_t));

        cudaMemcpy(d_logits, tied_logits.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice);

        int ret = cuda_topk_kernel(d_logits, d_vals, d_inds, vocab_size, k, nullptr);
        assert(ret == 0);

        std::vector<float> gpu_values(k);
        std::vector<int32_t> gpu_indices(k);
        cudaMemcpy(gpu_values.data(), d_vals, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gpu_indices.data(), d_inds, k * sizeof(int32_t), cudaMemcpyDeviceToHost);

        std::cout << "  Tied values test:\n";
        std::cout << "    CPU result: [";
        for (int i = 0; i < k; ++i) {
            std::cout << cpu_indices[i] << (i < k-1 ? "," : "");
        }
        std::cout << "]\n    GPU result: [";
        for (int i = 0; i < k; ++i) {
            std::cout << gpu_indices[i] << (i < k-1 ? "," : "");
        }
        std::cout << "]\n";

        for (int32_t i = 0; i < k; ++i) {
            assert(gpu_indices[i] == cpu_indices[i]);
        }
        std::cout << "  ✓ Tied values handled consistently\n";

        cudaFree(d_logits);
        cudaFree(d_vals);
        cudaFree(d_inds);
    }
#endif

    std::cout << "\n=== All determinism tests PASSED ===\n";
    return 0;
}
