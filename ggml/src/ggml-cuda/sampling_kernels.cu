/*
 * sampling_kernels.cu
 *
 * GPU implementations (and safe fallbacks) for sampling helper kernels:
 *  - cuda_argmax_kernel
 *  - cuda_apply_penalties_kernel
 *  - cuda_softmax_kernel
 *
 * These are conservative, correct implementations intended as working
 * replacements until highly-optimized device-only kernels are added.
 */

#include "sampling.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

extern "C" {

// Simple block-level reduction argmax for a single vector (vocab)
__global__ static void argmax_kernel(const float * logits, int32_t * out_token, int32_t vocab_size) {
    extern __shared__ float sdata[]; // will hold pairs: val then idx
    float * svals = sdata; // size = blockDim.x
    int * sidx = (int*)&svals[blockDim.x];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float local_max = -INFINITY;
    int local_idx = -1;

    for (int i = gid; i < vocab_size; i += blockDim.x * gridDim.x) {
        float v = logits[i];
        if (v > local_max) { local_max = v; local_idx = i; }
    }

    svals[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    // reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (svals[tid + offset] > svals[tid]) {
                svals[tid] = svals[tid + offset];
                sidx[tid] = sidx[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        // atomic compare across blocks: use atomicMax on integer representation is unsafe for floats
        // So write to out_token only if first block, for correctness run with single block grid or rely on host fallback
        out_token[0] = sidx[0];
    }
}

int cuda_argmax_kernel(float * d_logits, int32_t * d_out_token, int32_t vocab_size, void * cuda_stream) {
    if (!d_logits || !d_out_token || vocab_size <= 0) return -1;

    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) stream = 0;

    // Launch configuration: use up to 1024 threads and several blocks
    int threads = 256;
    int blocks = std::min(1024, (vocab_size + threads - 1) / threads);
    size_t shared = threads * (sizeof(float) + sizeof(int));

    // If vocab is small, single block reduction is fine
    if (blocks <= 1) {
        argmax_kernel<<<1, threads, shared, stream>>>(d_logits, d_out_token, vocab_size);
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    // For larger vocab sizes, fall back to host copy for correctness (safe path)
    float * host_buf = (float*)malloc((size_t)vocab_size * sizeof(float));
    if (!host_buf) return -1;
    if (cudaMemcpyAsync(host_buf, d_logits, (size_t)vocab_size * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) { free(host_buf); return -1; }
    // synchronize to ensure copy complete
    if (cudaStreamSynchronize(stream) != cudaSuccess) { free(host_buf); return -1; }

    int32_t best = 0; float bestv = host_buf[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (host_buf[i] > bestv) { bestv = host_buf[i]; best = i; }
    }
    free(host_buf);

    if (cudaMemcpyAsync(d_out_token, &best, sizeof(int32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess) return -1;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return -1;
    return 0;
}

// Apply penalties: logits[i] -= alpha * penalties[i]
__global__ static void apply_penalties_simple_kernel(float * logits, const float * penalties, float alpha, int32_t vocab_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < vocab_size; i += blockDim.x * gridDim.x) {
        logits[i] -= alpha * penalties[i];
    }
}

int cuda_apply_penalties_kernel(float * d_logits, const float * d_penalties, float alpha, int32_t vocab_size, void * cuda_stream) {
    if (!d_logits || !d_penalties || vocab_size <= 0) return -1;
    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) stream = 0;

    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    blocks = std::min(blocks, 1024);

    apply_penalties_simple_kernel<<<blocks, threads, 0, stream>>>(d_logits, d_penalties, alpha, vocab_size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Softmax: safe host-fallback implementation (copies to host, computes softmax, copies back)
int cuda_softmax_kernel(const float * d_logits, float * d_probs, int32_t vocab_size, void * cuda_stream) {
    if (!d_logits || !d_probs || vocab_size <= 0) return -1;
    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) stream = 0;

    float * host_buf = (float*)malloc((size_t)vocab_size * sizeof(float));
    if (!host_buf) return -1;
    if (cudaMemcpyAsync(host_buf, d_logits, (size_t)vocab_size * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) { free(host_buf); return -1; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { free(host_buf); return -1; }

    // compute softmax on host
    float maxv = host_buf[0];
    for (int i = 1; i < vocab_size; ++i) if (host_buf[i] > maxv) maxv = host_buf[i];
    double sum = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        host_buf[i] = expf(host_buf[i] - maxv);
        sum += host_buf[i];
    }
    if (sum == 0.0) sum = 1.0;
    for (int i = 0; i < vocab_size; ++i) host_buf[i] = (float)(host_buf[i] / sum);

    if (cudaMemcpyAsync(d_probs, host_buf, (size_t)vocab_size * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess) { free(host_buf); return -1; }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { free(host_buf); return -1; }
    free(host_buf);
    return 0;
}

// Temperature scaling kernel: logits[i] /= temperature
__global__ static void temperature_scale_kernel(float * logits, float temperature, int32_t vocab_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < vocab_size; i += blockDim.x * gridDim.x) {
        if (temperature == 0.0f) continue;
        logits[i] = logits[i] / temperature;
    }
}

int cuda_temperature_scale_kernel(float * d_logits, float temperature, int32_t top_k, int32_t vocab_size, void * cuda_stream) {
    (void)top_k;  // currently top_k filtering is not implemented on device
    if (!d_logits || vocab_size <= 0) return -1;
    cudaStream_t stream = (cudaStream_t) cuda_stream;
    if (stream == NULL) stream = 0;

    // apply simple in-place temperature scaling
    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    blocks = max(1, min(blocks, 1024));
    temperature_scale_kernel<<<blocks, threads, 0, stream>>>(d_logits, temperature, vocab_size);

    // currently top_k filtering is not implemented on device; leave logits unchanged for top_k>0
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // extern "C"
