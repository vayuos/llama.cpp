# GPU Sampling Optimization Integration Guide

## Overview

This document describes how to integrate GPU-accelerated token sampling into llama.cpp to eliminate the GPU-to-CPU logits transfer bottleneck, reducing per-token latency by 5-10ms.

## Current State

The codebase already has:
- ✅ Penalty application kernel in `ggml/src/ggml-cuda/sampling.cuh`
- ✅ CUDA kernel framework in place
- ✅ Backend integration hooks

## What's Missing

GPU-side implementations for:
1. **Argmax (Greedy Sampling)** - Find token index with highest logit
2. **Softmax** - Compute probability distribution from logits
3. **Categorical Sampling** - Sample from probability distribution
4. **Temperature Scaling** - Apply temperature to logits

## Implementation Roadmap

### Phase 1: Kernel Implementation (High Priority)

**File:** `ggml/src/ggml-cuda/sampling.cuh` (extend existing)

### 1. Argmax Kernel
```cuda
// Parallel reduction to find max logit and its index
__global__ void kernel_argmax(
    const float * d_logits,
    int32_t * d_out_token,
    int32_t vocab_size
)
```

**Key Optimization:**
- Use warp-level shuffles for register-resident reduction
- Eliminate shared memory bank conflicts
- Expected latency: 100-200 μs for vocab_size=50k

### 2. Softmax Kernel
```cuda
// Numerically stable softmax with parallel reductions
// Step 1: Find max (for stability)
// Step 2: Compute exp and partial sums
// Step 3: Normalize by total sum
```

**Key Optimization:**
- Use two-pass online reduction to avoid numerical overflow
- Process all vocab tokens in parallel (one block per vocab_size)
- Expected latency: 500-1000 μs for vocab_size=50k

### 3. Categorical Sampling Kernel
```cuda
// Inverse transform sampling using cumulative probabilities
// Uses curand for random number generation
__global__ void kernel_sample_categorical(
    const float * d_probs,  // [vocab_size] normalized probabilities
    int32_t * d_out_token,
    int32_t vocab_size,
    uint64_t seed
)
```

**Key Optimization:**
- Single thread per warp for sequential cumsum (cache-friendly)
- Vectorized random number generation
- Expected latency: 300-500 μs

### Phase 2: Integration with llama_sampling (Medium Priority)

**Files to Modify:**
- `include/llama.h` - Add GPU sampling context struct
- `src/llama.cpp` - Add initialization/cleanup
- `tools/server/server-context.cpp` - Call GPU sampling in decode loop

**Integration Points:**

```cpp
// In llama_sampling_context (extend existing struct)
struct cuda_sampling_gpu_context {
    float * d_logits;           // Device memory for logits
    float * d_penalties;        // Device memory for penalties
    float * d_probs;            // Device memory for probabilities
    int32_t * d_sampled_token;  // Device memory for output token
};

// New functions to add
int llama_sampling_init_gpu(llama_sampling_context * ctx);
int llama_sampling_free_gpu(llama_sampling_context * ctx);
int llama_sample_token_gpu(
    llama_sampling_context * ctx,
    llama_token_data_array * candidates,
    float temperature,
    int32_t top_k,
    float penalty_alpha
);
```

### Phase 3: Hot-Path Optimization (Low Priority)

**Performance Profile Areas:**
1. Batch sampling loop in `update_slots()` [server-context.cpp:2600+]
2. Logits copy loop [optimize to use GPU sampling directly]
3. llama_sample_token calls [redirect to GPU when available]

## Expected Performance Impact

| Metric | Baseline | With GPU Sampling | Improvement |
|--------|----------|-------------------|------------|
| TPOT (ms) | 20-30 ms | 10-15 ms | **33-50%** |
| D2H Copy | ~5-8 ms | 0 ms | **5-8 ms saved** |
| CPU Sampling | ~2-3 ms | 0 ms | **2-3 ms saved** |
| GPU Util | 70-80% | 90-95% | **Better overlap** |

## Testing Checklist

- [ ] Argmax kernel produces identical results to CPU `std::max_element`
- [ ] Softmax kernel produces numerically stable results
- [ ] Categorical sampling produces correct distribution
- [ ] Temperature scaling works correctly
- [ ] Greedy sampling matches CPU baseline
- [ ] Top-K sampling matches CPU baseline
- [ ] Correctness on different vocab sizes (1k, 10k, 50k, 100k)
- [ ] Performance scales linearly with vocab_size
- [ ] No VRAM leaks in context init/cleanup
- [ ] Works on different GPU architectures (70, 80, 89)

## Build Instructions

Current build already supports GPU sampling framework:

```bash
./build_optimized.sh
```

To add kernel implementations:

```bash
cd build_optimized
cmake -DGGML_CUDA_SAMPLING=ON ..
make -j$(nproc)
```

## Verification Script

```bash
#!/bin/bash
# Verify GPU sampling correctness

python3 profile_baseline.py \
  --server http://localhost:8000 \
  --iterations 100 \
  --output baseline_with_gpu_sampling.json
```

Compare TPOT metrics before/after GPU sampling implementation.

## References

### CUDA Optimization Techniques
- [NVIDIA: Parallel Reduction in CUDA](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [Numerically Stable Softmax](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
- [curand for Sampling](https://docs.nvidia.com/cuda/curand/host-api-overview.html)

### llama.cpp Integration
- [Backend System](ggml/include/ggml-backend.h)
- [Sampling Context](include/llama.h#L1445)
- [Server Decode Loop](tools/server/server-context.cpp#L2600)

## Code Example: GPU Greedy Sampling

Once implemented, usage would be:

```cpp
// Initialize GPU context
llama_sampling_context * ctx = llama_sampling_init(model, params);
llama_sampling_init_gpu(ctx);

// Main inference loop
while (generating_tokens) {
    // Compute logits (on GPU)
    llama_decode(lctx, batch);
    
    // Sample on GPU (no D2H copy!)
    llama_token token = llama_sample_token_gpu(
        ctx,
        lctx->get_logits(),  // GPU pointer
        temperature,
        top_k,
        penalty_alpha
    );
    
    // Continue generation
}

// Cleanup
llama_sampling_free_gpu(ctx);
llama_sampling_free(ctx);
```

## Troubleshooting

**Issue:** GPU sampling produces different results than CPU
- **Solution:** Check numerical stability of softmax (max subtraction)
- **Check:** d_logits is in correct memory layout (row-major)

**Issue:** Kernel times out on large vocab
- **Solution:** Increase shared memory allocation or use global memory reduction
- **Check:** Grid/block size configuration in CMake

**Issue:** VRAM exhaustion
- **Solution:** Re-use device memory buffers across samples
- **Profile:** Check memory allocation in init function

## Future Optimizations

1. **Graph Capture:** Capture GPU sampling as CUDA graph for 10-20% faster replay
2. **Fused Kernels:** Combine softmax + sample in single kernel
3. **Dynamic Dispatch:** Choose algorithm based on vocabulary size
4. **Warp-Level Sampling:** One token per warp for higher throughput
5. **Multi-GPU:** Distribute sampling across GPUs for batch inference

---

**Last Updated:** February 10, 2026  
**Status:** Framework ready for implementation  
**Next Step:** Implement argmax kernel + integrate with llama_sampling
