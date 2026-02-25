# GPU-Exclusive Decode Architecture - Implementation Status

**Last Updated:** 2026-02-24
**Status:** In Progress - Completing Critical Gaps

## SECTION COMPLETION STATUS

### ✅ COMPLETED SECTIONS
1. **GPU Sampling** - CUDA kernels for top-k, top-p filtering (llama-topk-gpu.cpp, llama-topp-gpu.cpp)
2. **Backend Lock Enforcement** - Immutable backend during decode (llama-decode-backend-lock.cpp)
3. **Decode Phase Detection** - CPU/GPU phase boundary (llama-decode-logging-disable.cpp, multiple)
4. **MoE Expert Streaming** - Expert cache management (recent commit)
5. **KV Cache Compression** - FP8/Q8 support infrastructure (llama-kv-cache-iswa.cpp)
6. **Kernel Fusion** - RMSNorm+QKV fusion (llama-rnorm-matmul-fusion.cpp, llama-bias-activation-fusion.cpp)
7. **Graph Freeze** - Graph immutability (llama-graph-freeze-enforce.cpp)
8. **CPU Dequantization Elimination** - GPU-only dequant (llama-cpu-dequantization-elimination.cpp)
9. **Attention State GPU** - GPU-resident attention state (llama-attention-state-gpu.cpp)
10. **Token Persistent Execution** - Token-level persistence infrastructure

### 🔄 PARTIALLY COMPLETED SECTIONS
- **Persistent CUDA Graph Execution** - Framework exists, needs cudaGraphCapture/Replay implementation
- **GPU-Resident Sampling** - Kernels exist, needs RNG state management on GPU
- **SSM Acceleration** - CUDA kernel stub exists, needs full implementation
- **Stream-Ordered Execution** - Partial, needs cudaEventRecord/StreamWaitEvent conversion

### ⏳ NOT STARTED / NEEDS COMPLETION
1. **Decode Loop Offload** - Persistent kernel wrapper (Section 7)
2. **GPU Utilization Maximization** - Micro-batching infrastructure (Section 16)
3. **Memory Residency Guarantee** - Pre-decode verification layer (Section 17)
4. **Hybrid Pipeline Overlap** - Fallback implementation (Section 8)

## CRITICAL GAPS TO FILL (Priority Order)

### Priority 1: CUDA Graph Persistence
- [ ] Implement graph capture at decode start
- [ ] Implement graph replay per token
- [ ] Add graph caching/reuse logic
- [ ] Eliminate per-token cudaGraphCreate overhead

### Priority 2: GPU RNG State Management
- [ ] Move RNG state to GPU global memory
- [ ] Implement GPU-side RNG evolution
- [ ] Remove CPU RNG polling

### Priority 3: Persistent Decode Kernel
- [ ] Implement persistent kernel wrapper
- [ ] Move decode loop entirely to GPU
- [ ] Implement GPU-side termination logic

### Priority 4: Memory Residency Verification
- [ ] Add pre-decode layer/KV residency check
- [ ] Implement abort on non-resident memory
- [ ] Add memory pressure monitoring

### Priority 5: SSM Full Implementation
- [ ] Complete SSM convolution CUDA kernel
- [ ] Complete state update kernel
- [ ] Complete gated recurrence kernel
- [ ] Test with qwen3next models

## FILES THAT NEED CREATION/MODIFICATION

### New Files Needed:
1. `ggml/src/ggml-cuda/graph-executor.cu` - CUDA graph capture/replay
2. `src/llama-cuda-graph-cache.cpp` - Graph caching infrastructure
3. `ggml/src/ggml-cuda/rng-gpu-state.cu` - GPU RNG management
4. `src/llama-decode-persistent-kernel.cpp` - Persistent kernel orchestration
5. `src/llama-memory-residency-verify.cpp` - Pre-decode verification
6. `ggml/src/ggml-cuda/ssm-full.cu` - Complete SSM implementation

### Files to Modify:
- `ggml/src/ggml-cuda/sampling.cu` - Integrate GPU RNG
- `src/llama-context.cpp` - Add graph cache, RNG management
- `src/llama-sampler.cpp` - Use GPU RNG instead of CPU
- `CMakeLists.txt` - Add new CUDA files

## ARCHITECTURE LAYERS STATUS

```
GPU-EXCLUSIVE DECODE PIPELINE
================================

Layer 1: Control Plane (CPU)
├── Request parsing ✓
├── Context setup ✓
├── Graph construction ✓
├── ONE-TIME CUDA graph capture ⏳
└── Monitoring only ✓

Layer 2: Data Plane (GPU)
├── Embedding ✓
├── Forward pass (all layers) ✓
├── Attention (fused) ✓
├── MoE routing ✓
├── SSM operations ⏳ (kernel stub)
├── Sampling with GPU RNG ⏳
├── KV update ✓
└── Persistent execution ⏳

Layer 3: Synchronization
├── Stream-ordered execution ⏳
├── Event barriers (no device sync) ⏳
└── Async host observation ✓

Layer 4: Memory Management
├── KV compression (FP8/Q8) ✓
├── Expert LRU cache ✓
├── Pre-allocated arenas ✓
├── Residency verification ⏳
└── No per-token allocation ✓
```

## NEXT STEPS

1. Implement CUDA graph capture/replay infrastructure
2. Complete GPU RNG state management
3. Integrate persistent kernel framework
4. Add memory residency verification
5. Complete SSM implementation
6. Full integration testing
7. Performance benchmarking
8. PR creation

---

**Total Commits Required:** ~8-12 major commits to complete architecture
