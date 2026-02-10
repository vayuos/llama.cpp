# Optimization of llama.cpp for Decode Throughput

This README outlines the objectives, analysis, and implementation strategy for optimizing `llama.cpp` specifically for **single-sequence decode throughput** on a desktop-class NVIDIA GPU. The primary goal is to maximize tokens-per-second (t/s) and GPU utilization during the autoregressive generation phase, overcoming CPU-side orchestration bottlenecks.

Based on detailed system analysis in `systemchanges.md`.

## 1. Objectives

*   **Primary Goal**: significantly reduce CPU utilization and increase GPU utilization during the token-by-token decode phase.
*   **Target Metric**: Maximize sustained tokens/sec without compromising correctness or determinism.
*   **Constraint**: Maintain exact autoregressive semantics (no speculative decoding, no architectural changes).

## 2. Hardware Context

The optimization targets the following specific hardware profile:

*   **CPU**: x86_64 desktop processor (12 hardware threads available). Supports AVX/AVX2.
*   **GPU**: NVIDIA GeForce RTX 4060 Ti (Ada Lovelace, 16GB VRAM, CC 8.9).
*   **Memory**: Discrete CPU DRAM and GPU VRAM. No unified memory.
*   **Software**: Linux, CUDA, `llama-server` binary.

## 3. Performance Analysis: The "CPU Gap"

Analysis reveals that at **batch size = 1 (decode phase)**, the GPU is severely underutilized.

*   **Bottleneck**: The critical path is dominated by CPU-side orchestration, kernel launch overhead, and synchronization stops.
*   **Key Issue**: The GPU completes small kernels (MV, GEMV) faster than the CPU can prepare and launch the next set of operations.
*   **Synchronization**: Frequent CPU<->GPU sync points (sampling, KV cache updates, graph boundaries) force the GPU to idle.

## 4. Build Variants & Recommendations

Different build configurations define *where* math executes. For this specific hardware and objective:

| Build Variant | Recommendation | Reason |
| :--- | :--- | :--- |
| **`build_cuda_mmq_moe`** | **Preferred** | Best arithmetic efficiency, supports fused quantized kernels, highest decode t/s. |
| `build_cuda_dense` | Good Alternative | Useful for unquantized models where MMQ is not applicable. |
| `build_cuda_cublas_dense` | Avoid for Decode | Optimized for prefill/batching; high kernel launch overhead hurts single-stream decode. |
| `build_cpu_hybrid` | Avoid | CPU-bound; effectively zero GPU utilization during decode. |

**Recommendation**: Use **`build_cuda_mmq_moe`** for all large quantized models.

## 5. Optimization Strategy

To achieve the objectives, the following changes are prioritized:

### A. Architectural Changes
1.  **Reduce Synchronization**: Minimize `cudaDeviceSynchronize` calls. Use CUDA streams more effectively to overlap CPU work with GPU execution.
2.  **Sampling Offload**: Move the sampling pipeline (argmax, penalties, top-k/p) from CPU to GPU to remove the largest per-token synchronization barrier.
3.  **Graph Persistence**: Extend graph and resource lifetime to avoid per-token rebuilding overhead.
4.  **Hardware Residency**: Ensure Model Weights and KV Cache remain strictly GPU-resident. Zero host<->device copies during decode.

### B. Configuration Tuning
*   **Threads**: Use strictly fewer CPU threads (e.g., physical core count or less) to reduce context switching jitter.
*   **Batching**: Strict batch size = 1 for decode. No micro-batching.
*   **Context**: Use the smallest sufficient context length to minimize attention compute growth.

### C. Build-Time Options
*   Target `sm_89` (Ada) explicitly.
*   Enable `MMQ` and `Flash Attention` support.
*   Compile with `-O3 -march=native`.

## 6. Implementation Plan targeting `systemchanges.md`

Items identified for immediate code intervention:
*   **`ggml-cuda.cu`**: Stream optimization and sync reduction.
*   **`sampling.cpp`/`sampling.cu`**: Porting probability filtering and token selection to CUDA.
*   **`llama-kv-cache.cpp`**: Enforcing GPU residency.
*   **`server.cpp`**: Isolating HTTP/slot logic from the heavy decode loop.

## 7. Validation Methodology

Success is defined by:
1.  **Setup**: Single sequence, fixed seed, fixed prompt.
2.  **Metric**: Steady-state tokens/sec (ignoring prefill).
3.  **Criteria**:
    *   Increase in t/s (Target: 1.3x - 2.0x).
    *   Reduction in CPU usage (no longer pegged at 100%).
    *   Increase in GPU compute usage during decode.
    *   **Strict Correctness**: Output must match baseline bit-for-bit.
