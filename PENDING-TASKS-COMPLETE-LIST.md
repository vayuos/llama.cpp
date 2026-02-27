# Complete List of Pending Tasks - GPU-Exclusive Decode Implementation

**Last Updated:** 2026-02-27
**Total Pending Tasks:** 47
**Estimated Effort:** 150-200 hours
**Status:** Analysis Complete, Implementation Pending

---

## Critical Path Tasks (Must Do)

### Violation #7: Graph Autonomy Implementation (40-60 hours)

#### Phase 1: Eliminate CPU Decode Loop (6-8 hours)
- [ ] Task 1.1: Create `llama_gpu_exclusive_decode()` wrapper function
  - File: `src/llama-gpu-exclusive-decode-engine.cpp`
  - Status: Partially stubbed, needs full implementation
  - Effort: 2 hours

- [ ] Task 1.2: Remove `for` loop from `examples/simple/simple.cpp` (lines 168-201)
  - Current: CPU-driven token loop
  - Required: Single GPU entry point call
  - Effort: 1 hour

- [ ] Task 1.3: Implement GPU decode entry point
  - File: `src/llama-gpu-exclusive-decode-engine.cpp`
  - Create function: `llama_gpu_exclusive_engine_start_decode()`
  - Effort: 2 hours

- [ ] Task 1.4: Implement enforcement point for CPU loop detection
  - File: `src/llama-decode-loop-elimination.cpp`
  - Function: `llama_decode_loop_elimination_detect_cpu_owns_loop()`
  - Effort: 1 hour

- [ ] Task 1.5: Implement enforcement point for per-token decode calls
  - File: `src/llama-decode-loop-elimination.cpp`
  - Function: `llama_decode_loop_elimination_detect_per_token_decode_calls()`
  - Effort: 2 hours

#### Phase 2: Transfer Token Index to GPU (8-10 hours)
- [ ] Task 2.1: Define `gpu_decode_token_state` structure
  - File: `ggml/src/ggml-cuda/ggml-cuda.cu`
  - Create device-resident state
  - Effort: 2 hours

- [ ] Task 2.2: Implement `gpu_init_token_state()` CUDA kernel
  - Initialize GPU-resident state
  - File: `ggml/src/ggml-cuda/ggml-cuda.cu`
  - Effort: 1 hour

- [ ] Task 2.3: Implement `gpu_advance_token_index()` device function
  - Atomic increment on GPU
  - File: `ggml/src/ggml-cuda/ggml-cuda.cu`
  - Effort: 1 hour

- [ ] Task 2.4: Implement CPU-GPU token state synchronization
  - Copy state back to CPU after decode
  - Function: `gpu_get_tokens_produced()`
  - Effort: 2 hours

- [ ] Task 2.5: Implement enforcement: prevent CPU token index modification
  - File: `src/llama-token-persistent-execution.cpp`
  - Function: `llama_token_persistent_enforce_gpu_ownership()`
  - Effort: 2 hours

- [ ] Task 2.6: Implement enforcement: prevent CPU token index reads
  - File: `src/llama-token-persistent-execution.cpp`
  - Function: `llama_token_persistent_prevent_cpu_position_reads()`
  - Effort: 2 hours

#### Phase 3: GPU-Based Sampling (10-12 hours)
- [ ] Task 3.1: Implement `gpu_sample_temperature_kernel`
  - File: `ggml/src/ggml-cuda/sampling_kernel.cu` (NEW)
  - Apply temperature and sample via softmax
  - Effort: 3 hours

- [ ] Task 3.2: Implement `gpu_sample_greedy_kernel`
  - File: `ggml/src/ggml-cuda/sampling_kernel.cu` (NEW)
  - Argmax sampling on GPU
  - Effort: 2 hours

- [ ] Task 3.3: Implement `gpu_sample_topk_kernel`
  - Top-K sampling on GPU
  - File: `ggml/src/ggml-cuda/sampling_kernel.cu`
  - Effort: 3 hours

- [ ] Task 3.4: Implement `gpu_sample_topp_kernel`
  - Top-P (nucleus) sampling on GPU
  - File: `ggml/src/ggml-cuda/sampling_kernel.cu`
  - Effort: 3 hours

- [ ] Task 3.5: Integrate GPU sampling into GPU decode loop
  - Replace CPU sampler calls with GPU kernel calls
  - File: `ggml/src/ggml-cuda/ggml-cuda.cu`
  - Effort: 2 hours

- [ ] Task 3.6: Implement enforcement: block CPU samplers during decode
  - Add checks to all 6 CPU samplers
  - File: `src/llama-sampler.cpp`
  - Effort: 1 hour

#### Phase 4: Persistent CUDA Graphs (6-8 hours)
- [ ] Task 4.1: Implement single graph capture for entire decode
  - File: `src/llama-gpu-exclusive-decode-engine.cpp`
  - Function: `llama_gpu_exclusive_engine_prepare_decode()`
  - Effort: 2 hours

- [ ] Task 4.2: Implement graph instantiation
  - File: `ggml/src/ggml-cuda/ggml-cuda.cu`
  - Function: `ggml_cuda_graph_instantiate()`
  - Effort: 2 hours

- [ ] Task 4.3: Implement single graph launch (no per-token replays)
  - File: `src/llama-gpu-exclusive-decode-engine.cpp`
  - Function: `llama_gpu_exclusive_engine_start_decode()`
  - Effort: 2 hours

- [ ] Task 4.4: Implement enforcement: verify single graph constraint
  - File: `ggml/src/ggml-cuda/ggml-cuda.cu`
  - Function: `verify_single_launch_kernel()`
  - Effort: 2 hours

#### Phase 5: GPU Signal Interface (4-6 hours)
- [ ] Task 5.1: Implement GPU-to-CPU signaling via CUDA events
  - File: `ggml/src/ggml-cuda/ggml-cuda.cu`
  - Function: `gpu_signal_decode_complete()`
  - Effort: 2 hours

- [ ] Task 5.2: Implement CPU wait for decode complete (non-polling)
  - File: `src/llama-gpu-exclusive-decode-engine.cpp`
  - Function: `llama_gpu_exclusive_engine_wait_for_completion()`
  - Effort: 2 hours

- [ ] Task 5.3: Implement enforcement: detect CPU polling attempts
  - File: `src/llama-decode-loop-elimination.cpp`
  - Function: `llama_decode_loop_elimination_detect_cpu_polling_for_tokens()`
  - Effort: 2 hours

---

## High Priority Tasks (Should Do)

### Build and Compilation (8-10 hours)

- [ ] Task B1: Compile with all GPU-exclusive flags enabled
  - Run: `./scripts/build_variants_mmq_moe.sh`
  - Expected: Clean compilation with no errors
  - Effort: 2 hours

- [ ] Task B2: Resolve compilation errors (if any)
  - Fix missing includes, undefined references, etc.
  - Effort: 4 hours (estimate)

- [ ] Task B3: Verify all enforcement points compile
  - Check all new code in Phase 1-5 compiles
  - Effort: 1 hour

- [ ] Task B4: Link-time verification
  - Ensure all GPU-exclusive symbols are exported
  - Check libggml-cuda.so contains required symbols
  - Effort: 1 hour

### Simple Example Testing (4-6 hours)

- [ ] Task T1: Update `examples/simple/simple.cpp` to use GPU decode
  - Replace CPU loop with `llama_gpu_exclusive_decode()`
  - Effort: 1 hour

- [ ] Task T2: Test simple example with GPU decode
  - Load small model, generate 100 tokens
  - Verify GPU mode active
  - Effort: 2 hours

- [ ] Task T3: Compare outputs: CPU decode vs GPU decode
  - Ensure identical token sequences produced
  - Effort: 1 hour

- [ ] Task T4: Benchmark throughput improvement
  - Measure tokens/sec for CPU-driven vs GPU-autonomous
  - Target: 2-3x improvement
  - Effort: 2 hours

### Violation #1-6 Build Verification (6-8 hours)

- [ ] Task V1: Verify Violation #1 fix (CPU↔GPU Sync)
  - Check backend sync skip on decode path
  - File: `ggml/src/ggml-cuda/ggml-cuda.cu` (lines 3020-3033)
  - Verify in build cache
  - Effort: 1 hour

- [ ] Task V2: Verify Violation #2 fix (Host↔Device Transfers)
  - Check transfer guard active
  - File: `ggml/src/ggml-cuda/sampling_impl.cu` (lines 299-313)
  - Test with actual inference
  - Effort: 2 hours

- [ ] Task V3: Verify Violation #3 fix (CPU Sampling Infrastructure)
  - Check compile-time exclusion guards present
  - File: `src/llama-sampler.cpp`
  - Effort: 1 hour

- [ ] Task V4: Verify Violation #4 fix (CPU Sampling Code)
  - Run with LLAMA_CPU_SAMPLING_EXCLUDED=ON
  - Verify CPU samplers cannot be called
  - Effort: 1 hour

- [ ] Task V5: Verify Violation #5 fix (Hybrid KV Cache)
  - Check hard errors in all 7 KV locations
  - Test with GPU-exclusive mode
  - Effort: 2 hours

- [ ] Task V6: Verify Violation #6 fix (CPU Backend)
  - Check CPU backend not registered with flag
  - File: `ggml/src/ggml-backend-reg.cpp` (lines 197-219)
  - Effort: 1 hour

---

## Medium Priority Tasks (Nice to Have)

### Extended Testing (10-12 hours)

- [ ] Task E1: Test with various model sizes
  - Tiny (13B), Small (70B), Large (405B)
  - Measure throughput for each
  - Effort: 3 hours

- [ ] Task E2: Test with various sequence lengths
  - 10, 100, 500, 1000, 5000 tokens
  - Verify enforcement points work at all scales
  - Effort: 3 hours

- [ ] Task E3: Test with different samplers
  - Temperature, greedy, top-k, top-p
  - Verify GPU sampling works for each
  - Effort: 2 hours

- [ ] Task E4: Stress test - long-running inference
  - Generate 10,000+ tokens continuously
  - Monitor for memory leaks, signal issues
  - Effort: 2 hours

- [ ] Task E5: Test with batched inference
  - Multiple sequences in parallel
  - Verify GPU autonomy holds per-batch
  - Effort: 2 hours

### Integration Testing (8-10 hours)

- [ ] Task I1: Update `llama-server` to use GPU decode
  - Replace existing decode loop with GPU entry point
  - File: `src/server/server.cpp`
  - Effort: 3 hours

- [ ] Task I2: Test llama-server with GPU decode
  - Start server with GPU-exclusive flags
  - Send inference requests
  - Measure throughput improvement
  - Effort: 2 hours

- [ ] Task I3: Test parallel inference in server
  - Multiple concurrent requests
  - Verify GPU remains autonomous
  - Effort: 2 hours

- [ ] Task I4: Update `llama-bench` to support GPU decode
  - Add GPU-exclusive benchmark option
  - File: `examples/llama-bench/llama-bench.cpp`
  - Effort: 2 hours

### Optimization Tasks (8-10 hours)

- [ ] Task O1: Profile GPU kernel performance
  - Identify bottlenecks in autonomous loop
  - Effort: 2 hours

- [ ] Task O2: Optimize GPU sampling kernel
  - Reduce latency of temperature/greedy sampling
  - Effort: 3 hours

- [ ] Task O3: Optimize token index management
  - Reduce atomic operation overhead
  - Effort: 2 hours

- [ ] Task O4: Tune CUDA block/grid dimensions
  - Find optimal launch parameters
  - Effort: 2 hours

- [ ] Task O5: Memory optimization
  - Reduce device memory footprint
  - Optimize GPU-resident state
  - Effort: 2 hours

---

## Low Priority Tasks (Optional)

### Documentation Tasks (6-8 hours)

- [ ] Task D1: Create GPU-exclusive decode user guide
  - How to enable, configure, use
  - Effort: 2 hours

- [ ] Task D2: Document GPU autonomy architecture
  - Design decisions, tradeoffs
  - Effort: 2 hours

- [ ] Task D3: Create performance benchmark comparison
  - Before/after graphs
  - Effort: 2 hours

- [ ] Task D4: Create troubleshooting guide
  - Common issues and solutions
  - Effort: 1 hour

### Cleanup Tasks (4-6 hours)

- [ ] Task C1: Remove deprecated CPU loop code
  - Clean up unused variables, functions
  - Effort: 2 hours

- [ ] Task C2: Consolidate GPU sampling kernels
  - Move to single file if scattered
  - Effort: 1 hour

- [ ] Task C3: Code review and refactoring
  - Ensure consistency, readability
  - Effort: 2 hours

---

## Summary by Category

| Category | Count | Hours | Priority |
|----------|-------|-------|----------|
| Violation #7 Implementation | 23 | 40-60 | CRITICAL |
| Build & Compilation | 4 | 8-10 | HIGH |
| Simple Example Testing | 4 | 4-6 | HIGH |
| Violation #1-6 Verification | 6 | 6-8 | HIGH |
| Extended Testing | 5 | 10-12 | MEDIUM |
| Integration Testing | 4 | 8-10 | MEDIUM |
| Optimization | 5 | 8-10 | MEDIUM |
| Documentation | 4 | 6-8 | LOW |
| Cleanup | 3 | 4-6 | LOW |
| **TOTAL** | **58** | **100-130** | - |

---

## Critical Path (Minimum to Achieve Compliance)

**Required Tasks:** 23 + 4 + 4 + 6 = **37 tasks**
**Estimated Time:** 40-60 + 8-10 + 4-6 + 6-8 = **58-84 hours**
**Timeline:** 1-2 weeks full-time development

**Must Complete:**
1. All Phase 1-5 tasks (Violation #7 implementation)
2. Build and compilation tasks
3. Simple example testing
4. Violation #1-6 verification

**Critical Checkpoints:**
- [ ] Phase 1 complete: CPU loop eliminated
- [ ] Phase 2 complete: Token index on GPU
- [ ] Phase 3 complete: GPU sampling working
- [ ] Phase 4 complete: Persistent graphs
- [ ] Phase 5 complete: Signal interface working
- [ ] Build passes with no errors
- [ ] Simple example generates correct tokens
- [ ] Throughput improved 2-3x

---

## Stretch Goals (Optional Enhancements)

If time permits:
1. Extended testing with various models and sequences
2. Full server integration and testing
3. Performance profiling and optimization
4. Comprehensive documentation
5. Code cleanup and refactoring

---

## Current Status

**Completed:**
- ✅ Architecture analysis (all 7 violations identified)
- ✅ Violation #1-6 code fixes applied
- ✅ Build scripts updated with GPU-exclusive flags
- ✅ Violation #7 detailed analysis
- ✅ Implementation plan (5 phases)

**Not Started:**
- ❌ All implementation tasks (Phase 1-5)
- ❌ Build and testing
- ❌ Verification of existing fixes
- ❌ Integration and extended testing

**Next Action:** Begin Phase 1 (CPU loop elimination) implementation

---

**This document serves as the master task list for the GPU-exclusive decode implementation.**
