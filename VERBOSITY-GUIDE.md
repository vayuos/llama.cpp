# Maximum Verbosity Runtime Guide

**Updated:** 2026-02-27
**Build Scripts Updated:**
- scripts/build_variants_mmq_moe.sh ✅
- scripts/build_variants_mmq_moe_inc.sh ✅

---

## Quick Start - Maximum Debug Output

```bash
# Set all debug environment variables
export LLAMA_LOG_LEVEL=DEBUG
export GGML_LOG_LEVEL=DEBUG
export GGML_SCHED_DEBUG=1
export GGML_CUDA_DEBUG=1
export GGML_BACKEND_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1
export CUDA_VERBOSE_API_TRACE=1

# Run inference with verbose flags
./build_cuda_mmq_moe/bin/llama-server -m model.gguf --verbose 2>&1 | tee inference.log
```

---

## Environment Variables Reference

### Core Logging

#### `LLAMA_LOG_LEVEL=DEBUG`
- **Purpose:** Enable debug-level logging from llama.cpp core library
- **Levels:** QUIET, ERROR, WARN, INFO, DEBUG
- **Default:** WARN
- **Output:** CPU-side decisions, inference progress, sampler operations
- **When to use:** Always enable for debugging

**Example output:**
```
[DEBUG] llama_decode: n_tokens=1, pos=512
[DEBUG] sampler_apply_penalties: repeat_last_n=64
[DEBUG] sampler_sample: selected token 45632 (The)
```

---

#### `GGML_LOG_LEVEL=DEBUG`
- **Purpose:** Enable debug-level logging from GGML backend
- **Levels:** QUIET, ERROR, WARN, INFO, DEBUG
- **Default:** WARN
- **Output:** Backend operations, tensor operations, memory management
- **When to use:** Always enable for debugging

**Example output:**
```
[DEBUG] ggml_backend_dispatch: tensors scheduled
[DEBUG] ggml_backend_sched: offloading layers to GPU
[DEBUG] ggml_metal_get_device_memory: total: 8192 MB
```

---

#### `GGML_SCHED_DEBUG=1`
- **Purpose:** Enable GPU scheduler debug output
- **Output:** Task scheduling, GPU work queuing, synchronization points
- **When to use:** When debugging GPU workload distribution

**Example output:**
```
[SCHED] Task 1: matmul (512x768) → GPU 0
[SCHED] Task 2: add_bias (512x768) → GPU 0
[SCHED] Graph: 24 tasks, critical path: 156.2 ms
```

---

#### `GGML_CUDA_DEBUG=1`
- **Purpose:** Enable CUDA backend-specific debug output
- **Output:** CUDA kernel launches, memory allocations, graph operations
- **When to use:** When debugging GPU operations

**Example output:**
```
[CUDA] Launching kernel: matmul_f32 (1024x768x512)
[CUDA] Memory: allocated 256MB, total: 4.2GB / 8GB
[CUDA] Graph capture: 47 kernels, 12.3 ms replay time
```

---

#### `GGML_BACKEND_DEBUG=1`
- **Purpose:** Enable general backend infrastructure debug output
- **Output:** Backend selection, registration, switching
- **When to use:** When debugging backend behavior

**Example output:**
```
[BACKEND] Available: CUDA (device 0: RTX4060Ti), CPU
[BACKEND] Selected for decode: CUDA
[BACKEND] KV cache placement: GPU (CUDA)
```

---

### CUDA Runtime Debugging

#### `CUDA_LAUNCH_BLOCKING=1`
- **Purpose:** Make all CUDA operations synchronous
- **Effect:** CPU waits for GPU completion after each kernel
- **Performance:** Significantly slower (debugging only)
- **Benefit:** Errors reported immediately (easier debugging)
- **When to use:** When tracking down GPU errors or hangs

**Without:**
```
CPU launches kernel → CPU continues → GPU might error later
```

**With:**
```
CPU launches kernel → CPU waits → Error caught immediately
```

---

#### `CUDA_DEVICE_WAITS_ON_EXCEPTION=1`
- **Purpose:** GPU waits for exception handling
- **Effect:** GPU halts on errors instead of continuing
- **Benefit:** Prevents cascading errors from GPU problems
- **When to use:** Always during debugging

**Example:** If a memory access is out-of-bounds:
```
Without: GPU continues with garbage data, causes corruption downstream
With:    GPU halts, error reported to CPU immediately
```

---

#### `CUDA_VERBOSE_API_TRACE=1`
- **Purpose:** Log EVERY CUDA API call
- **Output:** Extremely detailed (can produce 1GB+ logs)
- **Useful for:** Deep kernel debugging, memory leak tracking
- **When to use:** Only for very specific debugging

**Example output:**
```
cudaMalloc(ptr=0x7fff..., size=1048576)
cudaMemcpy(dst=0x7fff..., src=0x555..., size=1048576, kind=H2D)
cudaLaunchKernel(kernel=0x405..., gridDim=(128,1,1), blockDim=(256,1,1))
```

⚠️ **WARNING:** This produces MASSIVE log files. Not recommended for long runs.

---

## Four Verbosity Configurations

### Configuration 1: Standard Debug

**Use:** Most common debugging scenario
**Performance:** 10-20% slower

```bash
export LLAMA_LOG_LEVEL=DEBUG
export GGML_LOG_LEVEL=DEBUG
export GGML_SCHED_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1

./bin/llama-server -m model.gguf --verbose
```

**What you see:**
- All llama.cpp debug messages
- GGML operations
- Scheduler decisions
- CUDA synchronization points
- Error details

**Good for:**
- General debugging
- Performance analysis
- Understanding inference flow
- Tracking architectural violations

---

### Configuration 2: GPU-Exclusive Decode Diagnostics

**Use:** When testing GPU-exclusive architecture
**Performance:** 20-50% slower
**Purpose:** Deep GPU and backend diagnostics

```bash
export LLAMA_LOG_LEVEL=DEBUG
export GGML_LOG_LEVEL=DEBUG
export GGML_SCHED_DEBUG=1
export GGML_CUDA_DEBUG=1
export GGML_BACKEND_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1
export CUDA_VERBOSE_API_TRACE=1

./bin/llama-server -m model.gguf --verbose
```

**What you see:**
- Everything from Configuration 1, PLUS:
- CUDA kernel launch details
- Backend selection decisions
- Graph capture/replay operations
- Memory allocations
- Every CUDA API call

**Good for:**
- Debugging GPU-exclusive violations
- Verifying GPU autonomy
- Tracking CUDA graph behavior
- Deep performance analysis

---

### Configuration 3: Production (Minimal Verbosity)

**Use:** Running inference for actual work
**Performance:** Normal (-5% overhead)
**Purpose:** Minimal output for clean logs

```bash
export LLAMA_LOG_LEVEL=INFO
export GGML_LOG_LEVEL=WARN

./bin/llama-server -m model.gguf
```

**What you see:**
- Only errors and important info
- No debug messages
- Clean, readable output

**Good for:**
- Production inference
- Benchmarking
- Normal operations
- Deployment

---

### Configuration 4: KV Cache & Sampling Detailed

**Use:** Debugging KV cache or sampling issues
**Performance:** 20-30% slower
**Output:** Log file for later analysis

```bash
export LLAMA_LOG_LEVEL=DEBUG
export GGML_LOG_LEVEL=DEBUG
export GGML_SCHED_DEBUG=1
export GGML_CUDA_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1
export CUDA_VERBOSE_API_TRACE=1

./bin/llama-server -m model.gguf --verbose 2>&1 | tee inference.log
```

**Output:**
- File: `inference.log` (contains all output)
- Screen: Real-time output (for monitoring)
- Both: Duplicated for analysis and live viewing

**Good for:**
- Detailed post-run analysis
- Comparing multiple runs
- Finding patterns in failures
- Long-term debugging

---

## What to Look For in Debug Output

### GPU-Exclusive Decode Indicators

#### ✅ Good Signs:
```
[DEBUG] GPU-exclusive decode mode ACTIVE
[DEBUG] offloaded N/N layers to GPU
[DEBUG] GPU sampling kernels initialized
[SCHED] All decode tasks on GPU 0
[CUDA] Graph launched once for entire sequence
```

#### ❌ Bad Signs:
```
[WARN] CPU fallback for sampling
[DEBUG] KV cache moved to CPU
[WARN] Transfer: logits GPU→CPU
[ERROR] CPU sampling called during decode
[SCHED] Tasks split: CPU and GPU
```

---

### Backend Selection

#### Expected Output:
```
[BACKEND] Available backends: CUDA, CPU
[BACKEND] Backend selection: CUDA (for decode)
[BACKEND] Backend lock acquired (decode critical path)
[BACKEND] CPU backend not registered (GPU-exclusive mode)
```

#### Problem Indicators:
```
[WARN] CUDA backend unavailable, using CPU
[ERROR] Failed to initialize GPU backend
[WARN] GPU memory insufficient, falling back to CPU
```

---

### Synchronization Points

#### Expected (Efficient):
```
[SCHED] 1000 GPU tasks in dependency graph
[CUDA] Graph instantiated: 234 kernels, 5.2ms
[CUDA] Graph launched (one-time)
[CUDA] Event: Decode complete
```

#### Problem (Inefficient):
```
[CUDA] Kernel 1 launched (per-token)
[CUDA] cudaStreamSynchronize()  ← Per-token sync!
[CUDA] Kernel 2 launched (per-token)
[CUDA] cudaStreamSynchronize()  ← Per-token sync!
```

---

### Memory Management

#### Expected:
```
[CUDA] Allocated: 6.2 GB for model weights
[CUDA] Allocated: 1.8 GB for KV cache
[CUDA] Total GPU memory used: 8.0 GB / 8.0 GB
[CUDA] No CPU fallback needed (GPU resident)
```

#### Problem:
```
[CUDA] Allocated: 6.2 GB for weights
[CUDA] Allocated: 0.5 GB for cache (limited!)
[WARN] KV cache partial on CPU
[WARN] Memory transfers: GPU↔CPU per token
```

---

## Log Analysis Tips

### 1. Search for Warnings/Errors
```bash
grep "\[WARN\]\|\[ERROR\]" inference.log
```

### 2. Count Synchronization Points
```bash
grep "cudaStreamSynchronize\|Stream sync" inference.log | wc -l
```

### 3. Check GPU Utilization Pattern
```bash
grep "cudaLaunchKernel\|Kernel launch" inference.log | wc -l
```

### 4. Monitor Memory Allocations
```bash
grep "Allocated\|Memory" inference.log | head -20
```

### 5. Track Token Progression
```bash
grep "token\|sampler" inference.log | head -50
```

---

## Common Debug Scenarios

### Scenario 1: "Is GPU decode actually working?"

```bash
# Run with this config
export GGML_BACKEND_DEBUG=1
export GGML_SCHED_DEBUG=1

./bin/llama-server -m model.gguf 2>&1 | grep "GPU\|GPU-exclusive\|BACKEND"
```

**Look for:**
- `GPU-exclusive decode mode ACTIVE`
- `All N/N layers offloaded to GPU`
- `Backend: CUDA`

---

### Scenario 2: "Why is inference slow?"

```bash
# Run with performance profiling
export GGML_SCHED_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1  # Accurate timing

time ./bin/llama-server -m model.gguf -n 100 2>&1 | grep "time\|ms\|performance"
```

**Look for:**
- Critical path timing
- Task distribution
- GPU utilization percentage

---

### Scenario 3: "Is KV cache on GPU?"

```bash
# Run with memory debug
export GGML_CUDA_DEBUG=1

./bin/llama-server -m model.gguf 2>&1 | grep -i "kv\|cache\|memory"
```

**Look for:**
- `KV cache on GPU` (good)
- `KV cache on CPU` (bad)
- Memory allocations for weights and cache

---

## Script Auto-Printing

Both build scripts now auto-print verbosity configurations at the end:

```bash
./scripts/build_variants_mmq_moe.sh
# ... build output ...
# Then prints verbosity options
```

You'll see all 4 configurations printed automatically after successful build.

---

## Tips for Efficient Debugging

1. **Start with Configuration 1** (Standard Debug)
   - Provides most useful info without overwhelming detail

2. **Save logs to files**
   - Use `2>&1 | tee logfile.log` to both see and save

3. **Use grep to filter**
   - Focus on specific keywords: GPU, kernel, memory, error

4. **Compare runs**
   - Run before/after changes with same config
   - Diff the log files

5. **Gradually increase verbosity**
   - Start with INFO level
   - Increase to DEBUG if needed
   - Add SCHED_DEBUG for scheduler issues
   - Add CUDA_DEBUG for GPU issues

6. **Disable CUDA_VERBOSE_API_TRACE for long runs**
   - It produces MASSIVE logs (GB per minute)
   - Only use for very short test runs

---

## Performance Impact Summary

| Config | Overhead | Use Case |
|--------|----------|----------|
| Configuration 1 (Standard) | -10% to -20% | General debugging |
| Configuration 2 (GPU Diag) | -20% to -50% | GPU-specific debugging |
| Configuration 3 (Minimal) | -5% | Production |
| Configuration 4 (Detailed) | -20% to -30% | KV/sampling debugging |

---

## Environment Variable Quick Reference

```bash
# Copy-paste ready full debug config:

export LLAMA_LOG_LEVEL=DEBUG
export GGML_LOG_LEVEL=DEBUG
export GGML_SCHED_DEBUG=1
export GGML_CUDA_DEBUG=1
export GGML_BACKEND_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1
export CUDA_VERBOSE_API_TRACE=1

./build_cuda_mmq_moe/bin/llama-server -m model.gguf --verbose 2>&1 | tee full_debug.log
```

This provides **maximum possible verbosity** for debugging GPU-exclusive decode architecture.
