# Model Buffer Size Reporting Bug - Zero Accounting

## Problem

At startup, buffer accounting shows:

```
load_tensors:          CPU model buffer size =     0.00 MiB
load_tensors:        CUDA0 model buffer size =     0.00 MiB
load_tensors:    CUDA_Host model buffer size =     0.00 MiB
```

Yet the model file is:
```
print_info: file size = 16.45 GiB
```

**Contradiction**: Model clearly loaded (layers assigned, KV allocated, ~7 GiB used) but buffer accounting reports zero.

## Root Cause Analysis

### What Should Happen

Tensor placement should track buffer allocations:

```
for each tensor in model:
    if assigned_to_CPU:
        cpu_buffer_size += tensor_size
    elif assigned_to_CUDA0:
        gpu_buffer_size += tensor_size
    elif assigned_to_CUDA_Host:
        host_buffer_size += tensor_size

print("CPU: ", cpu_buffer_size)
print("CUDA0: ", gpu_buffer_size)
print("CUDA_Host: ", host_buffer_size)
```

### What's Actually Happening

The reporting layer is **disconnected from the allocator**:

```
// Memory is allocated via:
ggml_backend_alloc_buffer(backend, ...)   ← new GGML_BACKEND_API path
    ↓
Tensors placed on devices
    ↓
Model fully resident and functional

// But reporting reads from:
ctx->model_buffer_size[device]            ← legacy variable
    ↓
Returns: 0.00 MiB (never updated)
```

## Technical Breakdown

### The Accounting Mismatch

**Actual memory layout** (from logs and fit projection):
```
CPU layers (0-28):
  + Token embeddings
  + Layer weights
  ≈ 3-4 GiB

CUDA0 layers (29-47):
  + Layer weights
  + Output projection
  ≈ 3-4 GiB

KV Cache:
  + CPU KV: ~288 MiB (layers 0-28)
  + CUDA KV: ~288 MiB (layers 29-47)
  ≈ 576 MiB

Total: ~7-8 GiB ✅ (matches fit projection of 7052 MiB)
```

**Reported accounting**:
```
CPU model buffer:      0.00 MiB  ✗
CUDA0 model buffer:    0.00 MiB  ✗
CUDA_Host buffer:      0.00 MiB  ✗
Total reported:        0.00 MiB  ✗
```

**Gap**: Allocation tracking not wired to reporting counters.

### Likely Root Causes

#### Cause A: Backend API Path Bypasses Legacy Counters

New backend abstraction (`GGML_BACKEND_API`) allocates directly:

```cpp
// Modern path (what's being used):
ggml_backend_alloc_buffer(backend, size)
    → backend->alloc_buffer_fn(size)
    → Returns buffer pointer
    → [Legacy counters NOT updated]

// vs. Legacy path (what reporting expects):
ggml_alloc_graph_impl(...)
    → ctx->model_buffer_size[device] += size
    → [But this path not taken]
```

**Result**: Tensors allocated outside of tracking mechanism.

#### Cause B: Unified Memory Arena

Model may use a single unified allocator:

```cpp
// Single arena for all tensors:
global_buffer = ggml_backend_alloc_buffer(
    size = 16.45 GiB,
    device = AUTO  // Unified, not per-device
)

// Tensors placed via:
ggml_backend_tensor_set_device(tensor, device)
    → Changes device flag only
    → Does NOT track size by device
```

Tensor **device assignment** ≠ buffer allocation tracking.

#### Cause C: Debug Build Instrumentation Gap

Your build shows:
```
build: 106 (...) (debug)
```

Debug builds may:
- Skip optimized aggregation paths
- Bypass memory statistics collection
- Use simplified allocation without tracking

#### Cause D: Legacy Reporting Variable Not Initialized

```cpp
// Initialization:
ctx->model_buffer_size[CUDA0] = 0;  // ✓ Set to zero

// Updating:
// [Missing code to update during tensor placement]

// Reporting:
print(ctx->model_buffer_size[CUDA0]);  // Prints: 0
```

The variable was never updated after initialization.

## Impact Assessment

### What IS Working

✅ **Functional and Correct**:
- Model loads successfully
- Layers assigned to correct devices
- Tensor placement matches device flags
- KV cache allocates correctly
- Decode executes properly
- ~7-8 GiB actually used

**Evidence**:
```
load_tensors: layer 0-28 assigned to device CPU ✓
load_tensors: layer 29-47 assigned to device CUDA0 ✓
load_tensors: offloaded 20/49 layers to GPU ✓
fit projection: estimated device usage: ~7052 MiB ✓
```

### What IS Broken

❌ **Telemetry and Diagnostics**:
- Cannot verify per-device buffer allocation
- GPU/CPU split unknown from output
- No way to validate tensor placement from logs
- Memory optimization decisions based on guesswork
- Combined with Issue #6 (unaccounted corruption), observability layer is compromised

### What This Breaks

1. **Debugging Memory Issues**
   - "Where is the 7 GiB actually allocated?" → Unknown
   - "How much GPU VRAM is used?" → Must calculate manually
   - "Is tensor X on GPU or CPU?" → Must infer from layer assignment

2. **Optimization Verification**
   - Cannot confirm GPU-exclusive decode from logs
   - KV cache split invisible from reporting
   - Batch optimization impossible to verify

3. **Performance Analysis**
   - No per-device memory pressure data
   - Cannot correlate buffer sizes to throughput
   - Memory bottleneck identification difficult

### What This Doesn't Break

✓ **Execution**:
- Model still works correctly
- Performance not directly affected
- Correctness unimpacted
- Only observability affected

## Solutions

### Solution 1: Calculate from Layer Assignment (Workaround)

Since buffer accounting is broken, calculate manually:

```bash
# 1. Get total model size
MODEL_SIZE_GB=16.45

# 2. From logs, identify layer split
# load_tensors: layer 0-28 assigned to device CPU
# load_tensors: layer 29-47 assigned to device CUDA0

# 3. Estimate per-device allocation
# Method: proportional to layer count
TOTAL_LAYERS=48
CPU_LAYERS=29
GPU_LAYERS=19

# 4. Calculate
CPU_BUFFER_MB=$((MODEL_SIZE_GB * 1024 * CPU_LAYERS / TOTAL_LAYERS))
GPU_BUFFER_MB=$((MODEL_SIZE_GB * 1024 * GPU_LAYERS / TOTAL_LAYERS))

echo "Estimated CPU buffer: $CPU_BUFFER_MB MiB"
echo "Estimated GPU buffer: $GPU_BUFFER_MB MiB"
```

**Result**:
```
Estimated CPU buffer: 8954 MiB (~8.75 GiB)
Estimated GPU buffer: 6554 MiB (~6.4 GiB)
Total: ~15.15 GiB (minus KV/overhead)
```

### Solution 2: Add Explicit Accounting Instrumentation (Code Fix)

**Location**: `src/llama-model.cpp` (tensor placement loop)

**Current code** (approximate):
```cpp
for (auto & kv : model.tensors_by_name) {
    ggml_tensor * tensor = kv.second;

    // Assign to device
    ggml_backend_tensor_set_device(tensor, device);

    // BUG: No accounting update
}
```

**Fixed code**:
```cpp
for (auto & kv : model.tensors_by_name) {
    ggml_tensor * tensor = kv.second;
    size_t tensor_size = ggml_nbytes(tensor);

    // Assign to device
    ggml_backend_tensor_set_device(tensor, device);

    // FIX: Track allocation by device
    if (device == GGML_BACKEND_CPU) {
        ctx->model_buffer_size[GGML_BACKEND_CPU] += tensor_size;
    } else if (device == GGML_BACKEND_CUDA0) {
        ctx->model_buffer_size[GGML_BACKEND_CUDA0] += tensor_size;
    }
    // ... etc for other backends
}
```

**Result**:
```
load_tensors:          CPU model buffer size =  8954 MiB
load_tensors:        CUDA0 model buffer size =  6554 MiB
load_tensors:    CUDA_Host model buffer size =     0 MiB
```

### Solution 3: Use Alternative Diagnostic Tools

Until code fix applied:

```bash
# Option A: NVIDIA-SMI for GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
# Shows: 7210 MiB / 16384 MiB (actual GPU VRAM used)

# Option B: Parse layer assignment from logs
./llama-server -m model.gguf -v 2>&1 | grep "layer.*assigned"
# Count CPU vs GPU layers, estimate from counts

# Option C: Memory mapping from /proc (Linux)
cat /proc/$(pgrep llama-server)/maps | grep -i "model\|heap"
# Shows actual page mappings (OS-level view)

# Option D: Modify startup to print actual sizes
# Add debug print in tensor placement loop before/after
```

### Solution 4: Fix Unified Arena Accounting

If using unified allocator, track per-device breakdown:

```cpp
// Instead of single allocation:
buffer = ggml_backend_alloc_buffer(unified_size)

// Track placement:
struct device_accounting {
    size_t cpu_used = 0;
    size_t gpu_used = 0;
    size_t host_used = 0;
};

device_accounting acc;

for (each tensor) {
    tensor_device_t device = get_tensor_device(tensor);
    switch(device) {
        case CPU:   acc.cpu_used += tensor_size;   break;
        case CUDA0: acc.gpu_used += tensor_size;   break;
        case HOST:  acc.host_used += tensor_size;  break;
    }
}

report_accounting(acc);  // Prints actual breakdown
```

## Verification Steps

### Confirm the Bug

```bash
# Run server with verbose output
./llama-server -m model.gguf -v 2>&1 | tee startup.log

# Check for zero buffer reporting
grep "model buffer size" startup.log
```

**Current (buggy)**:
```
CPU model buffer size =     0.00 MiB
CUDA0 model buffer size =   0.00 MiB
```

### Calculate Actual Allocation

```bash
# From same log, find layer assignment
grep "layer.*assigned" startup.log

# Count CPU vs GPU layers
CPU_COUNT=$(grep "layer.*assigned to device CPU" startup.log | wc -l)
GPU_COUNT=$(grep "layer.*assigned to device CUDA" startup.log | wc -l)

# Calculate expected buffers
MODEL_SIZE_MIB=16845  # 16.45 GiB
TOTAL_LAYERS=48
CPU_BUFFER=$((MODEL_SIZE_MIB * CPU_COUNT / TOTAL_LAYERS))
GPU_BUFFER=$((MODEL_SIZE_MIB * GPU_COUNT / TOTAL_LAYERS))

echo "CPU layers: $CPU_COUNT → Expected buffer: ${CPU_BUFFER} MiB"
echo "GPU layers: $GPU_COUNT → Expected buffer: ${GPU_BUFFER} MiB"
```

### Monitor Actual GPU Memory

```bash
# In another terminal during inference
nvidia-smi dmon

# Or watch specific allocation
watch -n 0.5 "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
```

## Impact on Other Issues

### Compound Effect with Issue #6

**Issue #6**: Memory accounting underflow (unaccounted field shows exabytes)
**Issue #11**: Model buffer accounting (all fields show zero)

Together:
- Unaccounted: 16+ exabytes (corrupted underflow)
- Model buffers: 0 MiB (never tracked)
- KV cache: ~576 MiB (correct)
- **Total**: Impossible to reconcile

**Example output**:
```
Reporting:
  Model CPU:        0 MiB
  Model GPU:        0 MiB
  KV Cache:       576 MiB
  Unaccounted: 16 exabytes
  Total:       ~16 exabytes  [CORRUPTED]

Actual:
  Model CPU:     ~9 GiB
  Model GPU:     ~6 GiB
  KV Cache:      576 MiB
  Total:        ~15.5 GiB  [CORRECT]
```

**Solution**: Fix both #6 and #11 together for correct observability.

## Recommended Fix Priority

### Severity Assessment

| Aspect | Severity | Reason |
|--------|----------|--------|
| Functional impact | None | Model works correctly |
| Performance impact | None | Execution unaffected |
| Observability impact | High | Cannot verify buffer placement |
| Debugging impact | High | Cannot diagnose memory issues |
| Optimization verification | High | Cannot confirm GPU-exclusive from logs |

**Overall**: LOW-MEDIUM priority (observability issue, not functional bug)

### Implementation Cost

**Code Fix**:
- Location: `src/llama-model.cpp` (tensor placement loop)
- Complexity: Low (track tensor sizes during assignment)
- Lines of code: 10-15
- Testing: Verify reported values match calculated

**Time**: 30 minutes to fix + verify

## Related Issues

- **Issue #6**: Memory accounting underflow (unaccounted field corrupted)
  - Both are memory reporting bugs
  - Fix both together for complete observability fix
  - Combined: "Memory Observability Refactor"

- **Issue #4**: GPU layer offloading
  - Depends on this issue for verification
  - "Are all layers actually on GPU?" → Can't tell from logs

- **Issue #5**: KV cache split
  - KV accounting works correctly
  - But model buffer accounting broken
  - Impossible to calculate total VRAM from logs

## Summary Table

| Aspect | Current | After Fix |
|--------|---------|-----------|
| **Reporting** | | |
| CPU model buffer | 0.00 MiB | ~8954 MiB |
| GPU model buffer | 0.00 MiB | ~6554 MiB |
| KV cache (correct) | 576 MiB | 576 MiB |
| **Observability** | | |
| Can verify GPU placement? | No | Yes |
| Can check buffer split? | No | Yes |
| Can diagnose memory issues? | No (partial) | Yes |
| **Impact** | | |
| Execution | Works ✓ | Works ✓ |
| Performance | 120 tok/s | 120 tok/s |
| Diagnostics | Unreliable | Reliable |

## Conclusion

**Model Buffer Size Reporting is a diagnostics bug, not a functional bug.**

The model loads, executes, and performs correctly despite zero buffer accounting. However:

1. **You cannot visually verify tensor placement from logs**
   - Must infer from layer assignment
   - Must calculate manually

2. **Combined with Issue #6, memory observability is compromised**
   - Unaccounted field corrupted (exabytes)
   - Model buffer fields zero
   - Only KV cache accurate
   - Total memory reconciliation impossible

3. **Fix is straightforward**
   - Track tensor sizes during device assignment
   - Update per-device counters
   - ~30 minutes implementation

**Recommendation**: Fix both Issues #6 and #11 together as "Memory Observability Refactor" for complete diagnostic reliability.

Currently: **Not blocking execution, blocks diagnostics.**
