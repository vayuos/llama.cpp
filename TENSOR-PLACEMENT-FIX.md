# Tensor Placement Fix - GPU-Exclusive Decode

## Problem

Runtime warning:
```
load_tensors: tensor 'token_embd.weight' (q4_K) cannot be used with preferred buffer type CUDA_Host,
using CPU instead
```

This causes embedding lookups to execute on CPU instead of GPU, severely degrading decode performance.

## Root Cause

In `src/llama-model.cpp` (lines 2797-2805), when MMAP is enabled, ALL tensors (including critical ones like embeddings) are forced from GPU/Host buffers to CPU buffers:

```cpp
// avoid using a host buffer when using mmap
auto * buft_dev = ggml_backend_buft_get_device(buft);
if (ml.use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        throw std::runtime_error("no CPU backend found");
    }
    buft = ggml_backend_dev_buffer_type(cpu_dev);  // ← Forces to CPU buffer
}
```

**Why this is problematic:**
1. MMAP (memory-mapped files) is used for efficient disk access
2. The code assumes MMAP + Host buffer = incompatible
3. It forcefully downgrades to CPU buffer
4. This includes `token_embd.weight`, which is accessed **every token**
5. Result: Every embedding lookup is CPU-bound instead of GPU-bound

## Impact on GPU-Exclusive Decode

The GPU-exclusive optimization requires:
- Token embedding on GPU: ✓ Needs to be GPU resident
- All transformer layers on GPU: ✓ Normally on GPU
- Decoding path: GPU → Embedding lookup (CPU!) → GPU

This creates a hybrid CPU/GPU pipeline that destroys the optimization's performance benefits.

## Solution Strategy

### Option A: Preserve GPU Placement for Critical Tensors (Recommended)

Modify tensor selection to preserve GPU placement for critical tensors when GPU-exclusive mode is enabled:

```cpp
// For GPU-exclusive decode: preserve GPU placement for critical tensors
if (info.layer == LLM_TENSOR_LAYER_INPUT && tn_tensor == LLM_TENSOR_TOKEN_EMBD) {
    // Keep token embeddings on GPU/Host even with mmap
    // They are small and frequently accessed
    if (gpu_exclusive_decode_enabled) {
        // Skip the CPU buffer downgrade for embeddings
        goto skip_mmap_downgrade;
    }
}

// avoid using a host buffer when using mmap
auto * buft_dev = ggml_backend_buft_get_device(buft);
if (ml.use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
    // ... downgrade to CPU buffer
}

skip_mmap_downgrade:
```

### Option B: Disable MMAP for GPU-Exclusive Mode

If GPU-exclusive decode is enabled, disable MMAP entirely since all tensors should be on GPU anyway:

```cpp
if (gpu_exclusive_decode_enabled && n_gpu_layers >= n_layer) {
    ml.use_mmap = false;  // Disable mmap for GPU-exclusive execution
}
```

### Option C: Per-Tensor Override

Allow specific tensors to maintain GPU placement even with MMAP:

```cpp
static const std::vector<std::string> GPU_CRITICAL_TENSORS = {
    "token_embd.weight",
    "output.weight"
};

bool is_critical = std::find(
    GPU_CRITICAL_TENSORS.begin(),
    GPU_CRITICAL_TENSORS.end(),
    tn.str()
) != GPU_CRITICAL_TENSORS.end();

if (is_critical && gpu_exclusive_decode_enabled) {
    // Preserve GPU placement for critical tensors
    skip_cpu_downgrade = true;
}
```

## Recommended Fix (Option A + Option B Combined)

1. **For GPU-Exclusive Mode**:
   - Disable MMAP when all layers are on GPU
   - Reason: All tensors will be in GPU memory anyway

2. **For Mixed Mode**:
   - Preserve GPU/Host buffers for embedding tensors
   - These are small and frequently accessed
   - Benefit outweighs any MMAP efficiency

## Implementation Location

File: `src/llama-model.cpp`
- Lines 2797-2805: Current problematic code
- Lines 2740-2765: Layer type detection (where we can identify embeddings)
- Lines 2691-2826: `create_tensor` lambda function

## Related Code

Key parameters to check:
- `n_gpu_layers`: Number of layers on GPU
- `n_layer`: Total number of layers
- `ml.use_mmap`: MMAP flag
- `info.layer`: Layer type (INPUT, OUTPUT, REPEATING)
- `tn_tensor`: Specific tensor type

## Performance Impact

### Before Fix
- Token embedding lookup: CPU
- GPU transfer overhead per token: ~50-100 μs
- Total decode time per token: +5-10%

### After Fix
- Token embedding lookup: GPU
- No transfer overhead
- Total decode time per token: -5-10%

## Verification

After implementing fix, check logs should show:

```
load_tensors: token_embd.weight allocated to CUDA device
or
load_tensors: token_embd.weight allocated to CUDA_Host (if mixed GPU)
```

NOT:
```
load_tensors: tensor 'token_embd.weight' cannot be used with preferred buffer type CUDA_Host,
using CPU instead
```

## Code Locations to Modify

1. `src/llama-model.cpp:2797-2805`
   - Add exception for critical GPU-resident tensors

2. `src/llama-model.cpp:2640-2690`
   - Add GPU-exclusive mode detection

3. `src/llama.h` or `src/llama-impl.h`
   - Add flag to track GPU-exclusive mode from parameters

## Testing Strategy

1. **Before**: Run decode with `-ngl 48 -m model.gguf`
   - Observe: `cannot be used with preferred buffer type CUDA_Host`

2. **After**: Run same command
   - Observe: No fallback warnings
   - Verify: `token_embd.weight` allocated to GPU

3. **Performance**: Compare tokens/sec before/after
   - Expected improvement: 5-10% on decode throughput

## Alternative: Disable MMAP Entirely

If implementation complexity is high, simplest fix:

```cpp
// In load_model_common() or similar initialization:
if (n_gpu_layers >= n_layer) {
    // GPU-exclusive mode: all tensors on GPU
    // No need for MMAP since everything fits in GPU memory
    model.use_mmap = false;
}
```

This is simpler but less nuanced - still resolves the core issue.

## References

- MMAP: Memory-mapped file I/O for efficient file access
- Host buffer: Pinned host memory for fast GPU access
- Device buffer: Direct GPU memory
- GPU-exclusive decode: All operations on GPU, no CPU fallback
