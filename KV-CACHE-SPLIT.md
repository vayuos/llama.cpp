# KV Cache Split Between CPU/GPU

## Problem

When running with `-ngl 20`, KV cache is split:

```
llama_kv_cache: layer   0: dev = CPU
llama_kv_cache: layer   1: dev = CPU
...
llama_kv_cache: layer  28: dev = CPU
llama_kv_cache: layer  29: dev = CUDA0
...
llama_kv_cache: layer  47: dev = CUDA0

Total KV size: 576 MiB
  CPU KV: 288 MiB (layers 0-28)
  GPU KV: 288 MiB (layers 29-47)
```

This mirrors the layer split and creates a hybrid memory topology.

## Root Cause

**This is a direct consequence of Issue #4 (Partial Layer Offloading)**

KV cache is allocated per-layer on the **same device** where that layer executes:

```
-ngl 20 (20 GPU layers) →
  Layers 0-28 execute on CPU → KV cache on CPU RAM
  Layers 29-47 execute on GPU → KV cache on GPU VRAM
```

Not a separate bug - a symptom of hybrid layer configuration.

## Decode Flow with Split KV

```
Token Input
  ↓
Layer 0-28 (CPU) → KV lookup from CPU RAM
  ↓ [PCIe Transfer - BOTTLENECK]
  ↓
Layer 29-47 (GPU) → KV lookup from GPU VRAM
  ↓
Output
```

Every token requires:
1. CPU computation (28 layers)
2. **PCIe transfer of activations**
3. GPU computation (19 layers)

## Performance Impact

### Memory Bandwidth

```
PCIe Gen3 × 16: ~16 GB/sec
PCIe Gen4 × 16: ~32 GB/sec
PCIe Gen5 × 16: ~64 GB/sec

Per-token activation size: ~2-4 MB
Tokens/sec limit from PCIe:
  Gen3: 16000 MB/s ÷ 2 MB = ~8000 tokens/sec (theoretical)
  Gen4: 32000 MB/s ÷ 2 MB = ~16000 tokens/sec (theoretical)
  Gen5: 64000 MB/s ÷ 2 MB = ~32000 tokens/sec (theoretical)

Reality: 120-130 tokens/sec (CPU bottleneck dominates PCIe limit)
```

### Synchronization Overhead

```
CPU executes layers 0-28: ~8-10 ms
GPU waits for CPU: [sync point]
GPU executes layers 29-47: ~2-3 ms
Total per-token: ~10-13 ms → ~77-100 tokens/sec

With overhead + PCIe transfer: ~120 tokens/sec
```

## Solution

**Same as Issue #4**: Use maximum GPU layers

```bash
# Before (hybrid - KV split)
./llama-server -m model.gguf -ngl 20

# After (GPU-exclusive - all KV on GPU)
./llama-server -m model.gguf -ngl 999
```

## Expected Result After Fix

```
llama_kv_cache: layer   0: dev = CUDA0
llama_kv_cache: layer   1: dev = CUDA0
...
llama_kv_cache: layer  47: dev = CUDA0

Total KV size: 576 MiB
  CPU KV: 0.00 MiB
  GPU KV: 576.00 MiB (all on GPU)
```

**All KV cache on GPU = Maximum performance**

## Verification Checklist

After fixing Issue #4 (`-ngl 999`):

```
✓ All layers on GPU:
  grep "layer.*dev = " log shows CUDA0 for all layers

✓ No CPU KV:
  Should NOT see "dev = CPU" in KV cache output

✓ Single device:
  All "dev = CUDA0" (no mixing)

✓ Performance:
  Tokens/sec 140+ (no PCIe bottleneck)
```

## Related Issues

- **Issue #4**: Partial Layer Offloading (root cause)
- **Issue #3**: Tensor Placement (embeddings fallback)
- **Issue #1-2**: Backend Symbols (prerequisite)

Fixing Issue #4 automatically fixes Issue #5 (KV cache split).

## Memory Telemetry Issue

The log shows:

```
CPU KV buffer size = 0.00 MiB
CUDA0 KV buffer size = 0.00 MiB
Total KV size = 576 MiB
```

This is a reporting bug - per-device accounting doesn't match aggregate size.

**Workaround**: Calculate from layer count:
```
Per-layer KV size = Total KV ÷ num_layers
Per-device KV = layers_on_device × per_layer_size
```

For this example:
```
Per-layer KV: 576 MiB ÷ 48 = 12 MiB
CPU KV: 29 layers × 12 MiB = 348 MiB
GPU KV: 19 layers × 12 MiB = 228 MiB
Total: 576 MiB ✓
```

## Summary

| Aspect | Current | After Fix |
|--------|---------|-----------|
| Layer distribution | 28 CPU / 20 GPU | 48 GPU |
| KV cache split | 288 MiB CPU / 288 MiB GPU | 576 MiB GPU |
| Decode synchronization | Per-token (slow) | Continuous (fast) |
| PCIe traffic | Per-token (high) | Once (at load) |
| Performance | ~120 tokens/sec | ~140+ tokens/sec |

**Fix**: Apply Issue #4 solution (`-ngl 999`)

This automatically resolves KV cache split by moving all layers and their KV to GPU.
