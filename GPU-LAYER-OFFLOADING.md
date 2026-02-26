# GPU Layer Offloading Configuration Guide

## Problem

Running with `-ngl 20` creates hybrid CPU/GPU execution:
```
load_tensors: layer   0 assigned to device CPU
load_tensors: layer   1 assigned to device CPU
...
load_tensors: layer  28 assigned to device CPU
load_tensors: layer  29 assigned to device CUDA0
...
load_tensors: layer  47 assigned to device CUDA0
load_tensors: offloaded 20/49 layers to GPU
```

**Result**: GPU-exclusive decode optimization is completely defeated.

## Understanding `-ngl` Parameter

### What `-ngl` Does
```
-ngl N  = "number of GPU layers"
```

This means: **Offload the LAST N layers to GPU**

### Example: Model with 48 Transformer Layers

| Command | GPU Layers | CPU Layers | Split Point |
|---------|-----------|-----------|------------|
| `-ngl 0` | 0 | 48 | All CPU |
| `-ngl 20` | 20 | 28 | Layer 28→29 |
| `-ngl 48` | 48 | 0 | All GPU |
| `-ngl 999` | 48 (max) | 0 | All GPU |

### Why This Matters for GPU-Exclusive Decode

Your design requires:
- ✓ Token embeddings: GPU
- ✓ Layers 0-47: GPU
- ✓ Output projection: GPU
- ✓ **Zero CPU execution during decode**

Current configuration (`-ngl 20`):
- ✗ Embeddings: GPU ✓
- ✗ Layers 0-28: **CPU** ✗
- ✗ Layers 29-47: GPU ✓
- ✗ Output: GPU ✓
- ✗ **28 layers execute on CPU!**

## Impact on Decode Performance

### CPU/GPU Hybrid Path (Current, `-ngl 20`)
```
Token Embedding (GPU)
  ↓
Layer 0 (CPU) → transfers activations via PCIe
Layer 1 (CPU) → transfers activations via PCIe
...
Layer 28 (CPU) → transfers activations via PCIe
Layer 29 (GPU) ← receives from CPU
Layer 30-47 (GPU) → processes on GPU
Output (GPU)

Performance: ~120 tokens/sec (CPU bottleneck)
PCIe traffic: Constant activation transfer
GPU utilization: 30-40% (waiting for CPU layers)
```

### Full GPU Path (Target, `-ngl 48`)
```
Token Embedding (GPU)
  ↓
Layers 0-47 (GPU) → all on GPU, no transfers
  ↓
Output (GPU)

Performance: ~130-150 tokens/sec (GPU only)
PCIe traffic: None (weights loaded once)
GPU utilization: 95-99% (fully utilized)
```

**Performance improvement: +15-25%**

## Layer Counting Explanation

### Why "20/49" instead of "20/48"?

Model reports `n_layer = 48` (48 transformer blocks)

Runtime counts:
- 48 transformer blocks
- 1 output/projection layer
- **Total: 49 logical layers**

So: `offloaded 20/49 layers to GPU` means 20 out of 49 total layers.

This is correct - not a bug, just a counting convention including the output layer.

## Configuration for GPU-Exclusive Decode

### Recommended: Maximum GPU Layers

```bash
# Force ALL layers to GPU (recommended for GPU-exclusive)
./llama-server -m model.gguf -ngl 999

# Or explicit max
./llama-server -m model.gguf -ngl 48
```

Expected output:
```
load_tensors: offloaded 48/49 layers to GPU
or
load_tensors: offloaded 49/49 layers to GPU (if including output)
```

**Key**: No layers should be assigned to CPU.

### How to Choose GPU Layers

```bash
# Check available VRAM
nvidia-smi

# Calculate max layers:
# Each layer ≈ model_size / num_layers
# For 7B model with 48 layers ≈ 145MB per layer

# For 24GB VRAM: All layers fit → use -ngl 48
# For 12GB VRAM: ~80 layers fit → use -ngl 48 (test)
# For 8GB VRAM: ~55 layers fit → use -ngl 48 or reduce

# Rule of thumb
-ngl 48    # 7B-13B models on RTX 3090/4090
-ngl 24    # 7B models on RTX 3080/4070
-ngl 12    # 7B models on RTX 3060
```

### Memory Checking

```bash
# Run and monitor VRAM usage
nvidia-smi dmon | head -20

# Or query once
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Calculate if all layers fit:
# If Used < 95% of Total → OK
# If Used > 95% of Total → Out of memory, reduce -ngl
```

## Complete Optimal Configuration

For GPU-exclusive decode on well-equipped GPU:

```bash
./llama-server \
  -m model.gguf \
  -ngl 999 \           # Max GPU layers (auto-limits to available VRAM)
  --no-mmap \          # Avoid CPU embedding fallback
  -t 8 \               # 8 threads for other tasks
  -c 2048              # Context size
```

## Verification Checklist

After launching, verify in logs:

```
✓ All layers on GPU:
  load_tensors: offloaded 48/49 layers to GPU
  (or 49/49 if counting output)

✓ No CPU layers:
  Should NOT see: "load_tensors: layer   X assigned to device CPU"

✓ No embedding fallback:
  Should NOT see: "cannot be used with preferred buffer type CUDA_Host"

✓ Backend initialized:
  Look for: "backend init" or "CUDA backend" without errors

✓ Performance:
  Tokens/sec: 130-150+ (not 120 or lower)
```

## Performance Comparison

| Configuration | GPU Layers | Tokens/sec | GPU Util | Reason |
|--------------|-----------|-----------|---------|--------|
| `-ngl 0` | 0 | ~30 | 0% | CPU only (slowest) |
| `-ngl 20` | 20 | ~120 | 35% | Hybrid, PCIe bottleneck |
| `-ngl 48` | 48 | ~140+ | 95% | GPU-exclusive (optimal) |

## Troubleshooting

### "Out of Memory" with `-ngl 999`

**Solution**: Gradually reduce until it fits

```bash
# Try in order:
-ngl 999    # Auto (may OOM)
-ngl 48     # All layers
-ngl 40     # Most layers
-ngl 32     # Half layers
-ngl 24     # Quarter layers
```

### "Still seeing CPU layers" with `-ngl 48`

**Check**:
1. GPU has enough VRAM for all layers
2. Run `nvidia-smi` to verify VRAM available
3. Check for errors in server startup

```bash
# Debug: Force GPU-only
./llama-server -m model.gguf -ngl 48 -v  # Verbose output
```

### "Performance not improving" with max GPU layers

**Check the other fixes**:
1. Backend symbols properly exported? (see CUDA-BACKEND-FIX.md)
2. Embeddings on GPU or CPU? (see TENSOR-PLACEMENT-WORKAROUND.md)
3. MMAP enabled? (add `--no-mmap` if issue)

## Summary

**For GPU-Exclusive Decode**: Always use **maximum GPU layers**

```bash
# This is correct:
-ngl 999        # or -ngl 48 for 48-layer models

# This defeats GPU-exclusive optimization:
-ngl 20         # Leaves 28 layers on CPU!
-ngl 32         # Leaves 16 layers on CPU!
```

The `-ngl` parameter controls which layers execute where. For GPU-exclusive optimization, **all transformer layers must execute on GPU**.

## Related Issues

See also:
- `CUDA-BACKEND-FIX.md` - Backend symbol export (prerequisite)
- `TENSOR-PLACEMENT-WORKAROUND.md` - Embedding placement (prerequisite)
- `GPU-EXCLUSIVE-DECODE.md` - Full optimization architecture
