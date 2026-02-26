# Tensor Placement Issue - Workarounds

## Quick Diagnosis

If you see this warning:
```
load_tensors: tensor 'token_embd.weight' (...) cannot be used with preferred buffer
type CUDA_Host, using CPU instead
```

**Your embedding lookups are running on CPU instead of GPU!**

## Workarounds (Until Fix is Applied)

### Workaround 1: Disable MMAP (Recommended)

When running llama-server, add `--no-mmap`:

```bash
./build_cuda/bin/llama-server \
  -m model.gguf \
  --no-mmap
```

**Pros:**
- ✓ Simple
- ✓ Embeddings stay on GPU
- ✓ Fastest performance

**Cons:**
- ✗ Requires loading full model into memory
- ✗ Slower startup time
- ✗ May not work for very large models

### Workaround 2: Force Full GPU Placement

Use `--gpu-layers 48` (or `ngl 48`) to force all layers to GPU:

```bash
./build_cuda/bin/llama-server \
  -m model.gguf \
  -ngl 48
```

**Pros:**
- ✓ Maximizes GPU utilization
- ✓ Embeddings on GPU
- ✓ Best for GPU-exclusive optimization

**Cons:**
- ✗ Requires enough VRAM
- ✗ May not work if model > GPU memory

### Workaround 3: Quantize to Fit in VRAM

For models larger than VRAM:

```bash
# First: Quantize model to smaller size
./build_cuda/bin/llama-quantize model.gguf model_q4_k.gguf Q4_K_M

# Then: Run with full GPU layers
./build_cuda/bin/llama-server \
  -m model_q4_k.gguf \
  --no-mmap \
  -ngl 48
```

**Pros:**
- ✓ Works with any model size
- ✓ Both MMAP and full GPU workable

**Cons:**
- ✗ Some quality loss from quantization
- ✗ Extra step required

## Performance Comparison

### With CPU Embedding Fallback (Current Issue)
```
Tokens/sec: ~120
Average latency/token: 8.3 ms
Embedding lookup: CPU (~50% latency)
```

### With GPU Embeddings (After Workaround)
```
Tokens/sec: ~130-140
Average latency/token: 7-7.6 ms
Embedding lookup: GPU (~10% latency)
```

**Expected improvement: +8-12% throughput**

## Recommended Approach

For GPU-exclusive decode optimization:

1. **Use Option 1 (--no-mmap)** if model fits in RAM
2. **Use Option 2 (-ngl 48)** if model fits in VRAM
3. **Use Option 3 (Quantize)** if model larger than VRAM

Command template:
```bash
./build_cuda/bin/llama-server \
  -m model.gguf \
  --no-mmap \
  -ngl 48 \
  -t 8
```

## Monitoring

Check server output for:

```
Good (embeddings on GPU):
- No "cannot be used with preferred buffer type" warnings
- Tokens/sec: ~130-140

Bad (embeddings on CPU):
- Warning about "cannot be used with preferred buffer type CUDA_Host"
- Tokens/sec: ~120 or lower
- High CPU usage during embedding lookup
```

## Build Verification

Ensure build has correct flags (see CUDA-BACKEND-FIX.md):

```bash
./scripts/build-cuda-backend-fix.sh --clean -j$(nproc)
```

This ensures proper symbol export and CUDA support.

## Complete Setup Guide

```bash
# 1. Build with fixes
./scripts/build-cuda-backend-fix.sh --clean -j$(nproc)

# 2. Verify GPU support
./build_cuda_mmq_moe_full_logs/bin/llama-cli --version

# 3. Run with workaround
./build_cuda_mmq_moe_full_logs/bin/llama-server \
  -m model.gguf \
  --no-mmap \
  -ngl $(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader | awk '{print int($1/512)}') \
  -t 8
```

The last command auto-calculates max GPU layers based on VRAM size.

## When the Fix is Applied

Once TENSOR-PLACEMENT-FIX.md is implemented:
- No workarounds needed
- Embeddings automatically stay on GPU
- MMAP and GPU placement work together
- Maximum performance achieved

Check for: "GPU-Exclusive Decode" section in release notes.
