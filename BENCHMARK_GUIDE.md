# 🚀 Performance Benchmark Guide

## Overview

Three benchmark scripts are provided to measure and verify llama.cpp optimizations:

1. **FULL_BENCHMARK.sh** ⭐ RECOMMENDED
   - Complete end-to-end benchmark with actual API requests
   - Measures real throughput (tok/sec)
   - Tests short, medium, and long prompts
   - Captures all system diagnostics
   - **Runtime**: ~5-10 minutes

2. **RUN_BENCHMARK.sh**
   - Simpler benchmark script
   - Good for quick checks
   - **Runtime**: ~2-3 minutes

3. **EXTRACT_METRICS.sh**
   - Analyzes existing log files
   - No server restart needed
   - Good for extracting data from previous runs

---

## Quick Start

### Step 1: Make Scripts Executable
```bash
cd ~/llama/llama.cpp
chmod +x FULL_BENCHMARK.sh RUN_BENCHMARK.sh EXTRACT_METRICS.sh
```

### Step 2: Run Full Benchmark (RECOMMENDED)
```bash
./FULL_BENCHMARK.sh
```

This will:
1. ✅ Kill any existing servers
2. ✅ Start optimized llama-server with all flags
3. ✅ Run 3 benchmark tests (short, medium, long prompts)
4. ✅ Measure generation throughput in real-time
5. ✅ Extract GPU metrics from logs
6. ✅ Generate comprehensive results

**Output files:**
- `server_logs_benchmark_YYYYMMDD_HHMMSS.txt` - Full server log
- `benchmark_results_YYYYMMDD_HHMMSS.txt` - Metric summary

---

## Expected Results

### Optimizations Applied
- ✅ **GPU Layers**: All 49/49 on GPU (100% offload)
- ✅ **Token Embeddings**: On GPU (ROCm0)
- ✅ **KV Cache**: On GPU (102 MiB)
- ✅ **Context**: Optimized to 8,192 tokens
- ✅ **Batch**: 4,096 tokens
- ✅ **Ubatch**: 1,024 tokens
- ✅ **Flash Attention**: Enabled

### Performance Targets

| Metric | Baseline | Expected | Improvement |
|--------|----------|----------|-------------|
| Prompt Throughput | 405 tok/sec | 475-550 tok/sec | +17-35% |
| Gen Speed | ~60 tok/sec | ~60+ tok/sec | Stable |
| GPU Memory | 88% | 88-89% | Optimal |

---

## Interpreting Results

### Good Results ✅
```
Time: 4,523ms | Tokens: 256 | Speed: 56.56 tok/sec
✅ No OOM errors
✅ No GPU errors
offloaded 49/49 layers to GPU
```

### Warning Signs ⚠️
```
Time: 2,145ms | Tokens: 128 | Speed: 59.67 tok/sec  ← Too low (check CPU bottleneck)
⚠️ OOM DETECTED!  ← Reduce context or batch size
⚠️ GPU ERRORS DETECTED!  ← Check ROCm installation
```

---

## Detailed Options

### For Custom Configurations

Edit FULL_BENCHMARK.sh and modify these variables:

```bash
PORT=8080                    # Change if needed
LONG_PROMPT="..."           # Customize test prompts
```

Server parameters in FULL_BENCHMARK.sh:
```bash
-c 8192         # Context size (reduce if OOM)
-b 4096         # Batch size (reduce if OOM)
-ub 1024        # Ubatch (keep smaller than batch)
-ngl 999        # GPU layers (999 = all)
--no-mmap       # CRITICAL for ROCm embeddings
```

---

## Troubleshooting

### Server Won't Start
```bash
# Check if port is in use
lsof -i :8080

# Kill any existing server
pkill -f llama-server

# Run with explicit error output
./llama-server -m /path/to/model.gguf --verbose
```

### Low Throughput
Check these in order:
1. **Are all layers on GPU?**
   ```bash
   grep "offloaded.*layers" server_logs_benchmark_*.txt
   # Should show: offloaded 49/49 layers to GPU
   ```

2. **Is token embedding on GPU?**
   ```bash
   grep "ROCm0 model buffer" server_logs_benchmark_*.txt
   # Should show: ROCm0 model buffer size = ~42000 MiB
   ```

3. **Is KV cache on GPU?**
   ```bash
   grep "ROCm0 KV buffer" server_logs_benchmark_*.txt
   # Should show: ROCm0 KV buffer size = 102.00 MiB
   ```

4. **Check CPU/GPU split**
   ```bash
   grep "buffer size" server_logs_benchmark_*.txt
   # CPU (ROCm_Host) should be < 200 MiB
   ```

### OOM Errors
Reduce context and batch:
```bash
# Edit FULL_BENCHMARK.sh, change:
-c 8192    → -c 4096
-b 4096    → -b 2048
```

### GPU Errors (ROCm/HIP)
```bash
# Check ROCm installation
rocm-smi

# Check available GPU memory
rocm-smi --showmeminfo

# Try environment variable
export GGML_HIP_PINNED_MEM=1
./FULL_BENCHMARK.sh
```

---

## Performance Comparison

### Against Baseline (405 tok/sec)

| Optimization | Impact | Cumulative |
|--------------|--------|-----------|
| GPU-exclusive decode (-ngl 999) | +15-25% | 466-506 |
| Token embeddings on GPU | +5-10% | 489-556 |
| Context optimization (8K) | Already applied | ~510-560 |
| Batch/Ubatch tuning | Already applied | ~510-560 |

**Final Expected Range**: **475-560 tok/sec** (target: +17-38% improvement)

---

## File Locations

```
~/llama/llama.cpp/
├── FULL_BENCHMARK.sh                    ← Run this
├── server_logs_benchmark_*.txt           ← Server logs (auto-generated)
├── benchmark_results_*.txt               ← Metric summary (auto-generated)
├── server_logs_all_optimizations_*.txt   ← Previous run logs
└── /home/vayuos/models/qwen/
    └── Qwen3-Coder-Next-UD-Q4_K_XL.gguf
```

---

## Next Steps

After running benchmark:

1. **Check results file**
   ```bash
   cat benchmark_results_*.txt
   ```

2. **Compare to baseline**
   - Baseline: 405 tok/sec
   - Target: 475-560 tok/sec
   - Calculate: (new_speed - 405) / 405 * 100 = % improvement

3. **If not meeting targets**:
   - Run `./EXTRACT_METRICS.sh` to analyze in detail
   - Check GPU metrics (layer offloading, memory placement)
   - Review troubleshooting section above

4. **Commit results**
   ```bash
   git add benchmark_results_*.txt server_logs_benchmark_*.txt
   git commit -m "benchmark: optimization verification with X tok/sec"
   ```

---

## References

- Commit 8c18344: Token embedding buffer placement fix
- BUILD_ALL_OPTIMIZATIONS.sh: Full build pipeline
- OPTIMIZATION_SUMMARY.md: Configuration guide
- ADVANCED_OPTIMIZATIONS.md: Technical analysis
