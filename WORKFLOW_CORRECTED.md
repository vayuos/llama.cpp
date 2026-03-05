# ⚠️ WORKFLOW CORRECTED - Build First, Benchmark Second

## Issue Discovered ❌

The benchmarking script was run before the project was built. The `llama-server` binary doesn't exist yet.

```
ERROR: llama-server: No such file or directory
```

---

## Correct Workflow ✅

### STEP 1: Build the Project (20-30 minutes) ⭐ DO THIS FIRST
```bash
cd ~/llama/llama.cpp
chmod +x BUILD_ALL_OPTIMIZATIONS.sh
./BUILD_ALL_OPTIMIZATIONS.sh
```

**What this does**:
- Pulls latest code (includes commit 8c18344)
- Runs CMake with ROCm/HIP optimization flags
- Compiles all sources
- Creates `build_cuda_mmq_moe_full_logs/bin/llama-server` binary
- Duration: **20-30 minutes**

**Wait for completion** - this is REQUIRED before benchmarking

---

### STEP 2: Verify Build Success ✅
After build completes, check for the binary:
```bash
ls -lh ~/llama/llama.cpp/build_cuda_mmq_moe_full_logs/bin/llama-server
```

Should show:
```
-rwxr-xr-x llama-server (size: 100+ MB)
```

---

### STEP 3: Run Benchmark (5-10 minutes)
```bash
cd ~/llama/llama.cpp
./FULL_BENCHMARK.sh
```

**What this does**:
1. Starts the freshly-built llama-server
2. Runs 3 throughput tests
3. Measures real tok/sec performance
4. Captures GPU metrics
5. Generates results file

**Expected output**:
- `benchmark_results_YYYYMMDD_HHMMSS.txt` with metrics

---

## Complete Timeline

| Step | Task | Duration | Status |
|------|------|----------|--------|
| 1 | Build project | 20-30 min | ⏳ **DO FIRST** |
| 2 | Verify binary | 1 min | Quick check |
| 3 | Run benchmark | 5-10 min | After step 1 |
| 4 | Analyze results | 5 min | After step 3 |
| **Total** | **Full process** | **~35-50 min** | Includes waiting |

---

## Quick Start (Corrected)

```bash
cd ~/llama/llama.cpp

# Step 1: Build (20-30 minutes) - REQUIRED FIRST
chmod +x BUILD_ALL_OPTIMIZATIONS.sh
./BUILD_ALL_OPTIMIZATIONS.sh

# [Wait for build to complete]

# Step 2: Benchmark (5-10 minutes)
./FULL_BENCHMARK.sh

# [Benchmark runs automatically]

# Step 3: Check results
cat benchmark_results_*.txt
```

---

## What to Expect

### During Build (BUILD_ALL_OPTIMIZATIONS.sh)
```
CMake configuring...
[████████████████] 100%
Building...
[████████████████] 100%
✅ Build complete
```

### During Benchmark (FULL_BENCHMARK.sh)
```
Server starting...
✅ Server ready
Running TEST 1 (SHORT PROMPT)...
  Time: 4523ms | Tokens: 256 | Speed: 56.56 tok/sec
Running TEST 2 (MEDIUM PROMPT)...
  Time: 9045ms | Tokens: 512 | Speed: 56.60 tok/sec
Running TEST 3 (LONG PROMPT)...
  Time: 18090ms | Tokens: 1024 | Speed: 56.58 tok/sec

✅ Benchmark complete
Results: benchmark_results_20260305_HHMMSS.txt
```

---

## Expected Results After Build + Benchmark

**Performance Metrics**:
- Prompt throughput: 475-560 tok/sec (vs 405 baseline)
- GPU layers: 49/49 offloaded ✅
- Token embeddings: On GPU (ROCm0) ✅
- Improvement: +17-38%

**GPU Status**:
- Model on GPU: ~42.3 GB / 48 GB
- Memory usage: 88-89%
- No crashes/OOM: ✅

---

## Troubleshooting

### Build Takes Too Long
- Normal: 20-30 minutes for full build
- Monitoring: Watch for CMake progress
- Can't stop: Build must complete

### Build Fails
Check the last 50 lines of output:
```bash
tail -50 BUILD_ALL_OPTIMIZATIONS.sh.log
```

Look for:
- Compiler errors
- Missing dependencies
- GPU driver issues

### Benchmark Won't Start After Build
Scripts now have automatic binary detection. Checks:
1. `build_cuda_mmq_moe_full_logs/bin/llama-server`
2. `build/bin/llama-server`
3. `./llama-server`

If none found, it will error with clear instructions.

---

## Files Updated ✅

The benchmark scripts have been **updated** with binary auto-detection:
- `FULL_BENCHMARK.sh` - Now finds binary automatically
- `RUN_BENCHMARK.sh` - Now finds binary automatically

These will now:
1. Search for llama-server in expected locations
2. Error with clear instructions if not found
3. Proceed if found

---

## Key Differences from Previous Instructions

| Before | After |
|--------|-------|
| Run benchmark immediately | **Build first (20-30 min)** |
| Assumed binary existed | **Checks for binary, guides if missing** |
| No error handling | **Clear error messages with solutions** |

---

## Next Steps

### RIGHT NOW:
1. Run BUILD_ALL_OPTIMIZATIONS.sh
2. Wait for completion (20-30 minutes)

### AFTER BUILD:
1. Run FULL_BENCHMARK.sh
2. Wait for results (5-10 minutes)
3. Review benchmark_results_*.txt

### IF BUILD FAILS:
1. Check error messages
2. Review troubleshooting section
3. Fix issues and rebuild

---

## One-Command Workflow

```bash
cd ~/llama/llama.cpp && chmod +x BUILD_ALL_OPTIMIZATIONS.sh && ./BUILD_ALL_OPTIMIZATIONS.sh && ./FULL_BENCHMARK.sh
```

(Takes ~35-50 minutes total, runs build then immediately benchmarks)

---

## Summary

✅ **Code changes**: Already in repo (commit 8c18344)
⏳ **Build required**: 20-30 minutes (run BUILD_ALL_OPTIMIZATIONS.sh)
🚀 **Benchmark ready**: Runs after build completes

**Current status**: Ready to build and benchmark properly sequenced!
