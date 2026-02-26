# Immediate Actions - Quick Start Guide

## 🚀 Phase 1: Execute Now (5 minutes)

### Step 1: Update Your Server Command

**OLD (Current):**
```bash
./llama-server -m model.gguf -ngl 20 -t 8
```

**NEW (Optimized):**
```bash
./llama-server -m model.gguf -ngl 999 --no-mmap -c 16384 -t 8
```

### Step 2: Run & Observe

```bash
./llama-server -m model.gguf -ngl 999 --no-mmap -c 16384 -t 8 2>&1 | tee test_run.log
```

### Step 3: Verify in Logs

Check for these lines:
```bash
grep "offloaded.*layers to GPU" test_run.log
# Expected: "offloaded 48/49 layers to GPU"

grep "n_ctx_seq\|n_ctx_train" test_run.log
# Expected: "n_ctx_seq (16384)"

grep "cannot be used with preferred buffer type" test_run.log
# Expected: NO MATCHES (if matches, embeddings still on CPU)
```

### Step 4: Measure Throughput

Note your throughput:
```bash
grep "tokens/sec\|tokens / sec" test_run.log
# Should be significantly higher than baseline
```

**Expected gain: +30-70% throughput**

---

## 🔧 Phase 2: Schedule for Later (1-2 hours)

When ready, rebuild with backend symbol export:

```bash
./scripts/build-cuda-backend-fix.sh --clean -j$(nproc)
```

This will:
1. Clean previous build
2. Configure with `BUILD_SHARED_LIBS=ON`
3. Build CUDA & CPU backends with symbol export
4. Verify symbols are exported

**After rebuild:** Re-run Phase 1 test to confirm improvements persist

---

## 📊 What You Should See

### BEFORE (Current):
```
offloaded 20/49 layers to GPU
token_embd.weight... cannot be used with preferred buffer type CUDA_Host, using CPU instead
n_ctx_seq (6144) < n_ctx_train (262144)
Throughput: ~30 tokens/sec
```

### AFTER Phase 1 (5 min):
```
offloaded 48/49 layers to GPU ✓
(NO embedding warnings) ✓
n_ctx_seq (16384) ✓
Throughput: ~50-65 tokens/sec ✓
```

### AFTER Phase 2 (1-2 hrs):
```
All of the above +
Backend symbols: ✓ Verified
Throughput: ~65+ tokens/sec ✓
```

---

## 🐛 Troubleshooting

### If throughput doesn't improve after Phase 1:

1. **Verify parameters are actually used:**
   ```bash
   ps aux | grep llama-server
   # Should show: -ngl 999 --no-mmap -c 16384
   ```

2. **Check log for GPU layer count:**
   ```bash
   grep "gpu_layers\|offloaded" server_debug.log
   # If still shows 20, parameters not applied
   ```

3. **Check for GPU memory errors:**
   ```bash
   grep -i "out of memory\|oom\|error" server_debug.log | head -20
   ```

### If you get OOM errors with `-c 16384`:

Reduce context size:
```bash
./llama-server -m model.gguf -ngl 999 --no-mmap -c 8192 -t 8
```

Or remove `--no-mmap` (use MMAP despite embedded fallback):
```bash
./llama-server -m model.gguf -ngl 999 -c 16384 -t 8
```

### If `--no-mmap` causes crashes:

Try without it (embeddings will be on CPU but system more stable):
```bash
./llama-server -m model.gguf -ngl 999 -c 16384 -t 8
```

---

## ✅ Checklist

- [ ] Updated server command with `-ngl 999 --no-mmap -c 16384`
- [ ] Ran server and captured output
- [ ] Verified `offloaded 48/49 layers to GPU`
- [ ] Verified no embedding buffer type errors
- [ ] Measured throughput improvement
- [ ] Documented baseline vs. new performance
- [ ] Scheduled Phase 2 rebuild

---

## 📈 Expected Results

| Metric | Baseline | After Phase 1 | After Phase 2 |
|--------|----------|---------------|---------------|
| GPU Layers | 20/49 | 48/49 | 48/49 |
| Embeddings | CPU | GPU | GPU |
| Context | 6144 | 16384 | 16384 |
| Throughput | ~30 tps | ~50-65 tps | ~65+ tps |
| Improvement | - | +67% | +100%+ |

---

## 💡 Key Insight

The changes are:
1. **Configuration only** (Phase 1) - zero code changes
2. **Reversible** - if something breaks, just revert the command
3. **Safe** - no destructive operations
4. **Fast** - Phase 1 takes 1 minute

**Start with Phase 1 now, schedule Phase 2 for when you have time.**

---

**Next Action:** Run the new server command and measure throughput! 🚀
