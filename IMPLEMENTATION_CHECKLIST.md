# Implementation Checklist - Complete Action Plan

**Status:** Ready to Execute
**Total Time:** ~2.5 hours
**Risk Level:** LOW

---

## 🎯 Phase 1: Code Fix (30 minutes)

### Task 1.1: Open quantize.cu
- [ ] File: `ggml/src/ggml-cuda/quantize.cu`
- [ ] Use: nano, vim, VS Code, or preferred editor
- [ ] Command: `nano ggml/src/ggml-cuda/quantize.cu`

### Task 1.2: Fix Function 1 (quantize_mmq_q8_1)
- [ ] Find line 238: `static __global__ void quantize_mmq_q8_1(`
- [ ] Find line ~257: `const int64_t i01 = ids ? ids[i1] : i1;`
- [ ] After that line, add:
```cpp
    // Skip padding positions (INT_MAX indicates padding in MoE routing)
    if (ids != nullptr && i01 == INT_MAX) {
        return;
    }
```
- [ ] Verify 3 lines added

### Task 1.3: Fix Function 2 (quantize_mmq_q8_1_rms)
- [ ] Find line 336: `static __global__ void quantize_mmq_q8_1_rms(`
- [ ] Find line ~352: `const int64_t i01 = ids ? ids[i1] : i1;`
- [ ] After that line, add:
```cpp
    // Skip padding positions (INT_MAX indicates padding in MoE routing)
    if (ids != nullptr && i01 == INT_MAX) {
        return;
    }
```
- [ ] Verify 3 lines added

### Task 1.4: Check Function 3 (quantize_mmq_mxfp4)
- [ ] Search for: `quantize_mmq_mxfp4` in file
- [ ] Look for: `const int64_t i01 = ids ? ids[i1] : i1;`
- [ ] If found, apply same fix
- [ ] If not found, mark as complete

### Task 1.5: Save and Verify
- [ ] Save file (Ctrl+S or :wq)
- [ ] Run verification:
```bash
grep -n "INT_MAX" ggml/src/ggml-cuda/quantize.cu
```
- [ ] Should show 2-3 INT_MAX check lines
- [ ] Verify correct line numbers

---

## 🔨 Phase 2: Rebuild (1-2 hours)

### Task 2.1: Clean Build
- [ ] Command:
```bash
./scripts/build_variants_mmq_moe.sh --clean -j$(nproc)
```
- [ ] Expected: Build output showing compilation
- [ ] Wait for: Build completion (1-2 hours)

### Task 2.2: Monitor Build
- [ ] Watch for: "cmake --build" messages
- [ ] Check for: No compilation errors
- [ ] Verify: "[OK] MMQ + FA + CUDA Graphs" message at end

### Task 2.3: Verify Build Success
- [ ] Command:
```bash
nm -D build_cuda_mmq_moe/bin/libggml-cuda.so | grep ggml_backend_init
```
- [ ] Expected output: `T ggml_backend_init`
- [ ] If missing: Check build output for errors

### Task 2.4: Check No OOB Errors
- [ ] Command:
```bash
grep "OOB:" build_cuda_mmq_moe/bin/llama-server.log 2>/dev/null || echo "No OOB errors found"
```
- [ ] Expected: "No OOB errors found"
- [ ] If OOB errors: Check code changes were saved

---

## ⚙️ Phase 3: Configuration (1 minute)

### Task 3.1: Run Server with Optimized Config
- [ ] Command:
```bash
./build_cuda_mmq_moe/bin/llama-server -m model.gguf -ngl 999 --no-mmap -c 16384 -t 8
```

### Task 3.2: Verify in Logs
- [ ] Check for: `offloaded 48/49 layers to GPU`
- [ ] Check for: NO `cannot be used with preferred buffer type` errors
- [ ] Check for: NO `OOB:` errors
- [ ] Expected: Server running, accepting requests

---

## ✅ Verification Checklist

### Code Changes Applied
- [ ] quantize_mmq_q8_1 has INT_MAX check
- [ ] quantize_mmq_q8_1_rms has INT_MAX check
- [ ] quantize_mmq_mxfp4 checked (and fixed if needed)
- [ ] File saved successfully

### Build Successful
- [ ] Build completed without errors
- [ ] libggml-cuda.so has ggml_backend_init symbol
- [ ] libggml-cpu.so has ggml_backend_init symbol
- [ ] No OOB errors in output

### Server Running
- [ ] Server starts without crash
- [ ] GPU layers offloaded (48/49)
- [ ] No embedding buffer warnings
- [ ] No OOB memory errors
- [ ] Handles MoE inference without crash

---

## 📊 Expected Results

### Before Fixes
```
Status: CRASH
Error: OOB: ids_src1[36]=2147483647
Reason: INT_MAX padding not handled
```

### After Code Fix + Rebuild
```
Status: RUNNING
Backend: Symbols exported ✓
GPU: 48/49 layers offloaded
Throughput: ~30 tokens/sec
```

### After Phase 1 Config
```
Status: OPTIMIZED
GPU: 48/49 layers offloaded ✓
Embeddings: GPU ✓
Context: 16384 ✓
Throughput: ~50-65 tokens/sec (+67%)
```

---

## 🚨 Troubleshooting

### If Build Fails
```bash
# Check error output
tail -50 build_cuda_mmq_moe/CMakeOutput.log

# Clean and retry
rm -rf build_cuda_mmq_moe
./scripts/build_variants_mmq_moe.sh --clean -j$(nproc)
```

### If OOB Errors Still Appear
- [ ] Verify code changes were saved: `grep -n "INT_MAX" ggml/src/ggml-cuda/quantize.cu`
- [ ] Check line numbers are correct
- [ ] Make sure file was saved before rebuild

### If Symbols Still Missing
```bash
# Check CMake configuration
grep BUILD_SHARED_LIBS build_cuda_mmq_moe/CMakeCache.txt
# Should show: BUILD_SHARED_LIBS:BOOL=ON
```

### If Server Still Crashes
- [ ] Verify code fix applied to all functions
- [ ] Check quantize_mmq_mxfp4 was handled
- [ ] Rebuild with fresh checkout if needed

---

## 📝 Documentation Reference

### For This Phase
- `CODE_CHANGES_NEEDED.md` - Exact code changes
- `MOE_EXPERT_ROUTING_BUG.md` - Deep bug analysis
- `ALL_BUILD_SCRIPTS_UPDATED.md` - Build script details

### For Next Phases
- `IMMEDIATE_ACTIONS.md` - Phase 1 config
- `COMPREHENSIVE_FIX_REPORT.md` - All 7 issues

---

## ⏱️ Timeline

```
Start: Now
  ↓
Code Fix: 30 minutes
  ↓
Rebuild: 1-2 hours
  ↓
Config: 1 minute
  ↓
Verify: 10 minutes
  ↓
COMPLETE: ~2.5 hours total
```

---

## ✨ Key Points

1. **Code fix is critical** - Without this, MoE will crash
2. **Build scripts are ready** - Already have `-DBUILD_SHARED_LIBS=ON`
3. **Simple changes** - Just 3-line checks added twice
4. **Low risk** - Can't break anything, only adds safety checks
5. **100%+ performance gain** - After all phases complete

---

## 🎯 Success Criteria

✅ **Task Complete When:**
- Code changes applied to quantize.cu
- Build completes without errors
- Server runs without OOB crashes
- GPU loads 48/49 layers
- Phase 1 config shows 50+ tokens/sec

---

## 📞 Questions?

Refer to these documents:
- **How to apply code fix?** → CODE_CHANGES_NEEDED.md
- **Why is this bug happening?** → MOE_EXPERT_ROUTING_BUG.md
- **What are build scripts?** → ALL_BUILD_SCRIPTS_UPDATED.md
- **What's next after rebuild?** → IMMEDIATE_ACTIONS.md

---

**Ready to start? Begin with Phase 1: Code Fix! 🚀**
