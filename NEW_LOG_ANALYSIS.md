# New Server Debug Log Analysis

**Log File:** server_debug.log
**Lines:** 4,157
**Date:** Latest run
**Status:** ⚠️ Critical issues found

---

## 📊 Summary

New analysis reveals **4 critical issues**, including a new MoE expert routing bug that causes application crash.

---

## 🔴 Critical Issues Found

### Issue 1: Backend Symbol Export (STILL NOT FIXED)

**Log Evidence (Lines 7-8):**
```
load_backend: failed to find ggml_backend_init in /home/viren/llama/llama.cpp/build_cuda_mmq_moe_full_logs/bin/libggml-cuda.so
load_backend: failed to find ggml_backend_init in /home/viren/llama/llama.cpp/build_cuda_mmq_moe_full_logs/bin/libggml-cpu.so
```

**Status:** ❌ **NOT FIXED YET**
**Root Cause:** Build scripts updated but not executed
**Solution:** Run one of the updated build scripts:
```bash
./scripts/build_variants_mmq_moe.sh --clean -j$(nproc)
```

---

### Issue 2: Embeddings Fallback to CPU

**Log Evidence (Lines 790, 1647):**
```
load_tensors: tensor 'token_embd.weight' (q4_K) (and 319 others) cannot be used with preferred buffer type CUDA_Host, using CPU instead
```

**Status:** ❌ **Still happening**
**Solution:** Use Phase 1 fix:
```bash
./llama-server -m model.gguf -ngl 999 --no-mmap -c 16384 -t 8
```

---

### Issue 3: Decode Admission Failures

**Log Evidence (Lines 1736-1749):**
```
FATAL: Decode admission REJECTED - DECODE_CRITICAL_OP_ON_CPU
ERROR: Cannot lock admission in state INELIGIBLE (must be ELIGIBLE)
decode: Decode admission REJECTED. GPU-exclusive execution cannot be guaranteed. Falling back to hybrid CPU/GPU execution.
```

**Status:** ⚠️ **Blocking GPU-exclusive execution**
**Root Cause:** GPU not fully configured (Issues #1, #4)
**Will be fixed by:** Phase 1 + Phase 2

---

### Issue 4: MoE Expert Routing Out-of-Bounds (NEW! - CRITICAL)

**Log Evidence (Lines 4147-4155):**
```
OOB: ids_src1[36]=2147483647  max_access=4398046511103 >= limit=18432  s11=2048
OOB: ids_src1[37]=2147483647  max_access=4398046511103 >= limit=18432  s11=2048
OOB: ids_src1[38]=2147483647  max_access=4398046511103 >= limit=18432  s11=2048
OOB: ids_src1[39]=2147483647  max_access=4398046511103 >= limit=18432  s11=2048
max_ids_src1=2147483647  max_access=(max_id*s11+ne10-1)=4398046511103  src1_total_floats=18432  OOB=YES!
viren-pc: The application encountered a device error and CUDA_DEVICE_WAITS_ON_EXCEPTION is set.
```

**Status:** 🔴 **CRITICAL - Causes Application Crash**

**What's Happening:**
1. MoE expert indices contain INT_MAX (2147483647) at positions 36-39
2. These INT_MAX values are PADDING INDICATORS (not real expert indices)
3. The kernel tries to process them as valid indices
4. Results in out-of-bounds memory access
5. Application crashes

**Root Cause:** The MMQ kernel for MoE doesn't check for padding indicators (INT_MAX)

**Solution:** The kernel needs to skip INT_MAX values in expert indices

**Code Location:** Likely in `ggml/src/ggml-cuda/mmq.cu` or related MoE kernel

**Fix Type:** Code change (kernel logic)

---

## 📋 Issues Comparison: Old vs New Log

| Issue | Old Log | New Log | Status |
|-------|---------|---------|--------|
| #1 Backend Symbols | ✅ Found | ✅ Still present | Not rebuilt yet |
| #4 Embeddings CPU | ✅ Found | ✅ Still present | Needs config change |
| Decode Admission | ✅ Found | ✅ More detailed | Awaiting fixes |
| **MoE Expert OOB** | ❓ Not visible | ✅ **CRITICAL NEW** | **Requires code fix** |

---

## 🎯 The New MoE Expert Routing Bug

### Understanding the Error

```
ids_src1[36]=2147483647    ← INT_MAX (padding)
ids_src1[37]=2147483647    ← INT_MAX (padding)
ids_src1[38]=2147483647    ← INT_MAX (padding)
ids_src1[39]=2147483647    ← INT_MAX (padding)

max_access=4398046511103 >= limit=18432  ← Out of bounds!
```

### The Problem

1. **Padding Indicators:** INT_MAX marks invalid/padding positions
2. **Kernel Behavior:** Kernel tries to use these as expert indices
3. **Memory Access:** Calculates `expert_id * stride`, gets massive address
4. **OOB Check:** Address far exceeds buffer size (18432 bytes)
5. **Crash:** CUDA device error

### The Fix Needed

The kernel must check if `ids_src1[i] == INT_MAX` before using it:

**Pseudo-code:**
```cpp
for (int i = 0; i < num_experts; i++) {
    if (ids_src1[i] == INT_MAX) {
        // Skip padding positions
        continue;
    }
    // Process valid expert
    process_expert(ids_src1[i]);
}
```

---

## 🔧 What Needs To Be Done

### Immediate (5 minutes)

**Apply Phase 1 config changes:**
```bash
./llama-server -m model.gguf -ngl 999 --no-mmap -c 16384 -t 8
```

**Expected:** Reduces some issues, but MoE crash will persist

### Short-term (1-2 hours)

**Rebuild with symbol export fix:**
```bash
./scripts/build_variants_mmq_moe.sh --clean -j$(nproc)
```

**Expected:** Fixes backend symbols, but MoE crash persists

### Code Fix Needed (TBD)

**Fix MoE expert routing kernel:**
- File: `ggml/src/ggml-cuda/mmq.cu` or similar
- Issue: Kernel doesn't skip INT_MAX padding indicators
- Change: Add check for INT_MAX values before memory access

---

## 📊 Impact Assessment

| Issue | Severity | Blockers | Fix Time |
|-------|----------|----------|----------|
| Backend Symbols | CRITICAL | Build | 1-2h |
| Embeddings CPU | HIGH | Config | 1min |
| Decode Admission | HIGH | Dependent on others | Auto |
| **MoE Expert OOB** | **CRITICAL** | **Code change** | **Unknown** |

---

## 🚨 MoE Expert Routing - Detailed Analysis

### Current Stack

```
Process ubatch
  ↓
GPU-exclusive decode
  ↓
MMQ kernel for MoE (mul_mat_q with expert routing)
  ↓
For each token, route to active experts
  ↓
ids_src1[i] contains expert index
  ↓
❌ PROBLEM: ids_src1[36..39] = INT_MAX (padding)
  ↓
Kernel tries: expert_buffer[INT_MAX * stride]
  ↓
💥 Out of bounds access → CRASH
```

### Why INT_MAX Values Exist

```cpp
// In expert routing:
for (int i = 0; i < MAX_EXPERTS; i++) {
    if (i < num_active_experts) {
        ids[i] = active_expert_id[i];
    } else {
        ids[i] = INT_MAX;  // Marks padding
    }
}
```

The padding is intentional but the kernel doesn't handle it!

---

## ✅ Verification Commands

### Check if backend symbols are fixed
```bash
nm -D build_cuda_mmq_moe_full_logs/bin/libggml-cuda.so | grep ggml_backend_init
# Should show: T ggml_backend_init (not "failed to find")
```

### Check if embeddings still on CPU
```bash
grep "cannot be used with preferred buffer type" server_debug.log
# Should NOT appear if Phase 1 applied
```

### Check for MoE expert OOB
```bash
grep "OOB:" server_debug.log
# Should NOT appear after code fix
```

---

## 📝 Next Steps

### Step 1: Verify MoE Issue Confirmed
- ✅ Confirmed: INT_MAX values at indices 36-39
- ✅ Confirmed: Out-of-bounds access
- ✅ Confirmed: Causes application crash

### Step 2: Identify Kernel Location
Search for MoE kernel handling in:
- `ggml/src/ggml-cuda/mmq.cu`
- `ggml/src/ggml-cuda/common.cuh`
- Related expert routing code

### Step 3: Implement Check
Add guard for INT_MAX before expert processing

### Step 4: Rebuild and Verify
```bash
./scripts/build_variants_mmq_moe.sh --clean -j$(nproc)
# Should build without MoE OOB crash
```

---

## 🎓 Key Findings

1. **Build Scripts NOT YET EXECUTED**
   - Symbol export fix ready but not applied
   - Build scripts updated but source not rebuilt

2. **Phase 1 Config NOT YET APPLIED**
   - `-ngl 999 --no-mmap -c 16384` not used
   - Embeddings still on CPU

3. **NEW CRITICAL BUG DISCOVERED**
   - MoE expert routing kernel crashes on padding
   - INT_MAX values not handled properly
   - Requires kernel code fix

---

## 💡 Summary

The log confirms all previous findings PLUS reveals a NEW critical bug:

**The MoE kernel doesn't skip padding indicators (INT_MAX), causing out-of-bounds access and application crash.**

**Priority Actions:**
1. ✅ Use updated build scripts (fixes symbols)
2. ✅ Apply Phase 1 config (fixes embeddings)
3. 🔧 **FIX MoE KERNEL (prevents crash)**

---

**Status:** Analysis complete, new bug identified
**Next:** Locate and fix MoE kernel INT_MAX handling
**Impact:** Without this fix, MoE models will crash
