# Code Changes Required - Implementation Guide

**Status:** 🔴 MoE bug fix REQUIRED before rebuild
**Priority:** CRITICAL
**Estimated Time:** 30 minutes
**Risk Level:** LOW

---

## 🔴 CRITICAL: MoE Expert Routing Bug Fix

### File to Modify
```
ggml/src/ggml-cuda/quantize.cu
```

### Functions Requiring Fix
1. `quantize_mmq_q8_1` (line 238)
2. `quantize_mmq_q8_1_rms` (line 336)
3. `quantize_mmq_mxfp4` (needs verification)

---

## 📝 Change #1: quantize_mmq_q8_1 Function

**Location:** `ggml/src/ggml-cuda/quantize.cu`, lines 238-334

### Before (BROKEN - Crashes):
```cpp
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2) {

    const int64_t i1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i1 >= ne1) return;

    const int64_t i01 = ids ? ids[i1] : i1;
    // ❌ BUG: ids[i1] could be INT_MAX (padding), causes OOB crash

    // ... rest of function uses i01 as array index ...
```

### After (FIXED - Handles Padding):
```cpp
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2) {

    const int64_t i1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i1 >= ne1) return;

    const int64_t i01 = ids ? ids[i1] : i1;

    // ✅ FIX: Skip padding positions (INT_MAX indicates padding in MoE routing)
    if (ids != nullptr && i01 == INT_MAX) {
        return;
    }

    // ... rest of function uses i01 as array index ...
```

**Change Summary:**
- Add 3 lines after line 259
- Check if `ids` is not null AND `i01 == INT_MAX`
- Return early to skip processing

---

## 📝 Change #2: quantize_mmq_q8_1_rms Function

**Location:** `ggml/src/ggml-cuda/quantize.cu`, lines 336-435

### Before:
```cpp
static __global__ void quantize_mmq_q8_1_rms(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const float eps, const float * rms_w) {

    const int64_t i1 = threadIdx.x + blockIdx.x * blockDim.x;

    if (i1 >= ne1) return;

    const int64_t i01 = ids ? ids[i1] : i1;
    // ❌ BUG: Same INT_MAX issue as above
```

### After:
```cpp
static __global__ void quantize_mmq_q8_1_rms(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const float eps, const float * rms_w) {

    const int64_t i1 = threadIdx.x + blockIdx.x * blockDim.x;

    if (i1 >= ne1) return;

    const int64_t i01 = ids ? ids[i1] : i1;

    // ✅ FIX: Skip padding positions (INT_MAX indicates padding in MoE routing)
    if (ids != nullptr && i01 == INT_MAX) {
        return;
    }
```

**Change Summary:**
- Same fix as above
- Add 3 lines after the `const int64_t i01 = ...` line
- Location: Around line 352 (after `const int64_t i01 = ids ? ids[i1] : i1;`)

---

## 📝 Change #3: Check quantize_mmq_mxfp4

**Location:** `ggml/src/ggml-cuda/quantize.cu` (search for this function)

**Action:** Search for similar pattern:
```cpp
const int64_t i01 = ids ? ids[i1] : i1;
```

If found, apply the same fix:
```cpp
// Skip padding positions
if (ids != nullptr && i01 == INT_MAX) {
    return;
}
```

**How to find:**
```bash
grep -n "quantize_mmq_mxfp4" ggml/src/ggml-cuda/quantize.cu
```

---

## 🔧 Implementation Steps

### Step 1: Edit quantize.cu

```bash
# Open the file in your editor
nano ggml/src/ggml-cuda/quantize.cu
# or use your preferred editor (vim, code, etc.)
```

### Step 2: Find Function 1 - quantize_mmq_q8_1

Search for: `static __global__ void quantize_mmq_q8_1(`

Around line 238, find:
```cpp
const int64_t i01 = ids ? ids[i1] : i1;
```

After this line, add:
```cpp
    // Skip padding positions (INT_MAX indicates padding in MoE routing)
    if (ids != nullptr && i01 == INT_MAX) {
        return;
    }
```

### Step 3: Find Function 2 - quantize_mmq_q8_1_rms

Search for: `static __global__ void quantize_mmq_q8_1_rms(`

Around line 336, find:
```cpp
const int64_t i01 = ids ? ids[i1] : i1;
```

After this line, add:
```cpp
    // Skip padding positions (INT_MAX indicates padding in MoE routing)
    if (ids != nullptr && i01 == INT_MAX) {
        return;
    }
```

### Step 4: Check quantize_mmq_mxfp4

Search for: `quantize_mmq_mxfp4` in the file

Look for the pattern:
```cpp
const int64_t i01 = ids ? ids[i1] : i1;
```

If found, apply the same fix

### Step 5: Save and Verify

```bash
# Save the file
# Verify changes look correct
grep -n "INT_MAX" ggml/src/ggml-cuda/quantize.cu
# Should show your new INT_MAX checks
```

---

## ✅ Verification Before Rebuild

### Check the changes are in place:
```bash
grep -A2 "const int64_t i01 = ids" ggml/src/ggml-cuda/quantize.cu
# Should show INT_MAX check right after
```

### Expected output:
```
const int64_t i01 = ids ? ids[i1] : i1;

// Skip padding positions (INT_MAX indicates padding in MoE routing)
if (ids != nullptr && i01 == INT_MAX) {
    return;
}
```

---

## 🔨 Rebuild After Changes

```bash
# Use updated build script with MoE fix now in place
./scripts/build_variants_mmq_moe.sh --clean -j$(nproc)
```

---

## ✨ Summary of Changes

| Function | Location | Change | Lines |
|----------|----------|--------|-------|
| quantize_mmq_q8_1 | Line 238 | Add INT_MAX check | 3 |
| quantize_mmq_q8_1_rms | Line 336 | Add INT_MAX check | 3 |
| quantize_mmq_mxfp4 | TBD | Check & fix if needed | 3 |

**Total changes:** ~9 lines across 2-3 functions

---

## 🎯 Why These Changes

### The Bug
```
ids[36] = INT_MAX (2147483647)  ← Padding indicator
↓
Used as array index
↓
Calculates: buffer[2147483647 * stride]  ← MASSIVE address
↓
Out of bounds → CRASH
```

### The Fix
```
if (ids && i01 == INT_MAX) {
    return;  ← Skip this thread entirely
}
```

---

## ⚠️ Important Notes

1. **This is the ONLY code change needed**
   - Build scripts are already updated
   - No other source files need modification

2. **Must be applied BEFORE rebuild**
   - Without this fix, rebuild will still crash

3. **Low risk**
   - Simple check
   - No complex logic
   - Only skips invalid indices

4. **Required for MoE models**
   - Non-MoE models may work without this
   - But MoE will definitely crash without it

---

## 🚀 After This Fix

1. **Run rebuild:**
   ```bash
   ./scripts/build_variants_mmq_moe.sh --clean -j$(nproc)
   ```

2. **Apply Phase 1 config:**
   ```bash
   ./llama-server -m model.gguf -ngl 999 --no-mmap -c 16384 -t 8
   ```

3. **Expected result:**
   - No more OOB crashes
   - ~50-65 tokens/sec (+67% improvement)

---

**Status:** Ready to implement
**Complexity:** LOW (copy-paste 3-line check twice)
**Impact:** CRITICAL (prevents MoE crashes)
**Time:** 10 minutes to apply, 1-2 hours to rebuild
