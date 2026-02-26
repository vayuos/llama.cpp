# MoE Expert Routing Bug - Root Cause & Fix

**Status:** 🔴 CRITICAL - Causes Application Crash
**Location:** `ggml/src/ggml-cuda/quantize.cu` (lines 238-257)
**Issue:** INT_MAX padding values not checked before use as array indices

---

## 🔍 Root Cause Analysis

### The Problem Code

**File:** `ggml/src/ggml-cuda/quantize.cu`
**Function:** `quantize_mmq_q8_1` (kernel)
**Line 257:**

```cpp
const int64_t i01 = ids ? ids[i1] : i1;
```

### What's Wrong

1. **Retrieves expert index** from `ids[i1]`
2. **No validation** of the value
3. **When `ids[i1] == INT_MAX`** (padding indicator):
   - INT_MAX = 2147483647
   - Used directly as array index
   - Calculates `buffer[INT_MAX * stride]`
   - Results in massive out-of-bounds address
   - Crashes with OOB error

### The Bug Flow

```
1. Load ids[i1] from array
   └─ Value could be INT_MAX (padding)

2. Use directly as index
   └─ i01 = 2147483647

3. Calculate offset
   └─ ptr = base + i01 * stride
   └─ = base + 2147483647 * stride (MASSIVE!)

4. Access memory
   └─ Out of bounds!
   └─ CUDA crash!
```

---

## 📋 Complete Function Analysis

**File:** `ggml/src/ggml-cuda/quantize.cu`
**Lines:** 238-334

```cpp
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2) {

    const int64_t i1 = blockIdx.x * blockDim.x + threadIdx.x;  // Thread index in expert dimension

    if (i1 >= ne1) return;  // Boundary check (but only ne1, not checking if id is padding!)

    // ... [quantization loops] ...

    // 🐛 BUG HERE - Gets expert index without checking for INT_MAX padding:
    const int64_t i01 = ids ? ids[i1] : i1;  // ← NO CHECK FOR INT_MAX!

    // Then later uses i01 as array index:
    // const float * src = x + i01 * s01;  // ← CRASH if i01 = INT_MAX
```

---

## ✅ The Fix

**Add INT_MAX check before using the expert index:**

### Option 1: Skip Padding Entirely (Recommended)

```cpp
// After retrieving expert index
const int64_t i01 = ids ? ids[i1] : i1;

// ADD THIS CHECK:
if (ids && i01 == INT_MAX) {
    // Skip padding positions
    return;
}

// Then process normally
// ...
```

### Option 2: Check in Loop

```cpp
for (int64_t i1 = blockIdx.x; i1 < ne1; i1 += gridDim.x) {
    const int64_t i01 = ids ? ids[i1] : i1;

    // ADD THIS CHECK:
    if (ids && i01 == INT_MAX) {
        continue;  // Skip padding
    }

    // Process this expert
    // ...
}
```

### Option 3: Check in Helper Function

```cpp
static inline bool is_padding_expert(int32_t id) {
    return id == INT_MAX;
}

// Then in kernel:
const int64_t i01 = ids ? ids[i1] : i1;
if (ids && is_padding_expert(i01)) {
    return;  // Skip
}
```

---

## 📍 Where to Apply Fix

### Primary Location (CRITICAL)

**File:** `ggml/src/ggml-cuda/quantize.cu`

**Function 1:** `quantize_mmq_q8_1` (line 238)
```cpp
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2) {

    const int64_t i1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i1 >= ne1) return;

    // ← ADD FIX HERE:
    // if (ids && ids[i1] == INT_MAX) return;

    // ... rest of function
```

**Function 2:** `quantize_mmq_q8_1_rms` (line 336)
- Same fix needed

**Function 3:** `quantize_mmq_mxfp4` (likely similar issue)
- Check for similar pattern

---

## 🔧 Implementation Steps

### Step 1: Add INT_MAX constant if not defined

```cpp
// At top of quantize.cu or common.h:
#ifndef INT_MAX
#define INT_MAX 2147483647
#endif
```

### Step 2: Add check in quantize_mmq_q8_1

**Before:** (Line 257)
```cpp
const int64_t i1 = blockIdx.x * blockDim.x + threadIdx.x;
if (i1 >= ne1) return;
const int64_t i01 = ids ? ids[i1] : i1;
```

**After:**
```cpp
const int64_t i1 = blockIdx.x * blockDim.x + threadIdx.x;
if (i1 >= ne1) return;
const int64_t i01 = ids ? ids[i1] : i1;

// Skip padding positions (INT_MAX indicates padding in MoE routing)
if (ids != nullptr && i01 == INT_MAX) {
    return;
}
```

### Step 3: Apply same fix to quantize_mmq_q8_1_rms

Find similar pattern and apply same check.

### Step 4: Check quantize_mmq_mxfp4

Search for similar id indexing and apply fix if needed.

---

## 🧪 Verification

### Before Fix
```bash
./build_cuda_mmq_moe_full_logs/bin/llama-server -m model.gguf -ngl 999 -c 16384
# Output:
# OOB: ids_src1[36]=2147483647  max_access=4398046511103 >= limit=18432
# ...device error and CUDA_DEVICE_WAITS_ON_EXCEPTION is set...
# 💥 CRASH
```

### After Fix
```bash
# Rebuild with fix
./scripts/build_variants_mmq_moe.sh --clean -j$(nproc)

./build_cuda_mmq_moe_full_logs/bin/llama-server -m model.gguf -ngl 999 -c 16384
# Should run without crash
# No OOB errors
# ✅ SUCCESS
```

---

## 📊 Impact Assessment

| Aspect | Details |
|--------|---------|
| **Severity** | 🔴 CRITICAL (causes crash) |
| **Scope** | All MoE models on CUDA |
| **Workaround** | Use CPU only (not viable) |
| **Fix Complexity** | LOW (simple check) |
| **Lines to change** | ~3-5 per function |
| **Functions affected** | 3-4 quantize kernels |

---

## 💡 Understanding INT_MAX Padding

### Why INT_MAX is used

In MoE expert routing:
```cpp
// When routing tokens to experts
for (int i = 0; i < MAX_EXPERTS_PER_TOKEN; i++) {
    if (i < active_experts) {
        expert_ids[i] = actual_expert_id[i];
    } else {
        expert_ids[i] = INT_MAX;  // Padding marker
    }
}
```

### What it means

- **INT_MAX (2147483647)** = "This slot is padding, ignore it"
- **Valid expert ID** = 0 to (num_experts - 1)
- **Processing issue:** Kernel treats INT_MAX as valid index

### The Fix Philosophy

**Process only valid expert indices, skip padding:**
```cpp
if (expert_id != INT_MAX) {
    process_expert(expert_id);
}
```

---

## ✅ Checklist

- [ ] Locate `ggml/src/ggml-cuda/quantize.cu`
- [ ] Find `quantize_mmq_q8_1` function (line 238)
- [ ] Add INT_MAX check after line 257
- [ ] Apply same fix to `quantize_mmq_q8_1_rms` (line 336)
- [ ] Check `quantize_mmq_mxfp4` for similar issue
- [ ] Rebuild: `./scripts/build_variants_mmq_moe.sh --clean -j$(nproc)`
- [ ] Test: Run server with MoE model
- [ ] Verify: No OOB errors, no crash

---

## 🎯 Summary

**The Bug:**
- Kernel uses INT_MAX padding values as array indices
- Causes massive out-of-bounds address calculation
- Results in CUDA device error and application crash

**The Fix:**
- Add simple check: `if (ids && i01 == INT_MAX) return;`
- Skip processing for padding positions
- ~3-5 lines per function

**Status:**
- ✅ Root cause identified
- ✅ Fix location identified
- ⏳ Awaiting implementation

---

**Critical:** This bug prevents all MoE model inference on CUDA
