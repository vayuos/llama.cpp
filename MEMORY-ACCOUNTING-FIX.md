# Memory Accounting Corruption Fix

## Problem

Memory breakdown reporting shows corrupted "unaccounted" value:

```
llama_memory_breakdown_print:
  - CUDA0 (RTX 4060 Ti) | 16196 = 16062 + (7052 = 6824 + 228 + 0) + 17592186037496
                                                                        ^^^^^^^^^^^^^^
                                                                        16 exabytes!
                                                                        (Impossible)
```

Expected: `unaccounted ≈ 0-100 MiB`
Actual: `unaccounted ≈ 16 exabytes` (2^44 MiB)

## Root Cause

**File**: `src/llama-context.cpp` line 4539
**Bug**: Unsigned integer underflow

```cpp
const size_t unaccounted = total - self - free;
```

When `self + free > total` (measurement timing issue), the subtraction underflows:
- `total - self - free` becomes negative
- Unsigned arithmetic wraps around
- Result: Huge positive number (2^64 - small_number)

## Technical Details

### Memory Equation

```
total = free + self + unaccounted

Where:
  total      = Total GPU memory
  free       = Available GPU memory (from cudaMemGetInfo)
  self       = model + context + compute (allocated by llama)
  unaccounted = residual (OS overhead, other apps, etc.)
```

### Why Underflow Happens

1. **Timing Issue**: `cudaMemGetInfo()` called at different times
   - Before: free = 8GB, total = 16GB
   - Allocate: 8GB
   - After: free = 0GB
   - Calculation: 16 - 8 - 0 = 8 ✓ OK
   - But if kernel freed memory in between: 16 - 8 - 0.5 = 7.5 ✓ Still OK

2. **Measurement Drift**: Small deviations accumulate
   - If somehow `self + free` exceeds `total` by even 1 byte
   - With `size_t` (unsigned): `total - (self + free)` wraps
   - Result: `2^64 - 1 = 18446744073709551615` bytes ≈ 16 exabytes

### Why This Matters

1. **Debug Information Unreliable**
   - Cannot trust memory planner output
   - Performance tuning decisions based on corrupt data
   - Diagnostics show false information

2. **Potential Issues**
   - If real underflow (not timing), indicates double-counting
   - Some memory allocated but not tracked
   - GPU overcommit possible

3. **Silent Failure**
   - Doesn't crash (just prints bad number)
   - Appears to work normally
   - Easy to miss during development

## Solution

### Code Fix

Replace line 4539 in `src/llama-context.cpp`:

**Before** (Buggy):
```cpp
const size_t unaccounted = total - self - free;
```

**After** (Fixed):
```cpp
// Prevent unsigned integer underflow
int64_t unaccounted_signed = static_cast<int64_t>(total) - static_cast<int64_t>(self) - static_cast<int64_t>(free);
const size_t unaccounted = std::max(0LL, unaccounted_signed) / sizeof(size_t) * sizeof(size_t);
// Or more simply:
const size_t unaccounted = (total >= self + free) ? (total - self - free) : 0;
```

### Recommended Fix (Most Robust)

```cpp
// Safe memory accounting avoiding underflow
size_t accounted = self + free;
const size_t unaccounted = (accounted <= total) ? (total - accounted) : 0;
```

### Alternative: Debug-Only Fix

If this is debug-only output, validate before printing:

```cpp
// Validate that accounting makes sense
const size_t accounted = self + free;
if (accounted > total) {
    LLAMA_LOG_WARN("Memory accounting error: allocated (%zu) + free (%zu) > total (%zu)\n",
                   self, free, total);
    // Print with clamped values
    const size_t unaccounted = 0;  // Mark as unknown
} else {
    const size_t unaccounted = total - accounted;
}
```

## Where to Apply Fix

**File**: `src/llama-context.cpp`
**Function**: `llama_memory_breakdown_print()`
**Line**: 4539

```cpp
// Line 4535-4544
size_t free, total;
ggml_backend_dev_memory(dev, &free, &total);

const size_t self        = mb.model + mb.context + mb.compute;
// const size_t unaccounted = total - self - free;  // ← BUGGY
const size_t unaccounted = (total >= self + free) ? (total - self - free) : 0;  // ← FIXED
```

## Verification After Fix

After applying fix, the output should show:

```
llama_memory_breakdown_print:
  - CUDA0 (RTX 4060 Ti) | 16196 = 16062 + (7052 = 6824 + 228 + 0) + 82
                                                                        ^^
                                                                        Reasonable!
```

Expected:
- `unaccounted`: 0-500 MiB (OS overhead, driver memory, etc.)
- NOT: 16+ exabytes

## Impact

### Severity: Medium
- ✗ Corrupts debug output
- ✓ Doesn't crash runtime
- ✗ Affects performance tuning decisions
- ✓ Allocation logic still works correctly

### Scope
- Memory breakdown printing only
- Actual GPU memory management unaffected
- But diagnostics unreliable for optimization

## Testing

After fix, run and check:

```bash
# Build with fix
cmake --build . -j$(nproc)

# Run and capture output
./llama-server -m model.gguf > server.log 2>&1

# Verify memory accounting
grep "unaccounted" server.log
# Should show: reasonable MiB values (< 1000)
# NOT: values > 1000000 MiB
```

## Related Issues

- Issue #1-5: GPU performance issues
- Issue #6: Memory accounting corruption (this)

All issues should be fixed for proper GPU-exclusive decode operation.

## Summary

**Bug**: Unsigned integer underflow in memory accounting
**Location**: `src/llama-context.cpp:4539`
**Fix**: Add bounds check before subtraction
**Time to Fix**: < 5 minutes
**Testing**: Verify output shows reasonable unaccounted MiB values
