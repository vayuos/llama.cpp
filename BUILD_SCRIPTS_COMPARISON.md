# Build Scripts Comparison

## Side-by-Side: Clean vs Incremental Build

### Option 1: Full Clean Build
**File**: `scripts/build_cuda_cublas_dense_debug.sh`
**Size**: 3.7K
**Use Case**: Starting fresh, CMake configuration changes, fixing build issues

```bash
./scripts/build_cuda_cublas_dense_debug.sh
```

**What it does**:
```
✓ Remove entire build directory (rm -rf)
✓ Create fresh build directory
✓ Run full CMake configuration
✓ Compile all files (full build)
✓ Verify build settings
```

**Time**: ~10-15 minutes (first build)

**Pros**:
- ✅ Clean slate - no stale build artifacts
- ✅ Guaranteed fresh configuration
- ✅ Reliable when switching branches/configurations
- ✅ Detects all missing dependencies

**Cons**:
- ❌ Slow - recompiles everything
- ❌ Long wait time
- ❌ Not ideal for iterative development

---

### Option 2: Incremental Build (NEW)
**File**: `scripts/build_cuda_cublas_dense_debug_inc.sh`
**Size**: 4.5K
**Use Case**: Development, quick rebuilds, iterative changes

```bash
./scripts/build_cuda_cublas_dense_debug_inc.sh
```

**What it does**:
```
✓ Check if build directory exists
✓ Skip rm -rf (reuse existing build)
✓ Only reconfigure if needed (CMake)
✓ Incremental build (only changed files)
✓ Verify build settings
```

**Time**: ~30 seconds to 2 minutes (subsequent builds)

**Pros**:
- ✅ Fast - only rebuilds changed files
- ✅ Perfect for iterative development
- ✅ Reuses build artifacts
- ✅ Auto-detects if reconfigure needed
- ✅ First run handles setup automatically

**Cons**:
- ❌ May fail if build is corrupted
- ❌ Requires existing build directory first time
- ❌ Stale artifacts if files change externally

---

## Decision Tree: Which to Use?

```
Are you building for the first time?
├─ YES → Use CLEAN BUILD (full)
│        ./scripts/build_cuda_cublas_dense_debug.sh
│
└─ NO → Are you in active development?
    ├─ YES (frequent rebuilds) → Use INCREMENTAL (fast)
    │                             ./scripts/build_cuda_cublas_dense_debug_inc.sh
    │
    └─ NO → Did you change CMakeLists.txt?
        ├─ YES → Use INCREMENTAL (auto-reconfigures)
        │         ./scripts/build_cuda_cublas_dense_debug_inc.sh
        │
        └─ NO → Use INCREMENTAL (fastest)
                ./scripts/build_cuda_cublas_dense_debug_inc.sh
```

---

## Recommended Workflow

### First Time
```bash
# Full clean build (takes ~15 min)
./scripts/build_cuda_cublas_dense_debug.sh
```

### Development / Iteration
```bash
# Quick incremental rebuilds (takes ~30 sec - 2 min)
./scripts/build_cuda_cublas_dense_debug_inc.sh
./scripts/build_cuda_cublas_dense_debug_inc.sh
./scripts/build_cuda_cublas_dense_debug_inc.sh
```

### After Major Changes
```bash
# Go back to clean build
./scripts/build_cuda_cublas_dense_debug.sh
```

---

## Feature Comparison Table

| Feature | Clean | Incremental |
|---------|-------|-------------|
| **Deletes build dir** | ✅ Yes | ❌ No |
| **Fresh CMake config** | ✅ Always | ⚠️ When needed |
| **Recompile everything** | ✅ Yes | ❌ No |
| **First-time setup** | ✅ Works | ✅ Auto-detects |
| **Speed** | 🐢 Slow | ⚡ Fast |
| **Development** | ❌ Not ideal | ✅ Perfect |
| **Debugging** | ✅ Safe | ⚠️ Use clean if stuck |
| **Build artifacts** | 🗑️ Discarded | ♻️ Reused |
| **Time (first)** | ~15 min | ~15 min |
| **Time (subsequent)** | ~15 min | ~30 sec - 2 min |

---

## When to Use Each

### Use CLEAN Build When:
```
- Building for the first time
- Switching Git branches
- After git pull with major changes
- CMakeLists.txt was modified
- Build is acting strange/corrupted
- Need 100% guaranteed fresh build
- Troubleshooting build issues
```

### Use INCREMENTAL Build When:
```
- Making source code changes
- Doing iterative development
- Testing small fixes
- Rebuilding after minor changes
- Want fastest possible rebuild
- Already have a working build directory
```

---

## Troubleshooting

### Incremental Build Fails
```bash
# Solution 1: Go back to clean build
./scripts/build_cuda_cublas_dense_debug.sh

# Solution 2: Remove only build artifacts, keep CMake cache
cd build_cuda_mmq_moe_full_logs
cmake --build . --target clean
cmake --build . --config Debug -j 12
```

### Incremental Says "Configuration up-to-date" But Build Fails
```bash
# Force reconfigure:
rm -f build_cuda_mmq_moe_full_logs/CMakeCache.txt
./scripts/build_cuda_cublas_dense_debug_inc.sh
```

### Need to Switch Build Options
```bash
# Use clean build for major changes
./scripts/build_cuda_cublas_dense_debug.sh
```

---

## Build Environment Variables

Both scripts support:

```bash
# Control parallel jobs (default: auto-detect)
NUM_JOBS=8 ./scripts/build_cuda_cublas_dense_debug_inc.sh

# Custom build directory
BUILD_DIR=/custom/path ./script/build_cuda_cublas_dense_debug_inc.sh
```

---

## Performance Comparison

### First Build (No Cache)
```
Clean:        15-20 min (compile everything)
Incremental:  15-20 min (compile everything, same as clean)
```

### Subsequent Build (Minimal Changes)
```
Clean:        15-20 min (recompile everything again)
Incremental:  30 sec - 2 min (only rebuild changed files)
```

### Typical 5-File Change Cycle
```
Clean:        5 × 15-20 min = 75-100 minutes ❌
Incremental:  5 × 1 min = 5 minutes ✅

Time saved: 70-95 minutes per cycle!
```

---

## Advanced Usage

### Parallel Jobs
```bash
# Auto-detect CPU count (recommended)
./scripts/build_cuda_cublas_dense_debug_inc.sh

# Override CPU count
NUM_JOBS=16 ./scripts/build_cuda_cublas_dense_debug_inc.sh

# Single-threaded (debugging)
NUM_JOBS=1 ./scripts/build_cuda_cublas_dense_debug_inc.sh
```

### Monitor Progress
```bash
# With progress bar
./scripts/build_cuda_cublas_dense_debug_inc.sh

# Watch compilation in real-time
watch -n 1 "ls -lh build_cuda_mmq_moe_full_logs/CMakeFiles/"
```

### Partial Rebuild
```bash
cd build_cuda_mmq_moe_full_logs

# Rebuild only changed CMake targets
cmake --build . --config Debug

# Rebuild specific target
cmake --build . --config Debug --target llama-server

# Rebuild with verbose output
cmake --build . --config Debug --verbose
```

---

## Summary

**Start with CLEAN**, then switch to **INCREMENTAL** for development:

```bash
# First time
./scripts/build_cuda_cublas_dense_debug.sh        # 15-20 min

# Development cycle
./scripts/build_cuda_cublas_dense_debug_inc.sh    # 30 sec - 2 min (fast!)
./scripts/build_cuda_cublas_dense_debug_inc.sh    # 30 sec - 2 min (fast!)
./scripts/build_cuda_cublas_dense_debug_inc.sh    # 30 sec - 2 min (fast!)
```

This cuts development iteration time from **45-60 minutes** to **~5 minutes** for typical 5-change cycles.
