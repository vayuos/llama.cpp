# Backend Symbol Export Fix (CUDA & CPU)

## Problem

Runtime error when loading backend:
```
load_backend: failed to find ggml_backend_init in libggml-cuda.so
load_backend: failed to find ggml_backend_init in libggml-cpu.so
```

This occurs because the `ggml_backend_init` symbol is not exported from the shared library.
Both CUDA and CPU backends use the same mechanism and require the same fix.

## Root Cause Analysis

### 1. Symbol Export Mechanism

In `ggml/include/ggml-backend.h` (lines 6-17):
```c
#ifdef GGML_BACKEND_SHARED
    #if defined(_WIN32)
    #    ifdef GGML_BACKEND_BUILD
    #        define GGML_BACKEND_API __declspec(dllexport) extern
    #    else
    #        define GGML_BACKEND_API __declspec(dllimport) extern
    #    endif
    #else
    #        define GGML_BACKEND_API __attribute__ ((visibility ("default"))) extern
    #    endif
#else
    #    define GGML_BACKEND_API extern
#endif
```

For symbols to be exported:
- `GGML_BACKEND_SHARED` must be defined
- `GGML_BACKEND_BUILD` must be defined during backend compilation
- On Linux: uses `__attribute__ ((visibility ("default")))`
- On Windows: uses `__declspec(dllexport)`

### 2. CMake Configuration

In `ggml/src/CMakeLists.txt` (lines 268-271), the backend library is configured as:
```cmake
if (${BUILD_SHARED_LIBS})
    target_compile_definitions(${backend} PRIVATE GGML_BACKEND_BUILD)
    target_compile_definitions(${backend} PUBLIC  GGML_BACKEND_SHARED)
endif()
```

This is **correct**, but **only works if `BUILD_SHARED_LIBS` is explicitly set to ON**.

### 3. The Bug

The `build_cuda_mmq_moe_full_logs` directory is empty, indicating no successful build occurred. Previous builds likely used:
- `-DGGML_STATIC=ON` (static build)
- `-DBUILD_SHARED_LIBS=OFF` (default on Windows/MINGW)
- Wrong CMake configuration flags

Result: The backend library is built as **static** or **not properly configured for shared export**, causing symbol loss.

## Solution

### Build With Correct Flags

```bash
# Clean previous build
rm -rf build_cuda_mmq_moe_full_logs

# Configure with SHARED library mode enabled
cmake -S . -B build_cuda_mmq_moe_full_logs \
  -DGGML_CUDA=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native

# Build
cmake --build build_cuda_mmq_moe_full_logs -j$(nproc) --config Release
```

### Key Flags Explained

| Flag | Purpose | Why Important |
|------|---------|---------------|
| `-DGGML_CUDA=ON` | Enable CUDA backend | Required to build libggml-cuda.so |
| `-DBUILD_SHARED_LIBS=ON` | Build as shared libraries | **CRITICAL** - enables symbol export via GGML_BACKEND_SHARED |
| `-DCMAKE_BUILD_TYPE=Release` | Release build | Performance optimization |
| `-DCMAKE_CUDA_ARCHITECTURES=native` | Auto-detect GPU | Works with connected GPUs |

### DO NOT USE

❌ `-DGGML_STATIC=ON` - Builds static library, no symbol export
❌ `-DGGML_BACKEND_DL=ON` without `BUILD_SHARED_LIBS=ON` - Will fail
❌ Default MINGW settings on Windows - Defaults to `BUILD_SHARED_LIBS=OFF`

## Verification

After build, verify symbol export:

```bash
# On Linux/WSL:
nm -D build_cuda_mmq_moe_full_logs/bin/libggml-cuda.so | grep ggml_backend_init
# Output should show: ........... T ggml_backend_init

# On Windows (MinGW):
nm build_cuda_mmq_moe_full_logs/bin/libggml-cuda.dll | grep ggml_backend_init
# Output should show symbol table entry
```

### Expected Output
```
000000000003e4d0 T ggml_backend_init
```

The `T` indicates the symbol is in the text segment and properly exported.

## Configuration Details

### Default Behavior by Platform

| Platform | Default BUILD_SHARED_LIBS | Issue |
|----------|---------------------------|-------|
| Linux | ON | ✅ Works correctly |
| macOS | ON | ✅ Works correctly |
| Windows (MSVC) | ON | ✅ Works correctly |
| Windows (MinGW) | OFF | ❌ Static only |
| EMSCRIPTEN | OFF | ✅ Not applicable |

## Post-Build Checklist

After successful build:

- [ ] Verify `build_cuda_mmq_moe_full_logs/bin/libggml-cuda.so` (or `.dll` on Windows) exists
- [ ] Check size is > 50MB (indicates full library)
- [ ] Verify symbol with `nm -D libggml-cuda.so | grep ggml_backend_init`
- [ ] Test with llama-server:
  ```bash
  ./build_cuda_mmq_moe_full_logs/bin/llama-server -m model.gguf
  ```
- [ ] Verify logs show "init cuda backend" without symbol errors

## Advanced: Manual Symbol Check

If you need to inspect the build in detail:

```bash
# List all exported symbols
nm -D build_cuda_mmq_moe_full_logs/bin/libggml-cuda.so | head -20

# Check library type
file build_cuda_mmq_moe_full_logs/bin/libggml-cuda.so

# Check compiler flags used
strings build_cuda_mmq_moe_full_logs/bin/libggml-cuda.so | grep gcc
```

## Related Files

- `ggml/include/ggml-backend.h` - Symbol export definitions
- `ggml/src/CMakeLists.txt` - Backend library configuration
- `ggml/src/ggml-cuda/CMakeLists.txt` - CUDA backend build
- `CMakeLists.txt` - Main project configuration
- `ggml/CMakeLists.txt` - GGML library configuration

## Common Issues

### "Symbol not found after fix"
- Verify `BUILD_SHARED_LIBS=ON` in CMake output
- Check no `-DGGML_STATIC=ON` flag was used
- Rebuild: `rm -rf build_cuda* && cmake ...`

### "Build fails with 'undefined reference'"
- Ensure CUDA Toolkit is installed
- Check `which nvcc` returns valid path
- Try `-DCMAKE_CUDA_ARCHITECTURES=70` (for older CUDA)

### "Cannot find libgpuruntime"
- CUDA Toolkit not properly installed
- Set `CUDA_PATH` environment variable
- Update CUDA Toolkit installation

## Summary

**The fix requires:**
1. Clean build directory
2. Add `-DBUILD_SHARED_LIBS=ON` flag
3. Rebuild project
4. Verify symbol export

This ensures the backend library is built with proper symbol visibility and can be loaded at runtime.
