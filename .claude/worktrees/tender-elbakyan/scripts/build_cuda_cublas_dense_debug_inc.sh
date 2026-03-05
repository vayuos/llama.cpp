#!/bin/bash
set -euo pipefail

# ============================================================
# INCREMENTAL DEBUG BUILD (MMQ + MoE) WITH RUNTIME LOGGING
#
# - Reuses existing build directory
# - Only rebuilds changed files
# - MMQ forced
# - cuBLAS disabled
# - Flash Attention enabled
# - CUDA graphs enabled
# - Full CPU + GPU debug symbols
# - Scheduler logging compiled in
# - Tests disabled (fixes linker errors)
# - No runtime wrapper
#
# Target GPU: RTX 4060 Ti (sm_89)
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_cuda_mmq_moe_full_logs"

echo "==================================================="
echo "INCREMENTAL DEBUG BUILD (MMQ + MoE + Runtime Logging)"
echo "Source : ${ROOT_DIR}"
echo "Build  : ${BUILD_DIR}"
echo "Target : RTX 4060 Ti (sm_89)"
echo "==================================================="

# ------------------------------------------------------------
# Check if build directory exists
# If not, create it (first-time setup)
# ------------------------------------------------------------
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Build directory not found. Creating..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    # First time: need to configure
    echo "Running initial CMake configuration..."
    NEEDS_CONFIG=1
else
    echo "Build directory found. Using existing configuration..."
    cd "${BUILD_DIR}"
    NEEDS_CONFIG=0
fi

# Only reconfigure if:
# 1. First time setup, OR
# 2. CMakeCache.txt doesn't exist, OR
# 3. CMakeLists.txt changed
if [ $NEEDS_CONFIG -eq 1 ] || [ ! -f "CMakeCache.txt" ] || [ "../CMakeLists.txt" -nt "CMakeCache.txt" ]; then
    echo "Reconfiguring CMake..."

    # Host Debug Flags
    COMMON_CXX_FLAGS="-O0 -g3 -fno-omit-frame-pointer -march=native"

    # CUDA Debug Flags
    CUDA_FLAGS="-lineinfo -g -O0"

    # CMake Configure
    # CRITICAL: -DBUILD_SHARED_LIBS=ON enables backend symbol export
    # This fixes: "failed to find ggml_backend_init in libggml-cuda.so"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
        -DCMAKE_CUDA_FLAGS="${CUDA_FLAGS}" \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        \
        -DBUILD_SHARED_LIBS=ON \
        \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FORCE_MMQ=ON \
        -DGGML_CUDA_FORCE_CUBLAS=OFF \
        \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_CUDA_GRAPHS=ON \
        \
        -DGGML_CUDA_NO_VMM=OFF \
        -DGGML_SCHED_MAX_COPIES=1 \
        \
        -DGGML_CPU_REPACK=ON \
        -DGGML_BLAS=OFF \
        -DGGML_OPENMP=ON \
        -DGGML_CCACHE=OFF \
        \
        -DGGML_DEBUG=ON \
        -DGGML_DEBUG_CUDA=ON \
        \
        -DGGML_CUDA_SAMPLING=ON \
        \
        -DLLAMA_GPU_EXCLUSIVE_DECODE=ON \
        -DLLAMA_CPU_SAMPLING_EXCLUDED=ON \
        -DLLAMA_KV_HYBRID_EXCLUDED=ON \
        \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DGGML_CPU_ALL=ON \
        -DGGML_BACKEND_DL=OFF
else
    echo "CMake configuration up-to-date. Skipping reconfigure."
fi

# ------------------------------------------------------------
# Incremental Build (only rebuild changed files)
# Parallel jobs: detect CPU count
# ------------------------------------------------------------
NUM_JOBS=$(nproc 2>/dev/null || echo 4)
echo "Building with ${NUM_JOBS} parallel jobs..."

cmake --build . --config Debug -j "${NUM_JOBS}"

# Capture build exit code
BUILD_EXIT=$?

if [ $BUILD_EXIT -ne 0 ]; then
    echo "---------------------------------------------------"
    echo "[ERROR] BUILD FAILED"
    echo "---------------------------------------------------"
    exit $BUILD_EXIT
fi

# ------------------------------------------------------------
# Verify invariants (only if build succeeded)
# ------------------------------------------------------------
CACHE_FILE="CMakeCache.txt"

echo "Verifying build configuration..."
grep -q "GGML_CUDA_FORCE_MMQ:BOOL=ON" "${CACHE_FILE}" || { echo "FATAL: MMQ not forced"; exit 1; }
grep -q "GGML_CUDA_FORCE_CUBLAS:BOOL=OFF" "${CACHE_FILE}" || { echo "FATAL: cuBLAS incorrectly enabled"; exit 1; }
grep -q "GGML_CUDA_FA:BOOL=ON" "${CACHE_FILE}" || { echo "FATAL: Flash Attention missing"; exit 1; }
grep -q "GGML_CUDA_GRAPHS:BOOL=ON" "${CACHE_FILE}" || { echo "FATAL: CUDA Graphs disabled"; exit 1; }

echo "---------------------------------------------------"
echo "[OK] INCREMENTAL DEBUG BUILD COMPLETE"
echo ""
echo "Run manually with full runtime logs using:"
echo ""
echo "export LLAMA_LOG_LEVEL=DEBUG"
echo "export GGML_LOG_LEVEL=DEBUG"
echo "export GGML_SCHED_DEBUG=1"
echo "export CUDA_LAUNCH_BLOCKING=1"
echo "export CUDA_DEVICE_WAITS_ON_EXCEPTION=1"
echo ""
echo "./bin/llama-server -m /path/to/model.gguf --verbose"
echo "---------------------------------------------------"
