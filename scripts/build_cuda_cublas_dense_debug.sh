#!/bin/bash
set -euo pipefail

# ============================================================
# FULL DEBUG BUILD (MMQ + MoE) WITH RUNTIME LOGGING ENABLED
#
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
echo "FULL DEBUG BUILD (MMQ + MoE + Runtime Logging)"
echo "Source : ${ROOT_DIR}"
echo "Build  : ${BUILD_DIR}"
echo "Target : RTX 4060 Ti (sm_89)"
echo "==================================================="

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# ------------------------------------------------------------
# Host Debug Flags
# ------------------------------------------------------------
COMMON_CXX_FLAGS="-O0 -g3 -fno-omit-frame-pointer -march=native"

# ------------------------------------------------------------
# CUDA Debug Flags
# ------------------------------------------------------------
# -G enables device debug
# -lineinfo kept for profiling correlation (will be ignored by -G)
# ------------------------------------------------------------
CUDA_FLAGS="-lineinfo -g -O0"

# ------------------------------------------------------------
# CMake Configure
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------
cmake --build . --config Debug -j 12

# ------------------------------------------------------------
# Verify invariants
# ------------------------------------------------------------
CACHE_FILE="CMakeCache.txt"

grep -q "GGML_CUDA_FORCE_MMQ:BOOL=ON" "${CACHE_FILE}" || { echo "FATAL: MMQ not forced"; exit 1; }
grep -q "GGML_CUDA_FORCE_CUBLAS:BOOL=OFF" "${CACHE_FILE}" || { echo "FATAL: cuBLAS incorrectly enabled"; exit 1; }
grep -q "GGML_CUDA_FA:BOOL=ON" "${CACHE_FILE}" || { echo "FATAL: Flash Attention missing"; exit 1; }
grep -q "GGML_CUDA_GRAPHS:BOOL=ON" "${CACHE_FILE}" || { echo "FATAL: CUDA Graphs disabled"; exit 1; }

echo "---------------------------------------------------"
echo "[OK] FULL DEBUG BUILD COMPLETE"
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
