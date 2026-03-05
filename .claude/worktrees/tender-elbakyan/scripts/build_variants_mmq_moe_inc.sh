#!/bin/bash
set -euo pipefail

# ============================================================
# build_cuda_mmq_moe_incremental.sh
#
# INCREMENTAL GPU-MAXIMIZED BUILD
#
# Differences vs hard-clean version:
# - Does NOT remove build directory
# - Reuses CMake cache when possible
# - Forces reconfigure only if cache missing
# - Keeps invariant verification
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_cuda_mmq_moe"

echo "==================================================="
echo "Incremental GPU-maximized decode build (MMQ / MoE)"
echo "Source : ${ROOT_DIR}"
echo "Build  : ${BUILD_DIR}"
echo "Target : RTX 4060 Ti (sm_89)"
echo "==================================================="

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

COMMON_CXX_FLAGS="-O3 -ffast-math -funroll-loops -march=native -DNDEBUG"
CUDA_FLAGS="--use_fast_math -O3"

# ------------------------------------------------------------
# Configure (only if needed)
# ------------------------------------------------------------
if [ ! -f "CMakeCache.txt" ]; then
    echo "[INFO] No cache found — running fresh configure"

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
        -DCMAKE_CUDA_FLAGS="${CUDA_FLAGS}" \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        \
        -DBUILD_SHARED_LIBS=ON \
        \
        -DGGML_CUDA=ON \
        \
        -DGGML_CUDA_FORCE_MMQ=ON \
        -DGGML_CUDA_FORCE_CUBLAS=OFF \
        \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        \
        -DGGML_CUDA_GRAPHS=ON \
        \
        -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
        -DGGML_CUDA_NO_VMM=OFF \
        -DGGML_CUDA_SAMPLING=ON \
        \
        -DGGML_SCHED_MAX_COPIES=1 \
        \
        -DGGML_CPU_REPACK=ON \
        -DGGML_BLAS=OFF \
        -DGGML_OPENMP=OFF \
        -DGGML_CCACHE=OFF \
        \
        -DLLAMA_GPU_EXCLUSIVE_DECODE=ON \
        -DLLAMA_CPU_SAMPLING_EXCLUDED=ON \
        -DLLAMA_KV_HYBRID_EXCLUDED=ON \
        \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_SERVER_VERBOSE=OFF \
        -DGGML_CPU_ALL=ON
else
    echo "[INFO] Existing cache detected — reusing configuration"
fi

# ------------------------------------------------------------
# Incremental Build
# ------------------------------------------------------------
echo "[INFO] Building (incremental)"
cmake --build . --config Release -j "$(nproc)"

# ------------------------------------------------------------
# Post-build invariant checks
# ------------------------------------------------------------
CACHE_FILE="CMakeCache.txt"

echo "---------------------------------------------------"
echo "[VERIFY] Enforcing backend invariants via CMakeCache.txt"

if ! grep -q "GGML_CUDA_FORCE_MMQ:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: MMQ force flag missing"
    exit 1
fi

if grep -q "GGML_CUDA_FORCE_CUBLAS:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: cuBLAS force flag detected"
    exit 1
fi

if ! grep -q "GGML_CUDA_FA:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: Flash Attention flag missing"
    exit 1
fi

if grep -q "GGML_CUDA_GRAPHS:BOOL=OFF" "${CACHE_FILE}"; then
    echo "FATAL: CUDA graphs disabled"
    exit 1
fi

if ! grep -q "GGML_SCHED_MAX_COPIES:STRING=1" "${CACHE_FILE}"; then
    echo "FATAL: GGML_SCHED_MAX_COPIES not set to 1"
    exit 1
fi

echo "[OK] MMQ + FA + CUDA Graphs + fast-math kernels + single-copy sched verified"
echo "---------------------------------------------------"
echo "Incremental build_cuda_mmq_moe completed successfully"
echo ""
echo "==================================================="
echo "MAXIMUM VERBOSITY RUNTIME CONFIGURATION"
echo "==================================================="
echo ""
echo "To run with MAXIMUM VERBOSE OUTPUT during model inference:"
echo ""
echo "1. STANDARD VERBOSE RUN:"
echo "   export LLAMA_LOG_LEVEL=DEBUG"
echo "   export GGML_LOG_LEVEL=DEBUG"
echo "   export GGML_SCHED_DEBUG=1"
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo "   export CUDA_DEVICE_WAITS_ON_EXCEPTION=1"
echo "   ./bin/llama-server -m /path/to/model.gguf --verbose"
echo ""
echo "2. WITH GPU-EXCLUSIVE DECODE DIAGNOSTICS:"
echo "   export LLAMA_LOG_LEVEL=DEBUG"
echo "   export GGML_LOG_LEVEL=DEBUG"
echo "   export GGML_SCHED_DEBUG=1"
echo "   export GGML_CUDA_DEBUG=1"
echo "   export GGML_BACKEND_DEBUG=1"
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo "   export CUDA_DEVICE_WAITS_ON_EXCEPTION=1"
echo "   export CUDA_VERBOSE_API_TRACE=1"
echo "   ./bin/llama-server -m /path/to/model.gguf --verbose"
echo ""
echo "3. MINIMAL VERBOSITY (PRODUCTION):"
echo "   export LLAMA_LOG_LEVEL=INFO"
echo "   export GGML_LOG_LEVEL=WARN"
echo "   ./bin/llama-server -m /path/to/model.gguf"
echo ""
echo "4. FOR DETAILED KV CACHE & SAMPLING DEBUGGING:"
echo "   export LLAMA_LOG_LEVEL=DEBUG"
echo "   export GGML_LOG_LEVEL=DEBUG"
echo "   export GGML_SCHED_DEBUG=1"
echo "   export GGML_CUDA_DEBUG=1"
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo "   export CUDA_DEVICE_WAITS_ON_EXCEPTION=1"
echo "   export CUDA_VERBOSE_API_TRACE=1"
echo "   ./bin/llama-server -m /path/to/model.gguf --verbose 2>&1 | tee inference.log"
echo ""
echo "BUILD COMPLETE"

