#!/bin/bash
set -euo pipefail

# ============================================================
# build_variants_nvd_mmq_moe.sh
#
# GPU-MAXIMIZED BUILD SCRIPT (NVIDIA CUDA)
# Variant: build_nvd_mmq_moe
#
# PURPOSE:
# - MMQ fused kernels for ALL quantized decode matmul
# - Flash Attention (all quants) for attention ops
# - CUDA graphs for reduced kernel launch overhead
# - GPU-exclusive decode path per systemchanges.md
#
# TARGET:
# - Quantized models (Q4–Q8, K-variants, IQ)
# - MoE models
# - NVIDIA GeForce RTX 4060 Ti (Ada Lovelace, sm_89)
#
# CHANGES vs HIP:
# - CMAKE_CUDA_ARCHITECTURES: 89 (Native Ada support)
# - GGML_CUDA=ON (Enable NVIDIA backend)
# - GGML_SCHED_MAX_COPIES=1 (single GPU, single sequence)
# - GGML_CUDA_GRAPHS=ON (Enable CUDA Graphs)
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_mmq_moe"

echo "==================================================="
echo "GPU-maximized decode build (MMQ / MoE - NVIDIA CUDA)"
echo "Source : ${ROOT_DIR}"
echo "Build  : ${BUILD_DIR}"
echo "Target : NVIDIA GeForce RTX 4060 Ti (sm_89)"
echo "==================================================="

# ------------------------------------------------------------
# Hard clean — stale cache can shadow newly set flags
# ------------------------------------------------------------
if [ -d "${BUILD_DIR}" ]; then
    echo "[INFO] Removing existing build directory"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# ------------------------------------------------------------
# Host C++ compiler flags (applies to CPU-side code)
# -O3 -ffast-math: maximise float throughput on CPU side
# -march=native:  use all available CPU SIMD
# -DNDEBUG:       eliminate assertions and logging overhead
# ------------------------------------------------------------
COMMON_CXX_FLAGS="-O3 -ffast-math -fno-finite-math-only -funroll-loops -march=native -DNDEBUG"

# ------------------------------------------------------------
# CMake configuration
# ------------------------------------------------------------
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
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

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------
cmake --build . --config Release -j "$(nproc)"

# ------------------------------------------------------------
# Post-build invariant checks
# ------------------------------------------------------------
CACHE_FILE="CMakeCache.txt"

echo "---------------------------------------------------"
echo "[VERIFY] Enforcing backend invariants via CMakeCache.txt"

# MMQ must be forced
if ! grep -q "GGML_CUDA_FORCE_MMQ:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: MMQ force flag missing in ${CACHE_FILE}"
    exit 1
fi

# cuBLAS must NOT be forced
if grep -q "GGML_CUDA_FORCE_CUBLAS:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: cuBLAS force flag detected in ${CACHE_FILE}"
    exit 1
fi

# Flash Attention must be enabled
if ! grep -q "GGML_CUDA_FA:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: Flash Attention flag missing in ${CACHE_FILE}"
    exit 1
fi

# CUDA graphs must be enabled
if grep -q "GGML_CUDA_GRAPHS:BOOL=OFF" "${CACHE_FILE}"; then
    echo "FATAL: CUDA graphs are disabled — required for max GPU throughput"
    exit 1
fi

# Scheduler copies must be 1 (single GPU single sequence)
if ! grep -q "GGML_SCHED_MAX_COPIES:STRING=1" "${CACHE_FILE}"; then
    echo "FATAL: GGML_SCHED_MAX_COPIES not set to 1 in ${CACHE_FILE}"
    exit 1
fi

echo "[OK] MMQ + FA + CUDA Graphs + fast-math kernels verified"
echo "---------------------------------------------------"
echo "FINAL build_nvd_mmq_moe completed successfully"
echo ""
echo "==================================================="
echo "MAXIMUM VERBOSITY RUNTIME CONFIGURATION"
echo "==================================================="
echo ""
echo "To run with MAXIMUM VERBOSE OUTPUT during model inference (NVIDIA):"
echo ""
echo "1. STANDARD VERBOSE RUN:"
echo "   export GGML_CUDA_GRAPHS=0"
echo "   export LLAMA_LOG_LEVEL=DEBUG"
echo "   export GGML_LOG_LEVEL=DEBUG"
echo "   ./bin/llama-server -m /path/to/model.gguf --verbose"
echo ""
echo "2. WITH KERNEL TIMING & TRACING:"
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo "   ./bin/llama-server -m /path/to/model.gguf"
echo ""
echo "BUILD COMPLETE"
