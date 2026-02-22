#!/bin/bash
set -euo pipefail

# ============================================================
# build_variants_mmq_moe.sh
#
# GPU-MAXIMIZED BUILD SCRIPT
# Variant: build_cuda_mmq_moe
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
# - RTX 4060 Ti (Ada Lovelace, sm_89, cc=8.9)
#
# CHANGES vs v1:
# - CMAKE_CUDA_FLAGS: --use_fast_math (GPU kernels get fast reciprocal/rsqrt/trig)
# - GGML_SCHED_MAX_COPIES=1 (single GPU, single sequence — no pipeline copies needed)
# - GGML_CUDA_NO_VMM=OFF (explicit; VMM available on Ada, needed for memory reuse)
# - GGML_CPU_REPACK=ON (explicit; Q4_0 → Q4_X_X for CPU ops)
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_cuda_mmq_moe"

echo "==================================================="
echo "GPU-maximized decode build (MMQ / MoE)"
echo "Source : ${ROOT_DIR}"
echo "Build  : ${BUILD_DIR}"
echo "Target : RTX 4060 Ti (sm_89)"
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
COMMON_CXX_FLAGS="-O3 -ffast-math -funroll-loops -march=native -DNDEBUG"

# ------------------------------------------------------------
# CUDA compiler flags (applies to GPU kernel compilation)
# --use_fast_math: enables fast reciprocal, rsqrt, exp, sin/cos
#                  on Ada hardware — directly speeds up MMQ+attn kernels
# -O3:            max PTXAS optimisation level
# ------------------------------------------------------------
CUDA_FLAGS="--use_fast_math -O3"

# ------------------------------------------------------------
# CMake configuration
# ------------------------------------------------------------
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -DCMAKE_CUDA_FLAGS="${CUDA_FLAGS}" \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
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
    \
    -DGGML_SCHED_MAX_COPIES=1 \
    \
    -DGGML_CPU_REPACK=ON \
    -DGGML_BLAS=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_CCACHE=OFF \
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

echo "[OK] MMQ + FA + CUDA Graphs + fast-math kernels + single-copy sched verified"
echo "---------------------------------------------------"
echo "FINAL build_cuda_mmq_moe completed successfully"
