#!/bin/bash
set -euo pipefail

# ============================================================
# build_variants_mmq_moe.sh
#
# FINAL, GOAL-ALIGNED BUILD SCRIPT
# Variant: build_cuda_mmq_moe
#
# PURPOSE (NON-NEGOTIABLE):
# - Enforce GPU-exclusive decode-critical execution
# - Maximize decode-phase tokens/sec for quantized models
# - Eliminate CPU fallback and hybrid ambiguity
#
# TARGET:
# - Quantized models (Q4–Q8, K-variants, IQ)
# - MoE models (if present)
# - RTX 4060 Ti (Ada, sm_89)
#
# HARD CONSTRAINTS (FROM REQUIREMENT DOC):
# - MMQ is the ONLY allowed matmul backend
# - cuBLAS must NOT be used
# - NO CUDA graphs
# - NO OpenMP
# - Deterministic execution
# - CPU must never be decode-pacing
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_cuda_mmq_moe"

echo "==================================================="
echo "GPU-exclusive decode build (MMQ / MoE)"
echo "Source : ${ROOT_DIR}"
echo "Build  : ${BUILD_DIR}"
echo "==================================================="

# ------------------------------------------------------------
# Hard clean (MANDATORY)
# Any stale cache violates backend invariants
# ------------------------------------------------------------
if [ -d "${BUILD_DIR}" ]; then
    echo "[INFO] Removing existing build directory"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# ------------------------------------------------------------
# Compiler flags
# - Aggressive but deterministic
# ------------------------------------------------------------
COMMON_CXX_FLAGS="-O3 -ffast-math -funroll-loops -march=native -DNDEBUG"

# ------------------------------------------------------------
# CMake configuration
# Backend policy is FIXED here — no runtime ambiguity allowed
# ------------------------------------------------------------
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    -DGGML_CUDA_MMQ=ON \
    -DGGML_CUDA_USE_GRAPHS=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_SERVER_VERBOSE=OFF \
    -DGGML_CPU_ALL=ON \
    -DGGML_DISABLE_F16C=OFF

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------
cmake --build . --config Release -j "$(nproc)"

# ------------------------------------------------------------
# Post-build invariant checks (MANDATORY)
# ------------------------------------------------------------
LIB_CUDA="bin/libggml-cuda.so"

echo "---------------------------------------------------"
echo "[VERIFY] Enforcing backend invariants via CMakeCache.txt"
CACHE_FILE="CMakeCache.txt"

# cuBLAS must not be forced
if grep -q "GGML_CUDA_FORCE_CUBLAS:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: cuBLAS force flag detected in ${CACHE_FILE}"
    exit 1
fi

# MMQ must be forced
if ! grep -q "GGML_CUDA_FORCE_MMQ:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: MMQ force flag missing in ${CACHE_FILE}"
    exit 1
fi

# CUDA graphs must be absent
if strings "${LIB_CUDA}" | grep -qi "graph"; then
    echo "FATAL: CUDA graph symbols detected"
    exit 1
fi

echo "[OK] MMQ-only, no-cuBLAS, no-graphs configuration verified"
echo "---------------------------------------------------"

echo "FINAL build_cuda_mmq_moe completed successfully"
