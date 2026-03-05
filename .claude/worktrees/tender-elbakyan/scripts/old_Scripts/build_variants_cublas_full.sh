#!/bin/bash
set -euo pipefail

# ============================================================
# build_variants_cublas.sh
#
# FULL CLEAN BUILD — GPU-EXCLUSIVE DECODE
# Variant: build_cuda_cublas_dense
#
# Enforces:
# - cuBLAS-only dense CUDA backend
# - No MMQ
# - No CUDA graphs
# - No OpenMP
# - Deterministic execution
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_cuda_cublas_dense"
GENERATOR="Ninja"

echo "==================================================="
echo "GPU-exclusive decode build (FULL CLEAN)"
echo "Source : ${ROOT_DIR}"
echo "Build  : ${BUILD_DIR}"
echo "==================================================="

# ------------------------------------------------------------
# Toolchain sanity check
# ------------------------------------------------------------
command -v cmake >/dev/null 2>&1 || { echo "FATAL: cmake not found"; exit 1; }
command -v ninja >/dev/null 2>&1 || { echo "FATAL: ninja not found (install ninja-build)"; exit 1; }
command -v gcc >/dev/null 2>&1 || { echo "FATAL: gcc not found (install build-essential)"; exit 1; }
command -v g++ >/dev/null 2>&1 || { echo "FATAL: g++ not found (install build-essential)"; exit 1; }

# ------------------------------------------------------------
# Hard clean (MANDATORY)
# ------------------------------------------------------------
if [ -d "${BUILD_DIR}" ]; then
    echo "[INFO] Removing existing build directory"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# ------------------------------------------------------------
# Configure
# ------------------------------------------------------------
echo "[INFO] Configuring with ${GENERATOR}"

cmake .. \
    -G "${GENERATOR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DCMAKE_CUDA_STANDARD=20 \
    -DCMAKE_CUDA_STANDARD_REQUIRED=ON \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FORCE_CUBLAS=ON \
    -DGGML_CUDA_FORCE_MMQ=OFF \
    -DGGML_CUDA_MMQ=OFF \
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
cmake --build . -j "$(nproc)"

# ------------------------------------------------------------
# Post-build invariant checks
# ------------------------------------------------------------
echo "---------------------------------------------------"
echo "[VERIFY] Enforcing backend invariants"

CACHE_FILE="CMakeCache.txt"
LIB_CUDA="bin/libggml-cuda.so"

# MMQ must NOT be forced
if grep -q "GGML_CUDA_FORCE_MMQ:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: MMQ force flag detected"
    exit 1
fi

# cuBLAS MUST be forced
if ! grep -q "GGML_CUDA_FORCE_CUBLAS:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: cuBLAS force flag missing"
    exit 1
fi

# CUDA graphs must not exist in binary
if [ -f "${LIB_CUDA}" ]; then
    if strings "${LIB_CUDA}" | grep -qi "graph"; then
        echo "FATAL: CUDA graph symbols detected in libggml-cuda"
        exit 1
    fi
fi

echo "[OK] cuBLAS-only, no-MMQ, no-graphs configuration verified"
echo "---------------------------------------------------"

echo "FINAL build_cuda_cublas_dense completed successfully"

