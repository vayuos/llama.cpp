#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_cuda_cublas_dense"
GENERATOR="Ninja"

echo "==================================================="
echo "GPU-exclusive decode build (INCREMENTAL)"
echo "Source : ${ROOT_DIR}"
echo "Build  : ${BUILD_DIR}"
echo "==================================================="

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

RECONFIGURE=false

if [ ! -f "CMakeCache.txt" ]; then
    RECONFIGURE=true
else
    if ! grep -q "CMAKE_GENERATOR:INTERNAL=${GENERATOR}" CMakeCache.txt; then
        echo "[WARN] Generator mismatch detected. Reconfiguring..."
        rm -rf *
        RECONFIGURE=true
    fi
fi

if [ "$RECONFIGURE" = true ]; then
    echo "[INFO] Configuring with ${GENERATOR}"

    cmake .. \
        -G "${GENERATOR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_CXX_EXTENSIONS=OFF \
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
else
    echo "[INFO] Using existing CMake configuration"
fi

cmake --build . -j "$(nproc)"

echo "---------------------------------------------------"
echo "[OK] Incremental build complete"
echo "---------------------------------------------------"

