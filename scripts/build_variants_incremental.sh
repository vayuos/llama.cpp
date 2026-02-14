#!/bin/bash
set -e

# Usage: ./scripts/build_variants_incremental.sh
#
# GOAL:
# - Fast iteration with incremental rebuilds
# - Reuse existing build directories
# - Preserve object files and CUDA compilation artifacts

COMMON_CXX_FLAGS="-O3 -ffast-math -funroll-loops -march=native"

COMMON_CMAKE_FLAGS=(
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CXX_FLAGS=${COMMON_CXX_FLAGS}"
    "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"
    "-DCMAKE_CUDA_ARCHITECTURES=89"
    "-DCMAKE_CUDA_FLAGS=--use_fast_math"
)

build_variant_incremental() {
    local DIR="$1"
    shift
    local VARIANT_FLAGS=("$@")

    echo "==================================================="
    echo "Incremental build variant: $DIR"
    echo "Flags: ${VARIANT_FLAGS[*]}"
    echo "==================================================="

    mkdir -p "$DIR"
    cd "$DIR"

    # Reconfigure only if necessary (CMake handles this efficiently)
    cmake .. \
        "${COMMON_CMAKE_FLAGS[@]}" \
        "${VARIANT_FLAGS[@]}"

    # Incremental build (only changed targets rebuild)
    cmake --build . --config Release -j "$(nproc)"

    cd ..
    echo "Finished incremental build $DIR"
    echo
}

# --------------------------------------------------------------------
# build_cuda_cublas_dense (Q6 / Q8 dense models)
# --------------------------------------------------------------------
build_variant_incremental "build_cuda_cublas_dense" \
    "-DGGML_CUDA=ON" \
    "-DGGML_CUDA_FORCE_CUBLAS=ON" \
    "-DGGML_CUDA_FORCE_MMQ=OFF" \
    "-DGGML_CUDA_USE_GRAPHS=OFF" \
    "-DGGML_BLAS=OFF" \
    "-DGGML_OPENMP=OFF"

# --------------------------------------------------------------------
# build_cuda_mmq_moe (Q4 / MoE models)
# --------------------------------------------------------------------
build_variant_incremental "build_cuda_mmq_moe" \
    "-DGGML_CUDA=ON" \
    "-DGGML_CUDA_FORCE_MMQ=ON" \
    "-DGGML_CUDA_FORCE_CUBLAS=OFF" \
    "-DGGML_CUDA_USE_GRAPHS=OFF" \
    "-DGGML_BLAS=OFF" \
    "-DGGML_OPENMP=OFF"

