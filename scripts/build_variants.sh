#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE=Release
JOBS="$(nproc)"

FORCE_REBUILD=1

function clean_build_dir() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        echo "FORCE flag detected. Cleaning build directory..."
        rm -rf "$dir"
    fi
    mkdir -p "$dir"
}

function verify_cublas_only() {
    local cache="$1/CMakeCache.txt"
    echo "Verifying cuBLAS build configuration..."
    grep -q "GGML_CUDA_FORCE_CUBLAS:BOOL=ON" "$cache"
    grep -q "GGML_CUDA_FORCE_MMQ:BOOL=OFF" "$cache"
    grep -q "LLAMA_CUDA_MMQ:BOOL=OFF" "$cache"
    echo "cuBLAS configuration verified."
}

function verify_mmq_only() {
    local cache="$1/CMakeCache.txt"
    echo "Verifying MMQ build configuration..."
    grep -q "GGML_CUDA_FORCE_MMQ:BOOL=ON" "$cache"
    grep -q "GGML_CUDA_FORCE_CUBLAS:BOOL=OFF" "$cache"
    grep -q "LLAMA_CUDA_MMQ:BOOL=ON" "$cache"
    echo "MMQ configuration verified."
}

build_variant() {
    local name="$1"
    local flags="$2"
    local verifier="$3"

    local build_dir="$ROOT_DIR/$name"

    echo "==================================================="
    echo "Processing variant: $name"
    echo "Flags: $flags"
    echo "==================================================="

    clean_build_dir "$build_dir"

    cmake -S "$ROOT_DIR" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DGGML_BLAS=OFF \
        -DGGML_OPENMP=OFF \
        -DGGML_CUDA_USE_GRAPHS=OFF \
        $flags

    cmake --build "$build_dir" -- -j"$JOBS"

    "$verifier" "$build_dir"
    echo "Finished building $name"
    echo "---------------------------------------------------"
    echo
}

# ===================================================
# Variant 1: cuBLAS dense only (NO MMQ)
# ===================================================
build_variant \
    build_cuda_cublas_dense \
    "-DGGML_CUDA=ON \
     -DGGML_CUDA_FORCE_CUBLAS=ON \
     -DGGML_CUDA_FORCE_MMQ=OFF \
     -DLLAMA_CUDA_MMQ=OFF" \
    verify_cublas_only

# ===================================================
# Variant 2: MMQ / MoE optimized (NO cuBLAS)
# ===================================================
build_variant \
    build_cuda_mmq_moe \
    "-DGGML_CUDA=ON \
     -DGGML_CUDA_FORCE_MMQ=ON \
     -DGGML_CUDA_FORCE_CUBLAS=OFF \
     -DLLAMA_CUDA_MMQ=ON" \
    verify_mmq_only
