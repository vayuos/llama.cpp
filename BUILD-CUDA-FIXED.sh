#!/bin/bash

# =====================================================
# CUDA Build Script - After CMake Fix
# =====================================================
#
# This script rebuilds the CUDA backend after the
# CMAKE_CUDA_ARCHITECTURES fix was applied.
#
# Usage:
#   bash BUILD-CUDA-FIXED.sh
#   OR if running from PowerShell:
#   bash.exe BUILD-CUDA-FIXED.sh

set -e

echo "=============================================="
echo "CUDA Build - Fixed CMAKE_CUDA_ARCHITECTURES"
echo "=============================================="
echo ""

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[INFO] Repository root: $SCRIPT_DIR"
echo ""

# Clean previous build
echo "[INFO] Cleaning previous CUDA build..."
rm -rf build_cuda CMakeCache.txt CMakeFiles cmake_install.cmake Makefile

echo ""
echo "[INFO] Configuration: CMAKE_CUDA_ARCHITECTURES now properly joined"
echo ""

# Configure CMake
echo "=============================================="
echo "Configuring CMake"
echo "=============================================="
cmake -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON

echo ""
echo "=============================================="
echo "Building CUDA Version"
echo "=============================================="
cd build_cuda
make -j12

echo ""
echo "=============================================="
echo "Build Complete!"
echo "=============================================="
echo ""
echo "Built artifacts:"
find bin -name "libllama*" -o -name "llama-*" | head -10

echo ""
echo "Verify build with:"
echo "  ./build_cuda/bin/llama-cli --version"
