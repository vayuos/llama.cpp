#!/bin/bash

# build_variants.sh - Script to automate multiple build variants for llama.cpp

set -e

BUILD_DIR=$1
CONFIG_FLAGS=$2

if [ -z "$BUILD_DIR" ]; then
    echo "Usage: $0 <build_directory> [cmake_flags]"
    echo "Example: $0 build_cuda_mmq_moe '-DGGML_CUDA=ON -DGGML_CUDA_FORCE_MMQ=ON'"
    exit 1
fi

echo "Creating build in $BUILD_DIR with flags: $CONFIG_FLAGS"

# Common base flags
BASE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CCACHE=ON"

# Configure
cmake -B "$BUILD_DIR" $BASE_FLAGS $CONFIG_FLAGS

# Build
cmake --build "$BUILD_DIR" --config Release -j$(nproc)

echo "Build $BUILD_DIR completed successfully."
