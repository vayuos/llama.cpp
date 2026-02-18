#!/bin/bash
# Comprehensive rebuild script with CUDA diagnostics

set -e

PROJECT_DIR="/home/viren/llama/llama.cpp"
BUILD_DIR="${PROJECT_DIR}/build_diagnostics"

echo "======================================================================"
echo "LLAMA.CPP BUILD DIAGNOSTICS & REBUILD"
echo "======================================================================"
echo ""

# Check CUDA availability
echo "Step 1: Checking CUDA Toolkit Installation"
echo "----------------------------------------------------------------------"
if command -v nvcc &> /dev/null; then
    echo "✓ NVCC found: $(which nvcc)"
    echo "  Version: $(nvcc --version | grep release)"
else
    echo "✗ NVCC NOT FOUND - CUDA toolkit may not be installed"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    echo "  GPU Info:"
    nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader | sed 's/^/    /'
else
    echo "✗ nvidia-smi NOT FOUND"
fi

echo ""
echo "Step 2: Checking CUDA Libraries"
echo "----------------------------------------------------------------------"

# Common CUDA library locations
for lib_path in /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu /opt/cuda/lib64; do
    if [ -d "$lib_path" ]; then
        echo "Checking $lib_path:"
        ls -la "$lib_path"/libcudart* 2>/dev/null | head -3 || echo "  (no libcudart found)"
        ls -la "$lib_path"/libnvrtc* 2>/dev/null | head -3 || echo "  (no libnvrtc found)"
        ls -la "$lib_path"/libcuda.so* 2>/dev/null | head -3 || echo "  (no libcuda.so found)"
    fi
done

echo ""
echo "Step 3: Clean Previous Build Artifacts"
echo "----------------------------------------------------------------------"
if [ -d "$BUILD_DIR" ]; then
    echo "Removing $BUILD_DIR..."
    rm -rf "$BUILD_DIR"
fi

echo ""
echo "Step 4: Create Fresh Build Directory"
echo "----------------------------------------------------------------------"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "Step 5: Configure CMake with Verbose Output"
echo "----------------------------------------------------------------------"
# Configure with verbose flags to see linker commands
cmake "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MESSAGE_LOG_LEVEL=DEBUG \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_COMMON=ON \
    2>&1 | tee cmake_config.log

echo ""
echo "Step 6: Compile with Verbose Output (First 100 lines of output)"
echo "----------------------------------------------------------------------"
echo "Building... (saving full output to build_verbose.log)"
make -j4 VERBOSE=1 2>&1 | tee build_verbose.log | head -100

echo ""
echo "Step 7: Extract Linker Errors (if any)"
echo "----------------------------------------------------------------------"
echo "Searching for linker errors in build output..."
grep -i "undefined reference\|linker error\|ld:" build_verbose.log | head -20 || echo "No linker errors found in output!"

echo ""
echo "======================================================================"
echo "DIAGNOSTIC COMPLETE"
echo "======================================================================"
echo ""
echo "Output files:"
echo "  - cmake_config.log: CMake configuration output"
echo "  - build_verbose.log: Full verbose build output"
echo ""
echo "If linking succeeded, binaries are in: $BUILD_DIR/bin"
echo "To see all executables built:"
ls -la "$BUILD_DIR/bin/" 2>/dev/null | tail -20

echo ""
echo "Next steps:"
echo "  1. Review the linker error details above"
echo "  2. Check if CUDA libraries are in standard paths (see Step 2 output)"
echo "  3. If CUDA toolkit not found, try: export CUDA_PATH=/path/to/cuda"
echo "  4. If successful, copy to original build: cp -r $BUILD_DIR/bin/* $PROJECT_DIR/build/bin/"
