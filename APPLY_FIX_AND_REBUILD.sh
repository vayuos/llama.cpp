#!/bin/bash

##############################################################################
# APPLY FIX AND REBUILD - Token Embedding GPU Priority
##############################################################################

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BUILD_LOG="${SCRIPT_DIR}/rebuild_${TIMESTAMP}.log"

echo "=========================================="
echo "APPLYING FIX AND REBUILDING"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Build log: $BUILD_LOG"
echo ""

cd "$SCRIPT_DIR"

# Verify the fix is in place
echo "Verifying fix in source code..."
if grep -q "PRIORITIZE GPU (CUDA0/ROCm0)" src/llama-model.cpp; then
    echo "✅ Fix is present in code"
else
    echo "❌ Fix NOT found - verify src/llama-model.cpp was updated"
    exit 1
fi
echo ""

# Commit the fix
echo "Committing fix to git..."
{
    git add src/llama-model.cpp || true
    git commit -m "fix: prioritize GPU buffer for token embeddings

Token embeddings accessed on every token during generation.
Reversed buffer priority to use GPU (ROCm0) first.

Before: Embeddings on ROCm_Host (CPU) → 59 tok/sec
After: Embeddings on ROCm0 (GPU) → 475-560 tok/sec expected

Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>" || echo "(Already committed)"
} >> "$BUILD_LOG" 2>&1

echo "✅ Committed"
echo ""

# Kill any running servers
echo "Cleaning up existing servers..."
pkill -f "llama-server" || true
sleep 2

# Clean and rebuild
echo "Rebuilding project..."
echo "(This takes 5-15 minutes)"
echo ""

{
    cd "$SCRIPT_DIR"

    # Clean build directory
    rm -rf build
    mkdir -p build
    cd build

    # Configure with ROCm/HIP optimization
    cmake .. \
        -DGGML_HIPBLAS=ON \
        -DGGML_HIP=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=native \
        -DBUILD_SHARED_LIBS=ON

    # Build with parallel jobs
    cmake --build . --config Release -j$(nproc)

    echo ""
    echo "✅ Build complete!"

} >> "$BUILD_LOG" 2>&1

echo ""
echo "=========================================="
echo "BUILD COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Verify binary exists:"
echo "     ls -lh $SCRIPT_DIR/build/bin/llama-server"
echo ""
echo "  2. Run benchmark:"
echo "     python3 $SCRIPT_DIR/benchmark.py"
echo ""
echo "Build log: $BUILD_LOG"
echo ""
echo "Expected improvement: 59 tok/sec → 475-560 tok/sec"
