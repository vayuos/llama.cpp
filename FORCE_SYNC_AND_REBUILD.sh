#!/bin/bash

# FORCE SYNC ALL FIXES AND REBUILD
# This script will:
# 1. Copy all fixed files to build directory (overwrite old ones)
# 2. Force rebuild by cleaning and recompiling

set -e  # Exit on error

SOURCE_DIR="/home/viren/source/llama.cpp/llama.cpp/src"
BUILD_DIR="/home/viren/llama/llama_x86/llama.cpp"
BUILD_SRC_DIR="$BUILD_DIR/src"

echo "================================================================================"
echo "FORCE SYNC ALL FIXES TO BUILD DIRECTORY"
echo "================================================================================"
echo ""

# Files to force sync
FILES=(
    "llama-debug-stripping.cpp"
    "llama-json-isolation.cpp"
    "llama-server-decode-isolation.h"
    "llama-config-freeze.h"
    "llama-graph-schedule-elimination.cpp"
    "llama-tensor-allocation-gpu.cpp"
    "llama-rnorm-matmul-fusion.cpp"
    "llama-bias-activation-fusion.cpp"
)

echo "Syncing files from: $SOURCE_DIR"
echo "Syncing to: $BUILD_SRC_DIR"
echo ""

SYNC_COUNT=0
for file in "${FILES[@]}"; do
    SOURCE_FILE="$SOURCE_DIR/$file"
    TARGET_FILE="$BUILD_SRC_DIR/$file"

    if [ ! -f "$SOURCE_FILE" ]; then
        echo "⚠️  Source not found: $file"
        continue
    fi

    # Force copy (overwrite)
    cp -f "$SOURCE_FILE" "$TARGET_FILE"
    if [ $? -eq 0 ]; then
        echo "✅ Synced: $file"
        SYNC_COUNT=$((SYNC_COUNT + 1))
    else
        echo "❌ Failed to sync: $file"
    fi
done

echo ""
echo "Synced: $SYNC_COUNT/${#FILES[@]} files"
echo ""

if [ $SYNC_COUNT -lt ${#FILES[@]} ]; then
    echo "⚠️  WARNING: Not all files synced successfully"
fi

# Now force rebuild
echo "================================================================================"
echo "FORCING REBUILD (Clean + Recompile)"
echo "================================================================================"
echo ""

cd "$BUILD_DIR/build"

if [ $? -ne 0 ]; then
    echo "❌ Build directory not found: $BUILD_DIR/build"
    exit 1
fi

echo "Build directory: $(pwd)"
echo ""

# Get CPU count
NPROC=$(nproc 2>/dev/null || echo 4)

echo "Cleaning old build objects..."
make clean > /dev/null 2>&1 || true

echo "Starting rebuild with -j$NPROC..."
echo ""

make -j$NPROC

BUILD_STATUS=$?

echo ""
echo "================================================================================"
if [ $BUILD_STATUS -eq 0 ]; then
    echo "✅ BUILD SUCCESSFUL"
else
    echo "❌ BUILD FAILED (exit code: $BUILD_STATUS)"
fi
echo "================================================================================"

exit $BUILD_STATUS
