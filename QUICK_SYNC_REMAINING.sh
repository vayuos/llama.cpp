#!/bin/bash

# Quick sync of remaining files that still have C++20 warnings

SOURCE_DIR="/home/viren/source/llama.cpp/llama.cpp/src"
BUILD_SRC_DIR="/home/viren/llama/llama_x86/llama.cpp/src"

echo "Syncing remaining files with C++20 warnings..."
echo ""

# Files that still have warnings
FILES=(
    "llama-tensor-allocation-gpu.cpp"
    "llama-rnorm-matmul-fusion.cpp"
    "llama-bias-activation-fusion.cpp"
    "llama-kernel-fusion-enforce.cpp"
)

for file in "${FILES[@]}"; do
    SOURCE="$SOURCE_DIR/$file"
    TARGET="$BUILD_SRC_DIR/$file"

    if [ -f "$SOURCE" ]; then
        cp -f "$SOURCE" "$TARGET"
        echo "✅ Synced: $file"
    else
        echo "⚠️  Not found: $SOURCE"
    fi
done

echo ""
echo "Sync complete. Resuming build..."
echo ""

cd /home/viren/llama/llama_x86/llama.cpp/build
NPROC=$(nproc 2>/dev/null || echo 4)
make -j$NPROC
