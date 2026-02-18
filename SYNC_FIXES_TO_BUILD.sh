#!/bin/bash

# Sync all fixes from /home/viren/source to /home/viren/llama/llama_x86 build directory
# This script ensures both build locations have the latest fixes

SOURCE_DIR="/home/viren/source/llama.cpp/llama.cpp/src"
BUILD_DIR="/home/viren/llama/llama_x86/llama.cpp/src"

echo "================================================================================"
echo "SYNCING FIXES TO BUILD DIRECTORY"
echo "================================================================================"
echo ""
echo "Source: $SOURCE_DIR"
echo "Target: $BUILD_DIR"
echo ""

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Check if target directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ Target directory not found: $BUILD_DIR"
    echo "   Build directory may not exist yet."
    echo "   Creating target directory..."
    mkdir -p "$BUILD_DIR"
    if [ $? -eq 0 ]; then
        echo "✓ Created target directory"
    else
        echo "❌ Failed to create target directory"
        exit 1
    fi
fi

# Files to sync (the 4 fixed files)
FILES=(
    "llama-debug-stripping.cpp"
    "llama-json-isolation.cpp"
    "llama-server-decode-isolation.h"
    "llama-config-freeze.h"
)

echo "Syncing files..."
echo ""

SYNC_COUNT=0
for file in "${FILES[@]}"; do
    SOURCE_FILE="$SOURCE_DIR/$file"
    TARGET_FILE="$BUILD_DIR/$file"

    # Check if source file exists
    if [ ! -f "$SOURCE_FILE" ]; then
        echo "⚠️  Source file not found: $SOURCE_FILE"
        continue
    fi

    # Copy file
    cp "$SOURCE_FILE" "$TARGET_FILE"
    if [ $? -eq 0 ]; then
        echo "✅ Synced: $file"
        SYNC_COUNT=$((SYNC_COUNT + 1))
    else
        echo "❌ Failed to sync: $file"
    fi
done

echo ""
echo "================================================================================"
echo "SYNC COMPLETE"
echo "================================================================================"
echo "Files synced: $SYNC_COUNT/${#FILES[@]}"
echo ""

if [ $SYNC_COUNT -eq ${#FILES[@]} ]; then
    echo "✅ All fixes successfully synced to build directory"
    echo ""
    echo "Next steps:"
    echo "  1. cd /home/viren/llama/llama_x86/llama.cpp/build"
    echo "  2. make -j\$(nproc)"
else
    echo "⚠️  Some files failed to sync"
    exit 1
fi

echo ""
