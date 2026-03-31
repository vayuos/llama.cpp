#!/bin/bash

# Configuration - All paths are now relative to the script's location
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
SOURCE_DIR="$SCRIPT_DIR"
# HARDCODED RULE: Support Code is sibling directory
SUPPORT_CODE_DIR="$(readlink -f "$SCRIPT_DIR/../support-code")"
TARGET_MODULES="$SUPPORT_CODE_DIR/node_modules"

echo "=== Enforcing Symlink Topology ==="
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_MODULES"

# 1. Prepare Target
if [ ! -d "$SUPPORT_CODE_DIR" ]; then
    echo "Creating support-code directory..."
    mkdir -p "$SUPPORT_CODE_DIR"
fi

# 2. Enforce Local Cleanliness
cd "$SOURCE_DIR" || exit 1

if [ -d "node_modules" ] && [ ! -L "node_modules" ]; then
    echo "WARNING: Found local node_modules directory. Removing to enforce symlink..."
    rm -rf "node_modules"
fi

if [ -L "node_modules" ]; then
    CURRENT_LINK=$(readlink "node_modules")
    if [[ "$CURRENT_LINK" == *support-code/node_modules ]]; then
        echo "✅ Symlink already correct."
    else
        echo "❌ Symlink incorrect ($CURRENT_LINK). Re-linking..."
        rm "node_modules"
        ln -s "$TARGET_MODULES" "node_modules"
    fi
else
    echo "Creating new symlink..."
    ln -s "$TARGET_MODULES" "node_modules"
fi

# 3. Validation
ls -ld node_modules
echo "=== Setup Complete: Topology Enforced ==="
