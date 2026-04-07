#!/bin/bash
# fix_schemas.sh - Instant fix for Gravitas Pipeline Error

EXT_DIR="/home/viren/.vscode/extensions/vayuos.gravitas-code-0.1.0"
SRC_SCHEMAS="/home/viren/llama/llama.cpp/extensions/gravitasCode/src/schemas"

echo "Checking installation at: $EXT_DIR"

if [ -d "$EXT_DIR" ]; then
    echo "Extension found. Copying schemas..."
    mkdir -p "$EXT_DIR/schemas"
    cp -r "$SRC_SCHEMAS"/* "$EXT_DIR/schemas/"
    echo "✅ Schemas copied to $EXT_DIR/schemas/"
    echo "Please restart VS Code for changes to take effect."
else
    echo "❌ Extension directory not found at $EXT_DIR"
    echo "Trying local root..."
    mkdir -p "/home/viren/llama/llama.cpp/extensions/gravitasCode/schemas"
    cp -r "$SRC_SCHEMAS"/* "/home/viren/llama/llama.cpp/extensions/gravitasCode/schemas/"
    echo "✅ Schemas copied to local extension root."
fi
