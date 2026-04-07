#!/bin/bash

# Initial setup for Gravitas extension
# Creates required directories and ensures correct permissions.

GRAVITAS_HOME="$HOME/.gravitas"
SOCKET_DIR="$GRAVITAS_HOME/sockets"
LOG_DIR="$GRAVITAS_HOME/logs"

echo "=== Gravitas Initial Setup (VayuForge Standard) ==="

echo "[1/4] Creating $GRAVITAS_HOME..."
mkdir -p "$GRAVITAS_HOME"
mkdir -p "$SOCKET_DIR"
mkdir -p "$LOG_DIR"
chmod 700 "$SOCKET_DIR" "$LOG_DIR"

echo "[2/4] Verifying Package Dependencies..."
if [ -f "package.json" ]; then
    if command -v bun &> /dev/null; then
        echo "Bun detected. Installing dependencies..."
        bun install
    elif command -v npm &> /dev/null; then
        echo "NPM detected. Installing dependencies..."
        npm install
    fi
else
    echo "Skipping dependency install (already in VSIX bundle)."
fi

echo "[3/4] Verifying Llama Infrastructure..."
if command -v llama-server &> /dev/null; then
    echo "llama-server found in PATH."
else
    echo "WARNING: llama-server not found in PATH. Please ensure it is installed."
fi

echo "[4/4] Finalizing configuration..."
echo "{\"version\": \"1.0\", \"initialized\": true}" > "$GRAVITAS_HOME/config.json"

echo "=== Setup complete. Gravitas is Ready. ==="
echo "Sockets will be managed in $SOCKET_DIR."
