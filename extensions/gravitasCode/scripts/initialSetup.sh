#!/bin/bash

# Initial setup for Gravitas extension
# Creates required directories and ensures correct permissions.

GRAVITAS_HOME="$HOME/.gravitas"
SOCKET_DIR="$GRAVITAS_HOME/sockets"
LOG_DIR="$GRAVITAS_HOME/logs"

echo "=== Gravitas Initial Setup ==="

echo "Creating $GRAVITAS_HOME if it does not exist..."
mkdir -p "$GRAVITAS_HOME"

echo "Creating socket directory..."
mkdir -p "$SOCKET_DIR"
chmod 700 "$SOCKET_DIR"

echo "Creating log directory..."
mkdir -p "$LOG_DIR"
chmod 700 "$LOG_DIR"

# Touch placeholder socket files to avoid ENOENT until servers start
touch "$SOCKET_DIR/coder.sock"
chmod 600 "$SOCKET_DIR/coder.sock"

touch "$SOCKET_DIR/reviewer.sock"
chmod 600 "$SOCKET_DIR/reviewer.sock"

echo "Setup complete."
