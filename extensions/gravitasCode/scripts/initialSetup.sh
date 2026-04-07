#!/bin/bash

# Initial setup for Gravitas extension
# Creates required directories and ensures correct permissions.

GRAVITAS_HOME="$HOME/.gravitas"
SOCKET_DIR="$GRAVITAS_HOME/sockets"
LOG_DIR="$GRAVITAS_HOME/logs"

echo "=== Gravitas Initial Setup ==="

echo "Creating $GRAVITAS_HOME if it does not exist..."
mkdir -p "$GRAVITAS_HOME"

# Correct initial setup: Ensure socket directory exists with safe permissions,
# but do NOT pre-create regular files as sockets.
echo "Creating socket directory..."
mkdir -p "$SOCKET_DIR"
chmod 700 "$SOCKET_DIR"

echo "Creating log directory..."
mkdir -p "$LOG_DIR"
chmod 700 "$LOG_DIR"

echo "Setup complete. Llama-server will create its own sockets in $SOCKET_DIR."
