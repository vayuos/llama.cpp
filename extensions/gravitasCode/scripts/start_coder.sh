#!/bin/bash
# Helper script to start the Gravitas Coder Backend
# Model: Qwen3.5-9B

BINARY="/home/viren/llama/llama.cpp/build_mmq_moe/bin/llama-server"
MODEL="/home/viren/models/qwen/Qwen3.5-9B-UD-Q6_K_XL.gguf"
SOCKET="/home/viren/.gravitas/sockets/coder.sock"
LOG_FILE="/home/viren/.gravitas/logs/coder_backend.log"

# Create log and socket directories if missing
mkdir -p /home/viren/.gravitas/logs
mkdir -p /home/viren/.gravitas/sockets

# Cleanup orphaned socket
rm -f "$SOCKET"

echo "Starting Coder Backend ($MODEL)..."
echo "Logging to $LOG_FILE"

nohup "$BINARY" \
  -m "$MODEL" \
  --host "$SOCKET" \
  --port 0 \
  -c 102400 \
  --temp 0.2 \
  --top-p 0.9 \
  --top-k 40 \
  -ngl 999 \
  --parallel 1 \
  > "$LOG_FILE" 2>&1 &

echo "Coder started in background. PID: $!"
