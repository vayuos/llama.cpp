#!/usr/bin/env bash

LLAMA_BIN="/home/viren/llama/llama.cpp/build/bin/llama-server"
MODEL="/home/viren/models/deepseekdeepseek-coder-33b-base.Q4_K_M.gguf"

"$LLAMA_BIN" \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port 8001 \
  -ngl 0 \
  -c 8192 \
  --threads 64 \
  --temp 0.0

