#!/usr/bin/env bash

LLAMA_BIN="/home/viren/llama/llama.cpp/build/bin/llama-server"
MODEL="/home/viren/models/qwen3/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"

CUDA_VISIBLE_DEVICES=0 \
"$LLAMA_BIN" \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  -ngl 36 \
  -c 8192 \
  --batch-size 1024 \
  --ubatch-size 256 \
  --threads 32 \
  --temp 0.2 \
  --top-p 0.9

