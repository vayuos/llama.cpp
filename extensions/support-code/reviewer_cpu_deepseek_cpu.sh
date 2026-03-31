#!/usr/bin/env bash

LLAMA_BIN="$HOME/llama/llama.cpp/build-cpu/bin/llama-server"
MODEL="$HOME/models/Phi/Phi-3-medium-4k-instruct.Q6_K.gguf"

# Hard-disable GPU (defensive)
unset CUDA_VISIBLE_DEVICES
unset GGML_CUDA
unset NVIDIA_VISIBLE_DEVICES

"$LLAMA_BIN" \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port 8020 \
  -ngl 0 \
  -c 4096 \
  --threads 64 \
  --threads-batch 64 \
  --temp 0.0 \
  --top-k 1 \
  --top-p 1.0 \
  --repeat-penalty 1.0
