#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# HARD BIND RTX 4060 Ti (sm_89)
###############################################################################
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1   # <-- RTX 4060 Ti, NOT PCI ID

###############################################################################
# Sanity check (must show ONLY RTX 4060 Ti)
###############################################################################
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

###############################################################################
# Binary & model
###############################################################################
LLAMA_BIN="$HOME/llama/llama.cpp/build-sm89/bin/llama-server"
MODEL="$HOME/models/qwen3/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"

###############################################################################
# Launch
###############################################################################
exec numactl --interleave=all \
"$LLAMA_BIN" \
  -m "$MODEL" \
  --host 127.0.0.1 \
  --port 8010 \
  -ngl 36 \
  -c 8192 \
  --batch-size 1024 \
  --ubatch-size 256 \
  --threads 32 \
  --threads-batch 32 \
  --temp 0.2 \
  --top-p 0.9 \
  --top-k 40 \
  --repeat-penalty 1.05 \
  --no-warmup \
  --metrics
