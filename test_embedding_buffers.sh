#!/bin/bash

# Quick test to verify token embedding buffer placement

set -e

MODEL="${HOME}/models/qwen/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
SERVER_IP="192.168.1.5"
SERVER_PORT="8080"

echo "=========================================="
echo "Testing Token Embedding Buffer Placement"
echo "=========================================="
echo ""

# Test 1: DEFAULT mode (with mmap) - should use ROCm_Host
echo "[TEST 1] DEFAULT mode (mmap enabled = ROCm_Host expected)"
echo "Command: ./build/bin/llama-server -m model.gguf -ngl 999 -c 4096 ..."
echo ""

timeout 12 ./build/bin/llama-server \
  -m "$MODEL" \
  --host "$SERVER_IP" --port "$SERVER_PORT" \
  -ngl 999 -c 4096 \
  --threads 8 \
  --flash-attn on \
  2>&1 | grep -E "TOKEN_EMBD|USING|offloaded|model buffer" | head -20

echo ""
echo "[TEST 1 Complete]"
echo ""
sleep 2

# Test 2: GPU-EXCLUSIVE mode (--no-mmap) - should use ROCm0
echo "[TEST 2] GPU-EXCLUSIVE mode (--no-mmap = ROCm0 expected)"
echo "Command: ./build/bin/llama-server -m model.gguf -ngl 999 -c 4096 --no-mmap ..."
echo ""

timeout 12 ./build/bin/llama-server \
  -m "$MODEL" \
  --host "$SERVER_IP" --port "$SERVER_PORT" \
  -ngl 999 -c 4096 \
  --threads 8 \
  --flash-attn on \
  --no-mmap \
  2>&1 | grep -E "TOKEN_EMBD|USING|offloaded|model buffer" | head -20

echo ""
echo "[TEST 2 Complete]"
echo ""

echo "=========================================="
echo "Analysis:"
echo "=========================================="
echo ""
echo "If the fix is working:"
echo "  ✓ TEST 1 should show model on ROCm0 (all layers offloaded)"
echo "  ✓ TEST 2 should also show model on ROCm0 (all layers offloaded)"
echo ""
echo "Both modes put model on GPU because -ngl 999 offloads all layers."
echo "The difference is in how token EMBEDDINGS are accessed during decode:"
echo ""
echo "  Default mode:     Embeddings via ROCm_Host (pinned CPU) = ~59 tok/sec"
echo "  GPU-exclusive:    Embeddings via ROCm0 (GPU) = TBD (needs benchmark)"
echo ""
