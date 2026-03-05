#!/bin/bash

# Test script to benchmark ROCm_Host vs ROCm0 buffer performance
# Tests the smart embedding buffer placement strategy

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${HOME}/models/qwen/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
SERVER_IP="192.168.1.5"
SERVER_PORT="8080"
BENCHMARK_PROMPTS=5
TOKENS_PER_PROMPT=100

# Check model exists
if [[ ! -f "$MODEL" ]]; then
    echo "❌ Model not found: $MODEL"
    exit 1
fi

echo "=========================================="
echo "Token Embedding Buffer Strategy Test"
echo "=========================================="
echo "Model: $(basename "$MODEL")"
echo "Benchmarks: $BENCHMARK_PROMPTS runs"
echo "Tokens/run: $TOKENS_PER_PROMPT"
echo ""

# Rebuild with current code
echo "[1/5] Building with smart buffer placement strategy..."
cd "$REPO_DIR"

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -flto=auto" \
  -DGGML_HIP=ON -DGGML_HIPBLAS=ON -DGGML_HIPBLASLT=ON \
  -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DAMDGPU_TARGETS=gfx1100 > /dev/null 2>&1

cmake --build build --config Release -j$(nproc) > /dev/null 2>&1
echo "✅ Build complete"
echo ""

# Test 1: Default (with mmap) = ROCm_Host buffer
echo "[2/5] Testing DEFAULT MODE (ROCm_Host buffer with mmap)..."
echo "      Expected: ~59 tok/sec (based on previous runs)"
echo ""

SERVER_LOG_DEFAULT="/tmp/embedding_test_default.log"
rm -f "$SERVER_LOG_DEFAULT"

# Start server in background with default settings (mmap enabled)
"./build/bin/llama-server" \
  -m "$MODEL" \
  --host "$SERVER_IP" --port "$SERVER_PORT" \
  -ngl 999 -c 8192 \
  --threads 8 --threads-batch 8 \
  --batch-size 1024 --ubatch-size 512 \
  --parallel 1 \
  --flash-attn on \
  2>&1 | tee "$SERVER_LOG_DEFAULT" &

SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"

# Wait for server to be ready
sleep 5
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start"
    cat "$SERVER_LOG_DEFAULT"
    exit 1
fi

# Check embedding buffer selection in logs
echo ""
echo "Checking buffer selection from logs..."
if grep -q "USING ROCm_Host" "$SERVER_LOG_DEFAULT"; then
    echo "✅ Confirmed: Using ROCm_Host (pinned CPU memory)"
elif grep -q "USING GPU BUFFER" "$SERVER_LOG_DEFAULT"; then
    echo "⚠️  WARNING: Using GPU buffer (should be ROCm_Host in default mode)"
else
    echo "⚠️  Could not confirm buffer selection from logs"
fi
echo ""

# Run benchmark
echo "Running benchmark (default mode)..."
RESULT_DEFAULT=$("$REPO_DIR/benchmark_server.sh" "$SERVER_IP" "$SERVER_PORT" "$BENCHMARK_PROMPTS" "$TOKENS_PER_PROMPT" 2>/dev/null | grep -E "^\s*Average" | awk '{print $3}')

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 2

echo "Result: $RESULT_DEFAULT tok/sec"
echo ""

# Test 2: GPU-exclusive (--no-mmap) = ROCm0 buffer
echo "[3/5] Testing GPU-EXCLUSIVE MODE (ROCm0 buffer with --no-mmap)..."
echo "      Expected: Better performance (if token indices also on GPU)"
echo ""

SERVER_LOG_GPU="$(mktemp /tmp/embedding_test_gpu_XXXX.log)"

# Start server with --no-mmap (GPU-exclusive)
"./build/bin/llama-server" \
  -m "$MODEL" \
  --host "$SERVER_IP" --port "$SERVER_PORT" \
  -ngl 999 -c 8192 \
  --threads 8 --threads-batch 8 \
  --batch-size 1024 --ubatch-size 512 \
  --parallel 1 \
  --flash-attn on \
  --no-mmap \
  2>&1 | tee "$SERVER_LOG_GPU" &

SERVER_PID=$!
echo "Server started (PID: $SERVER_PID) with --no-mmap"

sleep 5
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start"
    cat "$SERVER_LOG_GPU"
    exit 1
fi

# Check embedding buffer selection in logs
echo ""
echo "Checking buffer selection from logs..."
if grep -q "USING GPU BUFFER" "$SERVER_LOG_GPU"; then
    echo "✅ Confirmed: Using GPU BUFFER (ROCm0)"
elif grep -q "USING ROCm_Host" "$SERVER_LOG_GPU"; then
    echo "⚠️  WARNING: Using ROCm_Host (should be GPU in --no-mmap mode)"
else
    echo "⚠️  Could not confirm buffer selection from logs"
fi
echo ""

# Run benchmark
echo "Running benchmark (GPU-exclusive mode)..."
RESULT_GPU=$("$REPO_DIR/benchmark_server.sh" "$SERVER_IP" "$SERVER_PORT" "$BENCHMARK_PROMPTS" "$TOKENS_PER_PROMPT" 2>/dev/null | grep -E "^\s*Average" | awk '{print $3}')

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
sleep 2

echo "Result: $RESULT_GPU tok/sec"
echo ""

# Analysis
echo "=========================================="
echo "BENCHMARK RESULTS"
echo "=========================================="
echo ""
echo "Default Mode (ROCm_Host):"
echo "  Throughput: $RESULT_DEFAULT tok/sec"
echo "  Config: with mmap (default)"
echo ""
echo "GPU-Exclusive Mode (ROCm0):"
echo "  Throughput: $RESULT_GPU tok/sec"
echo "  Config: --no-mmap flag"
echo ""

if (( $(echo "$RESULT_GPU > $RESULT_DEFAULT" | bc -l) )); then
    IMPROVEMENT=$(echo "scale=1; (($RESULT_GPU - $RESULT_DEFAULT) / $RESULT_DEFAULT) * 100" | bc)
    echo "✅ GPU-EXCLUSIVE IS FASTER (+${IMPROVEMENT}%)"
    echo "   Smart strategy is working correctly!"
elif (( $(echo "$RESULT_GPU < $RESULT_DEFAULT" | bc -l) )); then
    DEGRADATION=$(echo "scale=1; (($RESULT_DEFAULT - $RESULT_GPU) / $RESULT_DEFAULT) * 100" | bc)
    echo "⚠️  GPU-EXCLUSIVE IS SLOWER (-${DEGRADATION}%)"
    echo "   Token indices may still be on CPU (cross-device sync overhead)"
    echo "   Need to investigate if token indices are properly moved to GPU"
else
    echo "≈ Performance is similar between modes"
fi

echo ""
echo "DIAGNOSTIC LOGS:"
echo "  Default mode: $SERVER_LOG_DEFAULT"
echo "  GPU mode: $SERVER_LOG_GPU"
echo ""
echo "To debug further:"
echo "  grep 'TOKEN_EMBD' <log_file>  # Check buffer selection"
echo "  grep 'load_tensors' <log_file>  # Check tensor placement"
echo ""
