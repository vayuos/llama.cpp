#!/bin/bash

# Direct test to verify ROCm0 GPU buffer is working with the smart strategy

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${HOME}/models/qwen/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
SERVER_IP="192.168.1.5"
SERVER_PORT="8080"

echo "=========================================="
echo "Testing ROCm0 GPU Buffer Implementation"
echo "=========================================="
echo ""

# Rebuild
echo "[1/3] Rebuilding with smart buffer strategy..."
cd "$REPO_DIR"

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -flto=auto" \
  -DGGML_HIP=ON -DGGML_HIPBLAS=ON -DGGML_HIPBLASLT=ON \
  -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  > /tmp/cmake_build.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed"
    tail -20 /tmp/cmake_build.log
    exit 1
fi

cmake --build build --config Release -j$(nproc) > /tmp/build.log 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    tail -20 /tmp/build.log
    exit 1
fi

echo "✅ Build successful"
echo ""

# Test 1: Default mode (with mmap) - should use ROCm_Host
echo "[2/3] Testing DEFAULT mode (mmap enabled = ROCm_Host buffer)..."
echo ""

LOG_DEFAULT="/tmp/rocm_test_default_$(date +%s).log"

timeout 15 ./build/bin/llama-server \
  -m "$MODEL" \
  --host "$SERVER_IP" --port "$SERVER_PORT" \
  -ngl 999 -c 4096 \
  --threads 8 --batch-size 512 \
  --flash-attn on \
  2>&1 | tee "$LOG_DEFAULT" &

SERVER_PID=$!
sleep 4

# Check logs for buffer selection
echo "Checking logs for TOKEN_EMBD buffer selection..."
if grep -q "USING ROCm_Host\|USING CUDA_Host\|USING HIP_Host" "$LOG_DEFAULT" 2>/dev/null; then
    echo "✅ DEFAULT MODE: Using Host Buffer (ROCm_Host/CUDA_Host)"
    grep "USING ROCm_Host\|USING CUDA_Host\|USING HIP_Host" "$LOG_DEFAULT" | head -1
elif grep -q "USING GPU BUFFER" "$LOG_DEFAULT" 2>/dev/null; then
    echo "⚠️  DEFAULT MODE: Using GPU Buffer (unexpected!)"
    grep "USING GPU BUFFER" "$LOG_DEFAULT" | head -1
else
    echo "⚠️  Could not determine buffer from logs"
fi

# Kill server
kill $SERVER_PID 2>/dev/null || true
sleep 2

echo ""

# Test 2: GPU-exclusive mode (--no-mmap) - should use ROCm0
echo "[3/3] Testing GPU-EXCLUSIVE mode (--no-mmap = ROCm0 GPU buffer)..."
echo ""

LOG_GPU="/tmp/rocm_test_gpu_$(date +%s).log"

timeout 15 ./build/bin/llama-server \
  -m "$MODEL" \
  --host "$SERVER_IP" --port "$SERVER_PORT" \
  -ngl 999 -c 4096 \
  --threads 8 --batch-size 512 \
  --flash-attn on \
  --no-mmap \
  2>&1 | tee "$LOG_GPU" &

SERVER_PID=$!
sleep 4

# Check logs for buffer selection
echo "Checking logs for TOKEN_EMBD buffer selection..."
if grep -q "USING GPU BUFFER" "$LOG_GPU" 2>/dev/null; then
    echo "✅ GPU-EXCLUSIVE MODE: Using GPU Buffer (ROCm0)"
    grep "USING GPU BUFFER" "$LOG_GPU" | head -1
elif grep -q "USING ROCm_Host\|USING CUDA_Host\|USING HIP_Host" "$LOG_GPU" 2>/dev/null; then
    echo "⚠️  GPU-EXCLUSIVE MODE: Using Host Buffer (unexpected!)"
    grep "USING ROCm_Host\|USING CUDA_Host\|USING HIP_Host" "$LOG_GPU" | head -1
else
    echo "⚠️  Could not determine buffer from logs"
fi

# Kill server
kill $SERVER_PID 2>/dev/null || true
sleep 1

echo ""
echo "=========================================="
echo "SMART STRATEGY VERIFICATION"
echo "=========================================="
echo ""

DEFAULT_USES_HOST=0
GPU_USES_GPU=0

if grep -q "USING ROCm_Host\|USING CUDA_Host\|USING HIP_Host" "$LOG_DEFAULT" 2>/dev/null; then
    DEFAULT_USES_HOST=1
fi

if grep -q "USING GPU BUFFER" "$LOG_GPU" 2>/dev/null; then
    GPU_USES_GPU=1
fi

if [ "$DEFAULT_USES_HOST" -eq 1 ] && [ "$GPU_USES_GPU" -eq 1 ]; then
    echo "✅ SMART STRATEGY IS WORKING CORRECTLY!"
    echo ""
    echo "  ✓ Default mode uses ROCm_Host (optimized for get_rows)"
    echo "  ✓ GPU-exclusive mode uses ROCm0 (GPU-only execution)"
    echo ""
    echo "Next: Run benchmarks to compare performance:"
    echo "  Default: ./build/bin/llama-server -m model.gguf -ngl 999 ..."
    echo "  GPU-ex:  ./build/bin/llama-server -m model.gguf -ngl 999 --no-mmap ..."
else
    echo "⚠️  SMART STRATEGY INCOMPLETE"
    echo ""
    echo "  Status:"
    echo "    Default uses host: $DEFAULT_USES_HOST"
    echo "    GPU-exclusive uses GPU: $GPU_USES_GPU"
    echo ""
    echo "Debug files:"
    echo "  Default log: $LOG_DEFAULT"
    echo "  GPU log: $LOG_GPU"
    echo ""
    echo "Run this to check buffer selection:"
    echo "  grep 'TOKEN_EMBD.*USING' $LOG_DEFAULT"
    echo "  grep 'TOKEN_EMBD.*USING' $LOG_GPU"
fi

echo ""
