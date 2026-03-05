#!/bin/bash

# Complete debug logging test for token embedding buffer strategy

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${HOME}/models/qwen/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
SERVER_IP="192.168.1.5"
SERVER_PORT="8080"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   TOKEN EMBEDDING BUFFER STRATEGY - FULL DEBUG TEST        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Clean build
echo -e "${YELLOW}[STEP 1/5] Clean Build with Debug Symbols${NC}"
echo "─────────────────────────────────────────────────────────"
cd "$REPO_DIR"

if [ -d "build" ]; then
    echo "Removing old build..."
    rm -rf build/
fi

echo "Configuring CMake..."
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -flto=auto -g" \
  -DGGML_HIP=ON \
  -DGGML_HIPBLAS=ON \
  -DGGML_HIPBLASLT=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DGGML_LTO=ON \
  -DGGML_REPACK=ON \
  -DGGML_CPU_REPACK=ON \
  -DGGML_AVX2=ON \
  -DGGML_FMA=ON \
  -DGGML_F16C=ON \
  -DGGML_BMI2=ON \
  -DGGML_OFFLOAD_KQV=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  > /tmp/cmake.log 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ CMake configuration failed${NC}"
    tail -20 /tmp/cmake.log
    exit 1
fi

echo "Building..."
cmake --build build --config Release -j$(nproc) > /tmp/build.log 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed${NC}"
    tail -20 /tmp/build.log
    exit 1
fi

echo -e "${GREEN}✓ Build successful${NC}"
echo ""

# Step 2: Setup environment
echo -e "${YELLOW}[STEP 2/5] Setting Up Environment Variables${NC}"
echo "─────────────────────────────────────────────────────────"

export GGML_HIP_PINNED_MEM=1
export GGML_HIP_PREFER_HOST_KV=1
export HSA_ENABLE_SDMA=0
export OMP_NUM_THREADS=8
export LLAMA_LOG_DEBUG=1
export LLAMA_LOG_VERBOSE=1

echo "GGML_HIP_PINNED_MEM=1"
echo "GGML_HIP_PREFER_HOST_KV=1"
echo "HSA_ENABLE_SDMA=0"
echo "OMP_NUM_THREADS=8"
echo "LLAMA_LOG_DEBUG=1"
echo "LLAMA_LOG_VERBOSE=1"
echo -e "${GREEN}✓ Environment ready${NC}"
echo ""

# Step 3: Test DEFAULT mode
echo -e "${YELLOW}[STEP 3/5] Testing DEFAULT Mode (mmap enabled = ROCm_Host expected)${NC}"
echo "─────────────────────────────────────────────────────────"
echo "Starting server..."
echo "Command:"
echo "  ./build/bin/llama-server -m model.gguf -ngl 999 -c 4096 (no --no-mmap)"
echo ""

LOG_DEFAULT="debug_test_default_$(date +%s).log"

timeout 20 ./build/bin/llama-server \
  -m "$MODEL" \
  --host "$SERVER_IP" --port "$SERVER_PORT" \
  -ngl 999 \
  -c 4096 \
  --threads 8 --threads-batch 8 \
  --batch-size 1024 --ubatch-size 512 \
  --parallel 1 \
  --flash-attn on \
  2>&1 | tee "$LOG_DEFAULT" &

SERVER_PID=$!
sleep 8

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ Server failed to start${NC}"
    cat "$LOG_DEFAULT" | tail -20
    exit 1
fi

echo "Server running (PID: $SERVER_PID)"
echo ""
echo "Extracting buffer selection from logs..."
echo ""

if grep -q "USING ROCm_Host\|USING CUDA_Host\|USING HIP_Host" "$LOG_DEFAULT" 2>/dev/null; then
    echo -e "${GREEN}✓ Using Host Buffer (ROCm_Host/CUDA_Host)${NC}"
    grep "TOKEN_EMBD.*USING.*Host" "$LOG_DEFAULT" | head -1
elif grep -q "USING GPU BUFFER" "$LOG_DEFAULT" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Using GPU Buffer (ROCm0)${NC}"
    echo "  (This is unexpected for DEFAULT mode)"
    grep "TOKEN_EMBD.*USING GPU" "$LOG_DEFAULT" | head -1
else
    echo -e "${YELLOW}⚠ Could not confirm buffer selection${NC}"
    echo "  Searching logs..."
    grep -i "token_embd\|USING" "$LOG_DEFAULT" | head -10
fi

echo ""
echo "Layer offloading:"
grep "offloaded.*layers" "$LOG_DEFAULT" | head -1

echo ""
echo "Model buffer:"
grep "model buffer size\|ROCm0 model buffer" "$LOG_DEFAULT" | head -1

kill $SERVER_PID 2>/dev/null || true
sleep 2

echo ""

# Step 4: Test GPU-EXCLUSIVE mode
echo -e "${YELLOW}[STEP 4/5] Testing GPU-EXCLUSIVE Mode (--no-mmap = ROCm0 expected)${NC}"
echo "─────────────────────────────────────────────────────────"
echo "Starting server with --no-mmap..."
echo "Command:"
echo "  ./build/bin/llama-server -m model.gguf -ngl 999 -c 4096 --no-mmap"
echo ""

LOG_GPU="debug_test_gpu_exclusive_$(date +%s).log"

timeout 20 ./build/bin/llama-server \
  -m "$MODEL" \
  --host "$SERVER_IP" --port "$SERVER_PORT" \
  -ngl 999 \
  -c 4096 \
  --threads 8 --threads-batch 8 \
  --batch-size 1024 --ubatch-size 512 \
  --parallel 1 \
  --flash-attn on \
  --no-mmap \
  2>&1 | tee "$LOG_GPU" &

SERVER_PID=$!
sleep 8

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ Server failed to start${NC}"
    cat "$LOG_GPU" | tail -20
    exit 1
fi

echo "Server running (PID: $SERVER_PID)"
echo ""
echo "Extracting buffer selection from logs..."
echo ""

if grep -q "USING GPU BUFFER\|USING ROCm0\|USING CUDA0" "$LOG_GPU" 2>/dev/null; then
    echo -e "${GREEN}✓ Using GPU Buffer (ROCm0)${NC}"
    grep "TOKEN_EMBD.*USING GPU" "$LOG_GPU" | head -1
elif grep -q "USING ROCm_Host\|USING CUDA_Host" "$LOG_GPU" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Using Host Buffer${NC}"
    echo "  (This may indicate --no-mmap flag not triggering GPU path)"
    grep "TOKEN_EMBD.*USING.*Host" "$LOG_GPU" | head -1
else
    echo -e "${YELLOW}⚠ Could not confirm buffer selection${NC}"
    echo "  Searching logs..."
    grep -i "token_embd\|USING" "$LOG_GPU" | head -10
fi

echo ""
echo "Layer offloading:"
grep "offloaded.*layers" "$LOG_GPU" | head -1

echo ""
echo "Model buffer:"
grep "model buffer size\|ROCm0 model buffer" "$LOG_GPU" | head -1

kill $SERVER_PID 2>/dev/null || true
sleep 2

echo ""

# Step 5: Summary
echo -e "${YELLOW}[STEP 5/5] Summary${NC}"
echo "─────────────────────────────────────────────────────────"
echo ""

echo -e "${BLUE}DEFAULT MODE (mmap enabled):${NC}"
if [ -f "$LOG_DEFAULT" ]; then
    echo "Log file: $LOG_DEFAULT"
    echo ""
    echo "Buffer selection:"
    if grep -q "USING ROCm_Host" "$LOG_DEFAULT"; then
        echo -e "  ${GREEN}✓${NC} ROCm_Host (Host Buffer)"
    elif grep -q "USING GPU BUFFER" "$LOG_DEFAULT"; then
        echo -e "  ${YELLOW}✗${NC} GPU Buffer (unexpected)"
    else
        echo -e "  ${YELLOW}?${NC} Unknown"
    fi
    echo ""
    echo "Full token embedding log:"
    grep -A 2 -B 2 "TOKEN_EMBD Q4_K" "$LOG_DEFAULT" 2>/dev/null | head -10
fi

echo ""
echo -e "${BLUE}GPU-EXCLUSIVE MODE (--no-mmap):${NC}"
if [ -f "$LOG_GPU" ]; then
    echo "Log file: $LOG_GPU"
    echo ""
    echo "Buffer selection:"
    if grep -q "USING GPU BUFFER" "$LOG_GPU"; then
        echo -e "  ${GREEN}✓${NC} GPU Buffer (ROCm0)"
    elif grep -q "USING ROCm_Host" "$LOG_GPU"; then
        echo -e "  ${YELLOW}✗${NC} Host Buffer (unexpected)"
    else
        echo -e "  ${YELLOW}?${NC} Unknown"
    fi
    echo ""
    echo "Full token embedding log:"
    grep -A 2 -B 2 "TOKEN_EMBD Q4_K" "$LOG_GPU" 2>/dev/null | head -10
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo -e "${BLUE}Full log files saved:${NC}"
echo "  $LOG_DEFAULT"
echo "  $LOG_GPU"
echo ""
echo "To view all buffer-related messages:"
echo "  grep -i 'TOKEN_EMBD\|USING\|buffer\|offloaded' $LOG_DEFAULT"
echo "  grep -i 'TOKEN_EMBD\|USING\|buffer\|offloaded' $LOG_GPU"
echo ""
