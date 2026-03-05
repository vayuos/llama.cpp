#!/bin/bash

# Phase 3B Performance Testing Script
# Tests Phase 3B (CUDA_Host KV) and GPU overloading optimizations
# Target: Reach 35+ t/s

set -e

MODEL_PATH="${HOME}/.lmstudio/models/lmstudio-community/qwen/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
PROMPT="Once upon a time, there was a developer who wanted to optimize "
N_PREDICT=100
TIMEOUT=120

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Phase 3B Performance Testing ===${NC}"
echo "Model: $MODEL_PATH"
echo "Context: 32768 tokens"
echo "Predictions: $N_PREDICT tokens"
echo ""

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Model not found at $MODEL_PATH${NC}"
    exit 1
fi

# Function to run test
run_test() {
    local name=$1
    local flags=$2
    local desc=$3

    echo -e "${YELLOW}Test: $name${NC}"
    echo "Flags: $flags"
    echo "Description: $desc"
    echo "Starting benchmark..."

    # Run server with timeout
    timeout $TIMEOUT ./llama-server \
        -m "$MODEL_PATH" \
        -c 32768 \
        --no-mmap \
        -t 8 \
        $flags \
        > /tmp/llama-test-$$.log 2>&1 &

    SERVER_PID=$!

    # Wait for server to start
    sleep 3

    # Send test prompt
    local response=$(curl -s http://localhost:8000/completion \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$PROMPT\", \"n_predict\": $N_PREDICT}" 2>/dev/null || echo "{}")

    # Kill server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true

    # Extract performance metrics from logs
    if [ -f /tmp/llama-test-$$.log ]; then
        local tps=$(grep -oP 'decode time.*\K[\d.]+\s*\(.*?t/s\)' /tmp/llama-test-$$.log | tail -1 || echo "N/A")
        echo -e "${GREEN}Result: $tps${NC}"
        echo ""
        rm -f /tmp/llama-test-$$.log
    else
        echo -e "${RED}No logs found${NC}"
        echo ""
    fi
}

# PHASE A: Baseline Tests
echo -e "${BLUE}### PHASE A: Baseline (Safe Tests) ###${NC}"
echo ""

run_test "Baseline (no Phase 3B)" \
    "-ngl 36 --no-prefer-cuda-host-kv" \
    "Current production config - CPU-GPU hybrid"

run_test "Baseline + Phase 3B" \
    "-ngl 36 --prefer-cuda-host-kv" \
    "Phase 3B fix - CUDA_Host for CPU-resident KV"

# PHASE B: GPU Overloading
echo -e "${BLUE}### PHASE B: GPU Overloading (Test GPU Memory) ###${NC}"
echo ""
echo "Monitor GPU memory with: watch -n 1 nvidia-smi"
echo ""

run_test "GPU Overload -ngl 40" \
    "-ngl 40 --prefer-cuda-host-kv" \
    "40 layers on GPU, 9 on CPU - medium GPU pressure"

run_test "GPU Overload -ngl 44" \
    "-ngl 44 --prefer-cuda-host-kv" \
    "44 layers on GPU, 5 on CPU - high GPU pressure"

run_test "GPU Overload -ngl 48" \
    "-ngl 48 --prefer-cuda-host-kv" \
    "48 layers on GPU, 1 on CPU - nearly GPU-exclusive"

run_test "GPU Auto-Limit -ngl 999" \
    "-ngl 999 --prefer-cuda-host-kv" \
    "Auto-limit to available VRAM (should match -ngl 48)"

# PHASE C: Ubatch Tuning (if GPU optimization successful)
echo -e "${BLUE}### PHASE C: Ubatch Tuning (Optional) ###${NC}"
echo "Only run if Phase B shows improvement"
echo ""

read -p "Test ubatch optimization? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    run_test "Ubatch 256" \
        "-ngl 48 --prefer-cuda-host-kv --ubatch-optimize 256" \
        "Small batches - lower latency"

    run_test "Ubatch 512" \
        "-ngl 48 --prefer-cuda-host-kv --ubatch-optimize 512" \
        "Balanced - medium latency/throughput"

    run_test "Ubatch 1024" \
        "-ngl 48 --prefer-cuda-host-kv --ubatch-optimize 1024" \
        "Large batches - higher throughput"
fi

echo -e "${BLUE}=== Testing Complete ===${NC}"
echo "Summary of results above. Check logs for detailed metrics."
