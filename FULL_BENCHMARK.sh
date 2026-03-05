#!/bin/bash

##############################################################################
# COMPLETE PERFORMANCE BENCHMARK WITH DETAILED METRICS
# Measures prompt and generation throughput
##############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SCRIPT_DIR}/server_logs_benchmark_${TIMESTAMP}.txt"
METRICS_FILE="${SCRIPT_DIR}/benchmark_results_${TIMESTAMP}.txt"

PORT=8080
HOST="127.0.0.1"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "LLAMA.CPP FULL PERFORMANCE BENCHMARK"
echo "==========================================${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "Metrics file: $METRICS_FILE"
echo ""

# Function to log
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$METRICS_FILE"
}

# Start logging
{
    echo "=========================================="
    echo "BENCHMARK START: $TIMESTAMP"
    echo "=========================================="
    echo ""

    # Kill any existing server
    echo "Cleaning up any existing servers..."
    pkill -f "llama-server" || true
    sleep 2

    # Start new server with optimizations
    echo "Starting optimized llama-server..."
    cd "$SCRIPT_DIR"

    timeout 600 ./llama-server \
        -m /home/vayuos/models/qwen/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
        -ngl 999 \
        -c 8192 \
        -b 4096 \
        -ub 1024 \
        --no-mmap \
        -t 1 \
        --cache-prompt \
        --cache-prompt-tokens 4096 \
        -p 1 \
        --port $PORT \
        --verbose >> "$LOG_FILE" 2>&1 &

    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    echo "Waiting for server warmup (20 seconds)..."
    sleep 20

    # Verify server is running
    if ! nc -z $HOST $PORT 2>/dev/null; then
        echo "ERROR: Server failed to start!"
        tail -50 "$LOG_FILE"
        exit 1
    fi

    echo "✅ Server is ready on $HOST:$PORT"
    echo ""

    # Benchmark configuration
    echo "=========================================="
    echo "BENCHMARK CONFIGURATION"
    echo "=========================================="
    echo "Server: $HOST:$PORT"
    echo "Model: Qwen3-Coder-Next-UD-Q4_K_XL (80B parameters, Q4_K)"
    echo "Context: 8192 tokens"
    echo "Batch: 4096 tokens"
    echo "Ubatch: 1024 tokens"
    echo "GPU Layers: 999 (all to GPU)"
    echo "Flash Attention: Enabled"
    echo ""

    # Test prompts of different sizes
    SHORT_PROMPT="What is Python?"
    MID_PROMPT="Explain how quicksort works. Include the algorithm, time complexity, and a Python implementation."
    LONG_PROMPT="Design a complete REST API for a task management system using Python and FastAPI. Include:\n1. Data models for tasks and users\n2. CRUD endpoints\n3. Authentication\n4. Error handling\n5. Database integration\n\nProvide complete code with documentation."

    echo "=========================================="
    echo "TEST 1: SHORT PROMPT THROUGHPUT"
    echo "=========================================="

    START_TIME=$(date +%s%N)
    RESPONSE1=$(curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"gpt-3.5-turbo\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$SHORT_PROMPT\"}],
            \"max_tokens\": 128,
            \"temperature\": 0.7,
            \"top_p\": 0.9
        }")
    END_TIME=$(date +%s%N)

    ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))  # Convert to milliseconds
    TOKENS=$(echo "$RESPONSE1" | grep -oP '"usage":\{[^}]*"completion_tokens":\K[0-9]+' || echo "0")
    PROMPT_TOKENS=$(echo "$RESPONSE1" | grep -oP '"usage":\{[^}]*"prompt_tokens":\K[0-9]+' || echo "0")

    if [ $ELAPSED -gt 0 ] && [ $TOKENS -gt 0 ]; then
        SPEED=$(echo "scale=2; ($TOKENS * 1000) / $ELAPSED" | bc)
        echo "Time: ${ELAPSED}ms | Tokens: $TOKENS | Speed: ${SPEED} tok/sec"
    fi
    echo ""

    echo "=========================================="
    echo "TEST 2: MEDIUM PROMPT THROUGHPUT"
    echo "=========================================="

    START_TIME=$(date +%s%N)
    RESPONSE2=$(curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"gpt-3.5-turbo\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$MID_PROMPT\"}],
            \"max_tokens\": 256,
            \"temperature\": 0.7,
            \"top_p\": 0.9
        }")
    END_TIME=$(date +%s%N)

    ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))
    TOKENS=$(echo "$RESPONSE2" | grep -oP '"usage":\{[^}]*"completion_tokens":\K[0-9]+' || echo "0")

    if [ $ELAPSED -gt 0 ] && [ $TOKENS -gt 0 ]; then
        SPEED=$(echo "scale=2; ($TOKENS * 1000) / $ELAPSED" | bc)
        echo "Time: ${ELAPSED}ms | Tokens: $TOKENS | Speed: ${SPEED} tok/sec"
    fi
    echo ""

    echo "=========================================="
    echo "TEST 3: LONG PROMPT THROUGHPUT"
    echo "=========================================="

    START_TIME=$(date +%s%N)
    RESPONSE3=$(curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"gpt-3.5-turbo\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$LONG_PROMPT\"}],
            \"max_tokens\": 512,
            \"temperature\": 0.7,
            \"top_p\": 0.9
        }")
    END_TIME=$(date +%s%N)

    ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))
    TOKENS=$(echo "$RESPONSE3" | grep -oP '"usage":\{[^}]*"completion_tokens":\K[0-9]+' || echo "0")

    if [ $ELAPSED -gt 0 ] && [ $TOKENS -gt 0 ]; then
        SPEED=$(echo "scale=2; ($TOKENS * 1000) / $ELAPSED" | bc)
        echo "Time: ${ELAPSED}ms | Tokens: $TOKENS | Speed: ${SPEED} tok/sec"
    fi
    echo ""

    # Extract GPU metrics from server log
    echo "=========================================="
    echo "GPU METRICS"
    echo "=========================================="
    echo ""
    echo "GPU Layer Offloading:"
    grep "offloaded.*layers" "$LOG_FILE" | tail -1 || echo "  (Data being collected...)"
    echo ""
    echo "Memory Configuration:"
    grep "buffer size" "$LOG_FILE" | head -5 || echo "  (Data being collected...)"
    echo ""

    echo "=========================================="
    echo "SYSTEM DIAGNOSTICS"
    echo "=========================================="
    echo "Checking for errors..."
    ERROR_COUNT=$(grep -ic "error\|failed\|exception" "$LOG_FILE" || echo 0)
    WARNING_COUNT=$(grep -ic "warning" "$LOG_FILE" || echo 0)

    echo "Errors: $ERROR_COUNT"
    echo "Warnings: $WARNING_COUNT"
    echo ""

    if grep -i "oom\|out of memory" "$LOG_FILE" >/dev/null 2>&1; then
        echo "⚠️  OOM DETECTED!"
    else
        echo "✅ No OOM errors"
    fi

    if grep -i "cuda error\|rocm error\|hip error" "$LOG_FILE" >/dev/null 2>&1; then
        echo "⚠️  GPU ERRORS DETECTED!"
    else
        echo "✅ No GPU errors"
    fi
    echo ""

    echo "=========================================="
    echo "BENCHMARK COMPLETE"
    echo "=========================================="
    echo ""
    echo "Full server log: $LOG_FILE"
    echo "Metrics summary: $METRICS_FILE"
    echo ""
    echo "Server still running on $HOST:$PORT"
    echo "To run more benchmarks: curl requests to http://$HOST:$PORT/v1/chat/completions"
    echo "To stop server: pkill -f llama-server"

} | tee "$METRICS_FILE"

echo ""
echo -e "${GREEN}✅ Benchmark complete!${NC}"
echo "Results saved to: $METRICS_FILE"
