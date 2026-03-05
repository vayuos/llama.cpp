#!/bin/bash

##############################################################################
# FIXED PERFORMANCE BENCHMARK - Uses only supported flags
##############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SCRIPT_DIR}/server_logs_fixed_benchmark_${TIMESTAMP}.txt"
METRICS_FILE="${SCRIPT_DIR}/metrics_fixed_benchmark_${TIMESTAMP}.txt"

PORT=8080
HOST="127.0.0.1"

echo "=========================================="
echo "LLAMA.CPP FIXED BENCHMARK"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "Metrics file: $METRICS_FILE"
echo ""

{
    echo "=========================================="
    echo "BENCHMARK START: $TIMESTAMP"
    echo "=========================================="
    echo ""

    # Kill any existing server
    echo "Cleaning up any existing servers..."
    pkill -f "llama-server" || true
    sleep 2

    # Find the llama-server binary
    LLAMA_SERVER=""
    if [ -f "$SCRIPT_DIR/build_cuda_mmq_moe_full_logs/bin/llama-server" ]; then
        LLAMA_SERVER="$SCRIPT_DIR/build_cuda_mmq_moe_full_logs/bin/llama-server"
    elif [ -f "$SCRIPT_DIR/build/bin/llama-server" ]; then
        LLAMA_SERVER="$SCRIPT_DIR/build/bin/llama-server"
    elif [ -f "$SCRIPT_DIR/llama-server" ]; then
        LLAMA_SERVER="$SCRIPT_DIR/llama-server"
    else
        echo "ERROR: llama-server binary not found!"
        echo "Searched locations:"
        echo "  1. $SCRIPT_DIR/build_cuda_mmq_moe_full_logs/bin/llama-server"
        echo "  2. $SCRIPT_DIR/build/bin/llama-server"
        echo "  3. $SCRIPT_DIR/llama-server"
        echo ""
        echo "Please run BUILD_ALL_OPTIMIZATIONS.sh first to compile the project:"
        echo "  cd $SCRIPT_DIR"
        echo "  ./BUILD_ALL_OPTIMIZATIONS.sh"
        exit 1
    fi

    echo "Using binary: $LLAMA_SERVER"
    echo ""

    # Start server with ONLY SUPPORTED FLAGS
    echo "Starting optimized llama-server..."
    cd "$SCRIPT_DIR"

    timeout 600 "$LLAMA_SERVER" \
        -m /home/vayuos/models/qwen/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
        -ngl 999 \
        -c 8192 \
        -b 4096 \
        -ub 1024 \
        --no-mmap \
        -t 1 \
        --cache-prompt \
        --port $PORT \
        --verbose >> "$LOG_FILE" 2>&1 &

    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    echo "Waiting 20 seconds for server warmup..."
    sleep 20

    # Verify server is running
    if ! nc -z $HOST $PORT 2>/dev/null; then
        echo "ERROR: Server failed to start!"
        echo "Last 30 lines of log:"
        tail -30 "$LOG_FILE"
        exit 1
    fi

    echo "✅ Server is ready on $HOST:$PORT"
    echo ""

    echo "=========================================="
    echo "BENCHMARK CONFIGURATION"
    echo "=========================================="
    echo "Server: $HOST:$PORT"
    echo "Model: Qwen3-Coder-Next-UD-Q4_K_XL"
    echo "Context: 8192 tokens"
    echo "Batch: 4096 tokens"
    echo "Ubatch: 1024 tokens"
    echo "GPU Layers: 999 (all)"
    echo ""

    # Test prompts
    SHORT_PROMPT="What is Python?"
    MID_PROMPT="Explain how quicksort works. Include algorithm, complexity, and Python code."
    LONG_PROMPT="Design a REST API for task management using FastAPI. Include data models, CRUD endpoints, auth, error handling, and database integration."

    echo "=========================================="
    echo "TEST 1: SHORT PROMPT"
    echo "=========================================="

    START_TIME=$(date +%s%N)
    RESPONSE1=$(curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"gpt-3.5-turbo\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$SHORT_PROMPT\"}],
            \"max_tokens\": 128,
            \"temperature\": 0.7
        }")
    END_TIME=$(date +%s%N)

    ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))
    TOKENS=$(echo "$RESPONSE1" | grep -oP '"completion_tokens":\s*\K[0-9]+' || echo "0")

    echo "Elapsed: ${ELAPSED}ms | Tokens: $TOKENS"
    if [ $ELAPSED -gt 0 ] && [ $TOKENS -gt 0 ]; then
        SPEED=$(echo "scale=2; ($TOKENS * 1000) / $ELAPSED" | bc)
        echo "Speed: ${SPEED} tok/sec"
    fi
    echo ""

    echo "=========================================="
    echo "TEST 2: MEDIUM PROMPT"
    echo "=========================================="

    START_TIME=$(date +%s%N)
    RESPONSE2=$(curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"gpt-3.5-turbo\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$MID_PROMPT\"}],
            \"max_tokens\": 256,
            \"temperature\": 0.7
        }")
    END_TIME=$(date +%s%N)

    ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))
    TOKENS=$(echo "$RESPONSE2" | grep -oP '"completion_tokens":\s*\K[0-9]+' || echo "0")

    echo "Elapsed: ${ELAPSED}ms | Tokens: $TOKENS"
    if [ $ELAPSED -gt 0 ] && [ $TOKENS -gt 0 ]; then
        SPEED=$(echo "scale=2; ($TOKENS * 1000) / $ELAPSED" | bc)
        echo "Speed: ${SPEED} tok/sec"
    fi
    echo ""

    echo "=========================================="
    echo "TEST 3: LONG PROMPT"
    echo "=========================================="

    START_TIME=$(date +%s%N)
    RESPONSE3=$(curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"gpt-3.5-turbo\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$LONG_PROMPT\"}],
            \"max_tokens\": 512,
            \"temperature\": 0.7
        }")
    END_TIME=$(date +%s%N)

    ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))
    TOKENS=$(echo "$RESPONSE3" | grep -oP '"completion_tokens":\s*\K[0-9]+' || echo "0")

    echo "Elapsed: ${ELAPSED}ms | Tokens: $TOKENS"
    if [ $ELAPSED -gt 0 ] && [ $TOKENS -gt 0 ]; then
        SPEED=$(echo "scale=2; ($TOKENS * 1000) / $ELAPSED" | bc)
        echo "Speed: ${SPEED} tok/sec"
    fi
    echo ""

    echo "=========================================="
    echo "GPU METRICS FROM LOG"
    echo "=========================================="
    echo ""

    echo "GPU Detection:"
    grep "found.*ROCm devices" "$LOG_FILE" | head -1 || echo "  (Searching...)"

    echo ""
    echo "Layer Offloading:"
    grep "offloaded.*layers" "$LOG_FILE" | tail -1 || echo "  (Data being collected...)"

    echo ""
    echo "Memory Configuration:"
    grep "buffer size.*=" "$LOG_FILE" | head -5 || echo "  (Data being collected...)"

    echo ""

    echo "=========================================="
    echo "ERROR CHECK"
    echo "=========================================="

    ERROR_COUNT=$(grep -ic "error\|failed\|exception" "$LOG_FILE" || echo 0)
    echo "Errors detected: $ERROR_COUNT"

    if grep -iq "oom\|out of memory" "$LOG_FILE"; then
        echo "⚠️  OOM ERROR DETECTED"
    else
        echo "✅ No OOM errors"
    fi

    if grep -iq "cuda error\|rocm error\|hip error" "$LOG_FILE"; then
        echo "⚠️  GPU ERRORS DETECTED"
    else
        echo "✅ No GPU errors"
    fi

    echo ""
    echo "=========================================="
    echo "BENCHMARK COMPLETE"
    echo "=========================================="
    echo ""
    echo "Server log: $LOG_FILE"
    echo "Metrics: $METRICS_FILE"
    echo ""
    echo "To view full log:"
    echo "  tail -100 $LOG_FILE"
    echo ""
    echo "To kill server:"
    echo "  pkill -f llama-server"

} | tee "$METRICS_FILE"

echo ""
echo "✅ Benchmark complete!"
