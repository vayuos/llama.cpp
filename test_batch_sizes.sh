#!/bin/bash
# Phase 2.3: Batch Size Tuning Test
# Tests different -n-batch values to find optimal throughput

MODEL="$HOME/.lmstudio/models/lmstudio-community/qwen/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
SERVER_BIN="$HOME/llama/llama.cpp/build_cuda_mmq_moe/bin/llama-server"
LOG_DIR="$HOME/llama/llama.cpp/batch_tuning_results"

# Create results directory
mkdir -p "$LOG_DIR"

# Test batch sizes
BATCH_SIZES=(128 256 512 1024)

echo "======================================================"
echo "Phase 2.3: Batch Size Tuning Test"
echo "Model: 30B-A3B on RTX 4060 Ti (36/12 GPU/CPU split)"
echo "======================================================"
echo ""

# Simple test: Send API request and measure tokens/sec from response
for BATCH in "${BATCH_SIZES[@]}"; do
    echo "Testing -n-batch $BATCH..."

    # Run server in background with this batch size
    LOG_FILE="$LOG_DIR/batch_${BATCH}.log"

    timeout 120 $SERVER_BIN \
        -m "$MODEL" \
        -c 8192 \
        -ngl 36 \
        --no-mmap \
        --flash-attn on \
        -t 8 \
        -n-batch $BATCH \
        --port 8090 \
        --verbose \
        2>&1 | tee "$LOG_FILE" &

    SERVER_PID=$!

    # Wait for server to start
    sleep 3

    # Send test request
    echo "  Sending test prompt..."
    curl -s http://localhost:8090/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Write a hello world program in Python"}],
            "max_tokens": 100
        }' > "$LOG_DIR/response_${BATCH}.json"

    # Extract decode throughput from response (if available)
    DECODE_SPEED=$(grep -o '"predicted_per_second":[0-9.]*' "$LOG_DIR/response_${BATCH}.json" | cut -d: -f2)

    if [ -z "$DECODE_SPEED" ]; then
        DECODE_SPEED="N/A"
    fi

    echo "  Decode speed: $DECODE_SPEED t/s"

    # Kill server
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null

    # Wait before next test
    sleep 2
    echo ""
done

echo "======================================================"
echo "Results saved to: $LOG_DIR/"
echo "Compare response_*.json files for decode speeds"
echo "Pick the batch size with highest tokens/sec"
echo "======================================================"
