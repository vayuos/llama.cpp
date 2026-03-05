#!/bin/bash

##############################################################################
# DIAGNOSTIC SCRIPT - Identify performance issues
##############################################################################

echo "=========================================="
echo "LLAMA.CPP DIAGNOSTICS"
echo "=========================================="
echo ""

# Kill existing server
pkill -f "llama-server" 2>/dev/null || true
sleep 2

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/vayuos/llama/llama.cpp/diagnostic_${TIMESTAMP}.txt"

echo "Running diagnostic with verbose output..."
echo "Log file: $LOG_FILE"
echo ""

# Start server with MAXIMUM verbosity and capture ALL output
{
    echo "=========================================="
    echo "STARTING SERVER WITH VERBOSE DIAGNOSTICS"
    echo "=========================================="
    echo ""

    timeout 180 /home/vayuos/llama/llama.cpp/build/bin/llama-server \
        -m /home/vayuos/models/qwen/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
        -ngl 999 \
        -c 8192 \
        -b 4096 \
        -ub 1024 \
        --no-mmap \
        -t 1 \
        --cache-prompt \
        --host 192.168.1.5 \
        --port 8080 \
        --verbose 2>&1

} | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "DIAGNOSTIC ANALYSIS"
echo "=========================================="
echo ""

echo "Searching for key metrics in log..."
echo ""

echo "1. GPU Detection:"
grep -i "rocm\|cuda\|gpu\|device" "$LOG_FILE" | head -5

echo ""
echo "2. Layer Offloading:"
grep -i "offload" "$LOG_FILE" | head -5

echo ""
echo "3. Token Embedding Placement:"
grep -i "token.*emb\|embedding" "$LOG_FILE" | head -5

echo ""
echo "4. Buffer Information:"
grep -i "buffer.*size" "$LOG_FILE" | head -10

echo ""
echo "5. Model Loading Status:"
grep -i "load.*model\|loading" "$LOG_FILE" | head -5

echo ""
echo "6. Errors (if any):"
grep -i "error\|failed" "$LOG_FILE" | head -5

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Full diagnostic log saved to: $LOG_FILE"
echo ""
echo "To check GPU layers:"
echo "  grep 'offload' $LOG_FILE"
echo ""
echo "To check embeddings:"
echo "  grep 'HIP0\|ROCm0\|token.*emb' $LOG_FILE"
echo ""
echo "To check for errors:"
echo "  grep -i 'error\|failed' $LOG_FILE"
