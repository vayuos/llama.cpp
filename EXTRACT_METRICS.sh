#!/bin/bash

##############################################################################
# EXTRACT PERFORMANCE METRICS FROM SERVER LOGS
##############################################################################

LOG_FILE="${1:-server_logs_all_optimizations_20260305_222035.txt}"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found: $LOG_FILE"
    exit 1
fi

echo "=========================================="
echo "PERFORMANCE METRICS EXTRACTION"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo ""

# Extract key configuration metrics
echo "=== OPTIMIZATION STATUS ==="
echo ""
echo "GPU Layer Offloading:"
grep "offloaded.*layers to GPU" "$LOG_FILE" || echo "  (No data found)"
echo ""

echo "Memory Configuration:"
grep "buffer size.*=.*MiB" "$LOG_FILE" | head -10 || echo "  (No data found)"
echo ""

echo "Context & Batch Configuration:"
grep -E "n_ctx|n_batch|n_ubatch|flash_attn" "$LOG_FILE" | grep ":" || echo "  (No data found)"
echo ""

echo "=== TOKENIZATION METRICS ==="
echo ""
grep "n_tokens.*=" "$LOG_FILE" | tail -5 || echo "  (No data found)"
echo ""

echo "=== THROUGHPUT ANALYSIS ==="
echo ""

# Try to extract timing information
echo "Processing timing data..."

# Look for eval time patterns
eval_times=$(grep -oP "eval time = \K[0-9.]+" "$LOG_FILE" || true)
if [ -n "$eval_times" ]; then
    echo "Eval times found:"
    echo "$eval_times" | head -10
    echo ""

    # Calculate throughput from eval times
    first_eval=$(echo "$eval_times" | head -1)
    if [ -n "$first_eval" ]; then
        echo "Calculating throughput from eval time..."
        # Throughput = tokens / eval_time_ms
        python3 << 'EOF'
import sys

eval_times = """$eval_times""".strip().split('\n')
if eval_times and eval_times[0]:
    try:
        eval_ms = float(eval_times[0])
        # Assuming 256 tokens generated (typical max_tokens)
        tokens_generated = 256
        tok_per_sec = (tokens_generated * 1000) / eval_ms
        print(f"Generation speed: {tok_per_sec:.2f} tok/sec")
    except:
        pass
EOF
    fi
else
    echo "No eval time data found in logs"
fi

echo ""

# Summary
echo "=== SUMMARY ==="
echo ""
echo "Current log coverage:"
wc -l "$LOG_FILE" | awk '{print "  Total lines:", $1}'
echo ""
echo "To get complete metrics:"
echo "  1. Run the server with verbose logging"
echo "  2. Make inference requests to completion"
echo "  3. Extract timing data from output"
echo ""

# Check for any errors
echo "=== ERROR CHECK ==="
error_count=$(grep -ic "error\|failed\|exception" "$LOG_FILE" || echo 0)
warning_count=$(grep -ic "warning" "$LOG_FILE" || echo 0)

echo "Errors found: $error_count"
echo "Warnings found: $warning_count"

if grep -i "oom\|out of memory" "$LOG_FILE" >/dev/null 2>&1; then
    echo "⚠️  OOM detected!"
else
    echo "✅ No OOM errors"
fi

if grep -i "cuda error\|rocm error\|hip error" "$LOG_FILE" >/dev/null 2>&1; then
    echo "⚠️  GPU errors detected!"
else
    echo "✅ No GPU errors"
fi
