# Complete Commands for Full Logging & Verification

## Step 1: Clean Build with Debug Symbols

```bash
cd ~/llama/llama.cpp

# Clean previous build
rm -rf build/

# Configure with all optimizations + debug info
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
  -DAMDGPU_TARGETS=gfx1100

# Build
cmake --build build --config Release -j$(nproc)
```

## Step 2: Set Environment Variables for Full Logging

```bash
# ROCm optimization flags
export GGML_HIP_PINNED_MEM=1
export GGML_HIP_PREFER_HOST_KV=1
export HSA_ENABLE_SDMA=0
export OMP_NUM_THREADS=8

# Enable all logging
export LLAMA_LOG_DEBUG=1
export LLAMA_LOG_VERBOSE=1
export GGML_LOG_LEVEL=DEBUG

# Optional: Enable GPU debugging (can be verbose)
export HSA_DEBUG=0  # Set to 1 for full HIP debug output (WARNING: very verbose)
```

## Step 3: Test DEFAULT Mode (ROCm_Host Buffer Expected)

**Terminal 1 - Start Server with Default Settings (mmap enabled):**

```bash
cd ~/llama/llama.cpp

# Set environment variables
export GGML_HIP_PINNED_MEM=1
export GGML_HIP_PREFER_HOST_KV=1
export HSA_ENABLE_SDMA=0
export OMP_NUM_THREADS=8
export LLAMA_LOG_DEBUG=1

# Run server with logging to file
./build/bin/llama-server \
  -m ~/models/qwen/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --host 192.168.1.5 --port 8080 \
  -ngl 999 \
  -c 4096 \
  --threads 8 --threads-batch 8 \
  --batch-size 1024 --ubatch-size 512 \
  --parallel 1 \
  --flash-attn on \
  2>&1 | tee server_default_mode_$(date +%s).log

# This will show:
# - TOKEN_EMBD buffer selection
# - "USING ROCm_Host" message
# - All layer offloading info
```

**Terminal 2 - Query for Logs (while server is running):**

```bash
# Wait 5-10 seconds for server to start, then run:

# Show all TOKEN_EMBD related logs
grep -i "token_embd\|USING\|buffer" server_default_mode_*.log | head -30

# Show model buffer placement
grep "model buffer\|offloaded\|load_tensors" server_default_mode_*.log | head -20

# Show full token embedding selection process
grep -A 5 -B 5 "TOKEN_EMBD Q4_K" server_default_mode_*.log
```

## Step 4: Test GPU-EXCLUSIVE Mode (ROCm0 Buffer Expected)

**Terminal 1 - Start Server with --no-mmap:**

```bash
cd ~/llama/llama.cpp

# Set environment variables
export GGML_HIP_PINNED_MEM=1
export GGML_HIP_PREFER_HOST_KV=1
export HSA_ENABLE_SDMA=0
export OMP_NUM_THREADS=8
export LLAMA_LOG_DEBUG=1

# Run server with --no-mmap flag (GPU-exclusive mode)
./build/bin/llama-server \
  -m ~/models/qwen/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --host 192.168.1.5 --port 8080 \
  -ngl 999 \
  -c 4096 \
  --threads 8 --threads-batch 8 \
  --batch-size 1024 --ubatch-size 512 \
  --parallel 1 \
  --flash-attn on \
  --no-mmap \
  2>&1 | tee server_gpu_exclusive_mode_$(date +%s).log

# This should show:
# - TOKEN_EMBD buffer selection
# - "USING GPU BUFFER" message
# - All layers on GPU
```

**Terminal 2 - Query for Logs (while server is running):**

```bash
# Show all TOKEN_EMBD related logs
grep -i "token_embd\|USING\|buffer" server_gpu_exclusive_mode_*.log | head -30

# Show model buffer placement
grep "model buffer\|offloaded\|load_tensors" server_gpu_exclusive_mode_*.log | head -20

# Show full token embedding selection process
grep -A 5 -B 5 "TOKEN_EMBD Q4_K" server_gpu_exclusive_mode_*.log
```

## Step 5: Benchmark Both Modes

### Benchmark Test Script

```bash
#!/bin/bash

PROMPT="Write a Python function to calculate fibonacci numbers recursively. Include docstring and type hints. Generate detailed explanation."
TOKENS=100

# Test default mode
echo "Testing DEFAULT mode (ROCm_Host)..."
time curl -s -X POST http://192.168.1.5:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"gpt-3.5-turbo\",
    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
    \"max_tokens\": $TOKENS,
    \"temperature\": 0.7
  }" | jq '.usage'

sleep 5

# Test GPU-exclusive mode (kill server, restart with --no-mmap, then benchmark)
echo ""
echo "Testing GPU-EXCLUSIVE mode (ROCm0)..."
# (Start server in new terminal with --no-mmap first)
time curl -s -X POST http://192.168.1.5:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"gpt-3.5-turbo\",
    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
    \"max_tokens\": $TOKENS,
    \"temperature\": 0.7
  }" | jq '.usage'
```

## Step 6: Analyze Logs

### Extract Key Information

```bash
# Find the exact buffer selection decisions
echo "=== BUFFER SELECTION DECISIONS ==="
grep -E "TOKEN_EMBD.*USING|prefer_gpu_exclusive|CUDA_Host|ROCm_Host|GPU BUFFER" server_*.log

# Show model layer offloading
echo ""
echo "=== LAYER OFFLOADING STATUS ==="
grep "offloaded.*layers\|offloaded.*GPU" server_*.log

# Show memory breakdown
echo ""
echo "=== MEMORY LAYOUT ==="
grep -E "model buffer|KV buffer|compute buffer|memory breakdown" server_*.log

# Compare modes
echo ""
echo "=== COMPARING MODES ==="
echo "DEFAULT MODE (mmap enabled):"
grep -m1 "prefer_gpu_exclusive\|TOKEN_EMBD.*USING" server_default_mode_*.log | head -5

echo ""
echo "GPU-EXCLUSIVE MODE (--no-mmap):"
grep -m1 "prefer_gpu_exclusive\|TOKEN_EMBD.*USING" server_gpu_exclusive_mode_*.log | head -5
```

## Step 7: Full Diagnostic Report

```bash
# Generate comprehensive diagnostic report
cat > analyze_buffers.sh << 'EOF'
#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     TOKEN EMBEDDING BUFFER STRATEGY DIAGNOSTIC REPORT      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

for logfile in server_*.log; do
    echo "File: $logfile"
    echo "─────────────────────────────────────────────────────────"

    # Extract mode
    if [[ "$logfile" == *"default"* ]]; then
        echo "Mode: DEFAULT (mmap enabled)"
    else
        echo "Mode: GPU-EXCLUSIVE (--no-mmap)"
    fi

    # Extract token embedding buffer decision
    echo "Token Embedding Buffer:"
    if grep -q "USING ROCm_Host\|USING CUDA_Host\|USING HIP_Host" "$logfile"; then
        echo "  ✓ ROCm_Host (Pinned CPU Memory)"
        grep "TOKEN_EMBD.*USING.*Host" "$logfile" | head -1
    elif grep -q "USING GPU BUFFER" "$logfile"; then
        echo "  ✓ ROCm0 (GPU Buffer)"
        grep "TOKEN_EMBD.*USING GPU" "$logfile" | head -1
    else
        echo "  ? Unknown/Not logged"
    fi

    # Extract layer offloading
    echo "Layer Offloading:"
    grep "offloaded.*GPU" "$logfile" | head -1

    # Extract model buffer size
    echo "Model Buffer Size:"
    grep "model buffer size\|ROCm0 model buffer" "$logfile" | head -1

    # Extract memory breakdown
    echo "Memory Breakdown:"
    grep "memory breakdown\|ROCm0.*PRO W7800" "$logfile" | head -2

    echo ""
done
EOF

chmod +x analyze_buffers.sh
./analyze_buffers.sh
```

## Quick Checklist

✅ **Step 1:** Clean build
```bash
rm -rf build && cmake -B build ... && cmake --build build -j$(nproc)
```

✅ **Step 2:** Set environment variables
```bash
export GGML_HIP_PINNED_MEM=1 GGML_HIP_PREFER_HOST_KV=1 HSA_ENABLE_SDMA=0 OMP_NUM_THREADS=8 LLAMA_LOG_DEBUG=1
```

✅ **Step 3:** Run DEFAULT mode
```bash
./build/bin/llama-server -m model.gguf -ngl 999 -c 4096 ... 2>&1 | tee default_$(date +%s).log
```

✅ **Step 4:** Run GPU-EXCLUSIVE mode
```bash
./build/bin/llama-server -m model.gguf -ngl 999 -c 4096 --no-mmap ... 2>&1 | tee gpu_exc_$(date +%s).log
```

✅ **Step 5:** Analyze logs
```bash
grep -i "TOKEN_EMBD\|USING\|offloaded" default_*.log gpu_exc_*.log
```

✅ **Step 6:** Benchmark
```bash
curl -X POST http://192.168.1.5:8080/v1/chat/completions ...
```

## Expected Output

**DEFAULT Mode should show:**
```
load_tensors: TOKEN_EMBD Q4_K **USING ROCm_Host/CUDA_Host** (avoids get_rows cross-device sync with CPU token indices)
```

**GPU-EXCLUSIVE Mode should show:**
```
load_tensors: TOKEN_EMBD Q4_K **USING GPU BUFFER** (no-mmap flag: GPU-exclusive execution requested)
```

---

Run these commands and share:
1. Log file outputs showing buffer selection
2. Benchmark results (tokens/sec) for each mode
3. Memory breakdown from logs
