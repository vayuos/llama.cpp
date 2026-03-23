#!/bin/bash

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                                                            ║
# ║          LLAMA.CPP FULL OPTIMIZATION BUILD & TEST SCRIPT                  ║
# ║                  For AMD Radeon PRO W7800 (ROCm/HIP)                       ║
# ║                                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║               🚀 APPLYING ALL OPTIMIZATIONS - PHASE 1: GIT                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Phase 1: Update code from GitHub
echo "Step 1: Pulling latest code from GitHub..."
cd ~/llama/llama.cpp
git pull origin main

if [ $? -eq 0 ]; then
    echo "✅ Code pulled successfully"
    echo ""
else
    echo "❌ Git pull failed"
    exit 1
fi

# Show what changed
echo "Latest commits:"
git log --oneline -3
echo ""

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║            🚀 APPLYING ALL OPTIMIZATIONS - PHASE 2: CLEANUP              ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Phase 2: Clean previous build
echo "Step 2: Cleaning previous build..."
rm -rf build/
echo "✅ Build directory cleaned"
echo ""

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║            🚀 APPLYING ALL OPTIMIZATIONS - PHASE 3: CMAKE                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Phase 3: CMake configuration with all optimizations
echo "Step 3: Running CMake with OPTIMIZED settings..."
echo ""
echo "Configuration:"
echo "  - Code fix: Token embeddings on GPU (ROCm support)"
echo "  - Context: 8,192 tokens (optimized from 32K)"
echo "  - Batch: 4,096 tokens (optimized from 2K)"
echo "  - Ubatch: 1,024 tokens (optimized from 768)"
echo "  - GPU: HIP (ROCm 7.2.0)"
echo "  - Target: gfx1100 (RDNA3)"
echo ""

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -flto=auto" \
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

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ CMake configuration completed successfully"
    echo ""
else
    echo ""
    echo "❌ CMake configuration failed"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║            🚀 APPLYING ALL OPTIMIZATIONS - PHASE 4: BUILD                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Phase 4: Build
echo "Step 4: Building llama-server (parallel compilation)..."
echo "This will take ~5-10 minutes..."
echo ""

start_time=$(date +%s)
cmake --build build --config Release -j$(nproc)
end_time=$(date +%s)
build_time=$((end_time - start_time))

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build completed successfully in ${build_time}s"
    echo ""
else
    echo ""
    echo "❌ Build failed"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║            🚀 APPLYING ALL OPTIMIZATIONS - PHASE 5: TEST                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Phase 5: Run server with optimized settings
echo "Step 5: Starting llama-server with ALL OPTIMIZATIONS..."
echo ""
echo "Environment Variables:"
echo "  • GGML_HIP_PINNED_MEM=1          (GPU pinned memory)"
echo "  • GGML_HIP_PREFER_HOST_KV=1      (Host-pinned KV cache)"
echo "  • HSA_ENABLE_SDMA=0              (Disable DMA, optimize latency)"
echo "  • OMP_NUM_THREADS=1              (Minimize CPU threads)"
echo ""
echo "Server Parameters:"
echo "  • Context: -c 8192               (optimized from 32,768)"
echo "  • Batch: --batch-size 4096       (optimized from 2,048)"
echo "  • Ubatch: --ubatch-size 1024     (optimized from 768)"
echo "  • Threads: 1 (minimized)"
echo "  • Cache RAM: --cache-ram 4096    (prompt caching)"
echo ""
echo "Expected Performance:"
echo "  ✓ Prompt throughput: 475-550 tokens/sec (was 405)"
echo "  ✓ Generation speed: ~60 tokens/sec (stable)"
echo "  ✓ GPU memory usage: ~43-44 GB / 48 GB"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

export GGML_HIP_PINNED_MEM=1
export GGML_HIP_PREFER_HOST_KV=1
export HSA_ENABLE_SDMA=0
export OMP_NUM_THREADS=1

./build/bin/llama-server \
  -m ~/models/qwen/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --host 192.168.1.5 \
  --port 8080 \
  -ngl 999 \
  -c 8192 \
  --threads 1 \
  --threads-batch 1 \
  --batch-size 4096 \
  --ubatch-size 1024 \
  --parallel 1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --no-mmap \
  --cache-ram 4096 2>&1 | tee "server_logs_all_optimizations_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                        ✅ OPTIMIZATION COMPLETE                           ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
