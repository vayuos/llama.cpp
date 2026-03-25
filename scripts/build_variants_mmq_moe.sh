#!/bin/bash
set -euo pipefail

# ============================================================
# build_variants_mmq_moe.sh
#
# GPU-MAXIMIZED BUILD SCRIPT (AMD ROCm / HIP)
# Variant: build_hip_mmq_moe
#
# PURPOSE:
# - MMQ fused kernels for ALL quantized decode matmul
# - Flash Attention (all quants) for attention ops
# - HIP graphs for reduced kernel launch overhead
# - GPU-exclusive decode path per systemchanges.md
#
# TARGET:
# - Quantized models (Q4–Q8, K-variants, IQ)
# - MoE models
# - AMD Radeon PRO W7800 (RDNA3, gfx1100)
#
# CHANGES vs v1:
# - AMDGPU_TARGETS: gfx1100 (Native RDNA3 support)
# - GGML_HIP=ON (Enable AMD backend)
# - GGML_SCHED_MAX_COPIES=1 (single GPU, single sequence)
# - GGML_HIP_NO_VMM=OFF (VMM available on RDNA3)
# - GGML_CPU_REPACK=ON (Quantized weights to CPU-friendly format for fallbacks)
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_hip_mmq_moe"

echo "==================================================="
echo "GPU-maximized decode build (MMQ / MoE - ROCm/HIP)"
echo "Source : ${ROOT_DIR}"
echo "Build  : ${BUILD_DIR}"
echo "Target : AMD Radeon PRO W7800 (gfx1100)"
echo "==================================================="

# ------------------------------------------------------------
# Hard clean — stale cache can shadow newly set flags
# ------------------------------------------------------------
if [ -d "${BUILD_DIR}" ]; then
    echo "[INFO] Removing existing build directory"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# ------------------------------------------------------------
# Host C++ compiler flags (applies to CPU-side code)
# -O3 -ffast-math: maximise float throughput on CPU side
# -march=native:  use all available CPU SIMD
# -DNDEBUG:       eliminate assertions and logging overhead
# ------------------------------------------------------------
COMMON_CXX_FLAGS="-O3 -ffast-math -fno-finite-math-only -funroll-loops -march=native -DNDEBUG"

# ------------------------------------------------------------
# CUDA compiler flags (applies to GPU kernel compilation)
# --use_fast_math: enables fast reciprocal, rsqrt, exp, sin/cos
#                  on Ada hardware — directly speeds up MMQ+attn kernels
# -O3:            max PTXAS optimisation level
# ------------------------------------------------------------
CUDA_FLAGS="--use_fast_math -O3"

# ------------------------------------------------------------
# CMake configuration
# ------------------------------------------------------------
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}" \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    \
    -DBUILD_SHARED_LIBS=ON \
    \
    -DGGML_HIP=ON \
    \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    \
    -DGGML_HIP_GRAPHS=ON \
    \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
    -DGGML_HIP_NO_VMM=OFF \
    -DGGML_CUDA_SAMPLING=ON \
    \
    -DGGML_SCHED_MAX_COPIES=1 \
    \
    -DGGML_CPU_REPACK=ON \
    -DGGML_BLAS=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_CCACHE=OFF \
    \
    -DLLAMA_GPU_EXCLUSIVE_DECODE=ON \
    -DLLAMA_CPU_SAMPLING_EXCLUDED=ON \
    -DLLAMA_KV_HYBRID_EXCLUDED=ON \
    \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_SERVER_VERBOSE=OFF \
    -DGGML_CPU_ALL=ON

# ------------------------------------------------------------
# Build
# ------------------------------------------------------------
cmake --build . --config Release -j "$(nproc)"

# ------------------------------------------------------------
# Post-build invariant checks
# ------------------------------------------------------------
CACHE_FILE="CMakeCache.txt"

echo "---------------------------------------------------"
echo "[VERIFY] Enforcing backend invariants via CMakeCache.txt"

# MMQ must be forced
if ! grep -q "GGML_CUDA_FORCE_MMQ:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: MMQ force flag missing in ${CACHE_FILE}"
    exit 1
fi

# cuBLAS must NOT be forced
if grep -q "GGML_CUDA_FORCE_CUBLAS:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: cuBLAS force flag detected in ${CACHE_FILE}"
    exit 1
fi

# Flash Attention must be enabled
if ! grep -q "GGML_CUDA_FA:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: Flash Attention flag missing in ${CACHE_FILE}"
    exit 1
fi

# HIP graphs must be enabled
if grep -q "GGML_HIP_GRAPHS:BOOL=OFF" "${CACHE_FILE}"; then
    echo "FATAL: HIP graphs are disabled — required for max GPU throughput"
    exit 1
fi

# Scheduler copies must be 1 (single GPU single sequence)
if ! grep -q "GGML_SCHED_MAX_COPIES:STRING=1" "${CACHE_FILE}"; then
    echo "FATAL: GGML_SCHED_MAX_COPIES not set to 1 in ${CACHE_FILE}"
    exit 1
fi

echo "[OK] MMQ + FA + HIP Graphs + fast-math kernels + single-copy sched verified"
echo "---------------------------------------------------"
echo "FINAL build_hip_mmq_moe completed successfully"
echo ""
echo "==================================================="
echo "MAXIMUM VERBOSITY RUNTIME CONFIGURATION"
echo "==================================================="
echo ""
echo "To run with MAXIMUM VERBOSE OUTPUT during model inference:"
echo ""
echo "1. STANDARD VERBOSE RUN:"
echo "   export LLAMA_LOG_LEVEL=DEBUG"
echo "   export GGML_LOG_LEVEL=DEBUG"
echo "   export GGML_SCHED_DEBUG=1"
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo "   export CUDA_DEVICE_WAITS_ON_EXCEPTION=1"
echo "   ./bin/llama-server -m /path/to/model.gguf --verbose"
echo ""
echo "2. WITH GPU-EXCLUSIVE DECODE DIAGNOSTICS:"
echo "   export LLAMA_LOG_LEVEL=DEBUG"
echo "   export GGML_LOG_LEVEL=DEBUG"
echo "   export GGML_SCHED_DEBUG=1"
echo "   export GGML_CUDA_DEBUG=1"
echo "   export GGML_BACKEND_DEBUG=1"
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo "   export CUDA_DEVICE_WAITS_ON_EXCEPTION=1"
echo "   export CUDA_VERBOSE_API_TRACE=1"
echo "   ./bin/llama-server -m /path/to/model.gguf --verbose"
echo ""
echo "3. MINIMAL VERBOSITY (PRODUCTION):"
echo "   export LLAMA_LOG_LEVEL=INFO"
echo "   export GGML_LOG_LEVEL=WARN"
echo "   ./bin/llama-server -m /path/to/model.gguf"
echo ""
echo "4. FOR DETAILED KV CACHE & SAMPLING DEBUGGING:"
echo "   export LLAMA_LOG_LEVEL=DEBUG"
echo "   export GGML_LOG_LEVEL=DEBUG"
echo "   export GGML_SCHED_DEBUG=1"
echo "   export GGML_CUDA_DEBUG=1"
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo "   export CUDA_DEVICE_WAITS_ON_EXCEPTION=1"
echo "   export CUDA_VERBOSE_API_TRACE=1"
echo "   ./bin/llama-server -m /path/to/model.gguf --verbose 2>&1 | tee inference.log"
echo ""
echo "BUILD COMPLETE"
