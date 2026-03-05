#!/bin/bash
set -euo pipefail

# ============================================================
# SAFE INCREMENTAL GPU BUILD (Makefiles)
# Auto-detects generator mismatch and repairs directory
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_cuda_cublas_dense"
CACHE_FILE="${BUILD_DIR}/CMakeCache.txt"

MODE="incremental"

for arg in "$@"; do
    case "$arg" in
        --clean)
            MODE="clean"
            ;;
        --reconfigure)
            MODE="reconfigure"
            ;;
    esac
done

echo "==================================================="
echo "GPU-maximized decode build (Make incremental)"
echo "Mode   : ${MODE}"
echo "Source : ${ROOT_DIR}"
echo "Build  : ${BUILD_DIR}"
echo "Target : RTX 4060 Ti (sm_89)"
echo "==================================================="

# ------------------------------------------------------------
# Clean mode
# ------------------------------------------------------------
if [ "${MODE}" = "clean" ]; then
    echo "[INFO] Performing full clean"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

COMMON_CXX_FLAGS="-O3 -ffast-math -fno-finite-math-only -funroll-loops -march=native -DNDEBUG"
CUDA_FLAGS="--use_fast_math -O3"

CMAKE_FLAGS=(
    -G "Unix Makefiles"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS}"
    -DCMAKE_CUDA_FLAGS="${CUDA_FLAGS}"
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
    -DCMAKE_CUDA_ARCHITECTURES=89

    -DBUILD_SHARED_LIBS=ON

    -DGGML_CUDA=ON
    -DGGML_CUDA_FORCE_CUBLAS=ON
    -DGGML_CUDA_FORCE_MMQ=OFF
    -DGGML_CUDA_FA=ON
    -DGGML_CUDA_FA_ALL_QUANTS=ON
    -DGGML_CUDA_GRAPHS=ON
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128
    -DGGML_CUDA_NO_VMM=OFF
    -DGGML_SCHED_MAX_COPIES=1
    -DGGML_CPU_REPACK=ON
    -DGGML_BLAS=OFF
    -DGGML_OPENMP=OFF
    -DGGML_CCACHE=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_SERVER_VERBOSE=OFF
    -DGGML_CPU_ALL=ON
)

RECONFIGURE=false

# ------------------------------------------------------------
# Detect missing cache
# ------------------------------------------------------------
if [ ! -f "${CACHE_FILE}" ]; then
    echo "[INFO] No CMakeCache.txt → configuring"
    RECONFIGURE=true
fi

# ------------------------------------------------------------
# Detect generator mismatch
# ------------------------------------------------------------
if [ -f "${CACHE_FILE}" ]; then
    CURRENT_GEN=$(grep CMAKE_GENERATOR "${CACHE_FILE}" | cut -d= -f2)
    if [[ "${CURRENT_GEN}" != "Unix Makefiles" ]]; then
        echo "[INFO] Generator mismatch (${CURRENT_GEN}) → rebuilding directory"
        cd "${ROOT_DIR}"
        rm -rf "${BUILD_DIR}"
        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
        RECONFIGURE=true
    fi
fi

# ------------------------------------------------------------
# Detect missing Makefile
# ------------------------------------------------------------
if [ ! -f "Makefile" ]; then
    echo "[INFO] Makefile missing → configuring"
    RECONFIGURE=true
fi

# ------------------------------------------------------------
# Forced reconfigure
# ------------------------------------------------------------
if [ "${MODE}" = "reconfigure" ]; then
    echo "[INFO] Forced reconfigure"
    RECONFIGURE=true
fi

# ------------------------------------------------------------
# Configure if required
# ------------------------------------------------------------
if [ "${RECONFIGURE}" = true ]; then
    echo "[CONFIGURE] Running CMake"
    cmake .. "${CMAKE_FLAGS[@]}"
else
    echo "[INFO] Reusing valid Makefile configuration"
fi

# ------------------------------------------------------------
# Incremental Make build
# ------------------------------------------------------------
echo "[BUILD] make -j$(nproc)"
make -j"$(nproc)"

# ------------------------------------------------------------
# Backend invariant verification
# ------------------------------------------------------------
echo "---------------------------------------------------"
echo "[VERIFY] Backend invariants"

if ! grep -q "GGML_CUDA_FORCE_CUBLAS:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: cuBLAS force flag missing"
    exit 1
fi

if grep -q "GGML_CUDA_FORCE_MMQ:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: MMQ force flag detected"
    exit 1
fi

if ! grep -q "GGML_CUDA_FA:BOOL=ON" "${CACHE_FILE}"; then
    echo "FATAL: Flash Attention disabled"
    exit 1
fi

if grep -q "GGML_CUDA_GRAPHS:BOOL=OFF" "${CACHE_FILE}"; then
    echo "FATAL: CUDA graphs disabled"
    exit 1
fi

if ! grep -q "GGML_SCHED_MAX_COPIES:STRING=1" "${CACHE_FILE}"; then
    echo "FATAL: Scheduler copies != 1"
    exit 1
fi

echo "[OK] cuBLAS + FA + CUDA Graphs + single-copy scheduler verified"
echo "---------------------------------------------------"
echo "INCREMENTAL Make build completed successfully"
