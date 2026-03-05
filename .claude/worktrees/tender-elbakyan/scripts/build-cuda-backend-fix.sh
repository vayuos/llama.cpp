#!/bin/bash
#
# Backend Symbol Export Fix - Build Script (CUDA & CPU)
#
# This script rebuilds backend libraries with proper symbol export configuration.
# It ensures libggml-cuda.so and libggml-cpu.so export ggml_backend_init and related symbols.
#
# Usage:
#   ./scripts/build-cuda-backend-fix.sh [options]
#
# Options:
#   -j<N>      Use N parallel jobs (default: nproc)
#   -d         Debug build (default: Release)
#   --verify   Only verify, don't build
#   --clean    Clean before build
#   -h, --help Show this help
#

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[✓]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $*"; }
log_error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }

show_help() {
    grep "^#" "$0" | grep -v "^#!/" | sed 's/# //' | head -30
}

# Defaults
JOBS=$(nproc 2>/dev/null || echo 4)
BUILD_TYPE="Release"
VERIFY_ONLY=0
CLEAN_BUILD=0
BUILD_DIR="build_cuda_mmq_moe_full_logs"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -j*)
            JOBS="${1#-j}"
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --verify)
            VERIFY_ONLY=1
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            ;;
    esac
done

cd "$REPO_ROOT"

log_info "CUDA Backend Symbol Export Fix"
log_info "Repository: $REPO_ROOT"
log_info "Build directory: $BUILD_DIR"
log_info "Parallel jobs: $JOBS"
log_info "Build type: $BUILD_TYPE"
echo ""

# Check prerequisites
log_info "Checking prerequisites..."
if ! command -v cmake &> /dev/null; then
    log_error "cmake not found. Install it with: sudo apt-get install cmake"
fi
log_success "cmake found: $(cmake --version | head -1)"

if ! command -v nvcc &> /dev/null; then
    log_warning "nvcc not found. Building CPU-only version."
    CUDA_FLAG="OFF"
else
    log_success "nvcc found: $(nvcc --version | tail -1)"
    CUDA_FLAG="ON"
fi

if command -v gcc &> /dev/null; then
    log_success "gcc found: $(gcc --version | head -1)"
else
    log_error "gcc not found"
fi

echo ""

# Verify existing build
if [ $VERIFY_ONLY -eq 1 ]; then
    log_info "Verify mode - checking existing build..."
    if [ ! -f "$BUILD_DIR/bin/libggml-cuda.so" ] && [ ! -f "$BUILD_DIR/bin/libggml-cuda.dll" ]; then
        log_error "Build not found at $BUILD_DIR/bin/"
    fi

    if [ -f "$BUILD_DIR/bin/libggml-cuda.so" ]; then
        LIB_PATH="$BUILD_DIR/bin/libggml-cuda.so"
    else
        LIB_PATH="$BUILD_DIR/bin/libggml-cuda.dll"
    fi

    log_info "Checking symbols in: $LIB_PATH"
    if nm -D "$LIB_PATH" 2>/dev/null | grep -q "ggml_backend_init"; then
        log_success "Symbol ggml_backend_init found!"
        nm -D "$LIB_PATH" | grep "ggml_backend_init"
        exit 0
    else
        log_error "Symbol ggml_backend_init NOT found"
    fi
fi

# Step 1: Clean
if [ $CLEAN_BUILD -eq 1 ] || [ ! -d "$BUILD_DIR" ]; then
    log_info "Step 1/5: Cleaning previous build..."
    rm -rf "$BUILD_DIR" CMakeCache.txt CMakeFiles cmake_install.cmake Makefile
    log_success "Clean complete"
else
    log_info "Step 1/5: Skipping clean (use --clean to force)"
fi

echo ""

# Step 2: Check CMake configuration
log_info "Step 2/5: Verifying CMake configuration..."
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    log_info "CMake not configured, will configure now"
else
    # Check if it has the right flags
    if grep -q "BUILD_SHARED_LIBS:BOOL=ON" "$BUILD_DIR/CMakeCache.txt" && \
       grep -q "GGML_CUDA:BOOL=$CUDA_FLAG" "$BUILD_DIR/CMakeCache.txt"; then
        log_success "CMake configuration valid"
    else
        log_warning "CMake configuration mismatch, reconfiguring..."
        rm -rf "$BUILD_DIR"
    fi
fi

echo ""

# Step 3: Configure CMake
log_info "Step 3/5: Configuring CMake..."
log_info "Flags:"
log_info "  -DGGML_CUDA=$CUDA_FLAG"
log_info "  -DBUILD_SHARED_LIBS=ON (CRITICAL for symbol export)"
log_info "  -DCMAKE_BUILD_TYPE=$BUILD_TYPE"
log_info "  -DCMAKE_CUDA_ARCHITECTURES=native"

cmake -S . -B "$BUILD_DIR" \
    -DGGML_CUDA="$CUDA_FLAG" \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES=native

if [ $? -ne 0 ]; then
    log_error "CMake configuration failed"
fi

log_success "CMake configuration successful"
echo ""

# Verify configuration
log_info "Verifying CMake configuration..."
if grep -q "BUILD_SHARED_LIBS:BOOL=ON" "$BUILD_DIR/CMakeCache.txt"; then
    log_success "BUILD_SHARED_LIBS=ON confirmed"
else
    log_error "BUILD_SHARED_LIBS not set to ON in CMake cache"
fi

if [ "$CUDA_FLAG" = "ON" ] && ! grep -q "GGML_CUDA:BOOL=ON" "$BUILD_DIR/CMakeCache.txt"; then
    log_error "GGML_CUDA not set to ON in CMake cache"
fi

echo ""

# Step 4: Build
log_info "Step 4/5: Building..."
cmake --build "$BUILD_DIR" -j"$JOBS" --config "$BUILD_TYPE"

if [ $? -ne 0 ]; then
    log_error "Build failed"
fi

log_success "Build completed"
echo ""

# Step 5: Verify symbol export
log_info "Step 5/5: Verifying symbol export..."

# Find the library
if [ -f "$BUILD_DIR/bin/libggml-cuda.so" ]; then
    LIB_PATH="$BUILD_DIR/bin/libggml-cuda.so"
    LIB_TYPE="libggml-cuda.so"
elif [ -f "$BUILD_DIR/bin/libggml-cuda.dll" ]; then
    LIB_PATH="$BUILD_DIR/bin/libggml-cuda.dll"
    LIB_TYPE="libggml-cuda.dll"
else
    log_error "libggml-cuda library not found in build output"
fi

log_info "Checking: $LIB_PATH"

if [ ! -f "$LIB_PATH" ]; then
    log_error "Library file not found: $LIB_PATH"
fi

log_info "Library size: $(stat -f%z "$LIB_PATH" 2>/dev/null || stat -c%s "$LIB_PATH" 2>/dev/null || echo "unknown")"
log_info "Library type: $(file "$LIB_PATH")"
echo ""

log_info "Checking for ggml_backend_init symbol in all backends..."
echo ""

# Verify all backend libraries
BACKENDS_OK=0
BACKENDS_TOTAL=0

for backend_lib in "$BUILD_DIR/bin"/libggml-*.so "$BUILD_DIR/bin"/libggml-*.dll 2>/dev/null; do
    if [ -f "$backend_lib" ]; then
        BACKEND_NAME=$(basename "$backend_lib" | sed 's/libggml-//g' | sed 's/\.so$//g' | sed 's/\.dll$//g' | tr '[:lower:]' '[:upper:]')
        ((BACKENDS_TOTAL++))

        if nm -D "$backend_lib" 2>/dev/null | grep -q "ggml_backend_init"; then
            log_success "$BACKEND_NAME: Symbol ggml_backend_init found"
            ((BACKENDS_OK++))
        else
            log_error "$BACKEND_NAME: Symbol ggml_backend_init NOT FOUND"
        fi
    fi
done

if [ $BACKENDS_TOTAL -eq 0 ]; then
    log_error "No backend libraries found"
fi

if [ $BACKENDS_OK -ne $BACKENDS_TOTAL ]; then
    echo ""
    log_error "Symbol verification failed: $BACKENDS_OK/$BACKENDS_TOTAL backends OK"
fi

# Summary
echo ""
echo "=============================================="
log_success "Backend Build Successful! ($BACKENDS_OK/$BACKENDS_TOTAL backends verified)"
echo "=============================================="
echo ""
log_info "Build artifacts:"
ls -lh "$BUILD_DIR/bin"/libggml-*.so "$BUILD_DIR/bin"/libggml-*.dll 2>/dev/null | grep -v "\.a$" || true
echo ""
log_info "Main libraries built:"
ls -lh "$BUILD_DIR/bin/"libllama* 2>/dev/null || true
echo ""
log_info "Next steps:"
echo "1. Test with llama-server:"
echo "   $BUILD_DIR/bin/llama-server -m model.gguf"
echo ""
echo "2. Verify backend initialization in logs:"
echo "   - Look for 'backend init' messages without symbol errors"
echo "   - Both CPU and CUDA backends should initialize correctly"
echo ""
echo "3. Benchmark performance:"
echo "   $BUILD_DIR/bin/llama-bench -m model.gguf -t 8"
echo ""

