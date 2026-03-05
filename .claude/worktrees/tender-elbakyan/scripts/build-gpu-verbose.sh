#!/usr/bin/env bash

################################################################################
# GPU Build Script (Final Stable Version)
#
# Console:  Only [ XX% ] progress lines
# Log file: Full verbose output
# Log path: build_<backend>/build-gpu-verbose.txt
################################################################################

set -e

################################################################################
# Locate Real llama.cpp Root (walk upward only)
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_DIR="$SCRIPT_DIR"
PROJECT_ROOT=""

while [ "$SEARCH_DIR" != "/" ]; do
    if [ -f "$SEARCH_DIR/CMakeLists.txt" ] \
       && [ -d "$SEARCH_DIR/ggml" ] \
       && [ -d "$SEARCH_DIR/common" ]; then
        PROJECT_ROOT="$SEARCH_DIR"
        break
    fi
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done

if [ -z "$PROJECT_ROOT" ]; then
    echo "ERROR: Could not locate llama.cpp project root."
    exit 1
fi

################################################################################
# Defaults
################################################################################

BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
VERBOSE=0
MAKE_THREADS=12

################################################################################
# Simple Console Logging
################################################################################

log() { echo "[INFO] $1"; }
error() { echo "[ERROR] $1"; }

################################################################################
# Parse Arguments
################################################################################

BUILD_BACKEND="cpu"

while [[ $# -gt 0 ]]; do
    case "$1" in
        cpu)  BUILD_BACKEND="cpu"; shift ;;
        cuda) BUILD_BACKEND="cuda"; shift ;;
        -j*)  MAKE_THREADS="${1#-j}"; shift ;;
        -v)   VERBOSE=1; shift ;;
        -h|--help) exit 0 ;;
        *) error "Unknown option: $1"; exit 1 ;;
    esac
done

BUILD_DIR="$PROJECT_ROOT/build_${BUILD_BACKEND}"
LOG_FILE="$BUILD_DIR/build-gpu-verbose.txt"

mkdir -p "$BUILD_DIR"
: > "$LOG_FILE"

echo ""
echo "================================"
echo "GPU Build"
echo "================================"
echo ""
log "Backend: $BUILD_BACKEND"
log "Threads: $MAKE_THREADS"
log "Project root: $PROJECT_ROOT"
log "Build dir: $BUILD_DIR"
log "Full log: $LOG_FILE"

################################################################################
# Requirements
################################################################################

for cmd in cmake g++ python3; do
    command -v "$cmd" >/dev/null 2>&1 || {
        error "Missing required tool: $cmd"
        exit 1
    }
done

if [ "$BUILD_BACKEND" = "cuda" ]; then
    command -v nvcc >/dev/null 2>&1 || {
        error "CUDA toolkit not found"
        exit 1
    }
fi

################################################################################
# Configure (single run, full log, show errors if fail)
################################################################################

echo ""
echo "================================"
echo "Configuring"
echo "================================"
echo ""

cd "$BUILD_DIR"

CMAKE_ARGS=(
    -G "Unix Makefiles"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

if [ "$BUILD_BACKEND" = "cuda" ]; then
    CMAKE_ARGS+=(-DGGML_CUDA=ON)
else
    CMAKE_ARGS+=(-DGGML_CUDA=OFF)
fi

CONFIG_TMP=$(mktemp)

if ! cmake "$PROJECT_ROOT" "${CMAKE_ARGS[@]}" \
     > "$CONFIG_TMP" 2>&1; then

    cat "$CONFIG_TMP" >> "$LOG_FILE"

    error "CMake configuration failed."
    echo ""
    echo "---- Last 20 lines ----"
    tail -n 20 "$CONFIG_TMP"
    echo "-----------------------"
    echo ""

    rm -f "$CONFIG_TMP"
    exit 1
fi

cat "$CONFIG_TMP" >> "$LOG_FILE"
rm -f "$CONFIG_TMP"

log "Configuration complete."

################################################################################
# Build (single execution, guaranteed full logging)
################################################################################

echo ""
echo "================================"
echo "Building"
echo "================================"
echo ""

if [ $VERBOSE -eq 1 ]; then

    {
        cmake --build . \
            --config "$BUILD_TYPE" \
            --parallel "$MAKE_THREADS" \
            --verbose
    } > >(tee -a "$LOG_FILE" \
            | grep --line-buffered -E '^\[[[:space:]]*[0-9]+%\]') \
      2> >(tee -a "$LOG_FILE" >&2)

    BUILD_STATUS=$?

    if [ $BUILD_STATUS -ne 0 ]; then
        error "Build failed. See full log:"
        echo "  $LOG_FILE"
        exit $BUILD_STATUS
    fi

else
    cmake --build . \
        --config "$BUILD_TYPE" \
        --parallel "$MAKE_THREADS"
fi

log "Build finished successfully."

################################################################################
# Done
################################################################################

echo ""
echo "================================"
echo "Done"
echo "================================"
echo ""
log "Full verbose output stored at:"
echo "  $LOG_FILE"

