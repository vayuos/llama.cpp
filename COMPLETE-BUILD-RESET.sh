#!/bin/bash

################################################################################
# Complete Build Reset Script
#
# This script performs a complete reset of the build system, removing:
# - All build directories (build_cpu, build_cuda, etc)
# - CMake cache files
# - Any other build artifacts
#
# Use this if you have persistent build errors that won't go away
#
# Usage:
#   bash COMPLETE-BUILD-RESET.sh
#   # Then rebuild with:
#   ./scripts/build-gpu-exclusive.sh cpu
#
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_header() {
    echo ""
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

print_header "Complete Build System Reset"

log_warning "This will remove ALL build directories and CMake cache"
echo ""
log_info "Directories to be removed:"
log_info "  - build_cpu/"
log_info "  - build_cuda/"
log_info "  - CMakeCache.txt"
log_info "  - CMakeFiles/"
log_info "  - cmake_install.cmake"
echo ""

# Confirm action
read -p "Continue with complete reset? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    log_warning "Reset cancelled"
    exit 0
fi

echo ""
log_info "Starting complete reset..."
echo ""

cd "$PROJECT_ROOT"

# List what we're removing
log_info "Files and directories to remove:"
find . -maxdepth 1 \( -name "build_*" -o -name "CMakeCache.txt" -o -name "CMakeFiles" -o -name "cmake_install.cmake" -o -name "Makefile" \) 2>/dev/null | while read item; do
    [ ! -z "$item" ] && echo "  $item"
done

echo ""
log_info "Removing build directories..."

# Remove build directories
for dir in build_cpu build_cuda; do
    if [ -d "$dir" ]; then
        log_info "Removing $dir..."
        rm -rf "$dir" || log_warning "Could not remove $dir"
    fi
done

# Remove CMake cache in root
log_info "Removing CMake cache files..."
rm -f CMakeCache.txt cmake_install.cmake 2>/dev/null || true
rm -rf CMakeFiles/ 2>/dev/null || true
rm -f Makefile 2>/dev/null || true

# Remove any other cmake generated files
find . -maxdepth 1 -name "*.cmake" -type f ! -name "CMakeLists.txt" -delete 2>/dev/null || true

echo ""
log_success "Build system completely reset!"
echo ""

print_header "Next Steps"

log_info "To rebuild with CPU-only (recommended):"
echo "  $ ./scripts/build-gpu-exclusive.sh cpu"
echo ""

log_info "To rebuild with CUDA (requires CUDA toolkit):"
echo "  $ ./scripts/build-gpu-exclusive.sh cuda"
echo ""

log_info "To verify cmake will be clean:"
echo "  $ ls -la | grep -i cmake"
echo "  $ ls -la build_*"
echo "  # (should show nothing or minimal output)"
echo ""

print_header "Complete Reset Done"

log_success "Ready to rebuild from scratch!"
