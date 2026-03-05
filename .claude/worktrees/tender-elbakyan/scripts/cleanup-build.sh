#!/bin/bash

################################################################################
# Build Cleanup Script
#
# This script cleans up failed or incomplete builds to allow fresh rebuilds.
# Use this if you have build errors and want to start over.
#
# Usage:
#   ./scripts/cleanup-build.sh          # Clean all builds
#   ./scripts/cleanup-build.sh cpu      # Clean CPU build only
#   ./scripts/cleanup-build.sh cuda     # Clean CUDA build only
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cleanup_build() {
    local build_type="$1"
    local build_dir="$PROJECT_ROOT/build_${build_type}"

    if [ ! -d "$build_dir" ]; then
        log_info "No $build_type build directory found at: $build_dir"
        return 0
    fi

    log_info "Cleaning $build_type build directory: $build_dir"
    log_info "This may take a moment..."

    # Force remove with retries for stubborn processes
    local retry_count=0
    while [ -d "$build_dir" ] && [ $retry_count -lt 3 ]; do
        if rm -rf "$build_dir" 2>/dev/null; then
            break
        fi
        retry_count=$((retry_count + 1))
        log_warning "Retry $retry_count: Removing files..."
        sleep 1
    done

    if [ ! -d "$build_dir" ]; then
        log_success "Cleaned $build_type build directory"
        return 0
    else
        log_error "Failed to clean $build_type build directory"
        log_error "Directory still exists: $build_dir"
        log_info "Try manually: rm -rf $build_dir"
        return 1
    fi
}

cleanup_all() {
    local failed=0

    echo ""
    log_info "Cleaning all build directories..."
    echo ""

    if ! cleanup_build "cpu"; then
        failed=1
    fi

    if ! cleanup_build "cuda"; then
        failed=1
    fi

    echo ""
    if [ $failed -eq 0 ]; then
        log_success "All build directories cleaned successfully"
        echo ""
        log_info "You can now rebuild with:"
        log_info "  ./scripts/build-gpu-exclusive.sh cpu"
        echo ""
        return 0
    else
        log_error "Some cleanup operations failed"
        return 1
    fi
}

main() {
    if [ $# -eq 0 ]; then
        # No arguments - clean all
        cleanup_all
    elif [ "$1" = "cpu" ]; then
        echo ""
        log_info "Cleaning CPU build..."
        echo ""
        cleanup_build "cpu"
        echo ""
        log_info "Rebuild with: ./scripts/build-gpu-exclusive.sh cpu"
        echo ""
    elif [ "$1" = "cuda" ]; then
        echo ""
        log_info "Cleaning CUDA build..."
        echo ""
        cleanup_build "cuda"
        echo ""
        log_info "Rebuild with: ./scripts/build-gpu-exclusive.sh cuda"
        echo ""
    elif [ "$1" = "help" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        cat << EOF

Build Cleanup Script

USAGE:
    $(basename "$0") [OPTION]

OPTIONS:
    (no argument)    Clean all build directories
    cpu              Clean CPU build only
    cuda             Clean CUDA build only
    help, -h, --help Show this help message

EXAMPLES:
    $(basename "$0")            # Clean all builds
    $(basename "$0") cpu        # Clean CPU build only
    $(basename "$0") cuda       # Clean CUDA build only

NOTES:
    - This removes entire build directories
    - Use when you have build errors and want to start fresh
    - After cleanup, rebuild with: ./scripts/build-gpu-exclusive.sh cpu

EOF
    else
        log_error "Unknown option: $1"
        echo ""
        log_info "Use '$(basename "$0") help' for usage information"
        exit 1
    fi
}

main "$@"
