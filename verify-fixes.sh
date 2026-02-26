#!/bin/bash

# Verification script to confirm all fixes are applied
# Run from: /home/viren/llama/llama.cpp
# Usage: ./verify-fixes.sh

echo "=========================================="
echo "GPU Optimization Fixes - Verification"
echo "=========================================="
echo ""

PASS=0
FAIL=0

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if pattern exists in file
check_fix() {
    local issue=$1
    local file=$2
    local pattern=$3
    local description=$4

    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Issue #$issue: $description"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} Issue #$issue: $description"
        echo "    File: $file"
        echo "    Pattern: $pattern"
        ((FAIL++))
    fi
}

echo "--- Code Fixes ---"
echo ""

# Issue #10 Fix A: Argsort padding initialization
check_fix "10a" "ggml/src/ggml-cuda/argsort.cu" \
    "ISSUE #10 FIX: Initialize padding indices to -1" \
    "Argsort padding initialization"

# Issue #10 Fix B: Bitonic sort padding handling
check_fix "10b" "ggml/src/ggml-cuda/argsort.cu" \
    "bool col_invalid" \
    "Bitonic sort padding handling"

# Issue #10 Fix C: Expert index clamping
check_fix "10c" "src/llama-graph.cpp" \
    "ggml_clamp.*selected_experts.*n_expert" \
    "Expert index clamping"

# Issue #10 Fix D: Expert validation error handling
check_fix "10d" "ggml/src/ggml-backend.cpp" \
    "GGML_LOG_ERROR.*Invalid expert ID" \
    "Expert validation with error messages"

# Issue #11 Fix: Buffer accounting logging
check_fix "11" "src/llama-context.cpp" \
    "ISSUE #11 FIX.*buffer accounting" \
    "Buffer accounting debug logging"

# Issue #3 Fix: Tensor placement (from previous session)
check_fix "3" "src/llama-model.cpp" \
    "ISSUE #3 FIX.*tensor placement" \
    "Tensor placement GPU preservation"

# Issue #6 Fix: Memory underflow (should already be there)
check_fix "6" "src/llama-context.cpp" \
    "total >= self.*free.*total - self - free : 0" \
    "Memory accounting underflow prevention"

echo ""
echo "--- Build Scripts ---"
echo ""

# Check if build scripts exist
if [ -f "scripts/build_cuda_cublas_dense_debug.sh" ]; then
    echo -e "${GREEN}✓${NC} Clean build script exists"
    ((PASS++))
else
    echo -e "${RED}✗${NC} Clean build script missing"
    ((FAIL++))
fi

if [ -f "scripts/build_cuda_cublas_dense_debug_inc.sh" ]; then
    echo -e "${GREEN}✓${NC} Incremental build script exists"
    ((PASS++))
else
    echo -e "${RED}✗${NC} Incremental build script missing"
    ((FAIL++))
fi

# Check if build scripts are executable
if [ -x "scripts/build_cuda_cublas_dense_debug.sh" ]; then
    echo -e "${GREEN}✓${NC} Clean build script is executable"
    ((PASS++))
else
    echo -e "${YELLOW}⚠${NC} Clean build script not executable"
    echo "    Run: chmod +x scripts/build_cuda_cublas_dense_debug.sh"
fi

if [ -x "scripts/build_cuda_cublas_dense_debug_inc.sh" ]; then
    echo -e "${GREEN}✓${NC} Incremental build script is executable"
    ((PASS++))
else
    echo -e "${YELLOW}⚠${NC} Incremental build script not executable"
    echo "    Run: chmod +x scripts/build_cuda_cublas_dense_debug_inc.sh"
fi

echo ""
echo "--- Documentation Files ---"
echo ""

docs=(
    "BUILD-ALL-FIXES.md:Master build guide"
    "ALL-FIXES-APPLIED.md:Complete fixes summary"
    "QUICK-START.md:Quick reference"
    "COMPILATION-STATUS-REPORT.md:Detailed status"
    "ISSUE-3-FIX-APPLIED.md:Tensor placement details"
)

for doc_entry in "${docs[@]}"; do
    doc="${doc_entry%:*}"
    desc="${doc_entry#*:}"

    if [ -f "$doc" ]; then
        echo -e "${GREEN}✓${NC} $doc"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $doc missing: $desc"
        ((FAIL++))
    fi
done

echo ""
echo "=========================================="
echo "Verification Results"
echo "=========================================="

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All fixes verified!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run build: ./scripts/build_cuda_cublas_dense_debug_inc.sh"
    echo "2. After build succeeds, verify:"
    echo "   ./build_cuda_mmq_moe_full_logs/bin/llama-server -m model.gguf -ngl 999 --no-mmap 2>&1 | head -30"
    echo "3. Check for 'offloaded 48/49 layers to GPU'"
    exit 0
else
    echo -e "${RED}✗ Some fixes are missing!${NC}"
    echo ""
    echo "Checks passed: $PASS"
    echo "Checks failed: $FAIL"
    echo ""
    echo "Please review the failures above and apply missing fixes."
    exit 1
fi
