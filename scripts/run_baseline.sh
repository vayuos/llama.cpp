#!/usr/bin/env bash
set -euo pipefail

# Simple helper to build (if needed) and run baseline profiler
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Build optimized if no binary exists
if [ ! -x "build/bin/main" ] && [ ! -x "build/bin/llama" ]; then
  echo "No built binary found — invoking build_optimized.sh"
  ./build_optimized.sh
fi

# Run baseline profiler (adjust server URL and iterations as needed)
PY=$(command -v python3 || command -v python)
$PY profile_baseline.py --server http://localhost:8000 --iterations 50 --output baseline_results.json

echo "Baseline profiling complete — results: baseline_results.json"
