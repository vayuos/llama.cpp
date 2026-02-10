#!/usr/bin/env python3
"""
audit_memory_copies.py - Audit script for Host<->Device memory copies

Detects potentially unnecessary D2H (Device to Host) and H2D (Host to Device)
copies in the decode loop and related functions.

Searches for:
  - cudaMemcpy calls
  - cudaMemcpyAsync calls
  - ggml_backend_tensor_copy calls
  - ggml_backend_synchronize calls in hot paths
"""

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class MemoryCopyOp:
    """Represents a detected memory copy operation"""
    file_path: str
    line_number: int
    function_name: str
    operation: str  # cudaMemcpy, cudaMemcpyAsync, ggml_backend_tensor_copy, etc.
    direction: str  # H2D, D2H, D2D, H2H, Unknown
    context: str    # Code snippet
    is_async: bool = False
    has_sync: bool = False  # If immediately followed by sync


CUDAMEMCPY_PATTERNS = {
    r"cudaMemcpy[^(]*\([^)]*cudaMemcpyHostToDevice[^)]*\)": ("cudaMemcpy", "H2D", False),
    r"cudaMemcpy[^(]*\([^)]*cudaMemcpyDeviceToHost[^)]*\)": ("cudaMemcpy", "D2H", False),
    r"cudaMemcpy[^(]*\([^)]*cudaMemcpyDeviceToDevice[^)]*\)": ("cudaMemcpy", "D2D", False),
    r"cudaMemcpyAsync[^(]*\([^)]*cudaMemcpyHostToDevice[^)]*\)": ("cudaMemcpyAsync", "H2D", True),
    r"cudaMemcpyAsync[^(]*\([^)]*cudaMemcpyDeviceToHost[^)]*\)": ("cudaMemcpyAsync", "D2H", True),
    r"cudaMemcpyAsync[^(]*\([^)]*cudaMemcpyDeviceToDevice[^)]*\)": ("cudaMemcpyAsync", "D2D", True),
}

GGML_PATTERNS = {
    r"ggml_backend_tensor_copy[^(]*\([^)]*\)": ("ggml_backend_tensor_copy", "Unknown", False),
    r"ggml_backend_synchronize[^(]*\([^)]*\)": ("ggml_backend_synchronize", "Sync", False),
}

SYNC_PATTERNS = {
    r"cudaDeviceSynchronize[^(]*\(\)": "cudaDeviceSynchronize",
    r"cudaStreamSynchronize[^(]*\([^)]*\)": "cudaStreamSynchronize",
    r"ggml_backend_synchronize[^(]*\([^)]*\)": "ggml_backend_synchronize",
}


def parse_function_name(lines: List[str], line_idx: int) -> str:
    """Extract function name from previous context"""
    for i in range(line_idx, max(0, line_idx - 30), -1):
        line = lines[i]
        # Look for function declaration patterns
        if re.search(r'^[a-zA-Z_][a-zA-Z0-9_\*\s]*\([^)]*\)\s*{', line):
            match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)', line)
            if match:
                return match.group(1)
        elif re.search(r'^[a-zA-Z_][a-zA-Z0-9_\*\s]+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)', line):
            match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)', line)
            if match:
                return match.group(1)
    
    return "unknown"


def get_context(lines: List[str], line_idx: int, context_lines: int = 2) -> str:
    """Get code context around a line"""
    start = max(0, line_idx - context_lines)
    end = min(len(lines), line_idx + context_lines + 1)
    context = []
    for i in range(start, end):
        prefix = ">>> " if i == line_idx else "    "
        context.append(f"{prefix}{i+1:4d}: {lines[i][:100]}")
    return "\n".join(context)


def audit_file(file_path: Path, search_functions: List[str] = None) -> List[MemoryCopyOp]:
    """Audit a single file for memory copy operations"""
    
    if not file_path.exists():
        return []
    
    if file_path.suffix not in ['.c', '.cu', '.cpp', '.cuh', '.h', '.hpp']:
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return []
    
    results = []
    content = ''.join(lines)
    
    # Search for cudaMemcpy patterns
    for pattern, (op, direction, is_async) in CUDAMEMCPY_PATTERNS.items():
        for match in re.finditer(pattern, content):
            # Find the line number
            line_idx = content[:match.start()].count('\n')
            
            # Check if there's a sync call nearby
            has_sync = False
            for i in range(line_idx, min(len(lines), line_idx + 5)):
                if re.search(r'(cudaDeviceSynchronize|cudaStreamSynchronize)', lines[i]):
                    has_sync = True
                    break
            
            func_name = parse_function_name(lines, line_idx)
            
            # Filter by function if specified
            if search_functions and func_name not in search_functions:
                continue
            
            results.append(MemoryCopyOp(
                file_path=str(file_path),
                line_number=line_idx + 1,
                function_name=func_name,
                operation=op,
                direction=direction,
                context=get_context(lines, line_idx),
                is_async=is_async,
                has_sync=has_sync,
            ))
    
    # Search for GGML patterns
    for pattern, (op, direction, is_async) in GGML_PATTERNS.items():
        for match in re.finditer(pattern, content):
            line_idx = content[:match.start()].count('\n')
            func_name = parse_function_name(lines, line_idx)
            
            if search_functions and func_name not in search_functions:
                continue
            
            results.append(MemoryCopyOp(
                file_path=str(file_path),
                line_number=line_idx + 1,
                function_name=func_name,
                operation=op,
                direction=direction,
                context=get_context(lines, line_idx),
            ))
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Audit Host<->Device memory copies in llama.cpp"
    )
    parser.add_argument("--root", type=str, default=".",
                       help="Root directory to search")
    parser.add_argument("--functions", type=str, nargs="+",
                       default=["llama_decode", "update_slots", "ggml_cuda_op_mul_mat"],
                       help="Functions to focus on")
    parser.add_argument("--d2h-only", action="store_true",
                       help="Show only D2H transfers")
    parser.add_argument("--h2d-only", action="store_true",
                       help="Show only H2D transfers")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    
    args = parser.parse_args()
    
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"Error: {root_path} does not exist")
        return 1
    
    print("=" * 70)
    print("MEMORY COPY AUDIT - Host<->Device Transfer Analysis")
    print("=" * 70)
    print(f"Root directory: {root_path}")
    print(f"Target functions: {args.functions}")
    print()
    
    all_results = []
    
    # Search for relevant source files
    search_patterns = [
        "src/**/*.cpp",
        "src/**/*.h",
        "tools/server/**/*.cpp",
        "ggml/src/**/*.cu",
        "ggml/src/**/*.cuh",
        "ggml/src/**/*.cpp",
    ]
    
    files_checked = 0
    for pattern in search_patterns:
        for file_path in root_path.glob(pattern):
            files_checked += 1
            results = audit_file(file_path, search_functions=args.functions)
            all_results.extend(results)
    
    # Filter results
    filtered_results = all_results
    if args.d2h_only:
        filtered_results = [r for r in filtered_results if r.direction == "D2H"]
    if args.h2d_only:
        filtered_results = [r for r in filtered_results if r.direction == "H2D"]
    
    print(f"Scanned {files_checked} files\n")
    
    if not filtered_results:
        print("✓ No memory copy operations detected in target functions")
        return 0
    
    # Print results
    d2h_count = sum(1 for r in filtered_results if r.direction == "D2H")
    h2d_count = sum(1 for r in filtered_results if r.direction == "H2D")
    
    print(f"Found {len(filtered_results)} memory copy operations:")
    print(f"  D2H (Device → Host): {d2h_count} [⚠️  CHECK NECESSITY]")
    print(f"  H2D (Host → Device): {h2d_count}")
    print()
    
    # Group by operation type
    by_type: Dict[str, List[MemoryCopyOp]] = {}
    for result in filtered_results:
        key = f"{result.operation}_{result.direction}"
        if key not in by_type:
            by_type[key] = []
        by_type[key].append(result)
    
    for op_type, ops in sorted(by_type.items()):
        print(f"\n{'='*70}")
        print(f"{op_type} - {len(ops)} occurrence(s)")
        print(f"{'='*70}")
        
        for op in ops:
            print(f"\n📍 {op.file_path}:{op.line_number}")
            print(f"   Function: {op.function_name}")
            print(f"   Operation: {op.operation} ({'async' if op.is_async else 'sync'})")
            if op.has_sync:
                print(f"   ⚠️  BLOCKING SYNCHRONIZATION DETECTED")
            print(f"\n{op.context}\n")
    
    # Summary and recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    
    critical_issues = []
    
    if d2h_count > 0:
        critical_issues.append(
            f"• {d2h_count} D2H transfer(s) detected. Verify if logits/embeddings "
            "truly need CPU access or if GPU-side sampling can be used."
        )
    
    async_count = sum(1 for r in filtered_results if r.is_async and not r.has_sync)
    if async_count > 0:
        critical_issues.append(
            f"• {async_count} async transfer(s) without explicit sync. "
            "Ensure proper event-based synchronization is used."
        )
    
    sync_count = sum(1 for r in filtered_results if r.has_sync)
    if sync_count > 0:
        critical_issues.append(
            f"• {sync_count} transfer(s) with device synchronization. "
            "Replace with stream events or pipeline-level sync."
        )
    
    if critical_issues:
        print("\nCRITICAL FINDINGS:")
        for issue in critical_issues:
            print(issue)
    else:
        print("\n✓ No critical issues detected")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
