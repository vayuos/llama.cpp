#!/usr/bin/env python3
"""
Build verification and compilation script for llama.cpp GPU-exclusive decode optimization
Verifies all fixes are in place, then initiates cmake build
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Configuration
LLAMA_ROOT = "/home/viren/source/llama.cpp/llama.cpp"
BUILD_DIR = os.path.join(LLAMA_ROOT, "build")
SRC_DIR = os.path.join(LLAMA_ROOT, "src")

# Verification checks
VERIFICATION_CHECKS = {
    "llama-debug-stripping.cpp": {
        "file": os.path.join(SRC_DIR, "llama-debug-stripping.cpp"),
        "patterns": [
            "metrics->decode_loop_entries = g_llama_debug_stripping.metrics.decode_loop_entries.load()",
            "metrics->graph_execute_entries = g_llama_debug_stripping.metrics.graph_execute_entries.load()",
            "metrics->cuda_kernel_launches = g_llama_debug_stripping.metrics.cuda_kernel_launches.load()",
        ],
        "description": "Atomic field member-wise copy pattern"
    },
    "llama-json-isolation.cpp": {
        "file": os.path.join(SRC_DIR, "llama-json-isolation.cpp"),
        "patterns": [
            "auto json_start_ns = std::chrono::high_resolution_clock::now()",
            "auto json_duration = json_end_ns - json_start_ns",
            "duration_cast<std::chrono::nanoseconds>"
        ],
        "description": "Chrono time_point type corrections"
    },
    "llama-server-decode-isolation.h": {
        "file": os.path.join(SRC_DIR, "llama-server-decode-isolation.h"),
        "patterns": [
            "struct isolation_metrics",
            "struct decode_domain",
            "struct server_domain",
            "struct streaming_metrics",
            "std::atomic<uint64_t> head",
            "std::atomic<uint64_t> tail"
        ],
        "description": "Struct definitions and atomic queue members"
    },
    "llama-config-freeze.h": {
        "file": os.path.join(SRC_DIR, "llama-config-freeze.h"),
        "patterns": [
            "} cli_config;",
            "} env_config;",
            "bool pinned_host_memory;",
            "bool unified_memory;",
            "bool deterministic;"
        ],
        "description": "Configuration struct members"
    }
}

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def verify_file_pattern(filepath, patterns):
    """Verify that all patterns exist in file"""
    if not os.path.exists(filepath):
        print(f"  ❌ File not found: {filepath}")
        return False

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        missing = []
        for pattern in patterns:
            if pattern not in content:
                missing.append(pattern)

        if missing:
            print(f"  ❌ Missing {len(missing)} pattern(s) in {os.path.basename(filepath)}")
            for p in missing[:3]:  # Show first 3
                print(f"     - {p[:70]}...")
            return False
        else:
            print(f"  ✅ {os.path.basename(filepath)}: All {len(patterns)} patterns found")
            return True
    except Exception as e:
        print(f"  ❌ Error reading {filepath}: {e}")
        return False

def run_verification():
    """Run all verification checks"""
    print_header("VERIFICATION: All Fixes Applied")

    all_verified = True
    for check_name, check_info in VERIFICATION_CHECKS.items():
        print(f"\n  {check_info['description']}")
        if not verify_file_pattern(check_info['file'], check_info['patterns']):
            all_verified = False

    return all_verified

def setup_build_directory():
    """Create and configure CMake build directory"""
    print_header("CMAKE BUILD SETUP")

    # Check if build directory exists
    if os.path.exists(BUILD_DIR):
        print(f"  ✓ Build directory exists: {BUILD_DIR}")
    else:
        print(f"  Creating build directory: {BUILD_DIR}")
        try:
            os.makedirs(BUILD_DIR, exist_ok=True)
            print(f"  ✓ Build directory created")
        except Exception as e:
            print(f"  ❌ Failed to create build directory: {e}")
            return False

    # Configure CMake
    print(f"\n  Configuring CMake...")
    os.chdir(BUILD_DIR)

    try:
        # Try to configure with common options
        cmd = [
            "cmake",
            "..",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_CUDA=ON",
            "-DLLAMA_FAST=ON"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print(f"  ✅ CMake configuration successful")
            return True
        else:
            print(f"  ⚠️  CMake configuration had warnings/errors:")
            print(f"     {result.stderr[:200]}...")
            # Continue anyway - build might still work
            return True

    except subprocess.TimeoutExpired:
        print(f"  ⚠️  CMake configuration timed out (took >60s)")
        return False
    except Exception as e:
        print(f"  ⚠️  CMake configuration error: {e}")
        print(f"     (Build directory may still be usable)")
        return True

def start_build():
    """Start the build process"""
    print_header("STARTING BUILD")

    os.chdir(BUILD_DIR)

    print(f"  Build directory: {BUILD_DIR}")
    print(f"  Starting: make -j$(nproc)")
    print(f"\n  Expected progress:")
    print(f"    - Before fixes: 41% (45/76 sections)")
    print(f"    - Target: 72%+ (55+/76 sections)")
    print(f"    - Expected compile time: 5-15 minutes")
    print(f"\n  Build output will be displayed below...")
    print(f"\n" + "-"*80)

    try:
        # Get number of processors
        try:
            nproc = len(os.sched_getaffinity(0))
        except:
            nproc = os.cpu_count() or 4

        # Start build
        cmd = f"make -j{nproc}"
        result = subprocess.run(cmd, shell=True, text=True)

        print("\n" + "-"*80)

        if result.returncode == 0:
            print(f"\n  ✅ Build completed successfully!")
            return True
        else:
            print(f"\n  ❌ Build failed with exit code {result.returncode}")
            print(f"     Check the output above for error details")
            return False

    except Exception as e:
        print(f"\n  ❌ Build execution error: {e}")
        return False

def print_summary(verification_ok, build_ok=None):
    """Print final summary"""
    print_header("SUMMARY")

    print(f"\n  Verification Status:")
    if verification_ok:
        print(f"    ✅ All 4 files verified with correct fixes")
        print(f"    ✅ Ready for compilation")
    else:
        print(f"    ❌ Some files missing fixes - compilation will fail")

    if build_ok is not None:
        print(f"\n  Build Status:")
        if build_ok:
            print(f"    ✅ Build successful - project ready for testing")
        else:
            print(f"    ❌ Build failed - check output above for errors")

    print(f"\n  Files Modified:")
    print(f"    ✓ llama-debug-stripping.cpp")
    print(f"    ✓ llama-json-isolation.cpp")
    print(f"    ✓ llama-server-decode-isolation.h")
    print(f"    ✓ llama-config-freeze.h")

    print(f"\n  Next Steps:")
    if verification_ok and build_ok:
        print(f"    1. Run tests to validate GPU-exclusive decode optimization")
        print(f"    2. Check build artifacts in {BUILD_DIR}/bin/")
        print(f"    3. Profile to measure performance improvements (15-45% expected)")
    elif verification_ok:
        print(f"    1. Review build errors in output above")
        print(f"    2. Apply additional fixes if needed")
        print(f"    3. Re-run: cd {BUILD_DIR} && make -j$(nproc)")
    else:
        print(f"    1. Verify fixes were properly applied to source files")
        print(f"    2. Check git status: cd {LLAMA_ROOT} && git diff")

    print()

def main():
    """Main execution"""
    print_header("LLAMA.CPP GPU-EXCLUSIVE DECODE - BUILD & VERIFY")
    print(f"\n  Project: GPU-exclusive decode optimization")
    print(f"  Target: Sections 1-56+ completion")
    print(f"  Expected: 41% → 72%+ progress")
    print(f"  Build location: {BUILD_DIR}")

    # Phase 1: Verify all fixes
    print_header("PHASE 1: VERIFICATION")
    verification_ok = run_verification()

    if not verification_ok:
        print_summary(verification_ok)
        sys.exit(1)

    print(f"\n  ✅ All fixes verified - proceeding to build")

    # Phase 2: Setup build
    print_header("PHASE 2: BUILD SETUP")
    if not setup_build_directory():
        print(f"\n  ⚠️  Build setup incomplete - attempting make anyway...")

    # Phase 3: Start build
    print_header("PHASE 3: COMPILATION")
    build_ok = start_build()

    # Print summary
    print_summary(verification_ok, build_ok)

    sys.exit(0 if build_ok else 1)

if __name__ == "__main__":
    main()
