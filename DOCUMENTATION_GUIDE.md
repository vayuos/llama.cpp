# Documentation Guide for GPU-Exclusive Decode Optimization

## Core Documentation Files

This project uses a streamlined documentation structure:

### 1. **systemchanges.md** (Primary Reference)
- Comprehensive documentation of Sections 1-9 implementation
- Contains objective definition, requirement analysis, implementation details
- Updated after each section completion
- Location: `./llama.cpp/systemchanges.md`

### 2. **CHANGES.md** (Implementation Status)
- Detailed status tracking for all 76 sections
- Per-section implementation details and metrics
- Summary statistics and completion progress
- Location: `./CHANGES.md` (in project root)

### 3. **CLAUDE.md** (Project Instructions)
- Quick reference for AI assistant instructions
- Location: `./llama.cpp/CLAUDE.md`

### 4. **Standard Project Files**
- `README.md` - Original llama.cpp project information
- `CONTRIBUTING.md` - Contribution guidelines
- `SECURITY.md` - Security information

## Legacy Files Removed

The following documentation files have been consolidated into systemchanges.md and removed to reduce clutter:

- All GPU_TOPK_*.md files (GPU Top-K implementation)
- All DECODE_BOUNDARY_*.md files (Previous decode boundary work)
- DECODE_ENFORCEMENT_VALIDATION.md
- DECODE_PATH_ISOLATION.md
- BACKEND_PURITY.md
- CUBLAS_PREVENTION.md
- DEBUG_STRIPPING.md
- PROBING_ELIMINATION.md
- STREAMING_ASYNC_DECOUPLING.md
- THREADING_DISCIPLINE_ENFORCEMENT.md
- KERNEL_FUSION_ENFORCEMENT.md
- MMQ_ENFORCEMENT.md
- TOPOLOGY_FREEZE_ENFORCEMENT.md
- GPU_SAMPLING_INTEGRATION.md
- REQUIREMENT_*.md files (old requirements)
- AGENTS.md
- IMPLEMENTATION_COMPLETE.md
- OPTIMIZATION_SUMMARY.md
- SystemAddOnChanges.md
- README_TOPK.md
- INTEGRATION_GUIDE_DEBUG_STRIPPING.md

## Active Implementation Files

All implementation for Sections 1-9 resides in:

```
./llama.cpp/src/
  ├── llama-decode-invariant-enforce.{h,cpp}          [Section 1]
  ├── llama-task-taxonomy.{h,cpp}                     [Section 2]
  ├── llama-decode-admission-control.{h,cpp}          [Section 3]
  ├── llama-decode-cpu-hard-failure.{h,cpp}           [Section 4]
  ├── llama-token-dependency-assert.{h,cpp}           [Section 5]
  ├── llama-backend-immutability-enforce.{h,cpp}      [Section 6]
  ├── llama-graph-backend-binding.{h,cpp}             [Section 7]
  ├── llama-fallback-elimination.{h,cpp}              [Section 8]
  ├── llama-cuda-support-enforce.{h,cpp}              [Section 9]
  └── [Future: Sections 10-76]
```

## Next Steps

- Section 10: Add decode-time backend lock
- Sections 11-15: Graph lifetime & execution control
- Sections 16-20: CPU control-path removal
- Sections 21-76: Remaining optimizations

## Contact

Refer to CLAUDE.md for AI assistant instructions and context.
