# Working Memory Snapshot (2026-03-05)

## Current repo state
- Branch: main
- HEAD: 2b2e428 (codex wrkspace)
- Clean working tree: no uncommitted changes.

## Most recent commits
1. 2b2e428 (2026-03-05): codex workspace marker file.
2. 820a527 (2026-03-05): AntiGravity local project setup.
3. 96c6bd3 (2026-03-05): claude workspace setup.
4. 88b754e (2026-03-05): pulled latest origin/main.
5. 7f43f68 (2026-03-01): implementation details docs/scripts.

## Last substantial technical work (Phase 3B/3C)
- ae25227: Fix CUDA_Host KV cache placement for CPU-resident layers.
  - Files: src/llama-kv-cache.cpp
- caaec8d: Disable Phase 3B/3C after measured perf regression.
  - Files: src/llama-context.cpp, src/llama-kv-cache.cpp, test_batch_sizes.sh, test_ids_pointer.patch
- bd54603: Add Phase 3 CLI/server parameters.
  - Files: common/arg.cpp, common/common.cpp, common/common.h
- 7f43f68: Added human-readable status/run docs.
  - Files: PHASE3B-SUMMARY.txt, QUICK-START.txt, test-phase3b-performance.sh

## Recent artifacts worth reopening
- PHASE3B-SUMMARY.txt
- QUICK-START.txt
- test-phase3b-performance.sh
- Logs: test_cmd1_baseline.log ... test_cmd5_full_opt_4096.log

## Suggested resume point
Start from PHASE3B-SUMMARY.txt, then rerun test-phase3b-performance.sh and compare against test_cmd*.log baselines before re-enabling any Phase 3B/3C path.