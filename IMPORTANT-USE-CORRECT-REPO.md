# ⚠️ IMPORTANT: Use the Correct Repository

## Current Situation

There are **TWO llama.cpp directories**:

1. **Git Repository (REAL SOURCE)** ✅
   ```
   /home/viren/llama/llama.cpp/
   ```
   - This is the AUTHORITATIVE source
   - Contains all commits and fixes
   - All build scripts are here
   - USE THIS ONE

2. **Mirror/Development Directory** (OUTDATED)
   ```
   /home/viren/source/llama.cpp/
   ```
   - Old mirror copy
   - Does NOT have latest fixes
   - Has broken cached builds
   - DO NOT USE THIS

## Problem You Were Facing

The build was running from `/home/viren/source/llama.cpp` instead of the git repo. This is why:
- Cleanup didn't help (it was in the wrong directory)
- CUDA linker errors persisted
- Fixes weren't being picked up

## Solution: Use the Git Repository

**All builds MUST run from:**
```bash
cd /home/viren/llama/llama.cpp
```

**Then build:**
```bash
./scripts/build-gpu-exclusive.sh cpu
```

## Verify You're in the Right Place

```bash
# Check you're in the git repo
pwd  # Should show: /home/viren/llama/llama.cpp

# Verify git is available
git status  # Should show git info

# Check build scripts exist
ls scripts/build-gpu-exclusive.sh  # Should exist

# Check latest fixes are present
ls BUILD-STATUS.md FIX-CUDA-LINKER-ERRORS.md  # Should both exist
```

## What to Do Now

1. **Change to git repository:**
   ```bash
   cd /home/viren/llama/llama.cpp
   ```

2. **Verify you're in the right place:**
   ```bash
   pwd
   ls BUILD-STATUS.md
   ```

3. **Build:**
   ```bash
   ./scripts/build-gpu-exclusive.sh cpu
   ```

## Why Two Directories?

- `/home/viren/llama/llama.cpp/` - Git repository (synced, version controlled)
- `/home/viren/source/llama.cpp/` - Development mirror (deprecated, out of sync)

The mirror was likely created for convenience but is now outdated and should not be used.

## Key Differences

| Feature | Git Repo | Mirror |
|---------|----------|--------|
| Latest commits | ✅ Yes | ❌ Outdated |
| Build fixes | ✅ All included | ❌ Missing |
| Build scripts | ✅ Present | ❌ Old versions |
| CMake config | ✅ Clean | ❌ Broken CUDA |
| Recommended | ✅ YES | ❌ NO |

## Summary

- **CORRECT**: `cd /home/viren/llama/llama.cpp && ./scripts/build-gpu-exclusive.sh cpu`
- **WRONG**: `cd /home/viren/source/llama.cpp && ./scripts/build-gpu-exclusive.sh cpu`

Always use the git repository at `/home/viren/llama/llama.cpp/`.

---

**Remember**: All future builds should use:
```bash
cd /home/viren/llama/llama.cpp
./scripts/build-gpu-exclusive.sh cpu
```

This is your single source of truth for the project.
