# 🛑 CRITICAL DEVELOPER RULES

> **READ THIS BEFORE RUNNING ANY COMMANDS**

## 1. Node Modules Topology (HIGHEST PRIORITY)

The `gravitas-code` directory **MUST NEVER** contain a physical `node_modules` directory.

### The Rule
*   `gravitas-code/node_modules` **MUST** be a symbolic link.
*   Target: `../support-code/node_modules`

### Implementation
*   **DO NOT** run `npm install` or `bun install` directly in this directory unless you are certain it respects the symlink.
*   **ALWAYS** use `./setup.sh` to manage dependencies. It contains the enforcement logic.
*   **NEVER** commit `node_modules` or `package-lock.json` if they reflect a local installation.

### Remediation
If you accidentally create a local `node_modules` folder:
1. Stop what you are doing.
2. Run: `rm -rf node_modules`
3. Run: `./setup.sh` (This will restore the correct symlink)

---
*This file exists to prevent the recurrence of topology violations. Do not delete logic from setup.sh that enforces this.*

## 2. Runtime Invariants (POLICY LOCK)

The following GPU bindings and execution models are **IMMUTABLE**.

### Agent-GPU Mapping
*   **Coder Agent** → **RTX 4060 Ti (sm_89)** [GPU 0]
    *   *Execution Mode*: Full GPU Offload
*   **Reviewer Agent** → **CPU Only**
    *   *Execution Mode*: CPU Only

### Critical Constraints
1.  **Isolation**: Coder must NEVER see GPU 1. Reviewer must NEVER see GPU 0.
2.  **Hybrid Semantics**: Reviewer "Hybrid" means `nGpuLayers` is limited to fit 2GB VRAM, forcing CPU fallback.
3.  **No Global State**: Never set `CUDA_VISIBLE_DEVICES` globally.

