# ADR-0001: Align Extension Ports with System Scripts

**Status**: Accepted

**Date**: 2026-03-30

## Context

The `gravitas-code` extension had default ports (`8089`, `18080`) that were inconsistent with the provided system LLM server scripts (`coder_gpu_qwen3.sh` at `8010` and `reviewer_cpu_deepseek.sh` at `8011`). Additionally, a property access bug in `src/agents/loop.ts` prevent the agentic loop from connecting to the configured ports.

To ensure "Mission-Critical" integrity and out-of-the-box connectivity, the extension configuration must align with the deployment infrastructure.

## Decision

1.  **Corrected property access** in `src/agents/loop.ts` from `config.coderModel.port` to `config.coder.port`.
2.  **Updated default ports** in `src/core/config.ts` from `8089`/`18080` to `8010` (Coder) and `8011` (Reviewer).
3.  **Updated workspace settings** in `.vscode/settings.json` to reflect the new port alignment.

## Consequences

*   **Positive**: Improved reliability and reduced configuration friction for new developers.
*   **Positive**: Code alignment with the VayuForge System 4 (Owner Development) hardware spec.
*   **Negative**: Users previously using `8089`/`18080` will need to update their server ports or manually override the new extension defaults.
*   **Risks**: If external server ports change again, the extension defaults will once again become stale.
