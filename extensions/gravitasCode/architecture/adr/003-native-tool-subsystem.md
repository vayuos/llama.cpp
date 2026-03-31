# ADR 003: Native Tool Subsystem

## Status
Accepted

## Context
Gravitas initially relied on shell command spawning for all tools (e.g., `ls`, `grep`). This made high-fidelity analytical tools (like codebase stats) difficult to implement, as they would require complex shell parsing or external script dependencies.

## Decision
We will establish a modular `src/tools/` subsystem using a central `NativeToolRegistry`. This allows for in-process TypeScript tool execution with identical telemetry to shell-level commands.

## Consequences
- **Positive**: Native TypeScript-based analytical tools (e.g., `CodebaseStatTool`).
- **Positive**: Simplified, high-fidelity tool execution with structured telemetry.
- **Positive**: Modular, extensible tool registration.
- **Negative**: Increased complexity of the initial `ToolWrapper` logic to handle dual-mode dispatch.
