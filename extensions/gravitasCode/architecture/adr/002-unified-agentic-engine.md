# ADR 002: Unified Agentic engine

## Status
Accepted

## Context
Prior to this ADR, the Gravitas codebase had disparate implementations of the autonomous coding/review loop (e.g., in `pipelineRun.ts` and `loop.ts`). This led to inconsistent telemetry, sampling drift, and maintenance overhead.

## Decision
We will unify all autonomous agentic logic into a single, high-fidelity `AgentLoopController`. All user objectives (Manual Spawn, Autonomous Pipeline) must delegate to this controller.

## Consequences
- **Positive**: Single point of truth for agentic logic.
- **Positive**: Consistent telemetry across all execution modes.
- **Positive**: Simplified enforcement of sampling parameters (Temperature, Top-P).
- **Negative**: Increased complexity in the initial `AgentLoopController` implementation to handle diverse task origins.
