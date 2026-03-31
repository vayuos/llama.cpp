# Gravitas Code

Deterministic Dual-Agent Code Intelligence (Antigravity Architecture) for VS Code.

## Architectural Invariants

> [!IMPORTANT]
> **Nothing works until everything is tested and validated.**

1. **Validation Gate**: The Chat UI and agentic pipeline are hard-locked until the system passes a multi-step validation suite.
2. **Deterministic Review**: The Reviewer agent must provide strictly parsable JSON output.
3. **No Auto-Start**: LLM servers never start automatically. They are explicitly controlled.
4. **Configuration Integrity**: Any manual or UI-driven configuration change immediately invalidates the system state, requiring re-validation.

## Getting Started

1. **Setup**: Run `Gravitas: Start Setup` to configure your LLM infrastructure (binaries, models, networking).
2. **Validate**: Run `Gravitas: Validate Setup` and follow the live validation pipeline.
3. **Control**: Use the Status Bar and Logs Panel to monitor system health.

## Development

```bash
npm run compile
# Launch VS Code with the extension
```
