# Gravitas Code Architecture

## Domain Layers

1. **Config**: Type-safe loading and validation (Zod).
2. **LLM**: Pure HTTP transport layer.
3. **Agents**: Orchestration logic (Coder/Reviewer).
4. **Context**: AST-aware structural intelligence.
5. **Diff/Review**: Deterministic manipulation and validation.
6. **UI**: Minimal, native VS Code elements.

## Flow
Coder (GPU) -> Normalizer -> Reviewer (CPU) -> Validator -> User
