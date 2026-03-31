# Security Policy

## Threat Model: Stateless Worker (LLM)
We treat LLMs as untrusted, stateless workers that might attempt prompt injection or generate malicious code.

### Mitigations
- **Deterministic Validation**: LLMs are never the final authority. The Reviewer enforces a strict JSON schema and deterministic rules.
- **Role Locking**: Reviewer cannot generate code; Coder cannot provide prose.
- **Local Containment**: All agents run behind local HTTP boundaries (`llama.cpp`). No outbound network access is required or allowed for the model logic.

## Supply Chain Integrity
- **Provenance**: Every patch application is logged and can be audited against the Reviewer's JSON report.
- **No Remote Dependencies**: The extension relies solely on local compute and forked, audited base projects.
