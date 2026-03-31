# Gravitas Code

**Deterministic Dual-Agent Code Intelligence** (Gravitas Architecture) for VS Code.

> [!CAUTION]
> **DEVELOPERS: READ `DEVELOPER_RULES.md` FIRST.**
> Strict topology rules apply to this codebase. Failure to follow them will break the build environment.

Gravitas Code is a professional-grade AI extension designed for high-integrity code manipulation. It employs a twin-agent architecture (Coder and Reviewer) to ensure all AI-generated code is validated against system invariants before being applied to your workspace.

---

## 🛡️ Architectural Invariants

> [!IMPORTANT]
> **Nothing works until everything is tested and validated.**

1. **Validation Gate**: The Chat UI and agentic pipeline are hard-locked until the system passes a multi-step validation suite (Ports, Files, and Server health).
2. **Deterministic Review**: Every code generation is reviewed by a separate model that must provide strictly parsable JSON feedback.
3. **Hardware Isolation**: Coder and Reviewer can be pinned to specific devices (e.g., Coder on GPU, Reviewer on CPU/different GPU) via configuration.
4. **Configuration Integrity**: Any manual or UI-driven configuration change immediately invalidates the system state, requiring re-validation.

---

## 🚀 Getting Started

### 1. Installation

The extension is packaged as a `.vsix` for local deployment.

**Using CLI:**
```bash
code --install-extension support-code/gravitas-code-0.1.0.vsix
```

**Uninstall:**
```bash
code --uninstall-extension VayuOS.gravitas-code
```

### 2. Dependency Setup

Run the included automated setup script to prepare the environment and handle local `node_modules` relocation for optimization:

```bash
cd gravitas-code
# This script installs Bun (if missing), handles dependencies, and fixes permissions.
sudo ./setup.sh
```

### 3. Initialize & Validate

1. **Setup Wizard**: Open the "Shield" icon in the Activity Bar or run `Gravitas: Validate Setup` to configure binaries and models.
2. **Validation Pipeline**: Run `Gravitas: Validate Setup`. You will see a live trace of the multi-gate safety checks.
3. **Execution**: Once validated, the **Gravitas Controller** (Chat) becomes available for agentic tasks.

---

## ✨ Features

- **Dual-Agent Loop**: Autonomous "Coder" generates solutions; a "Reviewer" verifies logic and format.
- **Setup Wizard**: Intuitive UI for complex infrastructure configuration.
- **Validation Engine**: Real-time port safety checks and model connectivity monitoring.
- **Centralized Logging**: Segregated logs for system, coder, reviewer, and pipeline activities.
- **Process Management**: Integrated control for local LLM server instances.
- **Rich Status Bar**: Real-time visibility into system validation and server status.

---

## 🛠️ Commands

| Command | Description |
|---------|-------------|
| `Gravitas: Validate Setup` | Triggers the safety validation pipeline. |
| `Gravitas: Run Pipeline` | (Validated only) Launches the dual-agent task loop. |
| `Gravitas: Start LLM Server` | Boots the configured local model servers. |
| `Gravitas: Stop LLM Server` | Safely terminates model processes. |

---

## 🧑‍💻 Development

### Setup
```bash
# Sourcing environment (NVM/Bun)
npm install
npm run compile
```

### Extension Structure
- `src/core`: Configuration and state management.
- `src/agents`: Logic for Coder and Reviewer agents.
- `src/validation`: Safety gates and invariant checks.
- `src/ui`: Webview providers for Chat, Setup, and Status.
- `src/process`: Health monitoring and process safety.

---

**Developed by VayuOS**
*Deterministic AI for Mission-Critical Codebases.*
