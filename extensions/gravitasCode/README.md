# Gravitas Code: Deterministic AI Intelligence

**Gravitas Code** is a production-grade VS Code extension engineered for high-integrity systems development. It operates on a **Deterministic Dual-Agent Architecture** (Coder + Reviewer), ensuring that every AI-generated modification is validated against strict system invariants before integration.

---

## 🛡️ Architectural Invariants (Laws of Gravity)

* **Validation Gate**
  The agent pipeline remains locked until a complete validation suite (Ports, Files, Server Health) succeeds.

* **Deterministic Review**
  All generated outputs are audited by a Reviewer model that must return strictly structured JSON.

* **Hardware Isolation**
  Agents can be bound to specific compute resources (e.g., Coder → GPU, Reviewer → CPU).

* **State Integrity**
  Any configuration change invalidates the system state and forces re-validation.

---

## 🚀 Setup & Initialization

### Installation

```bash
code --install-extension support-code/gravitas-code-0.1.0.vsix
code --uninstall-extension VayuOS.gravitas-code
```

### Environment Setup

```bash
cd gravitas-code
sudo ./setup.sh
```

This script installs dependencies, configures Bun, and ensures proper permissions.

---

### Validation Pipeline

1. Launch Setup Wizard (Activity Bar → Shield icon)
2. Run: `Gravitas: Validate Setup`
3. Upon success, the **Gravitas Controller** unlocks

---

## ✨ Core Capabilities

* Dual-agent generation + validation loop
* Real-time validation engine (ports, models, VRAM)
* Structured logging (system, agents, pipeline)
* Local process lifecycle control
* Live system status via UI

---

## 🛠️ Command Interface

| Command          | Description                   |
| ---------------- | ----------------------------- |
| Validate Setup   | Runs full validation pipeline |
| Run Pipeline     | Starts dual-agent execution   |
| Start LLM Server | Launches llama.cpp/Ollama     |
| Stop LLM Server  | Gracefully stops processes    |

---

## 🧑‍💻 System Structure

* `src/core` → configuration + state
* `src/agents` → coder/reviewer logic
* `src/validation` → invariant enforcement
* `src/ui` → chat + dashboard
* `src/process` → execution + monitoring

---

# Gravitas Chat System (Execution Interface)

## 1. Intelligent Composer

A structured, context-aware input system:

* **@mentions**

  * `@coder` → execution agent
  * `@reviewer` → validation agent
  * `@terminal` → shell execution

* **#references**

  * `#file`, `#git`, `#docs`

* **Multimodal input**

  * Vision-based analysis (UI bugs, diagrams)

---

## 2. Execution Surface

* Step-based progress tracking
* Interactive diffs (apply/reject)
* Embedded terminal output

---

## 3. Intelligence Layer

* Reasoning transparency (thought traces)
* Confidence scoring
* Predictive context preloading

---

## 4. Telemetry & Monitoring

* VRAM usage tracking
* Quantization visibility
* Tokens/sec monitoring

---

## 5. Safety & Control

* Autonomous write toggle
* Checkpoint-based rollback
* Chat branching for parallel strategies

---

## Comparison

| Feature      | Standard Chat | Gravitas              |
| ------------ | ------------- | --------------------- |
| Goal         | Answer        | Execute               |
| Context      | Prompt-only   | Full system           |
| Output       | Text          | Interactive artifacts |
| Verification | Manual        | Automated             |
| Autonomy     | None          | Full pipeline         |

---

# Checkpointed Context Loop (CCL)

## Problem

LLMs have:

* Fixed context window
* Performance degradation near limit
* Hard truncation

---

## Solution

A **deterministic context lifecycle system**:

1. Monitor context usage
2. Extract structured state
3. Compress context
4. Restart session
5. Rehydrate state
6. Continue execution

---

## Context Trigger

```text
trigger_at = 70–80% of total context
```

---

## State Model

### Task State

* Objective
* Phase
* Pending actions

### Code State

* Files modified
* Diffs
* Symbols

### Reasoning State

* Decisions
* Constraints
* Errors

---

## Checkpoint Format

```json
{
  "task": "Optimize llama.cpp throughput",
  "environment": {
    "model": "Qwen3.5-9B-Q4_K_M",
    "hardware": "RTX 4060 Ti"
  },
  "progress": [
    "Baseline measured",
    "Threads tuned"
  ],
  "decisions": [
    "Use GPU offloading"
  ],
  "pending": [
    "Tune batch size"
  ]
}
```

---

## Execution Flow

```text
Plan → Execute → Monitor
        ↓
Checkpoint → Compress → Save
        ↓
Restart → Rehydrate → Continue
```

---

## Optimization Strategies

* Hierarchical checkpoints
* Delta compression
* Symbol-priority retention
* Persistent storage (`~/gravitas/state/`)
* Crash recovery

---

## Design Constraints

* Deterministic structure
* No redundant reasoning
* Strict token budgeting

---

# Full Gravitas Feature Surface

## Intelligence

* Dual-agent reasoning
* Deterministic execution
* Constraint-aware logic
* Dependency graph awareness

---

## System Integration

* Full filesystem access
* Shell execution
* Process control
* Hardware telemetry

---

## Validation

* Multi-stage safety gates
* Pre/post execution checks
* Rollback mechanisms
* JSON validation contracts

---

## Interaction

* Pulse pattern (Signal → Execution → Artifact → Conclusion)
* Challenge system (defensive validation)

---

## Performance

* Prompt caching
* KV cache optimization
* Context compression
* Quantization awareness

---

## Developer Experience

* Live logs
* Diff-based edits
* Error diagnostics
* Root cause analysis

---

## UI/UX

* Setup wizard
* Status dashboard
* Agent monitor
* Artifact rendering

---

## Toolchain Integration

* Compiler (gcc/clang)
* Debugger (gdb/lldb)
* System tools
* Asset pipeline
* Build systems

---

## Observability

* Inference metrics
* Resource tracking
* Reasoning trace
* Validation signatures

---

## Advanced Capabilities

* Stateful memory
* Task orchestration
* Parallel agents
* Context hot reload
* Hardware-aware scheduling
* Failure recovery

---

# Reality Constraint

A chat interface alone cannot execute this system.

Required components:

* Local runtime (**llama.cpp**)
* Gravitas VS Code extension
* Shell + toolchain integration
* Validation engine

---

# llama.cpp Required Modifications

## 1. Context Metrics API

```cpp
struct llama_context_metrics {
    int32_t n_ctx_total;
    int32_t n_ctx_used;
    float kv_cache_utilization;
};
```

```cpp
llama_context_metrics llama_get_context_metrics(llama_context * ctx);
```

---

## 2. Checkpoint Interface

```cpp
typedef std::string (*llama_checkpoint_cb)(void *);
```

```cpp
void llama_set_checkpoint_callback(...);
```

---

## 3. Session Persistence

```cpp
bool llama_save_session_full(...);
bool llama_load_session_full(...);
```

---

## 4. Context Reset

```cpp
void llama_reset_context(llama_context * ctx);
```

---

## 5. State Injection

```cpp
void llama_inject_system_state(...);
```

---

## 6. Server Extensions

* `/metrics`
* `/checkpoint`
* `/restart`
* `/resume`

---

## 7. Context Guardrail

```cpp
if (n_ctx_used > threshold) return LLAMA_CONTEXT_NEAR_LIMIT;
```

---

## 8. Token Typing

```cpp
enum llama_token_type {
    USER,
    SYSTEM,
    CHECKPOINT
};
```

---

## 9. Hierarchical Caching

* Base prompt
* Checkpoint layer
* Active context

---

## 10. Thread Safety

```cpp
std::mutex ctx_lifecycle_mutex;
```

---

# Final Architecture

```text
Gravitas (Intelligence Layer)
        ↓
Checkpoint Engine
        ↓
llama.cpp (Execution Layer)
        ↓
Hardware (GPU/CPU)
```

---

## Final Insight

* llama.cpp → execution engine
* Gravitas → reasoning + control layer

Only together do they form a **deterministic, long-context, system-aware AI pipeline**.

---

**Developed by VayuOS — Deterministic AI for Mission-Critical Codebases** 
