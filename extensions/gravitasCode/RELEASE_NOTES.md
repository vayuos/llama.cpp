# Gravitas v0.1.0 - Final Release

**Release Date**: 2026-02-06  
**Package**: `gravitas-code-0.1.0.vsix` (330.62 KB)  
**Bundle Size**: 928.4 KB  
**MD5**: (see below)

## 🎉 What's New

### Premium Task Shell UI
- ✨ **Glassmorphism Design**: Semi-transparent backgrounds with backdrop blur
- 🌊 **Liquid Gradients**: Animated emerald (success) and ruby (failure) phase glows
- 📊 **Real-time Telemetry**: CPU/RAM badges in task header, resource limit warnings
- 📦 **Artifact Validation**: Automatic PASS/FAIL badges for produced files
- 💭 **Collapsible UI**: Click thoughts and tool blocks to expand/collapse

### Auto-Launch After Validation
- 🚀 Task Shell now **automatically opens** after validation passes
- 🎯 Creates a welcome demo task to showcase the premium UI
- ⚡ No manual command entry required

### Proper Server Cleanup
- ✅ llama-server processes are now **properly killed** when VS Code closes
- 🧹 Prevents orphaned processes consuming system resources
- 📝 Logs cleanup status for debugging

### Event-Sourced Architecture
- 📜 **JSONL Event Ledger**: Append-only, crash-safe event storage
- 🔄 **Pure Reducer**: Deterministic state derivation from event streams
- ✅ **Schema Validation**: All 34 event types validated against JSON Schema
- 🔐 **Integrity Checks**: SHA-256 stream hashing for drift detection

## 📦 Package Contents

```
gravitas-code-0.1.0.vsix (15 files, 330.62 KB)
├─ dist/extension.js [928.37 KB]
├─ media/
│  ├─ taskShell.css [10.55 KB] - Premium glassmorphism styles
│  ├─ taskShell.js [22.49 KB] - Event rendering engine
│  ├─ taskShell.html [0.74 KB] - UI structure
│  ├─ toolkit.js [412.96 KB] - VS Code webview toolkit
│  ├─ setup.html [22.05 KB] - Setup wizard
│  └─ validation.html [3.35 KB] - Validation panel
├─ img/logo_only.png [48.83 KB]
├─ .gravitas/schema.json [2.11 KB]
└─ setup.sh [1.28 KB]
```

## 🔧 Installation

```bash
code --install-extension gravitas-code-0.1.0.vsix
```

Or via VS Code UI:
1. Extensions → `...` → Install from VSIX
2. Select `gravitas-code-0.1.0.vsix`

## ✅ Verified Features

### Telemetry & Resource Guards
```
📊 Sample: RAM=110MB, CPU=100%, VRAM=0MB
⚠️ LIMIT EXCEEDED: CPU 100 (Limit 90) Severity=WARNING
```

### Artifact Validation Pipeline
```
📦 ArtifactProduced: test_artifact.txt
✅ ArtifactValidated: Status=PASS | Validator=fs-checker
```

### UI Aesthetics
- Liquid glass backgrounds with `backdrop-filter: blur(24px)`
- State-driven animations for active phases
- Inter font family with optimized spacing
- Scoped status classes to prevent style collisions

## 🐛 Bug Fixes

1. **Server Cleanup**: Fixed llama-server processes staying alive after VS Code exit
2. **Branding**: Removed all "Antigravity" references, replaced with "Gravitas"
3. **Schema Compliance**: Fixed `ArtifactProduced` event validation errors
4. **TaskStore Indexing**: Fixed `this.tasks.get()` error (changed to object indexing)
5. **Package Files**: Removed non-existent `bun.lock` from files pattern

## 📚 Documentation

- [INSTALL.md](file:///home/viren/runs/full-server/gravitas-code/INSTALL.md) - Installation guide
- [UI_GUIDE.md](file:///home/viren/runs/full-server/gravitas-code/UI_GUIDE.md) - UI access guide
- [README.md](file:///home/viren/runs/full-server/gravitas-code/README.md) - Architecture overview

## 🧪 Test Results

All tests passing:
```bash
🚀 Starting Gravitas Hardening Test Suite
--- Running Telemetry & Resource Guard Test ---
✅ PASS
--- Running Artifact Validation Pipeline Test ---
✅ PASS
🏁 All Tests Complete
```

## 🚀 Next Steps

1. Install the extension
2. Configure LLM paths in Setup Wizard
3. Run validation (servers will auto-start)
4. See the premium Task Shell UI automatically!

## 📝 Known Limitations

- Date-time format warnings in schema validation (benign)
- Requires VS Code ≥1.80.0
- LLM models must be locally available

---

**Support**: [GitHub Repository](https://github.com/vayuos/Gravitas-Code)  
**License**: See LICENSE file
