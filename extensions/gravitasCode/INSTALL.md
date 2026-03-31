# Gravitas v0.1.0 - Installation Guide

## Package Information

**File**: `gravitas-code-0.1.0.vsix`  
**Size**: 330 KB (329.54 KB uncompressed: 2.5 MB)  
**Format**: VS Code Extension Package (Zip archive)  
**MD5**: `ffb0bd15605d0ce0579437278ee70f44`  
**Build Date**: 2026-02-06 16:04:52 UTC

## Package Contents

```
gravitas-code-0.1.0.vsix (15 files)
├─ package.json [20.37 KB]
├─ README.md
├─ setup.sh [1.28 KB]
├─ .gravitas/
│  └─ schema.json [2.11 KB]
├─ dist/
│  └─ extension.js [923.43 KB]
├─ img/
│  └─ logo_only.png [48.83 KB]
└─ media/
   ├─ setup.html [22.05 KB]
   ├─ taskShell.css [10.55 KB]
   ├─ taskShell.html [0.74 KB]
   ├─ taskShell.js [22.49 KB]
   ├─ test-lab.html [5.35 KB]
   ├─ toolkit.js [412.96 KB]
   └─ validation.html [3.35 KB]
```

## Installation Methods

### Method 1: VS Code UI (Recommended)
1. Open VS Code
2. Go to Extensions view (`Ctrl+Shift+X` or `Cmd+Shift+X`)
3. Click the `...` menu (top-right)
4. Select **"Install from VSIX..."**
5. Navigate to `gravitas-code-0.1.0.vsix`
6. Click **Install**

### Method 2: Command Line
```bash
code --install-extension gravitas-code-0.1.0.vsix
```

### Method 3: Manual Installation
```bash
# Copy to VS Code extensions directory
cp gravitas-code-0.1.0.vsix ~/.vscode/extensions/
cd ~/.vscode/extensions/
unzip gravitas-code-0.1.0.vsix
```

## Post-Installation Setup

### 1. Initial Configuration
After installation, the **Setup Wizard** will appear automatically. Configure:
- **LLM Binary Path**: Path to `llama-server` binary
- **Coder Model**: Path to Qwen3-Coder-30B model
- **Reviewer Model**: Path to DeepSeek-Coder-33B model
- **Hardware Settings**: GPU layers, context size, threads

### 2. Validation
Click **"Validate Setup"** to run comprehensive checks:
- Binary existence and permissions
- Model file accessibility
- Port availability (8010, 8011)
- Memory requirements

### 3. Activation
Once validation passes, the extension will activate and show:
- **Agent Console**: Task Shell interface
- **Runtime Control**: Start/Stop LLM servers
- **Context Scope**: Workspace file tree
- **Presets & Profiles**: Configuration management

## Verification

### Check Installation
```bash
code --list-extensions | grep gravitas
# Should output: vayuos.gravitas-code
```

### Test Telemetry
```bash
cd ~/.vscode/extensions/vayuos.gravitas-code-*/
npx ts-node --transpile-only src/test/telemetryTest.ts
```

Expected output:
```
🚀 Starting Gravitas Hardening Test Suite
📊 Sample: RAM=110MB, CPU=100%, VRAM=0MB
✅ ArtifactValidated: Status=PASS
🏁 All Tests Complete
```

## Troubleshooting

### Extension Not Activating
1. Check VS Code version: `code --version` (requires ≥1.80.0)
2. Reload window: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Check logs: `Ctrl+Shift+P` → "Developer: Show Logs" → "Extension Host"

### Setup Wizard Not Appearing
1. Open Command Palette (`Ctrl+Shift+P`)
2. Run: **"Gravitas: Validate Setup"**
3. The wizard will re-appear

### LLM Servers Not Starting
1. Verify binary path: `which llama-server`
2. Check port availability: `lsof -i :8010` and `lsof -i :8011`
3. Review logs in **Runtime Control** panel

## Uninstallation

### Via UI
1. Extensions view → Find "Gravitas Code"
2. Click **Uninstall**

### Via Command Line
```bash
code --uninstall-extension vayuos.gravitas-code
```

### Clean Removal (including data)
```bash
code --uninstall-extension vayuos.gravitas-code
rm -rf ~/.vscode/extensions/vayuos.gravitas-code-*
rm -rf ~/.config/Code/User/globalStorage/vayuos.gravitas-code
```

## Next Steps

1. **Configure Models**: Set paths to your local LLM binaries
2. **Start Runtime**: Launch Coder and Reviewer servers
3. **Spawn Task**: Run `Gravitas: Spawn Task Shell` to create your first execution container
4. **Explore UI**: Review the glassmorphism design and real-time telemetry

---

**Support**: For issues, see the [GitHub repository](https://github.com/vayuos/Gravitas-Code)  
**License**: See LICENSE file in extension directory
