# Gravitas UI Guide - Accessing the Premium Task Shell

## Current State

You're currently seeing the **Agent Console** input field, which is the entry point. The **premium Task Shell UI** (with glassmorphism, liquid gradients, telemetry badges, and artifact validation) appears when you spawn a task.

## How to See the Full UI

### Method 1: Via Command Palette (Recommended)
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type: **"Gravitas: Spawn Task Shell"**
3. Press Enter
4. Enter your command when prompted

### Method 2: Via Agent Console Input
1. Click the input field that says "Enter command to spawn new task..."
2. Type any command, for example:
   ```
   Analyze the taskShell.css file and suggest improvements
   ```
3. Press Enter

### Method 3: Programmatic Test (Developer)
Run this command to spawn a demo task:
```bash
cd /home/viren/runs/full-server/gravitas-code
code --command gravitas.task.spawn
```

## What You'll See

Once a task is spawned, the Agent Console will transform to show:

### 1. **Task Header** (Glassmorphism Design)
```
┌─────────────────────────────────────────────────┐
│ 🎯 Task: Analyze taskShell.css                 │
│ Status: RUNNING  ⚡ CPU: 14%  💾 RAM: 412MB    │
│ Attempt #1 • Started 2s ago                     │
└─────────────────────────────────────────────────┘
```

### 2. **Execution Timeline** (Liquid Gradient Phases)
```
┌─ Phase: Coder - Analyzing File ────────────────┐
│ 💭 Thought: "I'm examining the CSS structure..." │
│                                                  │
│ 🔧 Tool: view_file                              │
│    ├─ Status: RUNNING                           │
│    └─ File: taskShell.css                       │
│                                                  │
│ 📦 Artifact: analysis_report.md                 │
│    └─ Validation: ✅ PASS                       │
└──────────────────────────────────────────────────┘
```

### 3. **Real-time Telemetry**
- CPU/RAM usage updates every few seconds
- Resource limit warnings (⚠️ if thresholds exceeded)
- VRAM tracking (if GPU enabled)

### 4. **Artifact Validation Badges**
- 🟢 **PASS**: File verified on disk
- 🔴 **FAIL**: File not found
- 🟡 **PENDING**: Validation in progress

## UI Features You Built

✨ **Glassmorphism**: Semi-transparent backgrounds with backdrop blur
✨ **Liquid Gradients**: Animated emerald (success) and ruby (failure) glows
✨ **State-Driven Animations**: Phases pulse when active
✨ **Premium Typography**: Inter font with optimized spacing
✨ **Collapsible Sections**: Click thoughts/tools to expand/collapse

## Quick Demo Command

Try this command in the Agent Console to see all features:
```
Create a simple hello.txt file with "Hello Gravitas" and validate it exists
```

This will trigger:
- Phase start (Coder)
- Thought emission
- Tool execution (write_to_file)
- Artifact production
- Artifact validation (PASS/FAIL badge)
- Phase completion

## Troubleshooting

**If the UI doesn't appear:**
1. Check that validation passed (you already did this ✅)
2. Ensure both Coder and Reviewer are running (they are ✅)
3. Reload VS Code window: `Ctrl+Shift+P` → "Developer: Reload Window"

**If you see errors:**
- Check the Output panel: `View` → `Output` → Select "Gravitas"
- Review logs: `/home/viren/.gravitas/logs/gravitas-2026-02-06.log`

## Visual Reference

The UI you built looks like this (from the mockup):
![Gravitas Premium UI](/home/viren/.gemini/antigravity/brain/aa97802e-12ef-4bd3-9053-0843b878b6e2/gravitas_ui_premium_mockup_1770373317187.png)

---

**Next Step**: Simply type a command in the Agent Console input and press Enter to see the full Task Shell UI in action!
