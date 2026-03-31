"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.populateDemoTask = populateDemoTask;
const vscode = __importStar(require("vscode"));
const taskManager_1 = require("../uiv2/taskManager");
const types_1 = require("../uiv2/types");
/**
 * Populates the current demo task with rich content to showcase premium UI features
 */
async function populateDemoTask() {
    const tm = taskManager_1.TaskManager.getInstance();
    // Get the most recent task
    const tasks = tm.getAllTasks();
    if (tasks.length === 0) {
        vscode.window.showWarningMessage('No tasks found. Please run validation first.');
        return;
    }
    const demoTask = tasks[tasks.length - 1];
    const taskId = demoTask.id;
    // Start the task if not already running
    if (demoTask.status !== types_1.TaskState.RUNNING) {
        tm.updateTaskState(taskId, types_1.TaskState.RUNNING);
    }
    // Start an attempt if none exist
    let attemptNo = 1;
    if (!demoTask.attempts || demoTask.attempts.length === 0) {
        const attempt = tm.startNextAttempt(taskId);
        attemptNo = attempt.attemptNo;
    }
    else {
        attemptNo = demoTask.attempts[demoTask.attempts.length - 1].attemptNo;
    }
    // Emit sample telemetry to show badges
    tm.sampleResources(taskId);
    // Create Phase 1: Coder - Architecture Analysis
    const phase1Id = 'demo-phase-coder';
    tm.emitEvent(taskId, {
        type: 'PhaseStarted',
        phaseId: phase1Id,
        attemptNo,
        actor: 'coder',
        title: '🎨 Analyzing Glassmorphism Architecture',
        startedAt: new Date().toISOString()
    });
    // Add thought with glassmorphism context
    tm.emitEvent(taskId, {
        type: 'ThoughtCompleted',
        phaseId: phase1Id,
        thoughtId: 'demo-thought-1',
        content: '💭 Examining the premium UI design system. The liquid glass effect uses `backdrop-filter: blur(24px)` with semi-transparent backgrounds. Telemetry badges display real-time CPU/RAM metrics. The reducer pattern ensures deterministic state derivation from the event stream.',
        endedAt: new Date().toISOString(),
        durationMs: 450,
        attemptNo
    });
    // Add telemetry sample showing resource usage
    tm.emitEvent(taskId, {
        type: 'ResourceUsageSampled',
        resources: { ramMb: 412, cpuPercent: 14, vramMb: 0 }
    });
    // Add tool execution: view_file
    tm.emitEvent(taskId, {
        type: 'ToolExecutionStarted',
        phaseId: phase1Id,
        toolExecId: 'demo-tool-1',
        commandLine: 'view_file media/taskShell.css',
        startedAt: new Date().toISOString(),
        workingDirectory: '/tmp',
        attemptNo
    });
    await new Promise(resolve => setTimeout(resolve, 500));
    tm.emitEvent(taskId, {
        type: 'ToolExecutionCompleted',
        phaseId: phase1Id,
        toolExecId: 'demo-tool-1',
        status: 'SUCCESS',
        endedAt: new Date().toISOString(),
        exitCode: 0,
        attemptNo
    });
    // Create artifact with validation
    const artifactId = 'demo-artifact-premium';
    tm.emitEvent(taskId, {
        type: 'ArtifactProduced',
        artifactId,
        name: 'gravitas_ui_analysis.md',
        path: '/tmp/gravitas_ui_analysis.md',
        artifactType: 'report',
        producedByAttempt: attemptNo,
        producedAt: new Date().toISOString()
    });
    // Validate artifact (PASS)
    tm.emitEvent(taskId, {
        type: 'ArtifactValidated',
        artifactId,
        status: 'PASS',
        validatorId: 'premium-ui-validator',
        message: '✅ All premium UI features verified: Glassmorphism ✓ Liquid gradients ✓ Telemetry badges ✓ Artifact validation ✓'
    });
    // Complete Phase 1
    tm.emitEvent(taskId, {
        type: 'PhaseCompleted',
        phaseId: phase1Id,
        status: 'COMPLETED',
        endedAt: new Date().toISOString(),
        attemptNo
    });
    // Create Phase 2: Reviewer - Quality Check
    const phase2Id = 'demo-phase-reviewer';
    tm.emitEvent(taskId, {
        type: 'PhaseStarted',
        phaseId: phase2Id,
        attemptNo,
        actor: 'reviewer',
        title: '🔍 Reviewing Event-Sourced Ledger',
        startedAt: new Date().toISOString()
    });
    tm.emitEvent(taskId, {
        type: 'ThoughtCompleted',
        phaseId: phase2Id,
        thoughtId: 'demo-thought-2',
        content: '🤔 Verifying event schema compliance. All 34 event types validated against JSON Schema. The JSONL ledger provides crash-safe, append-only persistence. SHA-256 hashing ensures integrity.',
        endedAt: new Date().toISOString(),
        durationMs: 600,
        attemptNo
    });
    // Add another telemetry sample
    tm.emitEvent(taskId, {
        type: 'ResourceUsageSampled',
        resources: { ramMb: 428, cpuPercent: 18, vramMb: 0 }
    });
    // Emit verdict
    tm.emitEvent(taskId, {
        type: 'ReviewerResultEmitted',
        phaseId: phase2Id,
        verdict: 'PASS',
        emittedAt: new Date().toISOString(),
        attemptNo,
        issues: []
    });
    // Complete Phase 2
    tm.emitEvent(taskId, {
        type: 'PhaseCompleted',
        phaseId: phase2Id,
        status: 'COMPLETED',
        endedAt: new Date().toISOString(),
        attemptNo
    });
    // Complete the task
    tm.emitEvent(taskId, {
        type: 'FinalSummaryEmitted',
        outcome: 'SUCCESS',
        emittedAt: new Date().toISOString(),
        attemptCount: 1
    });
    vscode.window.showInformationMessage('✨ Demo task populated with premium UI features!');
}
//# sourceMappingURL=populateDemo.js.map