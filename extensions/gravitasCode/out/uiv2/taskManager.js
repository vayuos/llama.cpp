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
exports.TaskManager = void 0;
const vscode = __importStar(require("vscode"));
const uuid_1 = require("uuid");
const types_1 = require("./types");
const eventValidator_1 = require("./eventValidator");
const reducer_1 = require("./reducer");
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const integrity_1 = require("./integrity");
class TaskManager {
    constructor(context) {
        this.tasks = {};
        this._onDidTaskUpdate = new vscode.EventEmitter();
        this._onDidEmitEvent = new vscode.EventEmitter();
        this.cpuUsageBaseline = process.cpuUsage();
        this.lastTelemetryTime = Date.now();
        this.onDidTaskUpdate = this._onDidTaskUpdate.event;
        this.onDidEmitEvent = this._onDidEmitEvent.event;
        this.context = context;
        this.eventValidator = eventValidator_1.EventValidator.getInstance();
        this.ensureBaseDir();
        this.loadTasks();
    }
    ensureBaseDir() {
        const baseDir = this.getStoragePath();
        if (!fs.existsSync(baseDir)) {
            fs.mkdirSync(baseDir, { recursive: true });
        }
    }
    getStoragePath() {
        // Preference: Workspace storage if available, otherwise global storage
        if (this.context.storageUri) {
            return this.context.storageUri.fsPath;
        }
        return path.join(this.context.globalStorageUri.fsPath, 'tasks');
    }
    getTaskEventsPath(taskId) {
        return path.join(this.getStoragePath(), taskId, 'events.jsonl');
    }
    static initialize(context) {
        if (!TaskManager.instance) {
            TaskManager.instance = new TaskManager(context);
        }
        return TaskManager.instance;
    }
    static getInstance() {
        if (!TaskManager.instance) {
            throw new Error('TaskManager not initialized. Call initialize() first.');
        }
        return TaskManager.instance;
    }
    loadTasks() {
        const baseDir = this.getStoragePath();
        if (!fs.existsSync(baseDir))
            return;
        const taskDirs = fs.readdirSync(baseDir, { withFileTypes: true })
            .filter(dirent => dirent.isDirectory())
            .map(dirent => dirent.name);
        for (const taskId of taskDirs) {
            try {
                const eventsPath = this.getTaskEventsPath(taskId);
                if (fs.existsSync(eventsPath)) {
                    const events = fs.readFileSync(eventsPath, 'utf8')
                        .split('\n')
                        .filter(line => line.trim())
                        .map(line => JSON.parse(line));
                    if (events.length > 0) {
                        this.tasks[taskId] = this.rebuildTaskFromEvents(taskId, events);
                    }
                }
            }
            catch (err) {
                console.error(`[TaskManager] Failed to restore task ${taskId}:`, err);
            }
        }
        console.log(`[TaskManager] Restored ${Object.keys(this.tasks).length} tasks from ledger.`);
    }
    rebuildTaskFromEvents(taskId, events) {
        return (0, reducer_1.reduceTask)(taskId, events);
    }
    findPhase(task, attemptNo, phaseId) {
        const attempt = task.attempts.find(a => a.attemptNo === attemptNo);
        return attempt?.phases.find(p => p.id === phaseId);
    }
    findCurrentPhase(task, attemptNo) {
        const attempt = task.attempts.find(a => a.attemptNo === attemptNo);
        return attempt?.phases[attempt.phases.length - 1];
    }
    async saveTasks() {
        // No longer saving full TaskStore to globalState! 
        // Meta-index can still live there if we want faster boot, but for now we follow Gap 2 strictly.
        // We might want to save JUST the task IDs to globalState for tracking order.
        await this.context.globalState.update('gravitas_tasks_v2_ids', Object.keys(this.tasks));
    }
    createTask(command, origin = 'user', parentTaskId, regenerationType) {
        const id = (0, uuid_1.v4)();
        const now = Date.now();
        const newTask = {
            id,
            createdAt: now,
            origin,
            command,
            status: types_1.TaskState.CREATED,
            updatedAt: now,
            parentTaskId,
            regenerationType,
            attempts: []
        };
        this.tasks[id] = newTask;
        // Ensure task dir exists
        const taskDir = path.join(this.getStoragePath(), id);
        if (!fs.existsSync(taskDir)) {
            fs.mkdirSync(taskDir, { recursive: true });
        }
        // Emit TaskCreated as the first ledger entry
        this.emitEvent(id, {
            type: 'TaskCreated',
            createdAt: new Date().toISOString(),
            origin,
            command,
            metadata: {}
        });
        this.saveTasks();
        this._fireUpdate(newTask);
        return newTask;
    }
    getTask(id) {
        return this.tasks[id];
    }
    getAllTasks() {
        // Return sorted by creation time desc
        return Object.values(this.tasks).sort((a, b) => b.createdAt - a.createdAt);
    }
    updateTaskState(id, newState) {
        const task = this.tasks[id];
        if (!task)
            return;
        // Simple validation: Enforce monotonic flow (basic check)
        // In a real strict system we'd check transitions map
        task.status = newState;
        task.updatedAt = Date.now();
        this.saveTasks();
        this._fireUpdate(task);
    }
    /**
     * Starts a new monotonic attempt.
     * Enforces that previous attempt is CLOSED (or closes it).
     */
    startNextAttempt(id) {
        const task = this.tasks[id];
        if (!task)
            throw new Error('Task not found');
        // Close previous if open
        const lastAttempt = task.attempts[task.attempts.length - 1];
        if (lastAttempt && lastAttempt.state === 'OPEN') {
            lastAttempt.state = 'CLOSED';
            lastAttempt.endedAt = Date.now();
            lastAttempt.verdict = lastAttempt.verdict || 'INCOMPLETE'; // Default if not set
        }
        const attemptNo = task.attempts.length + 1;
        const attemptId = (0, uuid_1.v4)();
        const attempt = {
            id: attemptId,
            attemptNo,
            startedAt: Date.now(),
            state: 'OPEN',
            phases: []
        };
        task.attempts.push(attempt);
        task.updatedAt = Date.now();
        this.saveTasks();
        this._fireUpdate(task);
        this.emitEvent(id, {
            type: 'AttemptStarted',
            attemptNo,
            startedAt: new Date().toISOString(),
            initiator: 'system'
        });
        // Implicitly start a System phase for setup
        this.startPhase(id, 'system', 'Attempt Initialization');
        return attempt;
    }
    /**
     * Starts a new Phase Block.
     */
    startPhase(taskId, actor, title) {
        const task = this.tasks[taskId];
        if (!task)
            throw new Error('Task not found');
        const attempt = this.getCurrentAttempt(task);
        if (!attempt || attempt.state === 'CLOSED')
            throw new Error('No open attempt');
        // Close previous phase
        const lastPhase = attempt.phases[attempt.phases.length - 1];
        if (lastPhase && lastPhase.status === 'RUNNING') {
            lastPhase.status = 'COMPLETED';
            lastPhase.endedAt = Date.now();
        }
        const phaseId = (0, uuid_1.v4)();
        const newPhase = {
            id: phaseId,
            actor,
            title,
            startedAt: Date.now(),
            status: 'RUNNING',
            events: []
        };
        attempt.phases.push(newPhase);
        task.updatedAt = Date.now();
        this.saveTasks();
        this.emitEvent(taskId, {
            type: 'PhaseStarted',
            attemptNo: attempt.attemptNo,
            phaseId,
            actor,
            title,
            startedAt: new Date().toISOString()
        });
        return phaseId;
    }
    bindAgent(taskId, phaseId, agentId, modelId, configFingerprint) {
        this.emitEvent(taskId, {
            type: 'AgentBoundToPhase',
            phaseId,
            agentId,
            modelId,
            configFingerprint
        });
    }
    recordPhaseMetrics(taskId, phaseId, durationMs, tokenCount, costEstimate) {
        this.emitEvent(taskId, {
            type: 'PhaseMetricsReported',
            phaseId,
            durationMs,
            tokenCount,
            costEstimate
        });
    }
    recordPolicyDecision(taskId, policyName, decision, reasoning) {
        this.emitEvent(taskId, {
            type: 'PolicyEvaluated',
            policyName,
            decision,
            reasoning
        });
    }
    recordUserAction(taskId, actionType, targetId, metadata) {
        this.emitEvent(taskId, {
            type: 'UserActionPerformed',
            actionType,
            targetId,
            metadata
        });
    }
    // GAP 1: Authoritative Emitters
    emitTaskEvent(taskId, type, payload) {
        this.emitEvent(taskId, { type, ...payload });
    }
    emitAttemptEvent(taskId, attemptNo, type, payload) {
        this.emitEvent(taskId, { attemptNo, type, ...payload });
    }
    emitPhaseEvent(taskId, phaseId, type, payload) {
        const task = this.tasks[taskId];
        const attempt = task ? this.getCurrentAttempt(task) : undefined;
        this.emitEvent(taskId, {
            attemptNo: attempt?.attemptNo || 1,
            phaseId,
            type,
            ...payload
        });
    }
    completePhase(taskId) {
        const task = this.tasks[taskId];
        if (!task)
            return;
        const attempt = this.getCurrentAttempt(task);
        if (!attempt)
            return;
        const phase = attempt.phases[attempt.phases.length - 1];
        if (phase && phase.status === 'RUNNING') {
            phase.status = 'COMPLETED';
            phase.endedAt = Date.now();
            this.saveTasks();
        }
    }
    /**
     * Seals the current attempt with a verdict.
     */
    completeAttempt(taskId, verdict) {
        const task = this.tasks[taskId];
        if (!task)
            return;
        const attempt = this.getCurrentAttempt(task);
        if (!attempt || attempt.state === 'CLOSED')
            return;
        // Close active phase
        this.completePhase(taskId);
        attempt.verdict = verdict;
        attempt.state = 'CLOSED';
        attempt.endedAt = Date.now();
        task.updatedAt = Date.now();
        this.saveTasks();
        this._fireUpdate(task);
        this.emitEvent(taskId, {
            type: 'AttemptClosed', // Using AttemptClosed.v1.json
            attemptNo: attempt.attemptNo,
            verdict: verdict,
            closedAt: new Date().toISOString(),
            summary: task.summary || 'Attempt finished'
        });
    }
    sampleResources(taskId) {
        const mem = process.memoryUsage();
        const ramMb = Math.round(mem.rss / 1024 / 1024);
        // Calculate CPU %
        const now = Date.now();
        const cpuUsage = process.cpuUsage(this.cpuUsageBaseline);
        const elapsedUs = (now - this.lastTelemetryTime) * 1000;
        const totalUs = cpuUsage.user + cpuUsage.system;
        const cpuPercent = elapsedUs > 0 ? Math.min(100, Math.round((totalUs / elapsedUs) * 100)) : 0;
        // Reset baseline
        this.cpuUsageBaseline = process.cpuUsage();
        this.lastTelemetryTime = now;
        this.emitEvent(taskId, {
            type: 'ResourceUsageSampled',
            resources: {
                ramMb,
                vramMb: 0, // Mock: GPU drivers are complex, keeping it 0 for now
                cpuPercent
            }
        });
        // Resource Limit Guard (Configurable thresholds)
        // Default: 1GB RAM, 90% CPU
        const RAM_LIMIT = 1024;
        const CPU_LIMIT = 90;
        if (ramMb > RAM_LIMIT) {
            this.emitEvent(taskId, {
                type: 'ResourceLimitExceeded',
                resourceType: 'RAM',
                limitValue: RAM_LIMIT,
                actualValue: ramMb,
                severity: 'CRITICAL'
            });
        }
        if (cpuPercent > CPU_LIMIT) {
            this.emitEvent(taskId, {
                type: 'ResourceLimitExceeded',
                resourceType: 'CPU',
                limitValue: CPU_LIMIT,
                actualValue: cpuPercent,
                severity: 'WARNING'
            });
        }
    }
    recordArtifact(taskId, filePath, type, metadata) {
        const artifactId = (0, uuid_1.v4)();
        const task = this.tasks[taskId];
        const attemptNo = task ? task.attempts.length : 1;
        const fileName = path.basename(filePath);
        // Ensure type is within enum: "file" | "package" | "diff" | "report" | "other"
        const allowedTypes = ["file", "package", "diff", "report", "other"];
        const artifactType = allowedTypes.includes(type) ? type : "other";
        this.emitEvent(taskId, {
            type: 'ArtifactProduced',
            artifactId,
            name: fileName,
            path: filePath,
            artifactType: artifactType,
            producedByAttempt: attemptNo,
            producedAt: new Date().toISOString(),
            timestamp: new Date().toISOString()
        });
        // Auto-validate existence (Phase 19)
        const exists = fs.existsSync(filePath);
        this.validateArtifact(taskId, artifactId, exists ? 'PASS' : 'FAIL', 'fs-checker', exists ? 'File verified on disk.' : 'File not found after production.', { path: filePath, size: exists ? fs.statSync(filePath).size : 0 });
    }
    validateArtifact(taskId, artifactId, status, validatorId, message, details) {
        this.emitEvent(taskId, {
            type: 'ArtifactValidated',
            artifactId,
            validatorId,
            status,
            message,
            details
        });
    }
    // Deprecated compat shim
    addAttempt(id) {
        return this.startNextAttempt(id).id;
    }
    emitEvent(id, eventPayload) {
        const task = this.tasks[id];
        if (!task)
            return;
        const attempt = this.getCurrentAttempt(task);
        if (!attempt || attempt.state === 'CLOSED')
            return;
        // Enrich with mandatory schema fields + Gaps: Auto-routing
        const currentPhase = attempt.phases[attempt.phases.length - 1];
        const event = {
            eventId: (0, uuid_1.v4)(),
            taskId: id,
            timestamp: new Date().toISOString(),
            attemptNo: attempt.attemptNo,
            phaseId: currentPhase?.id,
            ...eventPayload
        };
        // Validate
        const validation = this.eventValidator.validate(event.type, event);
        if (!validation.valid) {
            console.error(`[TaskManager] Event validation failed for ${event.type}:`, validation.errors);
            // In strict mode we might reject, but for now we log.
        }
        // PHASE ROUTING (Now handled internally by applyEvent for state consistency)
        this.tasks[id] = (0, reducer_1.applyEvent)(task, event);
        // PERSISTENCE (Gap 2: Persistent Ledger)
        try {
            const eventPath = this.getTaskEventsPath(id);
            const line = JSON.stringify(event) + '\n';
            fs.appendFileSync(eventPath, line, 'utf8');
        }
        catch (err) {
            console.error(`[TaskManager] Failed to persist event to ledger:`, err);
        }
        this.saveTasks();
        this._fireUpdate(task);
        this._onDidEmitEvent.fire({ taskId: id, event });
    }
    getCurrentAttempt(task) {
        return task.attempts[task.attempts.length - 1];
    }
    addTerminalChunk(id, data, toolExecId = 'root', stream = 'stdout') {
        const task = this.tasks[id];
        const attempt = task ? this.getCurrentAttempt(task) : null;
        const phase = attempt ? attempt.phases[attempt.phases.length - 1] : null;
        this.emitEvent(id, {
            type: 'ToolExecutionOutput',
            attemptNo: attempt?.attemptNo || 1,
            phaseId: phase?.id || 'none',
            toolExecId,
            timestamp: new Date().toISOString(),
            stream,
            text: data
        });
    }
    completeTask(id, summary) {
        const task = this.tasks[id];
        if (!task)
            return;
        task.summary = summary;
        task.status = types_1.TaskState.COMPLETED;
        task.updatedAt = Date.now();
        this.saveTasks();
        this._fireUpdate(task);
    }
    failTask(id, reason) {
        const task = this.tasks[id];
        if (!task)
            return;
        task.summary = `Failed: ${reason}`;
        task.status = types_1.TaskState.FAILED;
        task.updatedAt = Date.now();
        this.saveTasks();
        this._fireUpdate(task);
    }
    abortTask(id) {
        const task = this.tasks[id];
        if (!task)
            return;
        if (['COMPLETED', 'FAILED', 'ABORTED'].includes(task.status))
            return;
        // Close active attempt/phase?
        const attempt = this.getCurrentAttempt(task);
        if (attempt && attempt.state === 'OPEN') {
            attempt.verdict = 'INCOMPLETE';
            attempt.state = 'CLOSED';
            attempt.endedAt = Date.now();
        }
        task.status = types_1.TaskState.ABORTED;
        task.summary = 'Aborted by User';
        task.updatedAt = Date.now();
        this.emitEvent(id, {
            type: 'AbortTriggered',
            triggeredAt: new Date().toISOString(),
            triggeredBy: 'user',
            reasonCode: 'USER_ABORT',
            humanMessage: '⛔ System — Task Aborted by User'
        });
        this.saveTasks();
        this._fireUpdate(task);
    }
    clearAllTasks() {
        this.tasks = {};
        this.saveTasks();
        // Notify listeners? might need a clear event
    }
    async verifyTaskIntegrity(id) {
        const eventsPath = this.getTaskEventsPath(id);
        if (!fs.existsSync(eventsPath)) {
            return { verified: false, hash: '', driftDetails: 'Ledger file missing' };
        }
        try {
            const content = fs.readFileSync(eventsPath, 'utf8');
            const result = (0, integrity_1.verifyContentIntegrity)(content);
            if (result.driftDetails === 'Baseline established (No previous verification found)') {
                // Emit initial baseline
                this.emitEvent(id, {
                    type: 'ReplayVerified',
                    streamHash: result.hash,
                    status: 'VERIFIED'
                });
            }
            return result;
        }
        catch (err) {
            return { verified: false, hash: '', driftDetails: err.message };
        }
    }
    _fireUpdate(task) {
        this._onDidTaskUpdate.fire(task);
    }
}
exports.TaskManager = TaskManager;
//# sourceMappingURL=taskManager.js.map