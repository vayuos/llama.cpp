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
const os = __importStar(require("os"));
const integrity_1 = require("./integrity");
/**
 * Authoritative Task Store and Lifecycle Manager.
 * Implements Gap 1-3: State, Ledger, Pure Reducers.
 */
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
                        console.log(`[TaskManager] Recovered task ${taskId} with ${events.length} events.`);
                    }
                }
            }
            catch (err) {
                console.error(`[TaskManager] Failed to restore task ${taskId}:`, err);
            }
        }
    }
    rebuildTaskFromEvents(taskId, events) {
        return (0, reducer_1.reduceTask)(taskId, events);
    }
    getTask(id) {
        return this.tasks[id];
    }
    getAllTasks() {
        return Object.values(this.tasks).sort((a, b) => b.createdAt - a.createdAt);
    }
    getLastTask() {
        const all = this.getAllTasks();
        return all.length > 0 ? all[0] : undefined;
    }
    getCurrentAttempt(task) {
        return task.attempts[task.attempts.length - 1];
    }
    async saveTasks() {
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
        const taskDir = path.join(this.getStoragePath(), id);
        if (!fs.existsSync(taskDir)) {
            fs.mkdirSync(taskDir, { recursive: true });
        }
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
    updateTaskState(id, newState) {
        const task = this.tasks[id];
        if (!task)
            return;
        task.status = newState;
        task.updatedAt = Date.now();
        this.emitEvent(id, {
            type: 'TaskStateChanged',
            previousState: task.status, // previous state capture might be useful for history
            newState
        });
        this.saveTasks();
        this._fireUpdate(task);
    }
    startNextAttempt(id) {
        const task = this.tasks[id];
        if (!task)
            throw new Error('Task not found');
        const lastAttempt = this.getCurrentAttempt(task);
        if (lastAttempt && lastAttempt.state === 'OPEN') {
            lastAttempt.state = 'CLOSED';
            lastAttempt.endedAt = Date.now();
            lastAttempt.verdict = lastAttempt.verdict || 'INCOMPLETE';
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
        this.startPhase(id, 'system', 'Attempt Initialization');
        return attempt;
    }
    startPhase(taskId, actor, title) {
        const task = this.tasks[taskId];
        if (!task)
            throw new Error('Task not found');
        const attempt = this.getCurrentAttempt(task);
        if (!attempt || attempt.state === 'CLOSED')
            throw new Error('No open attempt');
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
            this.emitEvent(taskId, {
                type: 'PhaseCompleted',
                attemptNo: attempt.attemptNo,
                phaseId: phase.id,
                endedAt: new Date().toISOString(),
                status: 'COMPLETED'
            });
            this.saveTasks();
            this._fireUpdate(task);
        }
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
    recordArtifact(taskId, filePath, type, metadata) {
        const fileName = path.basename(filePath);
        this.emitEvent(taskId, {
            type: 'ArtifactProduced',
            artifactId: (0, uuid_1.v4)(),
            name: fileName,
            path: filePath,
            artifactType: type,
            producedAt: new Date().toISOString(),
            metadata
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
    emitEvent(id, event) {
        const task = this.tasks[id];
        if (!task)
            return;
        // Auto-enrich execution events with current attempt/phase if missing
        const isExecutionEvent = ['ToolExecutionStarted', 'ToolExecutionOutput', 'ToolExecutionCompleted', 'ThoughtEmitted', 'TerminalLog'].includes(event.type);
        if (isExecutionEvent && !event.attemptNo) {
            const currentAttempt = this.getCurrentAttempt(task);
            if (currentAttempt) {
                event.attemptNo = currentAttempt.attemptNo;
                if (!event.phaseId) {
                    const currentPhase = currentAttempt.phases[currentAttempt.phases.length - 1];
                    event.phaseId = currentPhase?.id || 'none';
                }
            }
        }
        if (!event.timestamp) {
            event.timestamp = new Date().toISOString();
        }
        if (!event.eventId) {
            event.eventId = (0, uuid_1.v4)();
        }
        if (!event.taskId) {
            event.taskId = id;
        }
        // VALIDATION
        const validation = this.eventValidator.validate(event.type, event);
        if (!validation.valid) {
            console.error(`[TaskManager] Event validation failed for ${event.type}:`, validation.errors);
        }
        // STATE REDUCTION
        this.tasks[id] = (0, reducer_1.applyEvent)(task, event);
        // PERSISTENCE
        try {
            const eventPath = this.getTaskEventsPath(id);
            const line = JSON.stringify(event) + '\n';
            fs.appendFileSync(eventPath, line, 'utf8');
            const logger = require('../core/logger').CentralLogger.getInstance();
            logger.debug('system', `Task ${id}: Persisted event ${event.type} (${line.length} bytes)`);
        }
        catch (err) {
            console.error(`[TaskManager] Failed to persist event:`, err);
        }
        this.saveTasks();
        this._fireUpdate(this.tasks[id]);
        this._onDidEmitEvent.fire({ taskId: id, event });
    }
    sampleResources(taskId) {
        const mem = process.memoryUsage();
        const ramMb = Math.round(mem.rss / 1024 / 1024);
        const now = Date.now();
        const cpuUsage = process.cpuUsage(this.cpuUsageBaseline);
        const elapsedUs = (now - this.lastTelemetryTime) * 1000;
        const totalUs = cpuUsage.user + cpuUsage.system;
        const cpuPercent = elapsedUs > 0 ? Math.min(100, Math.round((totalUs / elapsedUs) * 100)) : 0;
        this.cpuUsageBaseline = process.cpuUsage();
        this.lastTelemetryTime = now;
        // Try to enrich with real VRAM if Coder is running locally
        let vramMb = 0;
        try {
            const socketPath = path.join(os.homedir(), '.gravitas', 'sockets', 'coder.sock');
            if (fs.existsSync(socketPath)) {
                // We use a sync-like check to avoid blocking the sampling 
                // but since this is an async-friendly area, we just fire-and-forget or keep last known.
                // For this implementation, we'll just emit what we have and let pollHardwareMetrics handle higher-fidelity data.
            }
        }
        catch (e) { }
        this.emitEvent(taskId, {
            type: 'ResourceUsageSampled',
            resources: {
                ramMb,
                vramMb,
                cpuPercent
            }
        });
    }
    async pollHardwareMetrics(taskId) {
        try {
            const socketPath = path.join(require('os').homedir(), '.gravitas', 'sockets', 'coder.sock');
            if (!fs.existsSync(socketPath))
                return;
            const { LlamaHttpClient } = require('../../llm/llamaHttpClient');
            const client = new LlamaHttpClient(`unix://${socketPath}`);
            const metrics = await client.get('/metrics');
            const slots = await client.get('/slots');
            this.emitEvent(taskId, {
                type: 'HardwareMetricsEmitted',
                vramMb: this.extractVram(metrics),
                activeSlots: Array.isArray(slots) ? slots.filter((s) => s.status === 'processing').length : 0,
                totalSlots: Array.isArray(slots) ? slots.length : 0,
                tps: 0 // Calculated by UI
            });
        }
        catch (e) {
            // Silence polling errors
        }
    }
    extractVram(metrics) {
        if (typeof metrics !== 'string')
            return 0;
        // Find llama_vram_used_bytes or similar Prometheus metric
        const match = metrics.match(/vram_used_bytes\s+([0-9.]+)/) || metrics.match(/vram_used\s+([0-9.]+)/);
        if (match) {
            return Math.round(parseFloat(match[1]) / 1024 / 1024);
        }
        return 0;
    }
    emitStreamingChunk(taskId, chunk, stage) {
        const task = this.tasks[taskId];
        if (!task)
            return;
        // We emit it as a live event but don't necessarily persist every single token to disk 
        // to avoid killing disk I/O. The final block will be persisted via Attempt/Phase completion.
        this._onDidEmitEvent.fire({
            taskId,
            event: {
                type: 'StreamingChunkEmitted',
                taskId,
                chunk,
                stage,
                timestamp: new Date().toISOString(),
                eventId: (0, uuid_1.v4)()
            }
        });
    }
    addTerminalChunk(id, data, toolExecId = 'root', stream = 'stdout') {
        this.emitEvent(id, {
            type: 'ToolExecutionOutput',
            toolExecId,
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
    completeAttempt(taskId, verdict) {
        const task = this.tasks[taskId];
        if (!task)
            return;
        const attempt = this.getCurrentAttempt(task);
        if (!attempt || attempt.state === 'CLOSED')
            return;
        attempt.verdict = verdict;
        attempt.state = 'CLOSED';
        attempt.endedAt = Date.now();
        this.emitEvent(taskId, {
            type: 'AttemptClosed',
            attemptNo: attempt.attemptNo,
            verdict,
            closedAt: new Date().toISOString(),
            summary: 'Attempt finalized via completeAttempt'
        });
        this.saveTasks();
        this._fireUpdate(task);
    }
    abortTask(id) {
        const task = this.tasks[id];
        if (!task)
            return;
        if (['COMPLETED', 'FAILED', 'ABORTED'].includes(task.status))
            return;
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
        const baseDir = this.getStoragePath();
        if (fs.existsSync(baseDir)) {
            const taskDirs = fs.readdirSync(baseDir, { withFileTypes: true })
                .filter(dirent => dirent.isDirectory())
                .map(dirent => dirent.name);
            for (const taskId of taskDirs) {
                try {
                    const taskDir = path.join(baseDir, taskId);
                    fs.rmSync(taskDir, { recursive: true, force: true });
                }
                catch (err) {
                    console.error(`[TaskManager] Failed to delete task dir ${taskId}:`, err);
                }
            }
        }
        this.saveTasks();
        this._onDidTaskUpdate.fire({ id: 'cleared', status: types_1.TaskState.CREATED });
    }
    deleteTask(id) {
        if (this.tasks[id]) {
            delete this.tasks[id];
            // Delete from disk
            const taskDir = path.join(this.getStoragePath(), id);
            if (fs.existsSync(taskDir)) {
                try {
                    fs.rmSync(taskDir, { recursive: true, force: true });
                }
                catch (err) {
                    console.error(`[TaskManager] Failed to delete task dir ${id}:`, err);
                }
            }
            this.saveTasks();
            this._onDidTaskUpdate.fire({ id: 'deleted', status: types_1.TaskState.CREATED });
        }
    }
    async verifyTaskIntegrity(id) {
        const eventsPath = this.getTaskEventsPath(id);
        if (!fs.existsSync(eventsPath)) {
            return { verified: false, hash: '', driftDetails: 'Ledger file missing' };
        }
        try {
            const content = fs.readFileSync(eventsPath, 'utf8');
            const result = (0, integrity_1.verifyContentIntegrity)(content);
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