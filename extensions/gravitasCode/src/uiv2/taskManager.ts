import * as vscode from 'vscode';
import { v4 as uuidv4 } from 'uuid';
import { Task, TaskId, TaskState, TaskAttempt, TaskStore, TaskEvent } from './types';
import { EventValidator } from './eventValidator';
import { reduceTask, applyEvent } from './reducer';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { verifyContentIntegrity, IntegrityResult } from './integrity';

/**
 * Authoritative Task Store and Lifecycle Manager.
 * Implements Gap 1-3: State, Ledger, Pure Reducers.
 */
export class TaskManager {
    private static instance: TaskManager;
    private context: vscode.ExtensionContext;
    private tasks: TaskStore = {};
    private eventValidator: EventValidator;
    private _onDidTaskUpdate = new vscode.EventEmitter<Task>();
    private _onDidEmitEvent = new vscode.EventEmitter<{ taskId: TaskId; event: TaskEvent }>();
    private cpuUsageBaseline = process.cpuUsage();
    private lastTelemetryTime = Date.now();

    public readonly onDidTaskUpdate = this._onDidTaskUpdate.event;
    public readonly onDidEmitEvent = this._onDidEmitEvent.event;

    private constructor(context: vscode.ExtensionContext) {
        this.syncLog('TRACE: [TaskManager] Entering constructor...');
        this.context = context;
        this.syncLog('TRACE: [TaskManager] Creating EventValidator.getInstance()...');
        this.eventValidator = EventValidator.getInstance();
        this.syncLog('TRACE: [TaskManager] Ensuring Base Directory...');
        this.ensureBaseDir();
        this.syncLog('TRACE: [TaskManager] Loading Tasks from disk...');
        this.loadTasks();
        this.syncLog('TRACE: [TaskManager] Constructor finished.');
    }
    
    private syncLog(msg: string) {
        try {
            const fs = require('fs'), os = require('os'), path = require('path');
            const file = path.join(os.homedir(), '.gravitas', 'logs', 'boot_trace.log');
            fs.appendFileSync(file, `[${new Date().toISOString()}] ${msg}\n`, 'utf8');
        } catch(e) {}
    }

    private ensureBaseDir() {
        const baseDir = this.getStoragePath();
        if (!fs.existsSync(baseDir)) {
            fs.mkdirSync(baseDir, { recursive: true });
        }
    }

    private getStoragePath(): string {
        if (this.context.storageUri) {
            return this.context.storageUri.fsPath;
        }
        return path.join(this.context.globalStorageUri.fsPath, 'tasks');
    }

    private getTaskEventsPath(taskId: string): string {
        return path.join(this.getStoragePath(), taskId, 'events.jsonl');
    }

    public static initialize(context: vscode.ExtensionContext): TaskManager {
        if (!TaskManager.instance) {
            TaskManager.instance = new TaskManager(context);
        }
        return TaskManager.instance;
    }

    public static getInstance(): TaskManager {
        if (!TaskManager.instance) {
            throw new Error('TaskManager not initialized. Call initialize() first.');
        }
        return TaskManager.instance;
    }

    private loadTasks() {
        const baseDir = this.getStoragePath();
        this.syncLog(`TRACE: [TaskManager.loadTasks] Checking storage path: ${baseDir}`);
        if (!fs.existsSync(baseDir)) {
            this.syncLog('TRACE: [TaskManager.loadTasks] Storage path does not exist. Skipping recovery.');
            return;
        }

        this.syncLog(`TRACE: [TaskManager.loadTasks] Starting fs.readdirSync...`);

        const taskDirs = fs.readdirSync(baseDir, { withFileTypes: true })
            .filter(dirent => dirent.isDirectory())
            .map(dirent => dirent.name);
        
        this.syncLog(`TRACE: [TaskManager.loadTasks] Found ${taskDirs.length} potential task directories.`);

        for (const taskId of taskDirs) {
            try {
                const eventsPath = this.getTaskEventsPath(taskId);
                this.syncLog(`TRACE: [TaskManager.loadTasks] Loop index - eventsPath: ${eventsPath}`);
                if (fs.existsSync(eventsPath)) {
                    const stats = fs.statSync(eventsPath);
                    if (stats.size > 20 * 1024 * 1024) { // 20 MB safety limit
                        this.syncLog(`CRITICAL: Task ${taskId} events.jsonl is bloated (${stats.size} bytes). Skipping to prevent V8 crash.`);
                        try {
                            // Rename to prevent persistent crash loops on subsequent boots
                            fs.renameSync(eventsPath, eventsPath + '.corrupted');
                        } catch(e) {}
                        continue;
                    }

                    this.syncLog(`TRACE: [TaskManager.loadTasks] Reading JSON file...`);
                    const fileContent = fs.readFileSync(eventsPath, 'utf8');
                    this.syncLog(`TRACE: [TaskManager.loadTasks] Parse JSON (Length: ${fileContent.length})`);
                    const events = fileContent
                        .split('\n')
                        .filter(line => line.trim())
                        .map(line => JSON.parse(line) as TaskEvent);

                    if (events.length > 0) {
                        this.syncLog(`TRACE: [TaskManager.loadTasks] Calling rebuildTaskFromEvents`);
                        this.tasks[taskId] = this.rebuildTaskFromEvents(taskId, events);
                        this.syncLog(`TRACE: [TaskManager.loadTasks] Task ${taskId} recovered with ${events.length} events.`);
                    }
                } else {
                    this.syncLog(`TRACE: [TaskManager.loadTasks] No events file: ${eventsPath}`);
                }
            } catch (err) {
                console.error(`TRACE: [TaskManager] Failed to restore task ${taskId}:`, err);
            }
        }
        console.log('TRACE: [TaskManager] Task recovery sequence completed.');
    }

    private rebuildTaskFromEvents(taskId: string, events: TaskEvent[]): Task {
        return reduceTask(taskId, events);
    }

    public getTask(id: TaskId): Task | undefined {
        return this.tasks[id];
    }

    public getAllTasks(): Task[] {
        return Object.values(this.tasks).sort((a, b) => b.createdAt - a.createdAt);
    }

    public getLastTask(): Task | undefined {
        const all = this.getAllTasks();
        return all.length > 0 ? all[0] : undefined;
    }

    private getCurrentAttempt(task: Task): TaskAttempt | undefined {
        return task.attempts[task.attempts.length - 1];
    }

    private async saveTasks() {
        await this.context.globalState.update('gravitas_tasks_v2_ids', Object.keys(this.tasks));
    }

    public createTask(command: string, origin: 'user' | 'system' = 'user', parentTaskId?: string, regenerationType?: 'REPLAY' | 'REGENERATE_SAME' | 'REGENERATE_MODIFIED'): Task {
        const id = uuidv4();
        const now = Date.now();

        const newTask: Task = {
            id,
            createdAt: now,
            origin,
            command,
            status: TaskState.CREATED,
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

    public updateTaskState(id: TaskId, newState: TaskState) {
        const task = this.tasks[id];
        if (!task) return;
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

    public startNextAttempt(id: TaskId): TaskAttempt {
        const task = this.tasks[id];
        if (!task) throw new Error('Task not found');

        const lastAttempt = this.getCurrentAttempt(task);
        if (lastAttempt && lastAttempt.state === 'OPEN') {
            lastAttempt.state = 'CLOSED';
            lastAttempt.endedAt = Date.now();
            lastAttempt.verdict = lastAttempt.verdict || 'INCOMPLETE';
        }

        const attemptNo = task.attempts.length + 1;
        const attemptId = uuidv4();

        const attempt: TaskAttempt = {
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

    public startPhase(taskId: TaskId, actor: 'coder' | 'reviewer' | 'system', title: string): string {
        const task = this.tasks[taskId];
        if (!task) throw new Error('Task not found');
        const attempt = this.getCurrentAttempt(task);
        if (!attempt || attempt.state === 'CLOSED') throw new Error('No open attempt');

        const lastPhase = attempt.phases[attempt.phases.length - 1];
        if (lastPhase && lastPhase.status === 'RUNNING') {
            lastPhase.status = 'COMPLETED';
            lastPhase.endedAt = Date.now();
        }

        const phaseId = uuidv4();
        const newPhase: import('./types').TaskPhase = {
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

    public completePhase(taskId: TaskId) {
        const task = this.tasks[taskId];
        if (!task) return;
        const attempt = this.getCurrentAttempt(task);
        if (!attempt) return;

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

    public bindAgent(taskId: TaskId, phaseId: string, agentId: string, modelId: string, configFingerprint?: string) {
        this.emitEvent(taskId, {
            type: 'AgentBoundToPhase',
            phaseId,
            agentId,
            modelId,
            configFingerprint
        });
    }

    public recordPhaseMetrics(taskId: TaskId, phaseId: string, durationMs: number, tokenCount?: number, costEstimate?: number) {
        this.emitEvent(taskId, {
            type: 'PhaseMetricsReported',
            phaseId,
            durationMs,
            tokenCount,
            costEstimate
        });
    }

    public recordPolicyDecision(taskId: TaskId, policyName: string, decision: 'ALLOW' | 'DENY' | 'ABORT' | 'OVERRIDE', reasoning: string) {
        this.emitEvent(taskId, {
            type: 'PolicyEvaluated',
            policyName,
            decision,
            reasoning
        });
    }

    public recordArtifact(taskId: TaskId, filePath: string, type: string, metadata?: any) {
        const fileName = path.basename(filePath);
        this.emitEvent(taskId, {
            type: 'ArtifactProduced',
            artifactId: uuidv4(),
            name: fileName,
            path: filePath,
            artifactType: type as any,
            producedAt: new Date().toISOString(),
            metadata
        });
    }

    public recordUserAction(taskId: TaskId, actionType: 'REGENERATE_CLICKED' | 'ARTIFACT_VIEWED' | 'ABORT_CLICKED' | 'SETTINGS_CHANGED', targetId?: string, metadata?: any) {
        this.emitEvent(taskId, {
            type: 'UserActionPerformed',
            actionType,
            targetId,
            metadata
        });
    }

    public emitEvent(id: TaskId, event: any) {
        const task = this.tasks[id];
        if (!task) return;

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
            event.eventId = uuidv4();
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
        this.tasks[id] = applyEvent(task, event);

        // PERSISTENCE
        try {
            const eventPath = this.getTaskEventsPath(id);
            const line = JSON.stringify(event) + '\n';
            fs.appendFileSync(eventPath, line, 'utf8');
            const logger = require('../core/logger').CentralLogger.getInstance();
            logger.debug('system', `Task ${id}: Persisted event ${event.type} (${line.length} bytes)`);
        } catch (err) {
            console.error(`[TaskManager] Failed to persist event:`, err);
        }

        this.saveTasks();
        this._fireUpdate(this.tasks[id]);
        this._onDidEmitEvent.fire({ taskId: id, event });
    }

    public sampleResources(taskId: TaskId) {
        const mem = process.memoryUsage();
        const ramMb = Math.round(mem.rss / 1024 / 1024);
        const now = Date.now();
        const cpuUsage = process.cpuUsage(this.cpuUsageBaseline);
        const elapsedUs = (now - this.lastTelemetryTime) * 1000;
        const totalUs = cpuUsage.user + cpuUsage.system;
        const cpuPercent = elapsedUs > 0 ? Math.min(100, Math.round((totalUs / elapsedUs) * 100)) : 0;

        this.cpuUsageBaseline = process.cpuUsage();
        this.lastTelemetryTime = now;

        this.emitEvent(taskId, {
            type: 'ResourceUsageSampled',
            resources: {
                ramMb,
                vramMb: 0, // TelemetryService handles real VRAM
                cpuPercent
            }
        });
    }

    public emitStreamingChunk(taskId: TaskId, chunk: string, stage: 'thought' | 'implementation' | 'review') {
        const task = this.tasks[taskId];
        if (!task) return;
        
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
                eventId: uuidv4()
            } as any
        } as any);
    }

    public addTerminalChunk(id: TaskId, data: string, toolExecId: string = 'root', stream: 'stdout' | 'stderr' = 'stdout') {
        this.emitEvent(id, {
            type: 'ToolExecutionOutput',
            toolExecId,
            stream,
            text: data
        });
    }

    public completeTask(id: TaskId, summary: string) {
        const task = this.tasks[id];
        if (!task) return;
        task.summary = summary;
        task.status = TaskState.COMPLETED;
        task.updatedAt = Date.now();
        this.saveTasks();
        this._fireUpdate(task);
    }

    public failTask(id: TaskId, reason: string) {
        const task = this.tasks[id];
        if (!task) return;
        task.summary = `Failed: ${reason}`;
        task.status = TaskState.FAILED;
        task.updatedAt = Date.now();
        this.saveTasks();
        this._fireUpdate(task);
    }

    public completeAttempt(taskId: TaskId, verdict: import('./types').AttemptVerdict) {
        const task = this.tasks[taskId];
        if (!task) return;
        const attempt = this.getCurrentAttempt(task);
        if (!attempt || attempt.state === 'CLOSED') return;

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

    public abortTask(id: TaskId) {
        const task = this.tasks[id];
        if (!task) return;
        if (['COMPLETED', 'FAILED', 'ABORTED'].includes(task.status)) return;

        const attempt = this.getCurrentAttempt(task);
        if (attempt && attempt.state === 'OPEN') {
            attempt.verdict = 'INCOMPLETE';
            attempt.state = 'CLOSED';
            attempt.endedAt = Date.now();
        }

        task.status = TaskState.ABORTED;
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

    public clearAllTasks() {
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
                } catch (err) {
                    console.error(`[TaskManager] Failed to delete task dir ${taskId}:`, err);
                }
            }
        }
        this.saveTasks();
        this._onDidTaskUpdate.fire({ id: 'cleared', status: TaskState.CREATED } as any);
    }

    public deleteTask(id: TaskId) {
        if (this.tasks[id]) {
            delete this.tasks[id];
            
            // Delete from disk
            const taskDir = path.join(this.getStoragePath(), id);
            if (fs.existsSync(taskDir)) {
                try {
                    fs.rmSync(taskDir, { recursive: true, force: true });
                } catch (err) {
                    console.error(`[TaskManager] Failed to delete task dir ${id}:`, err);
                }
            }

            this.saveTasks();
            this._onDidTaskUpdate.fire({ id: 'deleted', status: TaskState.CREATED } as any);
        }
    }

    public async verifyTaskIntegrity(id: TaskId): Promise<IntegrityResult> {
        const eventsPath = this.getTaskEventsPath(id);
        if (!fs.existsSync(eventsPath)) {
            return { verified: false, hash: '', driftDetails: 'Ledger file missing' };
        }
        try {
            const content = fs.readFileSync(eventsPath, 'utf8');
            const result = verifyContentIntegrity(content);
            return result;
        } catch (err: any) {
            return { verified: false, hash: '', driftDetails: err.message };
        }
    }

    private _fireUpdate(task: Task) {
        this._onDidTaskUpdate.fire(task);
    }
}
