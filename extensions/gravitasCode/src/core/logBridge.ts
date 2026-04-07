import * as vscode from 'vscode';
import { CentralLogger, LogEntry, LogSource } from './logger';
import { TaskManager } from '../uiv2/taskManager';

export class LogBridge {
    private static instance: LogBridge;
    private disposables: vscode.Disposable[] = [];

    private constructor() {
        this.setupListener();
    }

    public static initialize() {
        if (!LogBridge.instance) {
            LogBridge.instance = new LogBridge();
            console.log('Gravitas Code: LogBridge initialized (connecting CentralLogger -> TaskManager)');
        }
    }

    private setupListener() {
        CentralLogger.getInstance().onDidLog((entry: LogEntry) => {
            this.handleLogEntry(entry);
        }, null, this.disposables);
    }

    private handleLogEntry(entry: LogEntry) {
        // 🛡️ Loop Prevention: Only forward user-facing logs (coder, reviewer, validation)
        // system and ui logs should stay in the Master Log/Output Channel to avoid infinite TaskManager loops.
        const allowedSources: LogSource[] = ['coder', 'reviewer'];
        if (!allowedSources.includes(entry.source)) {
            return;
        }

        const timestamp = new Date(entry.timestamp).toLocaleTimeString();
        const line = `[${timestamp}] [${entry.source.toUpperCase()}] ${entry.message}\n`;

        // We need to know WHICH task to log to.
        // Currently, logging is global. We will create a mechanism in TaskManager
        // to direct this to the "latest active task" or broadcast.
        // TaskManager.getAllTasks() -> sort by date -> get latest running?

        // For now, let's look for a running task.
        const tasks = TaskManager.getInstance().getAllTasks();
        const runningTask = tasks.find(t => t.status === 'RUNNING' || t.status === 'CREATED');

        if (runningTask) {
            TaskManager.getInstance().addTerminalChunk(runningTask.id, line);
        } else {
            // No active task to pipe logs to.
            // In the future we might want a "System Task" or global console.
        }
    }

    public dispose() {
        this.disposables.forEach(d => d.dispose());
    }
}
