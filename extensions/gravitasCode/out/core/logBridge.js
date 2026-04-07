"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LogBridge = void 0;
const logger_1 = require("./logger");
const taskManager_1 = require("../uiv2/taskManager");
class LogBridge {
    constructor() {
        this.disposables = [];
        this.setupListener();
    }
    static initialize() {
        if (!LogBridge.instance) {
            LogBridge.instance = new LogBridge();
            console.log('Gravitas Code: LogBridge initialized (connecting CentralLogger -> TaskManager)');
        }
    }
    setupListener() {
        logger_1.CentralLogger.getInstance().onDidLog((entry) => {
            this.handleLogEntry(entry);
        }, null, this.disposables);
    }
    handleLogEntry(entry) {
        // 🛡️ Loop Prevention: Only forward user-facing logs (coder, reviewer, validation)
        // system and ui logs should stay in the Master Log/Output Channel to avoid infinite TaskManager loops.
        const allowedSources = ['coder', 'reviewer'];
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
        const tasks = taskManager_1.TaskManager.getInstance().getAllTasks();
        const runningTask = tasks.find(t => t.status === 'RUNNING' || t.status === 'CREATED');
        if (runningTask) {
            taskManager_1.TaskManager.getInstance().addTerminalChunk(runningTask.id, line);
        }
        else {
            // No active task to pipe logs to.
            // In the future we might want a "System Task" or global console.
        }
    }
    dispose() {
        this.disposables.forEach(d => d.dispose());
    }
}
exports.LogBridge = LogBridge;
//# sourceMappingURL=logBridge.js.map