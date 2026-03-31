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
exports.ToolWrapper = void 0;
const cp = __importStar(require("child_process"));
const taskManager_1 = require("./taskManager");
const uuid_1 = require("uuid");
/**
 * Authoritative wrapper for external tool execution. (Gap 6)
 */
class ToolWrapper {
    constructor() {
        this.tm = taskManager_1.TaskManager.getInstance();
    }
    async execute(taskId, command, cwd, label) {
        const toolExecId = (0, uuid_1.v4)();
        const start = new Date().toISOString();
        // Emit Started
        this.tm.emitEvent(taskId, {
            type: 'ToolExecutionStarted',
            toolExecId,
            commandLine: command,
            workingDirectory: cwd,
            startedAt: start,
            commandLabel: label
        });
        return new Promise((resolve) => {
            let output = '';
            const child = cp.spawn(command, { shell: true, cwd });
            child.stdout.on('data', (data) => {
                const chunk = data.toString();
                output += chunk;
                this.tm.emitEvent(taskId, {
                    type: 'ToolExecutionOutput',
                    toolExecId,
                    stream: 'stdout',
                    text: chunk
                });
            });
            child.stderr.on('data', (data) => {
                const chunk = data.toString();
                output += chunk;
                this.tm.emitEvent(taskId, {
                    type: 'ToolExecutionOutput',
                    toolExecId,
                    stream: 'stderr',
                    text: chunk
                });
            });
            child.on('close', (code) => {
                const exitCode = code ?? 0;
                this.tm.emitEvent(taskId, {
                    type: 'ToolExecutionCompleted',
                    toolExecId,
                    endedAt: new Date().toISOString(),
                    exitCode,
                    status: exitCode === 0 ? 'SUCCESS' : 'FAILURE'
                });
                resolve({ exitCode, output });
            });
        });
    }
}
exports.ToolWrapper = ToolWrapper;
//# sourceMappingURL=toolWrapper.js.map