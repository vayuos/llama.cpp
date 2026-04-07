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
const registry_1 = require("../tools/registry");
/**
 * Authoritative wrapper for dual-mode tool execution.
 * (Gap 6: Hybrid Hardened)
 */
class ToolWrapper {
    constructor() {
        this.tm = taskManager_1.TaskManager.getInstance();
        this.registry = registry_1.NativeToolRegistry.getInstance();
    }
    /**
     * Executes a tool (Internal or External) securely and emits event-sourced telemetry.
     */
    async execute(taskId, command, args = [], cwd, label) {
        const toolExecId = (0, uuid_1.v4)();
        const start = new Date().toISOString();
        // 🛡️ Mode Selection: Check if the command matches a Native Tool
        if (this.registry.hasTool(command)) {
            return this.executeNative(taskId, toolExecId, command, args, start, label);
        }
        // --- FALLBACK: External Command (Spawn) ---
        return this.executeExternal(taskId, toolExecId, command, args, cwd, start, label);
    }
    /**
     * Executes an internal TypeScript-based tool.
     */
    async executeNative(taskId, toolExecId, name, args, start, label) {
        const tool = this.registry.getTool(name);
        // Emit Started
        this.tm.emitEvent(taskId, {
            type: 'ToolExecutionStarted',
            toolExecId,
            commandLine: `${name}(${args.join(', ')})`,
            workingDirectory: '[NATIVE_RUNTIME]',
            startedAt: start,
            commandLabel: label || tool.description
        });
        const logger = require('../core/logger').CentralLogger.getInstance();
        logger.info('system', `ToolWrapper: Starting NATIVE tool: ${name}(${args.join(', ')})`);
        try {
            // Attempt to parse JSON arguments if provided as first arg
            const parsedArgs = args.length > 0 ? JSON.parse(args[0]) : {};
            const result = await tool.execute(parsedArgs);
            const output = JSON.stringify(result, null, 2);
            this.tm.emitEvent(taskId, {
                type: 'ToolExecutionOutput',
                toolExecId,
                stream: 'stdout',
                text: output
            });
            logger.info('system', `ToolWrapper: NATIVE tool ${name} COMPLETED. Output: ${output.substring(0, 1000)}${output.length > 1000 ? '...' : ''}`);
            this.tm.emitEvent(taskId, {
                type: 'ToolExecutionCompleted',
                toolExecId,
                endedAt: new Date().toISOString(),
                exitCode: 0,
                status: 'SUCCESS'
            });
            return { exitCode: 0, output };
        }
        catch (err) {
            const errorMsg = `Native Tool Execution Error: ${err.message}`;
            this.tm.emitEvent(taskId, {
                type: 'ToolExecutionOutput',
                toolExecId,
                stream: 'stderr',
                text: errorMsg
            });
            this.tm.emitEvent(taskId, {
                type: 'ToolExecutionCompleted',
                toolExecId,
                endedAt: new Date().toISOString(),
                exitCode: 1,
                status: 'FAILURE'
            });
            return { exitCode: 1, output: errorMsg };
        }
    }
    /**
     * Executes an external shell-level command.
     */
    async executeExternal(taskId, toolExecId, command, args, cwd, start, label) {
        const options = {
            cwd: cwd || process.cwd(),
            shell: args.length === 0
        };
        this.tm.emitEvent(taskId, {
            type: 'ToolExecutionStarted',
            toolExecId,
            commandLine: args.length > 0 ? `${command} ${args.join(' ')}` : command,
            workingDirectory: options.cwd,
            startedAt: start,
            commandLabel: label || command
        });
        const logger = require('../core/logger').CentralLogger.getInstance();
        logger.info('system', `ToolWrapper: Spawning EXTERNAL process: ${command} ${args.join(' ')} (CWD: ${options.cwd})`);
        return new Promise((resolve) => {
            let output = '';
            const child = cp.spawn(command, args, options);
            child.stdout?.on('data', (data) => {
                const chunk = data.toString();
                output += chunk;
                this.tm.emitEvent(taskId, {
                    type: 'ToolExecutionOutput',
                    toolExecId,
                    stream: 'stdout',
                    text: chunk
                });
            });
            child.stderr?.on('data', (data) => {
                const chunk = data.toString();
                output += chunk;
                this.tm.emitEvent(taskId, {
                    type: 'ToolExecutionOutput',
                    toolExecId,
                    stream: 'stderr',
                    text: chunk
                });
            });
            child.on('error', (err) => {
                this.tm.emitEvent(taskId, {
                    type: 'ToolExecutionOutput',
                    toolExecId,
                    stream: 'stderr',
                    text: `Execution Error: ${err.message}`
                });
                const logger = require('../core/logger').CentralLogger.getInstance();
                logger.error('system', `ToolWrapper: Spawning EXTERNAL process error: ${err.message}`);
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
                logger.info('system', `ToolWrapper: EXTERNAL process ${command} COMPLETED. ExitCode: ${exitCode}. Final Output: ${output.substring(0, 1000)}${output.length > 1000 ? '...' : ''}`);
                resolve({ exitCode, output });
            });
        });
    }
}
exports.ToolWrapper = ToolWrapper;
//# sourceMappingURL=toolWrapper.js.map