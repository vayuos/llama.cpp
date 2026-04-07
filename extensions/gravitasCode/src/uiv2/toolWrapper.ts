import * as cp from 'child_process';
import { TaskId } from './types';
import { TaskManager } from './taskManager';
import { v4 as uuidv4 } from 'uuid';
import { NativeToolRegistry } from '../tools/registry';

/**
 * Authoritative wrapper for dual-mode tool execution.
 * (Gap 6: Hybrid Hardened)
 */
export class ToolWrapper {
    private tm: TaskManager;
    private registry: NativeToolRegistry;

    constructor() {
        this.tm = TaskManager.getInstance();
        this.registry = NativeToolRegistry.getInstance();
    }

    /**
     * Executes a tool (Internal or External) securely and emits event-sourced telemetry.
     */
    public async execute(taskId: TaskId, command: string, args: string[] = [], cwd?: string, label?: string): Promise<{ exitCode: number; output: string }> {
        const toolExecId = uuidv4();
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
    private async executeNative(taskId: TaskId, toolExecId: string, name: string, args: string[], start: string, label?: string): Promise<{ exitCode: number; output: string }> {
        const tool = this.registry.getTool(name)!;
        
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
        } catch (err: any) {
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
    private async executeExternal(taskId: TaskId, toolExecId: string, command: string, args: string[], cwd: string | undefined, start: string, label?: string): Promise<{ exitCode: number; output: string }> {
        const options: cp.SpawnOptions = { 
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
