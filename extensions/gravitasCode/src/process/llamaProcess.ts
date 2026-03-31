import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import { ModelConfigSchema, GravitasConfig } from '../core/config';
import { GravitasState, ProcessStatus } from '../core/state';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { resolveTilde, resolveBinaryPath } from '../utils/pathUtils';

export class LlamaProcess {
    private process: ChildProcess | null = null;
    private outputChannel: vscode.OutputChannel;
    private logStream: fs.WriteStream | null = null;
    private errorBuffer: string[] = [];

    constructor(private name: string, private type: 'coder' | 'reviewer') {
        this.outputChannel = vscode.window.createOutputChannel(`Gravitas: ${name}`);
    }

    public async start(binaryPath: string, modelConfig: any, additionalArgs: string[] = []): Promise<boolean> {
        if (this.process) {
            await this.stop();
        }

        // Reset buffer
        this.errorBuffer = [];

        // Setup file logging
        try {
            const logDir = path.join(os.homedir(), '.gravitas', 'logs');
            if (!fs.existsSync(logDir)) {
                fs.mkdirSync(logDir, { recursive: true });
            }
            this.logStream = fs.createWriteStream(path.join(logDir, `${this.type}.log`), { flags: 'a' });
        } catch (e) {
            this.outputChannel.appendLine(`Failed to setup file logging: ${e}`);
        }

        const mode = modelConfig.mode || 'cpu';
        const isCpuMode = mode === 'cpu';

        const binaryPathResolved = resolveBinaryPath(binaryPath);
        const modelPathResolved = resolveTilde(modelConfig.modelPath);

        const binaryArgs = [
            '-m', modelPathResolved,
            '--host', modelConfig.host || '127.0.0.1',
            '--port', modelConfig.port.toString(),
            '-c', modelConfig.contextSize.toString(),
            '--temp', modelConfig.temperature.toString(),
            '--parallel', '4', // Default to 4 parallel slots
            ...additionalArgs
        ];

        // Mode Logic: Force -ngl 0 if CPU mode, otherwise use config
        const nGpuLayers = isCpuMode ? 0 : (modelConfig.nGpuLayers ?? 0);
        binaryArgs.push('-ngl', nGpuLayers.toString());

        if (modelConfig.threads !== undefined) {
            binaryArgs.push('-t', modelConfig.threads.toString());
        }
        if (modelConfig.threadsBatch !== undefined) {
            binaryArgs.push('-tb', modelConfig.threadsBatch.toString());
        }
        if (modelConfig.batchSize !== undefined) {
            binaryArgs.push('-b', modelConfig.batchSize.toString());
        }
        if (modelConfig.ubatchSize !== undefined) {
            binaryArgs.push('-ub', modelConfig.ubatchSize.toString());
        }
        if (modelConfig.topP !== undefined) {
            binaryArgs.push('--top-p', modelConfig.topP.toString());
        }
        if (modelConfig.topK !== undefined) {
            binaryArgs.push('--top-k', modelConfig.topK.toString());
        }
        if (modelConfig.repeatPenalty !== undefined) {
            binaryArgs.push('--repeat-penalty', modelConfig.repeatPenalty.toString());
        }
        if (modelConfig.noWarmup) {
            binaryArgs.push('--no-warmup');
        }

        let cmd = binaryPath;
        let args = binaryArgs;

        // Handle prefix command (e.g. numactl)
        if (modelConfig.prefixCommand && modelConfig.prefixCommand.trim().length > 0) {
            const parts = modelConfig.prefixCommand.split(' ');
            cmd = parts[0];
            args = [...parts.slice(1), binaryPathResolved, ...binaryArgs];
        } else {
            cmd = binaryPathResolved;
        }

        const cudaVisibleDevices = isCpuMode ? '' : (modelConfig.cudaVisibleDevices || '0');
        const startMsg = `[${new Date().toISOString()}] Starting ${this.name} [Mode: ${mode.toUpperCase()}] with command: ${cmd} ${args.join(' ')}\n`;
        this.outputChannel.appendLine(startMsg);
        this.logStream?.write(startMsg);

        this.process = spawn(cmd, args, {
            env: { ...process.env, CUDA_VISIBLE_DEVICES: cudaVisibleDevices },
            detached: false
        });

        this.process.stdout?.on('data', (data) => {
            this.outputChannel.append(data.toString());
            this.logStream?.write(data);
        });
        this.process.stderr?.on('data', (data) => {
            const str = data.toString();
            this.outputChannel.append(str);
            this.logStream?.write(data);

            // Buffer last 10 lines
            const lines = str.split('\n');
            this.errorBuffer.push(...lines);
            if (this.errorBuffer.length > 20) {
                this.errorBuffer = this.errorBuffer.slice(this.errorBuffer.length - 20);
            }
        });

        this.process.on('close', (code) => {
            const closeMsg = `[${new Date().toISOString()}] Process ${this.name} exited with code ${code}\n`;
            this.outputChannel.appendLine(closeMsg);
            this.logStream?.write(closeMsg);
            this.process = null;
            this.updateStatus('stopped');
        });

        this.updateStatus('starting');
        return true;
    }

    public async stop(): Promise<void> {
        if (this.process) {
            this.process.kill();
            this.process = null;
        }
        if (this.logStream) {
            this.logStream.end();
            this.logStream = null;
        }
        this.updateStatus('stopped');
    }

    private updateStatus(status: ProcessStatus) {
        const state = GravitasState.getInstance();
        if (this.type === 'coder') {
            state.updateState({ coderStatus: status });
        } else {
            state.updateState({ reviewerStatus: status });
        }
    }

    public getLastError(): string {
        return this.errorBuffer.filter(l => l.trim().length > 0).join('\n');
    }

    public getName(): string { return this.name; }

    public getPid(): number | undefined {
        return this.process?.pid;
    }

    public getTelemetry(): string {
        // Simple heuristic: look for "eval time =" lines in recent logs
        // Example: "eval time = 123.45 ms / 10 token ( 81.00 tokens per second )"
        const lastRelevant = this.errorBuffer.reverse().find(l => l.includes('tokens per second'));
        if (lastRelevant) {
            const match = lastRelevant.match(/\(\s*([\d\.]+)\s*tokens per second/);
            return match ? `${match[1]} t/s` : '';
        }
        return '';
    }

}
