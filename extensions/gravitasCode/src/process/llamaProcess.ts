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
    private outputChannel: vscode.OutputChannel | null = null;
    private logStream: fs.WriteStream | null = null;
    private errorBuffer: string[] = [];
    private telemetryStr: string = 'Initializing...';
    private telemetryInterval: NodeJS.Timeout | null = null;

    constructor(private name: string, private type: 'coder' | 'reviewer') {
    }

    private getOutputChannel(): vscode.OutputChannel {
        if (!this.outputChannel) {
            this.outputChannel = vscode.window.createOutputChannel(`Gravitas: ${this.name}`);
        }
        return this.outputChannel;
    }

    public async start(binaryPath: string, modelConfig: any, additionalArgs: string[] = []): Promise<boolean> {
        if (this.process) {
            await this.stop();
        }

        if (this.telemetryInterval) clearInterval(this.telemetryInterval);
        this.telemetryStr = 'Spawning...';

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
            this.getOutputChannel().appendLine(`Failed to setup file logging: ${e}`);
        }

        const mode = modelConfig.mode || 'cpu';
        const isCpuMode = mode === 'cpu';

        const binaryPathResolved = resolveBinaryPath(binaryPath);
        const modelPathResolved = resolveTilde(modelConfig.modelPath);

        const socketDir = path.join(os.homedir(), '.gravitas', 'sockets');
        if (!fs.existsSync(socketDir)) {
            fs.mkdirSync(socketDir, { recursive: true });
        }
        
        // Remove existing socket if it exists to prevent bind errors
        const socketPath = path.join(socketDir, `${this.type}.sock`);
        if (fs.existsSync(socketPath)) {
            try { fs.unlinkSync(socketPath); } catch(e) {}
        }

        // Hardware Auto-Detection
        let nGpuLayers = modelConfig.nGpuLayers ?? (isCpuMode ? 0 : 33); // Default 33 for GPU
        let threads = modelConfig.threads ?? Math.max(1, os.cpus().length - 2);

        const binaryArgs = [
            '-m', modelPathResolved,
            '--host', socketPath,
            '--port', '0',
            '-c', modelConfig.contextSize.toString(),
            '--temp', modelConfig.temperature.toString(),
            '--parallel', '4', 
            '-ngl', nGpuLayers.toString(),
            '-t', threads.toString(),
            '--log-verbose',
            '--log-debug',
            ...additionalArgs
        ];
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

        const logger = require('../core/logger').CentralLogger.getInstance();
        logger.debug('system', `Spawning llama-server for ${this.name}: ${cmd} ${args.join(' ')}`);

        const cudaVisibleDevices = isCpuMode ? '' : (modelConfig.cudaVisibleDevices || '0');
        const startMsg = `[${new Date().toISOString()}] Starting ${this.name} [Mode: ${mode.toUpperCase()}] with command: ${cmd} ${args.join(' ')}\n`;
        this.getOutputChannel().appendLine(startMsg);
        this.logStream?.write(startMsg);

        logger.debug('system', `LlamaProcess [${this.name}]: BINARY_PATH: ${cmd}`);
        logger.debug('system', `LlamaProcess [${this.name}]: ARGS: ${JSON.stringify(args)}`);
        logger.debug('system', `LlamaProcess [${this.name}]: SPAWN_CMD: ${cmd} ${args.join(' ')}`);

        try {
            this.process = spawn(cmd, args, {
                env: { ...process.env, CUDA_VISIBLE_DEVICES: cudaVisibleDevices },
                detached: false
            });
            logger.info('system', `LlamaProcess [${this.name}]: Spawned with PID ${this.process.pid}`);
        } catch (e: any) {
            logger.error('system', `LlamaProcess [${this.name}]: Spawn failed: ${e.message}`);
            return false;
        }

        this.process.stdout?.on('data', (data) => {
            this.getOutputChannel().append(data.toString());
            this.logStream?.write(data);
        });
        this.process.stderr?.on('data', (data) => {
            const str = data.toString();
            this.getOutputChannel().append(str);
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
            this.getOutputChannel().appendLine(closeMsg);
            this.logStream?.write(closeMsg);
            this.process = null;
            this.updateStatus('stopped');
        });

        this.updateStatus('starting');
        this.telemetryInterval = setInterval(() => this.pollTelemetry(modelConfig), 2500);
        return true;
    }

    public async stop(): Promise<void> {
        if (this.process) {
            this.process.kill('SIGTERM');
            this.process = null;
        }
        if (this.telemetryInterval) {
            clearInterval(this.telemetryInterval);
            this.telemetryInterval = null;
        }
        if (this.logStream) {
            this.logStream.end();
            this.logStream = null;
        }
        
        const socketPath = path.join(os.homedir(), '.gravitas', 'sockets', `${this.type}.sock`);
        if (fs.existsSync(socketPath)) {
            try { fs.unlinkSync(socketPath); } catch(e) {}
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
        return this.telemetryStr;
    }

    private async pollTelemetry(config: any) {
        if (!this.process) return;
        try {
            // First run `nvidia-smi` to get VRAM
            const smi = await new Promise<string>((resolve) => {
                require('child_process').exec('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader', (err: any, stdout: string) => resolve(stdout || ''));
            });
            let vramStr = 'CPU Mode';
            const mode = config.mode || 'cpu';
            if (mode !== 'cpu' && smi && smi.trim() && !smi.includes('failed')) {
                const lines = smi.trim().split('\n');
                let deviceId = 0;
                if (config.cudaVisibleDevices && config.cudaVisibleDevices.toString().trim() !== '') {
                    deviceId = parseInt(config.cudaVisibleDevices.split(',')[0]);
                }
                
                if (lines[deviceId]) {
                    const [used, total] = lines[deviceId].split(',').map((s: string) => s.trim().split(' ')[0]);
                    const pct = Math.round((parseInt(used)/parseInt(total))*100);
                    vramStr = `VRAM: ${pct}% (${used}MB)`;
                }
            }

            // Now read /metrics from UDS to get TPS
            const sockPath = path.join(os.homedir(), '.gravitas', 'sockets', `${this.type}.sock`);
            if (fs.existsSync(sockPath)) {
                // Inline to avoid circular imports if any, using native http
                const { LlamaHttpClient } = require('../llm/llamaHttpClient');
                const client = new LlamaHttpClient(`unix://${sockPath}`);
                let tpsStr: string | null = null;
                try {
                    const metrics = await client.get('/metrics');
                    const tpsMatch = metrics.match(/predicted_tokens_seconds\s+([0-9.]+)/);
                    if (tpsMatch && parseFloat(tpsMatch[1]) > 0.1) {
                        tpsStr = `TPS: ${parseFloat(tpsMatch[1]).toFixed(1)}`;
                    } else if (parseFloat(tpsMatch?.[1] || "0") === 0) {
                        tpsStr = 'Idle';
                    }
                } catch(e) {}
                
                this.telemetryStr = `${vramStr}${tpsStr ? ' | ' + tpsStr : ''}`.trim();
            } else {
                 this.telemetryStr = vramStr;
            }
        } catch(e) {
            // Error silently ignored for telemetry ping
        }
    }
}
