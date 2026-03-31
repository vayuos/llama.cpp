import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import { ModelConfigSchema, GravitasConfig } from '../core/config';
import { GravitasState, ProcessStatus } from '../core/state';

export class LlamaProcess {
    private process: ChildProcess | null = null;
    private outputChannel: vscode.OutputChannel;

    constructor(private name: string, private type: 'coder' | 'reviewer') {
        this.outputChannel = vscode.window.createOutputChannel(`Gravitas: ${name}`);
    }

    public async start(binaryPath: string, modelConfig: any, additionalArgs: string[] = []): Promise<boolean> {
        if (this.process) {
            await this.stop();
        }

        const args = [
            '-m', modelConfig.modelPath,
            '--port', modelConfig.port.toString(),
            '-c', modelConfig.ctx.toString(),
            '--temp', modelConfig.temp.toString(),
            ...additionalArgs
        ];

        if (modelConfig.gpuLayers !== undefined) {
            args.push('-ngl', modelConfig.gpuLayers.toString());
        }
        if (modelConfig.threads !== undefined) {
            args.push('-t', modelConfig.threads.toString());
        }

        this.outputChannel.appendLine(`Starting ${this.name} with command: ${binaryPath} ${args.join(' ')}`);

        this.process = spawn(binaryPath, args, {
            env: { ...process.env, CUDA_VISIBLE_DEVICES: '0' }, // Default for now
            detached: false
        });

        this.process.stdout?.on('data', (data) => this.outputChannel.append(data.toString()));
        this.process.stderr?.on('data', (data) => this.outputChannel.append(data.toString()));

        this.process.on('close', (code) => {
            this.outputChannel.appendLine(`Process ${this.name} exited with code ${code}`);
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

    public getName(): string { return this.name; }
}
