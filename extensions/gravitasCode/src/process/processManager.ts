import axios from 'axios';
import { LlamaProcess } from './llamaProcess';
import { GravitasConfig } from '../core/config';
import { LlamaHttpClient } from '../llm/llamaHttpClient';
import { TelemetryService } from '../llm/telemetry';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';
import * as vscode from 'vscode';

export class UnifiedProcessManager {
    private static instance: UnifiedProcessManager;
    private coder: LlamaProcess;
    private reviewer: LlamaProcess;

    private constructor() {
        this.coder = new LlamaProcess('Coder Server', 'coder');
        this.reviewer = new LlamaProcess('Reviewer Server', 'reviewer');
    }

    public static getInstance(): UnifiedProcessManager {
        if (!UnifiedProcessManager.instance) {
            UnifiedProcessManager.instance = new UnifiedProcessManager();
        }
        return UnifiedProcessManager.instance;
    }

    public async startCoder(config: GravitasConfig): Promise<boolean> {
        if (config.connection.mode === 'remote') {
            vscode.window.showInformationMessage(`Remote Mode: Coder should be running on System3 (${config.connection.system3Ip}:8080)`);
            return true;
        }
        const c = config.coder as any;
        const monorepoBin = '/home/viren/llama/llama.cpp/build_cuda_mmq_moe/bin/llama-server';
        const binPath = c.binPath?.trim() ? c.binPath : (config as any).llamaBinPath || monorepoBin;
        return this.coder.start(binPath, c, []);
    }

    public async startReviewer(config: GravitasConfig): Promise<boolean> {
        if (config.connection.mode === 'remote') {
            vscode.window.showInformationMessage(`Remote Mode: Reviewer should be running on System3 (${config.connection.system3Ip}:18080)`);
            return true;
        }
        const r = config.reviewer as any;
        const monorepoBin = '/home/viren/llama/llama.cpp/build_cuda_mmq_moe/bin/llama-server';
        const binPath = r.binPath?.trim() ? r.binPath : (config as any).llamaBinPath || monorepoBin;
        return this.reviewer.start(binPath, r, []);
    }

    public async stopAll(): Promise<void> {
        await Promise.all([this.coder.stop(), this.reviewer.stop()]);
    }

    public getLastError(type: 'coder' | 'reviewer'): string {
        return type === 'coder' ? this.coder.getLastError() : this.reviewer.getLastError();
    }

    public async getLiveStatus(type: 'coder' | 'reviewer', config: GravitasConfig): Promise<{ running: boolean; external: boolean; pid?: number; telemetry?: string }> {
        const proc = type === 'coder' ? this.coder : this.reviewer;
        const localPid = proc.getPid();
        
        if (localPid) {
            return { running: true, external: false, pid: localPid, telemetry: proc.getTelemetry() };
        }

        // Pulse check for external or remote process
        const isRemote = config.connection.mode === 'remote';
        const telemetry = TelemetryService.getInstance().getTelemetry(type);
        
        if (telemetry.status === 'online') {
            return { 
                running: true, 
                external: !localPid, 
                telemetry: `${telemetry.vram} | ${telemetry.tps} | ${telemetry.latency}` 
            };
        }

        return { running: false, external: false };
    }

    public getProcessStatus(type: 'coder' | 'reviewer'): { pid?: number; telemetry?: string } {
        const proc = type === 'coder' ? this.coder : this.reviewer;
        return {
            pid: proc.getPid(),
            telemetry: proc.getTelemetry()
        };
    }
}
