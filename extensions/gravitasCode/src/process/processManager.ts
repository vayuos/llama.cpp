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
    private _coder?: LlamaProcess;
    private _reviewer?: LlamaProcess;

    private constructor() {}

    private getCoder(): LlamaProcess {
        if (!this._coder) {
            this._coder = new LlamaProcess('Coder Server', 'coder');
        }
        return this._coder;
    }

    private getReviewer(): LlamaProcess {
        if (!this._reviewer) {
            this._reviewer = new LlamaProcess('Reviewer Server', 'reviewer');
        }
        return this._reviewer;
    }

    public static getInstance(): UnifiedProcessManager {
        if (!UnifiedProcessManager.instance) {
            UnifiedProcessManager.instance = new UnifiedProcessManager();
        }
        return UnifiedProcessManager.instance;
    }

    private getLogger() {
        return require('../core/logger').CentralLogger.getInstance();
    }

    private getBinaryPath(config: GravitasConfig, modelConfig: any): string {
        const monorepoBin = '/home/viren/llama/llama.cpp/build_mmq_moe/bin/llama-server';
        
        // Priority 1: Specifically configured path in settings
        if (modelConfig.binPath?.trim()) return modelConfig.binPath.trim();
        
        // Priority 2: Global extension setting
        if ((config as any).llamaBinPath) return (config as any).llamaBinPath;

        // Priority 3: Strictly enforce use of the native monorepo build per "use it only" directive
        if (fs.existsSync(monorepoBin)) {
            return monorepoBin;
        }

        // Fallback (only if monorepoBin is missing): System PATH
        return 'llama-server';
    }

    public async startCoder(config: GravitasConfig): Promise<boolean> {
        this.getLogger().debug('system', `UnifiedProcessManager: Requested startCoder. Mode: ${config.connection.mode}`);
        if (config.connection.mode === 'remote') {
            this.getLogger().info('system', `Remote Mode: Using Coder on System3 (${config.connection.system3Ip})`);
            vscode.window.showInformationMessage(`Remote Mode: Coder should be running on System3 (${config.connection.system3Ip}:8080)`);
            return true;
        }
        const c = config.coder as any;
        const binPath = this.getBinaryPath(config, c);
        return this.getCoder().start(binPath, c, []);
    }

    public async startReviewer(config: GravitasConfig): Promise<boolean> {
        this.getLogger().debug('system', `UnifiedProcessManager: Requested startReviewer. Mode: ${config.connection.mode}`);
        if (config.connection.mode === 'remote') {
            this.getLogger().info('system', `Remote Mode: Using Reviewer on System3 (${config.connection.system3Ip})`);
            vscode.window.showInformationMessage(`Remote Mode: Reviewer should be running on System3 (${config.connection.system3Ip}:18080)`);
            return true;
        }
        const r = config.reviewer as any;
        const binPath = this.getBinaryPath(config, r);
        return this.getReviewer().start(binPath, r, []);
    }

    public async stopAll(): Promise<void> {
        this.getLogger().info('system', 'UnifiedProcessManager: Stopping all local LLM processes...');
        await Promise.all([this.getCoder().stop(), this.getReviewer().stop()]);
    }

    public getLastError(type: 'coder' | 'reviewer'): string {
        return type === 'coder' ? this.getCoder().getLastError() : this.getReviewer().getLastError();
    }

    public async getLiveStatus(type: 'coder' | 'reviewer', config: GravitasConfig): Promise<{ running: boolean; external: boolean; pid?: number; telemetry?: string }> {
        const proc = type === 'coder' ? this.getCoder() : this.getReviewer();
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
        const proc = type === 'coder' ? this.getCoder() : this.getReviewer();
        return {
            pid: proc.getPid(),
            telemetry: proc.getTelemetry()
        };
    }
}
