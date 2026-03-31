import { LlamaHttpClient } from './llamaHttpClient';
import { ConfigManager } from '../core/config';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import { exec } from 'child_process';
import * as vscode from 'vscode';

export interface TelemetryData {
    status: 'online' | 'offline' | 'error';
    vram: string;
    tps: string;
    promptTps: string;
    slots: string;
    latency: string;
    load: string;
}

export class TelemetryService {
    private static instance: TelemetryService;
    private pollInterval: NodeJS.Timeout | null = null;
    private state: Map<string, TelemetryData> = new Map();
    private _onDidUpdate = new vscode.EventEmitter<void>();
    public readonly onDidUpdate = this._onDidUpdate.event;

    private constructor() {}

    public static getInstance(): TelemetryService {
        if (!TelemetryService.instance) {
            TelemetryService.instance = new TelemetryService();
        }
        return TelemetryService.instance;
    }

    public startPolling() {
        if (this.pollInterval) return;
        this.pollInterval = setInterval(() => this.pollAll(), 5000);
        this.pollAll();
    }

    public stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    public getTelemetry(type: 'coder' | 'reviewer'): TelemetryData {
        return this.state.get(type) || {
            status: 'offline',
            vram: '0%',
            tps: '0 strategy',
            promptTps: '0',
            slots: '0%',
            latency: '---',
            load: 'Idle'
        };
    }

    private async pollAll() {
        const config = await ConfigManager.getInstance().loadConfig();
        if (!config) return;

        await this.pollAgent('coder', config.coder, config);
        await this.pollAgent('reviewer', config.reviewer, config);
        this._onDidUpdate.fire();
    }

    private async pollAgent(type: 'coder' | 'reviewer', modelConfig: any, fullConfig: any) {
        const isRemote = fullConfig.connection.mode === 'remote';
        const start = Date.now();
        
        try {
            const client = new LlamaHttpClient(modelConfig.baseUrl);
            const metrics = await client.get('/metrics');
            const latency = `${Date.now() - start}ms`;
            
            // Parse TPS
            const genTpsMatch = metrics.match(/predicted_tokens_seconds\s+([0-9.]+)/);
            const promptTpsMatch = metrics.match(/prompt_tokens_seconds\s+([0-9.]+)/);
            
            const genTps = genTpsMatch ? parseFloat(genTpsMatch[1]) : 0;
            const promptTps = promptTpsMatch ? parseFloat(promptTpsMatch[1]) : 0;

            // Parse Slots
            const slotsMatch = metrics.match(/kv_cache_usage_ratio\s+([0-9.]+)/);
            const slots = slotsMatch ? `${Math.round(parseFloat(slotsMatch[1])*100)}%` : '0%';

            // Parse Load/Activity
            let load = 'Idle';
            if (genTps > 0) load = 'Generating';
            else if (promptTps > 0) load = 'Processing';

            // Parse VRAM
            let vram = '0%';
            if (isRemote) {
                try {
                    const hw = await client.get('/v1/hardware');
                    if (hw && hw.vram) {
                        vram = `${Math.round(hw.vram.used / hw.vram.total * 100)}%`;
                    }
                } catch(e) { vram = '---'; }
            } else {
                vram = await this.getLocalVram(modelConfig);
            }

            this.state.set(type, {
                status: 'online',
                vram,
                tps: genTps > 0 ? `${genTps.toFixed(1)}` : '0.0',
                promptTps: promptTps > 0 ? `${promptTps.toFixed(1)}` : '0.0',
                slots,
                latency,
                load
            });
        } catch (e) {
            this.state.set(type, {
                status: 'offline',
                vram: '0%',
                tps: '0.0',
                promptTps: '0.0',
                slots: '0%',
                latency: '---',
                load: 'Offline'
            });
        }
    }

    private getLocalVram(config: any): Promise<string> {
        return new Promise((resolve) => {
            if (config.mode === 'cpu') return resolve('CPU');
            exec('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader', (err, stdout) => {
                if (err || !stdout) return resolve('GPU');
                const parts = stdout.split(',').map(s => s.trim());
                if (parts.length >= 2) {
                    const used = parseInt(parts[0]);
                    const total = parseInt(parts[1]);
                    resolve(`${Math.round((used/total)*100)}%`);
                } else {
                    resolve('GPU');
                }
            });
        });
    }
}
