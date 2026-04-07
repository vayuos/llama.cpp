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
        const logger = require('../core/logger').CentralLogger.getInstance();
        logger.debug('system', 'TelemetryService: Starting global polling cycle...');
        const config = await ConfigManager.getInstance().loadConfig();
        if (!config) return;

        await this.pollAgent('coder', config.coder, config);
        await this.pollAgent('reviewer', config.reviewer, config);
        this._onDidUpdate.fire();
        logger.debug('system', 'TelemetryService: Global polling cycle completed.');
    }

    private async pollAgent(type: 'coder' | 'reviewer', modelConfig: any, fullConfig: any) {
        const isRemote = fullConfig.connection.mode === 'remote';
        const start = Date.now();
        
        try {
            const client = new LlamaHttpClient(modelConfig.baseUrl);
            
            // --- RESILIENCE: Try health first ---
            let status: 'online' | 'offline' = 'offline';
            try {
                const health = await client.get('/health');
                if (health && health.status === 'ok') status = 'online';
            } catch (e) {
                // If health fails, it's truly offline
                throw e; 
            }

            let metrics: string = '';
            try {
                metrics = await client.get('/metrics');
            } catch (e) {
                // If metrics fails but health worked, we are still "online" but with no data
                metrics = '';
            }
            
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
                status: status,
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
            const logger = require('../core/logger').CentralLogger.getInstance();
            if (config.mode === 'cpu') return resolve('CPU');
            
            const cmd = 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader';
            logger.debug('system', `TelemetryService: Executing hardware check: ${cmd}`);
            
            exec(cmd, (err, stdout) => {
                if (err || !stdout) {
                    logger.warn('system', `TelemetryService: nvidia-smi failed: ${err?.message}`);
                    return resolve('GPU');
                }
                const parts = stdout.split(',').map(s => s.trim());
                if (parts.length >= 2) {
                    const used = parseInt(parts[0]);
                    const total = parseInt(parts[1]);
                    const pct = Math.round((used/total)*100);
                    logger.debug('system', `TelemetryService: VRAM Check Result: ${used}/${total} MiB (${pct}%)`);
                    resolve(`${pct}%`);
                } else {
                    resolve('GPU');
                }
            });
        });
    }
}
