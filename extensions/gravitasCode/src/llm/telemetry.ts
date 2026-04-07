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
    driver?: string;
    mode?: string;
}

export class TelemetryService {
    private static instance: TelemetryService;
    private pollInterval: NodeJS.Timeout | null = null;
    private state: Map<string, TelemetryData> = new Map();
    private _onDidUpdate = new vscode.EventEmitter<void>();
    public readonly onDidUpdate = this._onDidUpdate.event;
    private detectedDriver: 'nvidia' | 'amd' | 'generic' | null = null;

    private constructor() {}

    public static getInstance(): TelemetryService {
        if (!TelemetryService.instance) {
            TelemetryService.instance = new TelemetryService();
        }
        return TelemetryService.instance;
    }

    public startPolling() {
        if (this.pollInterval) return;
        this.pollInterval = setInterval(() => this.pollAll(), 1000);
        this.pollAll();
    }

    public stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    public getTelemetry(type: 'coder' | 'reviewer' | 'rag'): TelemetryData {
        return this.state.get(type) || {
            status: 'offline',
            vram: '0%',
            tps: '0.0',
            promptTps: '0',
            slots: '0%',
            latency: '---',
            load: 'Idle'
        };
    }

    private async pollAll() {
        const logger = require('../core/logger').CentralLogger.getInstance();
        logger.debug('system', 'TelemetryService: Starting global parallel polling cycle...');
        const config = await ConfigManager.getInstance().loadConfig();
        if (!config) return;

        // Concurrent polling for performance (Agents + RAG)
        await Promise.allSettled([
            this.pollAgent('coder', config.coder, config),
            this.pollAgent('reviewer', config.reviewer, config),
            this.pollRag(config)
        ]);

        this._onDidUpdate.fire();
        logger.debug('system', 'TelemetryService: Global parallel polling cycle completed.');
    }

    private async pollAgent(type: 'coder' | 'reviewer', modelConfig: any, fullConfig: any) {
        const isRemote = fullConfig.connection.mode === 'remote';
        const start = Date.now();
        const logger = require('../core/logger').CentralLogger.getInstance();
        
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
                metrics = '';
            }
            
            const latency = `${Date.now() - start}ms`;
            
            // Parse TPS (predicted_tokens_seconds)
            const genTpsMatch = metrics.match(/predicted_tokens_seconds\s+([0-9.]+)/);
            const promptTpsMatch = metrics.match(/prompt_tokens_seconds\s+([0-9.]+)/);
            
            const genTps = genTpsMatch ? parseFloat(genTpsMatch[1]) : 0.0;
            const promptTps = promptTpsMatch ? parseFloat(promptTpsMatch[1]) : 0.0;
 
            // Parse KV Cache (kv_cache_usage_ratio)
            const slotsMatch = metrics.match(/kv_cache_usage_ratio\s+([0-9.]+)/);
            const slotsPct = slotsMatch ? Math.round(parseFloat(slotsMatch[1]) * 100) : 0;
 
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
 
            const monitor = this.detectMonitor();
            this.state.set(type, {
                status: status,
                vram,
                tps: genTps > 0 ? `${genTps.toFixed(1)}` : '0.0',
                promptTps: promptTps > 0 ? `${promptTps.toFixed(1)}` : '0.0',
                slots: `${slotsPct}%`,
                latency,
                load,
                driver: monitor.toUpperCase(),
                mode: isRemote ? 'REMOTE' : 'LOCAL'
            });
        } catch (e: any) {
            logger.debug('system', `TelemetryService: Failed to poll ${type}. Offline. (Err: ${e.message})`);
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

    private detectMonitor(): 'nvidia' | 'amd' | 'generic' {
        if (this.detectedDriver) return this.detectedDriver;
        
        try {
            if (fs.existsSync('/usr/bin/nvidia-smi')) {
                this.detectedDriver = 'nvidia';
            } else if (fs.existsSync('/usr/bin/rocm-smi')) {
                this.detectedDriver = 'amd';
            } else {
                this.detectedDriver = 'generic';
            }
        } catch (e) {
            this.detectedDriver = 'generic';
        }
        return this.detectedDriver;
    }

    private getLocalVram(config: any): Promise<string> {
        return new Promise((resolve) => {
            const logger = require('../core/logger').CentralLogger.getInstance();
            if (config.mode === 'cpu') return resolve('CPU');
            
            const monitor = this.detectMonitor();
            let cmd = '';
            
            if (monitor === 'nvidia') {
                cmd = 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader';
            } else if (monitor === 'amd') {
                cmd = 'rocm-smi --showmeminfo vram --json';
            } else {
                return resolve('GPU');
            }

            logger.debug('system', `TelemetryService: Polling HW (${monitor}): ${cmd}`);
            
            exec(cmd, (err, stdout) => {
                if (err || !stdout) {
                    logger.warn('system', `TelemetryService: HW monitor failed: ${err?.message}`);
                    return resolve('GPU');
                }
                
                try {
                    if (monitor === 'nvidia') {
                        const parts = stdout.split(',').map(s => s.trim());
                        if (parts.length >= 2) {
                            const pct = Math.round((parseInt(parts[0]) / parseInt(parts[1])) * 100);
                            return resolve(`${pct}%`);
                        }
                    } else if (monitor === 'amd') {
                        // Rough AMD JSON parse logic
                        const data = JSON.parse(stdout);
                        const vram = data['GPU[0]']?.['VRAM Total Memory (B)'];
                        const used = data['GPU[0]']?.['VRAM Total Used (B)'];
                        if (vram && used) {
                            return resolve(`${Math.round((used/vram)*100)}%`);
                        }
                    }
                } catch (e) {}
                resolve('GPU');
            });
        });
    }

    private async pollRag(config: any) {
        const url = (config.vayuforge?.ragEndpoint || 'http://127.0.0.1:8081/retrieve').replace('/retrieve', '/health');
        try {
            const axios = require('axios');
            await axios.get(url, { timeout: 1000 });
            this.state.set('rag', { status: 'online' } as any);
        } catch (e) {
            this.state.set('rag', { status: 'offline' } as any);
        }
    }
}
