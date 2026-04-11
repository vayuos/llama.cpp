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
exports.TelemetryService = void 0;
const llamaHttpClient_1 = require("./llamaHttpClient");
const config_1 = require("../core/config");
const fs = __importStar(require("fs"));
const child_process_1 = require("child_process");
const vscode = __importStar(require("vscode"));
class TelemetryService {
    constructor() {
        this.pollInterval = null;
        this.state = new Map();
        this._onDidUpdate = new vscode.EventEmitter();
        this.onDidUpdate = this._onDidUpdate.event;
        this.detectedDriver = null;
    }
    static getInstance() {
        if (!TelemetryService.instance) {
            TelemetryService.instance = new TelemetryService();
        }
        return TelemetryService.instance;
    }
    startPolling() {
        if (this.pollInterval)
            return;
        this.pollInterval = setInterval(() => this.pollAll(), 1000);
        this.pollAll();
    }
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }
    getTelemetry(type) {
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
    async pollAll() {
        const logger = require('../core/logger').CentralLogger.getInstance();
        logger.debug('system', 'TelemetryService: Starting global parallel polling cycle...');
        const config = await config_1.ConfigManager.getInstance().loadConfig();
        if (!config)
            return;
        // Concurrent polling for performance (Agents + RAG)
        await Promise.allSettled([
            this.pollAgent('coder', config.coder, config),
            this.pollAgent('reviewer', config.reviewer, config),
            this.pollRag(config)
        ]);
        this._onDidUpdate.fire();
        logger.debug('system', 'TelemetryService: Global parallel polling cycle completed.');
    }
    async pollAgent(type, modelConfig, fullConfig) {
        const isRemote = fullConfig.connection.mode === 'remote';
        const start = Date.now();
        const logger = require('../core/logger').CentralLogger.getInstance();
        try {
            const client = new llamaHttpClient_1.LlamaHttpClient(modelConfig.baseUrl);
            // --- RESILIENCE: Try health first ---
            let status = 'offline';
            try {
                const health = await client.get('/health', 0, 2000); // 0 retries, 2s timeout
                if (health && health.status === 'ok')
                    status = 'online';
            }
            catch (e) {
                // If health fails, it's truly offline
                throw e;
            }
            let metrics = '';
            try {
                metrics = await client.get('/metrics', 0, 2000); // 0 retries, 2s timeout
            }
            catch (e) {
                metrics = '';
            }
            const latency = `${Date.now() - start}ms`;
            // Parse TPS (predicted_tokens_seconds) - handle various prefixes like llamacpp: or llama_
            const genTpsMatch = metrics.match(/(?:(?:llamacpp:|llama_)predicted_tokens_seconds|predicted_tokens_seconds)\s+([0-9.]+)/);
            const promptTpsMatch = metrics.match(/(?:(?:llamacpp:|llama_)prompt_tokens_seconds|prompt_tokens_seconds)\s+([0-9.]+)/);
            const genTps = genTpsMatch ? parseFloat(genTpsMatch[1]) : 0.0;
            const promptTps = promptTpsMatch ? parseFloat(promptTpsMatch[1]) : 0.0;
            // Parse KV Cache (kv_cache_usage_ratio)
            const slotsMatch = metrics.match(/(?:(?:llamacpp:|llama_)kv_cache_usage_ratio|kv_cache_usage_ratio)\s+([0-9.]+)/);
            const slotsPct = slotsMatch ? Math.round(parseFloat(slotsMatch[1]) * 100) : 0;
            // Parse Load/Activity
            let load = 'Idle';
            if (genTps > 0)
                load = 'Generating';
            else if (promptTps > 0)
                load = 'Processing';
            // Parse VRAM
            let vram = '0%';
            if (isRemote) {
                try {
                    const hw = await client.get('/v1/hardware', 0, 2000); // 0 retries, 2s timeout
                    if (hw && hw.vram) {
                        vram = `${Math.round(hw.vram.used / hw.vram.total * 100)}%`;
                    }
                }
                catch (e) {
                    vram = '---';
                }
            }
            else {
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
        }
        catch (e) {
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
    detectMonitor() {
        if (this.detectedDriver)
            return this.detectedDriver;
        try {
            if (fs.existsSync('/usr/bin/nvidia-smi')) {
                this.detectedDriver = 'nvidia';
            }
            else if (fs.existsSync('/usr/bin/rocm-smi')) {
                this.detectedDriver = 'amd';
            }
            else {
                this.detectedDriver = 'generic';
            }
        }
        catch (e) {
            this.detectedDriver = 'generic';
        }
        return this.detectedDriver;
    }
    getLocalVram(config) {
        return new Promise((resolve) => {
            const logger = require('../core/logger').CentralLogger.getInstance();
            if (config.mode === 'cpu')
                return resolve('CPU');
            const monitor = this.detectMonitor();
            let cmd = '';
            if (monitor === 'nvidia') {
                cmd = 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader';
            }
            else if (monitor === 'amd') {
                cmd = 'rocm-smi --showmeminfo vram --json';
            }
            else {
                return resolve('GPU');
            }
            logger.debug('system', `TelemetryService: Polling HW (${monitor}): ${cmd}`);
            (0, child_process_1.exec)(cmd, (err, stdout) => {
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
                    }
                    else if (monitor === 'amd') {
                        // Rough AMD JSON parse logic
                        const data = JSON.parse(stdout);
                        const vram = data['GPU[0]']?.['VRAM Total Memory (B)'];
                        const used = data['GPU[0]']?.['VRAM Total Used (B)'];
                        if (vram && used) {
                            return resolve(`${Math.round((used / vram) * 100)}%`);
                        }
                    }
                }
                catch (e) { }
                resolve('GPU');
            });
        });
    }
    async pollRag(config) {
        const url = (config.vayuforge?.ragEndpoint || 'http://127.0.0.1:8081/retrieve').replace('/retrieve', '/health');
        try {
            const axios = require('axios');
            await axios.get(url, { timeout: 1000 });
            this.state.set('rag', { status: 'online' });
        }
        catch (e) {
            this.state.set('rag', { status: 'offline' });
        }
    }
}
exports.TelemetryService = TelemetryService;
//# sourceMappingURL=telemetry.js.map