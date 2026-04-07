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
const child_process_1 = require("child_process");
const vscode = __importStar(require("vscode"));
class TelemetryService {
    constructor() {
        this.pollInterval = null;
        this.state = new Map();
        this._onDidUpdate = new vscode.EventEmitter();
        this.onDidUpdate = this._onDidUpdate.event;
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
        this.pollInterval = setInterval(() => this.pollAll(), 5000);
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
            tps: '0 strategy',
            promptTps: '0',
            slots: '0%',
            latency: '---',
            load: 'Idle'
        };
    }
    async pollAll() {
        const logger = require('../core/logger').CentralLogger.getInstance();
        logger.debug('system', 'TelemetryService: Starting global polling cycle...');
        const config = await config_1.ConfigManager.getInstance().loadConfig();
        if (!config)
            return;
        await this.pollAgent('coder', config.coder, config);
        await this.pollAgent('reviewer', config.reviewer, config);
        this._onDidUpdate.fire();
        logger.debug('system', 'TelemetryService: Global polling cycle completed.');
    }
    async pollAgent(type, modelConfig, fullConfig) {
        const isRemote = fullConfig.connection.mode === 'remote';
        const start = Date.now();
        try {
            const client = new llamaHttpClient_1.LlamaHttpClient(modelConfig.baseUrl);
            // --- RESILIENCE: Try health first ---
            let status = 'offline';
            try {
                const health = await client.get('/health');
                if (health && health.status === 'ok')
                    status = 'online';
            }
            catch (e) {
                // If health fails, it's truly offline
                throw e;
            }
            let metrics = '';
            try {
                metrics = await client.get('/metrics');
            }
            catch (e) {
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
            const slots = slotsMatch ? `${Math.round(parseFloat(slotsMatch[1]) * 100)}%` : '0%';
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
                    const hw = await client.get('/v1/hardware');
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
            this.state.set(type, {
                status: status,
                vram,
                tps: genTps > 0 ? `${genTps.toFixed(1)}` : '0.0',
                promptTps: promptTps > 0 ? `${promptTps.toFixed(1)}` : '0.0',
                slots,
                latency,
                load
            });
        }
        catch (e) {
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
    getLocalVram(config) {
        return new Promise((resolve) => {
            const logger = require('../core/logger').CentralLogger.getInstance();
            if (config.mode === 'cpu')
                return resolve('CPU');
            const cmd = 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader';
            logger.debug('system', `TelemetryService: Executing hardware check: ${cmd}`);
            (0, child_process_1.exec)(cmd, (err, stdout) => {
                if (err || !stdout) {
                    logger.warn('system', `TelemetryService: nvidia-smi failed: ${err?.message}`);
                    return resolve('GPU');
                }
                const parts = stdout.split(',').map(s => s.trim());
                if (parts.length >= 2) {
                    const used = parseInt(parts[0]);
                    const total = parseInt(parts[1]);
                    const pct = Math.round((used / total) * 100);
                    logger.debug('system', `TelemetryService: VRAM Check Result: ${used}/${total} MiB (${pct}%)`);
                    resolve(`${pct}%`);
                }
                else {
                    resolve('GPU');
                }
            });
        });
    }
}
exports.TelemetryService = TelemetryService;
//# sourceMappingURL=telemetry.js.map