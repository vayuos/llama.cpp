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
exports.LlamaProcess = void 0;
const vscode = __importStar(require("vscode"));
const child_process_1 = require("child_process");
const state_1 = require("../core/state");
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const os = __importStar(require("os"));
const pathUtils_1 = require("../utils/pathUtils");
class LlamaProcess {
    constructor(name, type) {
        this.name = name;
        this.type = type;
        this.process = null;
        this.outputChannel = null;
        this.logStream = null;
        this.errorBuffer = [];
        this.telemetryStr = 'Initializing...';
        this.telemetryInterval = null;
    }
    getOutputChannel() {
        if (!this.outputChannel) {
            this.outputChannel = vscode.window.createOutputChannel(`Gravitas: ${this.name}`);
        }
        return this.outputChannel;
    }
    async start(binaryPath, modelConfig, additionalArgs = []) {
        if (this.process) {
            await this.stop();
        }
        if (this.telemetryInterval)
            clearInterval(this.telemetryInterval);
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
        }
        catch (e) {
            this.getOutputChannel().appendLine(`Failed to setup file logging: ${e}`);
        }
        const mode = modelConfig.mode || 'cpu';
        const isCpuMode = mode === 'cpu';
        const binaryPathResolved = (0, pathUtils_1.resolveBinaryPath)(binaryPath);
        const modelPathResolved = (0, pathUtils_1.resolveTilde)(modelConfig.modelPath);
        const socketDir = path.join(os.homedir(), '.gravitas', 'sockets');
        if (!fs.existsSync(socketDir)) {
            fs.mkdirSync(socketDir, { recursive: true });
        }
        // Remove existing socket if it exists to prevent bind errors
        const socketPath = path.join(socketDir, `${this.type}.sock`);
        if (fs.existsSync(socketPath)) {
            try {
                fs.unlinkSync(socketPath);
            }
            catch (e) { }
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
        }
        else {
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
            this.process = (0, child_process_1.spawn)(cmd, args, {
                env: { ...process.env, CUDA_VISIBLE_DEVICES: cudaVisibleDevices },
                detached: false
            });
            logger.info('system', `LlamaProcess [${this.name}]: Spawned with PID ${this.process.pid}`);
        }
        catch (e) {
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
    async stop() {
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
            try {
                fs.unlinkSync(socketPath);
            }
            catch (e) { }
        }
        this.updateStatus('stopped');
    }
    updateStatus(status) {
        const state = state_1.GravitasState.getInstance();
        if (this.type === 'coder') {
            state.updateState({ coderStatus: status });
        }
        else {
            state.updateState({ reviewerStatus: status });
        }
    }
    getLastError() {
        return this.errorBuffer.filter(l => l.trim().length > 0).join('\n');
    }
    getName() { return this.name; }
    getPid() {
        return this.process?.pid;
    }
    getTelemetry() {
        return this.telemetryStr;
    }
    async pollTelemetry(config) {
        if (!this.process)
            return;
        try {
            // 1. GPU VRAM Monitoring
            let vramStr = 'CPU Mode';
            const mode = config.mode || 'cpu';
            if (mode !== 'cpu') {
                try {
                    const { execSync } = require('child_process');
                    const smi = execSync('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits', { encoding: 'utf8', timeout: 1000 });
                    if (smi && smi.trim()) {
                        const lines = smi.trim().split('\n');
                        let deviceId = 0;
                        if (config.cudaVisibleDevices && config.cudaVisibleDevices.toString().trim() !== '') {
                            deviceId = parseInt(config.cudaVisibleDevices.split(',')[0]);
                        }
                        if (lines[deviceId]) {
                            const [used, total] = lines[deviceId].split(',').map((s) => s.trim());
                            const pct = Math.round((parseInt(used) / parseInt(total)) * 100);
                            vramStr = `VRAM: ${pct}% (${used}MB)`;
                        }
                    }
                }
                catch (smiErr) {
                    vramStr = 'VRAM: N/A';
                }
            }
            // 2. Performance Metrics via Unix Domain Socket
            const socketDir = path.join(os.homedir(), '.gravitas', 'sockets');
            const sockPath = path.join(socketDir, `${this.type}.sock`);
            if (fs.existsSync(sockPath)) {
                // Use the dedicated LlamaHttpClient for UDS communication
                const { LlamaHttpClient } = require('../llm/llamaHttpClient');
                const client = new LlamaHttpClient(`unix://${sockPath}`);
                let tpsStr = null;
                try {
                    // llama-server /metrics endpoint returns Prometheus formatting
                    const metrics = await client.get('/metrics');
                    const tpsMatch = metrics.match(/predicted_tokens_seconds\s+([0-9.]+)/);
                    const kvMatch = metrics.match(/kv_cache_usage_ratio\s+([0-9.]+)/);
                    if (tpsMatch && parseFloat(tpsMatch[1]) > 0.01) {
                        tpsStr = `TPS: ${parseFloat(tpsMatch[1]).toFixed(1)}`;
                    }
                    else {
                        tpsStr = 'Idle';
                    }
                    if (kvMatch) {
                        const kvPct = Math.round(parseFloat(kvMatch[1]) * 100);
                        tpsStr += ` | KV: ${kvPct}%`;
                    }
                }
                catch (e) {
                    // If /metrics isn't ready yet, it's just 'Spawning...'
                    tpsStr = 'Booting...';
                }
                this.telemetryStr = `${vramStr}${tpsStr ? ' | ' + tpsStr : ''}`.trim();
            }
            else {
                this.telemetryStr = vramStr;
            }
        }
        catch (e) {
            this.telemetryStr = 'Status: Error';
        }
    }
}
exports.LlamaProcess = LlamaProcess;
//# sourceMappingURL=llamaProcess.js.map