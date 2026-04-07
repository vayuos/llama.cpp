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
exports.UnifiedProcessManager = void 0;
const llamaProcess_1 = require("./llamaProcess");
const telemetry_1 = require("../llm/telemetry");
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const vscode = __importStar(require("vscode"));
class UnifiedProcessManager {
    constructor() { }
    getCoder() {
        if (!this._coder) {
            this._coder = new llamaProcess_1.LlamaProcess('Coder Server', 'coder');
        }
        return this._coder;
    }
    getReviewer() {
        if (!this._reviewer) {
            this._reviewer = new llamaProcess_1.LlamaProcess('Reviewer Server', 'reviewer');
        }
        return this._reviewer;
    }
    static getInstance() {
        if (!UnifiedProcessManager.instance) {
            UnifiedProcessManager.instance = new UnifiedProcessManager();
        }
        return UnifiedProcessManager.instance;
    }
    getLogger() {
        return require('../core/logger').CentralLogger.getInstance();
    }
    getBinaryPath(config, modelConfig) {
        const monorepoBin = '/home/viren/llama/llama.cpp/build_cuda_mmq_moe/bin/llama-server';
        // Priority 1: Specifically configured path in settings
        if (modelConfig.binPath?.trim())
            return modelConfig.binPath.trim();
        // Priority 2: Global extension setting
        if (config.llamaBinPath)
            return config.llamaBinPath;
        // Priority 3: Check if llama-server is in the current directory or build folder relative to extension
        const localBuild = path.join(__dirname, '..', '..', '..', 'build', 'bin', 'llama-server');
        if (fs.existsSync(localBuild))
            return localBuild;
        // Priority 4: Hardcoded VyuOS/VyuForge monorepo path
        if (fs.existsSync(monorepoBin))
            return monorepoBin;
        // Priority 5: System PATH
        return 'llama-server';
    }
    async startCoder(config) {
        this.getLogger().debug('system', `UnifiedProcessManager: Requested startCoder. Mode: ${config.connection.mode}`);
        if (config.connection.mode === 'remote') {
            this.getLogger().info('system', `Remote Mode: Using Coder on System3 (${config.connection.system3Ip})`);
            vscode.window.showInformationMessage(`Remote Mode: Coder should be running on System3 (${config.connection.system3Ip}:8080)`);
            return true;
        }
        const c = config.coder;
        const binPath = this.getBinaryPath(config, c);
        return this.getCoder().start(binPath, c, []);
    }
    async startReviewer(config) {
        this.getLogger().debug('system', `UnifiedProcessManager: Requested startReviewer. Mode: ${config.connection.mode}`);
        if (config.connection.mode === 'remote') {
            this.getLogger().info('system', `Remote Mode: Using Reviewer on System3 (${config.connection.system3Ip})`);
            vscode.window.showInformationMessage(`Remote Mode: Reviewer should be running on System3 (${config.connection.system3Ip}:18080)`);
            return true;
        }
        const r = config.reviewer;
        const binPath = this.getBinaryPath(config, r);
        return this.getReviewer().start(binPath, r, []);
    }
    async stopAll() {
        this.getLogger().info('system', 'UnifiedProcessManager: Stopping all local LLM processes...');
        await Promise.all([this.getCoder().stop(), this.getReviewer().stop()]);
    }
    getLastError(type) {
        return type === 'coder' ? this.getCoder().getLastError() : this.getReviewer().getLastError();
    }
    async getLiveStatus(type, config) {
        const proc = type === 'coder' ? this.getCoder() : this.getReviewer();
        const localPid = proc.getPid();
        if (localPid) {
            return { running: true, external: false, pid: localPid, telemetry: proc.getTelemetry() };
        }
        // Pulse check for external or remote process
        const isRemote = config.connection.mode === 'remote';
        const telemetry = telemetry_1.TelemetryService.getInstance().getTelemetry(type);
        if (telemetry.status === 'online') {
            return {
                running: true,
                external: !localPid,
                telemetry: `${telemetry.vram} | ${telemetry.tps} | ${telemetry.latency}`
            };
        }
        return { running: false, external: false };
    }
    getProcessStatus(type) {
        const proc = type === 'coder' ? this.getCoder() : this.getReviewer();
        return {
            pid: proc.getPid(),
            telemetry: proc.getTelemetry()
        };
    }
}
exports.UnifiedProcessManager = UnifiedProcessManager;
//# sourceMappingURL=processManager.js.map