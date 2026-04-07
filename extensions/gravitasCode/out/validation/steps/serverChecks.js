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
exports.PromptTestStep = exports.ServerPingStep = void 0;
const processManager_1 = require("../../process/processManager");
const llamaHttpClient_1 = require("../../llm/llamaHttpClient");
const path = __importStar(require("path"));
const os = __importStar(require("os"));
const fs = __importStar(require("fs"));
class ServerPingStep {
    constructor(type) {
        this.type = type;
    }
    get name() { return `Ping ${this.type} server health endpoint`; }
    async execute(config) {
        const pm = processManager_1.UnifiedProcessManager.getInstance();
        const sockPath = path.join(os.homedir(), '.gravitas', 'sockets', `${this.type}.sock`);
        const endpoint = fs.existsSync(sockPath)
            ? `unix://${sockPath}`
            : `http://${config[this.type].host || '127.0.0.1'}:${config[this.type].port}`;
        const client = new llamaHttpClient_1.LlamaHttpClient(endpoint);
        // Start server
        if (this.type === 'coder')
            await pm.startCoder(config);
        else
            await pm.startReviewer(config);
        // Wait for health check (max 90s for CPU models)
        for (let i = 0; i < 90; i++) {
            try {
                await client.get('/v1/models');
                return { success: true, message: `${this.type} server is healthy.` };
            }
            catch (e) {
                // Check if process crashed
                // Check if process crashed
                const status = pm.getProcessStatus(this.type);
                if (!status.pid) {
                    const lastError = pm.getLastError(this.type);
                    return { success: false, message: `${this.type} server crashed! Logs:\n${lastError}` };
                }
                await new Promise(r => setTimeout(r, 1000));
            }
        }
        return { success: false, message: `${this.type} server failed to respond within 90s.` };
    }
    async rollback() {
        // Don't stop servers - keep them running for user to inspect
        // await UnifiedProcessManager.getInstance().stopAll();
    }
}
exports.ServerPingStep = ServerPingStep;
class PromptTestStep {
    constructor(type) {
        this.type = type;
    }
    get name() { return `Send test ${this.type} prompt`; }
    async execute(config) {
        const sockPath = path.join(os.homedir(), '.gravitas', 'sockets', `${this.type}.sock`);
        const endpoint = fs.existsSync(sockPath)
            ? `unix://${sockPath}`
            : `http://${config[this.type].host || '127.0.0.1'}:${config[this.type].port}`;
        const client = new llamaHttpClient_1.LlamaHttpClient(endpoint);
        const prompt = this.type === 'coder' ? 'print("hello")' : 'How are you?';
        // Retry for up to 60s for model load
        for (let i = 0; i < 30; i++) {
            try {
                const resp = await client.client.post('/v1/chat/completions', {
                    messages: [{ role: 'user', content: prompt }],
                    max_tokens: 10
                }, { timeout: 20000 }); // Increased timeout for inference
                if (resp.status === 200) {
                    return { success: true, message: `${this.type} prompt test passed.` };
                }
                return { success: false, message: `${this.type} returned status ${resp.status}` };
            }
            catch (e) {
                // If 503 or connection reset (model loading), wait and retry
                if (e.response?.status === 503 || e.code === 'ECONNRESET') {
                    await new Promise(r => setTimeout(r, 2000));
                    continue;
                }
                // If it's the last attempt, fail
                if (i === 29) {
                    return { success: false, message: `${this.type} prompt test failed: ${e.message}` };
                }
                // Recoverable network error?
                await new Promise(r => setTimeout(r, 1000));
            }
        }
        return { success: false, message: `${this.type} failed to load model within 60s.` };
    }
    async rollback() {
        // Don't stop servers - keep them running for user to inspect
        // await UnifiedProcessManager.getInstance().stopAll();
    }
}
exports.PromptTestStep = PromptTestStep;
//# sourceMappingURL=serverChecks.js.map