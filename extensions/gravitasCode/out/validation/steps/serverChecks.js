"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PromptTestStep = exports.ServerPingStep = void 0;
const axios_1 = __importDefault(require("axios"));
const processManager_1 = require("../../process/processManager");
class ServerPingStep {
    constructor(type) {
        this.type = type;
    }
    get name() { return `Ping ${this.type} server health endpoint`; }
    async execute(config) {
        const pm = processManager_1.UnifiedProcessManager.getInstance();
        const port = this.type === 'coder' ? config.coder.port : config.reviewer.port;
        const endpoint = `http://127.0.0.1:${port}/v1/models`;
        // Start server
        if (this.type === 'coder')
            await pm.startCoder(config);
        else
            await pm.startReviewer(config);
        // Wait for health check (max 90s for CPU models)
        for (let i = 0; i < 90; i++) {
            try {
                await axios_1.default.get(endpoint, { timeout: 1000 });
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
        const port = this.type === 'coder' ? config.coder.port : config.reviewer.port;
        const endpoint = `http://127.0.0.1:${port}/v1/chat/completions`;
        const prompt = this.type === 'coder' ? 'print("hello")' : 'How are you?';
        // Retry for up to 60s for model load
        for (let i = 0; i < 30; i++) {
            try {
                const resp = await axios_1.default.post(endpoint, {
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