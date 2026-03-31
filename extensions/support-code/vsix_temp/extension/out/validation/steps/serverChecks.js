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
        const port = this.type === 'coder' ? config.coderModel.port : config.reviewerModel.port;
        const endpoint = `http://127.0.0.1:${port}/v1/models`;
        // Start server
        if (this.type === 'coder')
            await pm.startCoder(config);
        else
            await pm.startReviewer(config);
        // Wait for health check (max 30s)
        for (let i = 0; i < 30; i++) {
            try {
                await axios_1.default.get(endpoint, { timeout: 1000 });
                return { success: true, message: `${this.type} server is healthy.` };
            }
            catch (e) {
                await new Promise(r => setTimeout(r, 1000));
            }
        }
        return { success: false, message: `${this.type} server failed to respond within 30s.` };
    }
    async rollback() {
        await processManager_1.UnifiedProcessManager.getInstance().stopAll();
    }
}
exports.ServerPingStep = ServerPingStep;
class PromptTestStep {
    constructor(type) {
        this.type = type;
    }
    get name() { return `Send test ${this.type} prompt`; }
    async execute(config) {
        const port = this.type === 'coder' ? config.coderModel.port : config.reviewerModel.port;
        const endpoint = `http://127.0.0.1:${port}/v1/chat/completions`;
        const prompt = this.type === 'coder' ? 'print("hello")' : 'How are you?';
        try {
            const resp = await axios_1.default.post(endpoint, {
                messages: [{ role: 'user', content: prompt }],
                max_tokens: 10
            }, { timeout: 10000 });
            if (resp.status === 200) {
                return { success: true, message: `${this.type} prompt test passed.` };
            }
            return { success: false, message: `${this.type} returned status ${resp.status}` };
        }
        catch (e) {
            return { success: false, message: `${this.type} prompt test failed: ${e.message}` };
        }
    }
    async rollback() {
        await processManager_1.UnifiedProcessManager.getInstance().stopAll();
    }
}
exports.PromptTestStep = PromptTestStep;
//# sourceMappingURL=serverChecks.js.map