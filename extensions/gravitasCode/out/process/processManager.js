"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnifiedProcessManager = void 0;
const llamaProcess_1 = require("./llamaProcess");
class UnifiedProcessManager {
    constructor() {
        this.coder = new llamaProcess_1.LlamaProcess('Coder Server', 'coder');
        this.reviewer = new llamaProcess_1.LlamaProcess('Reviewer Server', 'reviewer');
    }
    static getInstance() {
        if (!UnifiedProcessManager.instance) {
            UnifiedProcessManager.instance = new UnifiedProcessManager();
        }
        return UnifiedProcessManager.instance;
    }
    async startCoder(config) {
        // LlamaProcess now handles all args construction based on config
        return this.coder.start(config.llamaBinPath, config.coder, []);
    }
    async startReviewer(config) {
        // LlamaProcess now handles all args construction based on config
        return this.reviewer.start(config.llamaBinPath, config.reviewer, []);
    }
    async stopAll() {
        await Promise.all([this.coder.stop(), this.reviewer.stop()]);
    }
    getLastError(type) {
        return type === 'coder' ? this.coder.getLastError() : this.reviewer.getLastError();
    }
    getProcessStatus(type) {
        const proc = type === 'coder' ? this.coder : this.reviewer;
        return {
            pid: proc.getPid(),
            telemetry: proc.getTelemetry()
        };
    }
}
exports.UnifiedProcessManager = UnifiedProcessManager;
//# sourceMappingURL=processManager.js.map