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
        const args = [
            '--batch-size', config.coderModel.batch?.toString() || '512',
            '--ubatch-size', config.coderModel.ubatch?.toString() || '512',
            '--top-p', config.coderModel.topP?.toString() || '0.95'
        ];
        return this.coder.start(config.llamaBinaryPath, config.coderModel, args);
    }
    async startReviewer(config) {
        return this.reviewer.start(config.llamaBinaryPath, config.reviewerModel);
    }
    async stopAll() {
        await Promise.all([this.coder.stop(), this.reviewer.stop()]);
    }
}
exports.UnifiedProcessManager = UnifiedProcessManager;
//# sourceMappingURL=processManager.js.map