"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AgentLoopController = void 0;
const config_1 = require("../core/config");
const logger_1 = require("../core/logger");
const parser_1 = require("../review/parser");
const axios_1 = __importDefault(require("axios"));
class AgentLoopController {
    constructor() {
        this.logger = logger_1.CentralLogger.getInstance();
        this.maxIterations = 3;
    }
    async run(task) {
        const config = await config_1.ConfigManager.getInstance().loadConfig();
        if (!config)
            throw new Error('Config missing');
        let currentCode = '';
        let iteration = 0;
        while (iteration < this.maxIterations) {
            iteration++;
            this.logger.info('system', `Loop Iteration ${iteration}: Coding...`);
            // 1. Coder Generates Code
            currentCode = await this.generateCode(config, task, currentCode);
            // 2. Reviewer Checks Code
            this.logger.info('system', `Loop Iteration ${iteration}: Reviewing...`);
            const review = await this.getReview(config, currentCode);
            if (!review) {
                this.logger.error('system', 'Reviewer failed to provide deterministic output.');
                break;
            }
            this.logger.info('system', `Review Severity: ${review.severity}. Errors: ${review.issues.length}`);
            if (review.severity === 'minor' && review.issues.length === 0) {
                this.logger.info('system', 'Auto-validation successful. Code meets quality bar.');
                break;
            }
            // 3. Feedback Loop (Inject review into next coder prompt)
            task = `Previous code had issues: ${review.summary}. Please fix: ${review.recommendedChanges.join(', ')}`;
        }
        return currentCode;
    }
    async generateCode(config, prompt, context) {
        const url = `http://127.0.0.1:${config.coderModel.port}/v1/chat/completions`;
        const resp = await axios_1.default.post(url, {
            messages: [
                { role: 'system', content: 'You are a master software engineer. Output ONLY raw code, no talk.' },
                { role: 'user', content: prompt }
            ]
        });
        return resp.data.choices[0].message.content;
    }
    async getReview(config, code) {
        const url = `http://127.0.0.1:${config.reviewerModel.port}/v1/chat/completions`;
        const systemPrompt = `Review this code. Output ONLY valid JSON matching: { severity: "critical"|"major"|"minor", issues: [{description, severity, line}], recommendedChanges: [string], summary: string }`;
        try {
            const resp = await axios_1.default.post(url, {
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: code }
                ]
            });
            return parser_1.ReviewParser.parse(resp.data.choices[0].message.content);
        }
        catch (e) {
            return null;
        }
    }
}
exports.AgentLoopController = AgentLoopController;
//# sourceMappingURL=loop.js.map