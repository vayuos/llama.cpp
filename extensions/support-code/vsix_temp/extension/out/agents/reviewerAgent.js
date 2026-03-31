"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ReviewerAgent = void 0;
const systemRules_1 = require("../prompts/systemRules");
class ReviewerAgent {
    constructor(client, modelName) {
        this.client = client;
        this.modelName = modelName;
    }
    async reviewPatch(patch) {
        const prompt = (0, systemRules_1.formatReviewerPrompt)(patch);
        const fullPrompt = `${systemRules_1.REVIEWER_SYSTEM_PROMPT}\nModel: ${this.modelName}\n\n${prompt}`;
        // Reviewer uses stricter/CPU-safe options
        const options = {
            temperature: 0.1,
            stop: ["[END]"]
        };
        const response = await this.client.generate(fullPrompt, options);
        return response.content;
    }
}
exports.ReviewerAgent = ReviewerAgent;
//# sourceMappingURL=reviewerAgent.js.map