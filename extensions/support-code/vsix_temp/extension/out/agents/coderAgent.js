"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.CoderAgent = void 0;
const systemRules_1 = require("../prompts/systemRules");
class CoderAgent {
    constructor(client) {
        this.client = client;
    }
    async generatePatch(prompt) {
        const fullPrompt = `${systemRules_1.CODER_SYSTEM_PROMPT}\n\nTask: ${prompt}`;
        const response = await this.client.generate(fullPrompt);
        return response.content;
    }
}
exports.CoderAgent = CoderAgent;
//# sourceMappingURL=coderAgent.js.map