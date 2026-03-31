"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LLMClient = void 0;
const llamaHttpClient_1 = require("./llamaHttpClient");
class LLMClient {
    constructor(endpoint) {
        this.http = new llamaHttpClient_1.LlamaHttpClient(endpoint);
    }
    async generate(prompt, options = {}) {
        const data = await this.http.post('/completion', {
            prompt,
            n_predict: options.max_tokens || 1024,
            temperature: options.temperature || 0.2,
            stop: options.stop || []
        });
        // llama.cpp /completion returns { content: string, ... }
        return {
            content: data.content
        };
    }
}
exports.LLMClient = LLMClient;
//# sourceMappingURL=llmClient.js.map