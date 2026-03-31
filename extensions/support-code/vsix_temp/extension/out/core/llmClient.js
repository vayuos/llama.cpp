"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.LLMClient = void 0;
const axios_1 = __importDefault(require("axios"));
class LLMClient {
    constructor(endpoint) {
        this.endpoint = endpoint;
    }
    async checkHealth() {
        try {
            const response = await axios_1.default.get(`${this.endpoint}/health`);
            return response.status === 200;
        }
        catch (error) {
            return false;
        }
    }
    async generate(prompt, options = {}) {
        const response = await axios_1.default.post(`${this.endpoint}/completion`, {
            prompt,
            n_predict: 2048,
            temperature: 0.0,
            stream: false,
            ...options
        });
        return {
            content: response.data.content,
            tokens_predicted: response.data.tokens_predicted
        };
    }
    async *generateStream(prompt, options = {}) {
        const response = await axios_1.default.post(`${this.endpoint}/completion`, {
            prompt,
            n_predict: 2048,
            temperature: 0.0,
            stream: true,
            ...options
        }, { responseType: 'stream' });
        for await (const chunk of response.data) {
            const lines = chunk.toString().split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.substring(6));
                    yield data.content;
                    if (data.stop)
                        return;
                }
            }
        }
    }
}
exports.LLMClient = LLMClient;
//# sourceMappingURL=llmClient.js.map