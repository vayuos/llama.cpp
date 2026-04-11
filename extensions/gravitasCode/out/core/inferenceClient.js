"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.InferenceClient = void 0;
const logger_1 = require("./logger");
const llamaHttpClient_1 = require("../llm/llamaHttpClient");
class InferenceClient {
    constructor() {
        this.logger = logger_1.CentralLogger.getInstance();
    }
    async *streamCompletion(baseUrl, options) {
        try {
            const client = new llamaHttpClient_1.LlamaHttpClient(baseUrl);
            const response = await client.post('/completion', {
                ...options,
                stream: true
            });
            const stream = response.data;
            for await (const chunk of stream) {
                const lines = chunk.toString().split('\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.content) {
                                yield data.content;
                            }
                            if (data.stop) {
                                return;
                            }
                        }
                        catch (e) {
                            // Ignore partial JSON
                        }
                    }
                }
            }
        }
        catch (error) {
            this.logger.error('system', `Stream completion failed: ${error.message}`);
            throw error;
        }
    }
    async getCompletion(baseUrl, options) {
        try {
            const client = new llamaHttpClient_1.LlamaHttpClient(baseUrl);
            const data = await client.post('/completion', {
                ...options,
                stream: false
            });
            return data.content;
        }
        catch (error) {
            this.logger.error('system', `Completion failed: ${error.message}`);
            throw error;
        }
    }
}
exports.InferenceClient = InferenceClient;
//# sourceMappingURL=inferenceClient.js.map