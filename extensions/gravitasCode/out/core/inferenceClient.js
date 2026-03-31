"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.InferenceClient = void 0;
const axios_1 = __importDefault(require("axios"));
const logger_1 = require("./logger");
class InferenceClient {
    constructor() {
        this.logger = logger_1.CentralLogger.getInstance();
    }
    async *streamCompletion(baseUrl, options) {
        try {
            const response = await axios_1.default.post(`${baseUrl}/completion`, {
                ...options,
                stream: true
            }, {
                responseType: 'stream'
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
            const response = await axios_1.default.post(`${baseUrl}/completion`, {
                ...options,
                stream: false
            });
            return response.data.content;
        }
        catch (error) {
            this.logger.error('system', `Completion failed: ${error.message}`);
            throw error;
        }
    }
}
exports.InferenceClient = InferenceClient;
//# sourceMappingURL=inferenceClient.js.map