"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.LlamaHttpClient = void 0;
const axios_1 = __importDefault(require("axios"));
class LlamaHttpClient {
    constructor(endpoint) {
        this.client = axios_1.default.create({
            baseURL: endpoint,
            timeout: 30000,
            headers: { 'Content-Type': 'application/json' }
        });
    }
    async post(path, data) {
        const response = await this.client.post(path, data);
        return response.data;
    }
    async get(path) {
        const response = await this.client.get(path);
        return response.data;
    }
}
exports.LlamaHttpClient = LlamaHttpClient;
//# sourceMappingURL=llamaHttpClient.js.map