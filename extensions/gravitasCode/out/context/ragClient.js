"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RAGClient = void 0;
const axios_1 = __importDefault(require("axios"));
const config_1 = require("../core/config");
class RAGClient {
    constructor() { }
    static getInstance() {
        if (!RAGClient.instance) {
            RAGClient.instance = new RAGClient();
        }
        return RAGClient.instance;
    }
    async retrieve(query) {
        const config = config_1.ConfigManager.getInstance().getCachedConfig();
        if (!config || !config.vayuforge.ragEndpoint)
            return [];
        try {
            // Using a shorter timeout for RAG to avoid blocking the main agent flow
            const response = await axios_1.default.post(config.vayuforge.ragEndpoint, { query }, { timeout: 15000 });
            // Standard VayuForge RAG response structure
            if (response.data && Array.isArray(response.data.results)) {
                return response.data.results;
            }
            else if (Array.isArray(response.data)) {
                return response.data;
            }
            return [];
        }
        catch (e) {
            console.error(`RAG Retrieval Error: ${e}`);
            return [];
        }
    }
}
exports.RAGClient = RAGClient;
//# sourceMappingURL=ragClient.js.map