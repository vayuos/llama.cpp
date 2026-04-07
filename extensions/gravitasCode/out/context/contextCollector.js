"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ContextCollector = void 0;
const axios_1 = __importDefault(require("axios"));
class ContextCollector {
    async retrieve(query, config) {
        const logger = require('../core/logger').CentralLogger.getInstance();
        if (!config.vayuforge || !config.vayuforge.ragEndpoint) {
            logger.debug('system', 'ContextCollector: RAG endpoint not configured, skipping retrieval.');
            return '';
        }
        logger.debug('system', `ContextCollector: Attempting RAG retrieval for query: "${query.substring(0, 50)}..." at ${config.vayuforge.ragEndpoint}`);
        try {
            const response = await axios_1.default.post(config.vayuforge.ragEndpoint, {
                query: query
            }, { timeout: 10000 });
            if (response.data && response.data.context) {
                logger.debug('system', `ContextCollector: Retrieved ${response.data.context.length} chars (Standard Format)`);
                return response.data.context;
            }
            else if (Array.isArray(response.data)) {
                // Handle Continue.dev adapter format as fallback
                const sources = response.data.map((item) => item.name).join(', ');
                logger.debug('system', `ContextCollector: Detected Continue.dev format. Sources: [${sources}]`);
                return response.data.map((item) => `Source: ${item.name}\n${item.content}`).join('\n\n');
            }
            logger.warn('system', 'ContextCollector: RAG response was empty or malformed.');
            return '';
        }
        catch (error) {
            logger.error('system', `ContextCollector: VayuForge RAG error: ${error.message}`);
            return '';
        }
    }
}
exports.ContextCollector = ContextCollector;
//# sourceMappingURL=contextCollector.js.map