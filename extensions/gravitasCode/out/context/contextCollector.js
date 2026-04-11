"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ContextCollector = void 0;
const axios_1 = __importDefault(require("axios"));
const vscode = __importStar(require("vscode"));
class ContextCollector {
    constructor() {
        this.MAX_CONTEXT_CHARS = 32768; // 🛡️ Safety: Prevent context window overflow
    }
    async retrieve(query, config) {
        const logger = require('../core/logger').CentralLogger.getInstance();
        // 1. Local Baseline
        let localContext = await this.collectLocalContext();
        let finalContext = localContext;
        // 2. RAG Hybrid
        if (config.vayuforge && config.vayuforge.ragEndpoint) {
            try {
                const response = await axios_1.default.post(config.vayuforge.ragEndpoint, { query }, { timeout: 10000 });
                if (response.data && response.data.context) {
                    finalContext = `Remote RAG Context:\n${response.data.context}\n\n${localContext}`;
                }
                else if (Array.isArray(response.data)) {
                    const rag = response.data.map((item) => `Source: ${item.name}\n${item.content}`).join('\n\n');
                    finalContext = `Remote RAG Context:\n${rag}\n\n${localContext}`;
                }
            }
            catch (e) {
                logger.error('system', `ContextCollector: RAG Error: ${e.message}`);
            }
        }
        if (finalContext.length > this.MAX_CONTEXT_CHARS) {
            logger.warn('system', `ContextCollector: Truncating context from ${finalContext.length} chars to safety limit.`);
            return finalContext.substring(0, this.MAX_CONTEXT_CHARS) + '\n\n[TRUNCATED FOR TOKEN SAFETY]';
        }
        return finalContext;
    }
    async collectLocalContext() {
        let result = '--- Local Workspace Map ---\n';
        // 1. Workspace Structure
        const folders = vscode.workspace.workspaceFolders;
        if (folders) {
            result += `Active Workspace Roots (${folders.length}):\n`;
            for (const folder of folders) {
                result += `- Name: ${folder.name}, Path: ${folder.uri.fsPath}\n`;
            }
        }
        // 2. Active Editor Content (High-Fidelity focus)
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const doc = editor.document;
            const content = doc.getText();
            const fileName = doc.fileName;
            result += `\n--- Focus File (${fileName}) ---\n`;
            // Capture up to 500 lines for the active editor
            result += content.split('\n').slice(0, 500).join('\n');
            result += '\n--- End Focus File ---\n';
        }
        return result;
    }
}
exports.ContextCollector = ContextCollector;
//# sourceMappingURL=contextCollector.js.map