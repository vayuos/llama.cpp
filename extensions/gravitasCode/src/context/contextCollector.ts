import axios from 'axios';
import * as vscode from 'vscode';
import { GravitasConfig } from '../core/config';

export class ContextCollector {
    private readonly MAX_CONTEXT_CHARS = 32768; // 🛡️ Safety: Prevent context window overflow

    async retrieve(query: string, config: GravitasConfig): Promise<string> {
        const logger = require('../core/logger').CentralLogger.getInstance();
        
        // 1. Local Baseline
        let localContext = await this.collectLocalContext();
        
        let finalContext = localContext;
        // 2. RAG Hybrid
        if (config.vayuforge && config.vayuforge.ragEndpoint) {
            try {
                const response = await axios.post(config.vayuforge.ragEndpoint, { query }, { timeout: 10000 });
                if (response.data && response.data.context) {
                    finalContext = `Remote RAG Context:\n${response.data.context}\n\n${localContext}`;
                } else if (Array.isArray(response.data)) {
                    const rag = response.data.map((item: any) => `Source: ${item.name}\n${item.content}`).join('\n\n');
                    finalContext = `Remote RAG Context:\n${rag}\n\n${localContext}`;
                }
            } catch (e: any) {
                logger.error('system', `ContextCollector: RAG Error: ${e.message}`);
            }
        }

        if (finalContext.length > this.MAX_CONTEXT_CHARS) {
            logger.warn('system', `ContextCollector: Truncating context from ${finalContext.length} chars to safety limit.`);
            return finalContext.substring(0, this.MAX_CONTEXT_CHARS) + '\n\n[TRUNCATED FOR TOKEN SAFETY]';
        }
        return finalContext;
    }

    private async collectLocalContext(): Promise<string> {
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
