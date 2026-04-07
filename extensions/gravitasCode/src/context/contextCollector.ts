import axios from 'axios';
import * as vscode from 'vscode';
import { GravitasConfig } from '../core/config';

export class ContextCollector {
    async retrieve(query: string, config: GravitasConfig): Promise<string> {
        const logger = require('../core/logger').CentralLogger.getInstance();
        
        // 1. Always start with Local Context as a high-fidelity baseline
        let localContext = await this.collectLocalContext();
        
        // 2. Attempt RAG retrieval if configured
        if (!config.vayuforge || !config.vayuforge.ragEndpoint) {
            logger.debug('system', 'ContextCollector: RAG endpoint not configured, returning local context only.');
            return localContext;
        }

        logger.debug('system', `ContextCollector: Attempting RAG retrieval for query: "${query.substring(0, 50)}..." at ${config.vayuforge.ragEndpoint}`);

        try {
            const response = await axios.post(config.vayuforge.ragEndpoint, {
                query: query
            }, { 
                timeout: 10000,
                headers: { 'Accept': 'application/json' }
            });

            if (response.data && response.data.context) {
                logger.debug('system', `ContextCollector: Retrieved ${response.data.context.length} chars (RAG Hybrid Mode).`);
                return `Remote RAG Context:\n${response.data.context}\n\n${localContext}`;
            } else if (Array.isArray(response.data)) {
                const ragContext = response.data.map((item: any) => `Source: ${item.name}\n${item.content}`).join('\n\n');
                return `Remote RAG Context:\n${ragContext}\n\n${localContext}`;
            }
        } catch (error: any) {
            logger.error('system', `ContextCollector: RAG fallthrough to local only: ${error.message}`);
        }

        return localContext;
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
