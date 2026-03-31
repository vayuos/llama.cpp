import axios from 'axios';
import { ConfigManager } from '../core/config';

export interface RAGResult {
    content: string;
    source: string;
    relevance?: number;
}

export class RAGClient {
    private static instance: RAGClient;

    private constructor() {}

    public static getInstance(): RAGClient {
        if (!RAGClient.instance) {
            RAGClient.instance = new RAGClient();
        }
        return RAGClient.instance;
    }

    public async retrieve(query: string): Promise<RAGResult[]> {
        const config = ConfigManager.getInstance().getCachedConfig();
        if (!config || !config.vayuforge.ragEndpoint) return [];

        try {
            // Using a shorter timeout for RAG to avoid blocking the main agent flow
            const response = await axios.post(config.vayuforge.ragEndpoint, { query }, { timeout: 15000 });
            
            // Standard VayuForge RAG response structure
            if (response.data && Array.isArray(response.data.results)) {
                return response.data.results;
            } else if (Array.isArray(response.data)) {
                return response.data;
            }
            
            return [];
        } catch (e) {
            console.error(`RAG Retrieval Error: ${e}`);
            return [];
        }
    }
}
