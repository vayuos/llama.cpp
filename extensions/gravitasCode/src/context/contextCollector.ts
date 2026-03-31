import axios from 'axios';
import { GravitasConfig } from '../core/config';

export class ContextCollector {
    async retrieve(query: string, config: GravitasConfig): Promise<string> {
        if (!config.vayuforge || !config.vayuforge.ragEndpoint) {
            return '';
        }

        try {
            const response = await axios.post(config.vayuforge.ragEndpoint, {
                query: query
            }, { timeout: 10000 });

            if (response.data && response.data.context) {
                return response.data.context;
            } else if (Array.isArray(response.data)) {
                // Handle Continue.dev adapter format as fallback
                return response.data.map((item: any) => 
                    `Source: ${item.name}\n${item.content}`
                ).join('\n\n');
            }
            return '';
        } catch (error) {
            console.error('VayuForge RAG error:', error);
            return '';
        }
    }
}
