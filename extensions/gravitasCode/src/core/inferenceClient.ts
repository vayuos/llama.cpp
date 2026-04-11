import { ModelConfigSchema, GravitasConfig } from './config';
import { CentralLogger } from './logger';
import { LlamaHttpClient } from '../llm/llamaHttpClient';

export interface CompletionOptions {
    prompt: string;
    stream?: boolean;
    n_predict?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    repeat_penalty?: number;
    stop?: string[];
}

export class InferenceClient {
    private logger = CentralLogger.getInstance();

    async *streamCompletion(baseUrl: string, options: CompletionOptions): AsyncGenerator<string> {
        try {
            const client = new LlamaHttpClient(baseUrl);
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
                        } catch (e) {
                            // Ignore partial JSON
                        }
                    }
                }
            }
        } catch (error: any) {
            this.logger.error('system', `Stream completion failed: ${error.message}`);
            throw error;
        }
    }

    async getCompletion(baseUrl: string, options: CompletionOptions): Promise<string> {
        try {
            const client = new LlamaHttpClient(baseUrl);
            const data = await client.post('/completion', {
                ...options,
                stream: false
            });
            return data.content;
        } catch (error: any) {
            this.logger.error('system', `Completion failed: ${error.message}`);
            throw error;
        }
    }

}
