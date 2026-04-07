import { LlamaHttpClient } from './llamaHttpClient';

export interface LLMResponse {
    content: string;
    usage?: {
        prompt_tokens: number;
        completion_tokens: number;
    };
}

export interface LLMOptions {
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    repeat_penalty?: number;
    stop?: string[];
}

export class LLMClient {
    private http: LlamaHttpClient;

    constructor(endpoint: string) {
        this.http = new LlamaHttpClient(endpoint);
    }

    async generate(
        messages: { role: 'user' | 'assistant' | 'system', content: string }[], 
        options: LLMOptions = {},
        onChunk?: (text: string) => void
    ): Promise<LLMResponse> {
        const body = {
            messages: messages,
            stream: !!onChunk,
            max_tokens: options.max_tokens || 1024,
            temperature: options.temperature ?? 0.2,
            top_p: options.top_p ?? 0.9,
            top_k: options.top_k ?? 40,
            repeat_penalty: options.repeat_penalty ?? 1.1,
            stop: options.stop || [],
            cache_prompt: false // 🛡️ CRITICAL: Clear KV cache for this specific task
        };

        const logger = require('../core/logger').CentralLogger.getInstance();
        logger.debug('system', `Gravitas LLM Request: ${JSON.stringify(body, null, 2)}`);

        if (onChunk) {
            // Streaming mode: SSE
            const response = await (this.http as any).client.post('/v1/chat/completions', body, { responseType: 'stream' });
            let fullContent = '';
            
            return new Promise((resolve, reject) => {
                response.data.on('data', (chunk: Buffer) => {
                    const lines = chunk.toString().split('\n').filter(line => line.trim());
                    for (const line of lines) {
                        const message = line.replace(/^data: /, '');
                        if (message === '[DONE]') break;
                        try {
                            const parsed = JSON.parse(message);
                            const content = parsed.choices[0].delta?.content || '';
                            if (content) {
                                fullContent += content;
                                onChunk(content);
                            }
                        } catch (e) {
                            // Non-json line, skip
                        }
                    }
                });
                response.data.on('end', () => {
                    logger.debug('system', `Gravitas LLM Stream Completed. Total content length: ${fullContent.length} chars.`);
                    resolve({ content: fullContent });
                });
                response.data.on('error', (err: Error) => {
                    logger.error('system', `Gravitas LLM Stream Error: ${err.message}`);
                    reject(err);
                });
            });
        }

        const data = await this.http.post('/v1/chat/completions', body);
        logger.debug('system', `Gravitas LLM Response: ${JSON.stringify(data, null, 2)}`);

        if (!data || !data.choices || data.choices.length === 0) {
            logger.error('system', `Gravitas LLM Error: Malformed response from server. Data: ${JSON.stringify(data)}`);
            throw new Error('Malformed response from LLM server (choices missing).');
        }

        // OpenAI-compatible response { choices: [{ message: { content: string } }] }
        return {
            content: data.choices[0].message.content,
            usage: data.usage
        };
    }
}
