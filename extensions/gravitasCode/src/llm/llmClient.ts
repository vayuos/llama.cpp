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
        onChunk?: (text: string) => void,
        signal?: AbortSignal
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
            cache_prompt: false
        };

        if (onChunk) {
            return this.generateStreamingWithRetry(body, onChunk, 3, signal);
        }

        const logger = require('../core/logger').CentralLogger.getInstance();
        const data = await this.http.post('/v1/chat/completions', body, 3, signal);
        
        if (!data || !data.choices || data.choices.length === 0) {
            throw new Error('Malformed response from LLM server (choices missing).');
        }

        return {
            content: data.choices[0].message.content,
            usage: data.usage
        };
    }

    private async generateStreamingWithRetry(body: any, onChunk: (text: string) => void, retries: number, signal?: AbortSignal): Promise<LLMResponse> {
        const logger = require('../core/logger').CentralLogger.getInstance();
        try {
            const response = await (this.http as any).client.post('/v1/chat/completions', body, { responseType: 'stream', signal });
            let fullContent = '';
            let lineBuffer = '';
            
            return new Promise((resolve, reject) => {
                const onAbort = () => {
                    response.data.destroy();
                    reject(new Error('LLM Stream aborted.'));
                };
                if (signal) signal.addEventListener('abort', onAbort);

                response.data.on('data', (chunk: Buffer) => {
                    lineBuffer += chunk.toString();
                    const lines = lineBuffer.split('\n');
                    lineBuffer = lines.pop() || '';

                    for (const line of lines) {
                        const trimmedLine = line.trim();
                        if (!trimmedLine || !trimmedLine.startsWith('data: ')) continue;
                        const message = trimmedLine.replace(/^data: /, '');
                        if (message === '[DONE]') break;
                        try {
                            const parsed = JSON.parse(message);
                            const content = parsed.choices?.[0]?.delta?.content || '';
                            if (content) {
                                fullContent += content;
                                onChunk(content);
                            }
                        } catch (e) {}
                    }
                });
                response.data.on('end', () => {
                    if (signal) signal.removeEventListener('abort', onAbort);
                    resolve({ content: fullContent });
                });
                response.data.on('error', (err: any) => {
                    if (signal) signal.removeEventListener('abort', onAbort);
                    reject(err);
                });
            });
        } catch (e: any) {
            if (retries > 0 && this.isRetryable(e)) {
                logger.warn('system', `LLM Streaming failed (${e.message}). Retrying... (${retries} left)`);
                await new Promise(r => setTimeout(r, 1000));
                return this.generateStreamingWithRetry(body, onChunk, retries - 1, signal);
            }
            throw e;
        }
    }

    private isRetryable(e: any): boolean {
        const code = e.code;
        const message = e.message?.toLowerCase() || '';
        return (
            code === 'ECONNREFUSED' || 
            code === 'ECONNRESET' || 
            code === 'ETIMEDOUT' || 
            message.includes('socket hang up')
        );
    }
}
