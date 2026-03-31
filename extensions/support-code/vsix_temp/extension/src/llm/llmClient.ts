import { LlamaHttpClient } from './llamaHttpClient';

export interface LLMResponse {
    content: string;
    usage?: {
        prompt_tokens: number;
        completion_tokens: number;
    };
}

export class LLMClient {
    private http: LlamaHttpClient;

    constructor(endpoint: string) {
        this.http = new LlamaHttpClient(endpoint);
    }

    async generate(prompt: string, options: any = {}): Promise<LLMResponse> {
        const data = await this.http.post('/completion', {
            prompt,
            n_predict: options.max_tokens || 1024,
            temperature: options.temperature || 0.2,
            stop: options.stop || []
        });

        // llama.cpp /completion returns { content: string, ... }
        return {
            content: data.content
        };
    }
}
