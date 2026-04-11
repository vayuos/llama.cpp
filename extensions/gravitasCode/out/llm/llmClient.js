"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LLMClient = void 0;
const llamaHttpClient_1 = require("./llamaHttpClient");
class LLMClient {
    constructor(endpoint) {
        this.http = new llamaHttpClient_1.LlamaHttpClient(endpoint);
    }
    async generate(messages, options = {}, onChunk, signal) {
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
            const response = await this.http.client.post('/v1/chat/completions', body, { responseType: 'stream', signal });
            let fullContent = '';
            let lineBuffer = '';
            return new Promise((resolve, reject) => {
                const onAbort = () => {
                    response.data.destroy();
                    reject(new Error('LLM Stream aborted via signal.'));
                };
                if (signal)
                    signal.addEventListener('abort', onAbort);
                response.data.on('data', (chunk) => {
                    lineBuffer += chunk.toString();
                    const lines = lineBuffer.split('\n');
                    lineBuffer = lines.pop() || ''; // Keep the last partial line
                    for (const line of lines) {
                        const trimmedLine = line.trim();
                        if (!trimmedLine || !trimmedLine.startsWith('data: '))
                            continue;
                        const message = trimmedLine.replace(/^data: /, '');
                        if (message === '[DONE]')
                            break;
                        try {
                            const parsed = JSON.parse(message);
                            const content = parsed.choices?.[0]?.delta?.content || '';
                            if (content) {
                                fullContent += content;
                                onChunk(content);
                            }
                        }
                        catch (e) {
                            // Non-json or partial json on this line, skip or log
                        }
                    }
                });
                response.data.on('end', () => {
                    if (signal)
                        signal.removeEventListener('abort', onAbort);
                    logger.debug('system', `Gravitas LLM Stream Completed. Total content length: ${fullContent.length} chars.`);
                    resolve({ content: fullContent });
                });
                response.data.on('error', (err) => {
                    if (signal)
                        signal.removeEventListener('abort', onAbort);
                    logger.error('system', `Gravitas LLM Stream Error: ${err.message}`);
                    reject(err);
                });
            });
        }
        const data = await this.http.post('/v1/chat/completions', body, 2, signal);
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
exports.LLMClient = LLMClient;
//# sourceMappingURL=llmClient.js.map