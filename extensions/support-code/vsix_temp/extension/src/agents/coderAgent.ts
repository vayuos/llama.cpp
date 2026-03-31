import { LLMClient } from '../llm/llmClient';
import { CODER_SYSTEM_PROMPT } from '../prompts/systemRules';

export class CoderAgent {
    constructor(private client: LLMClient) { }

    async generatePatch(prompt: string): Promise<string> {
        const fullPrompt = `${CODER_SYSTEM_PROMPT}\n\nTask: ${prompt}`;
        const response = await this.client.generate(fullPrompt);
        return response.content;
    }
}
