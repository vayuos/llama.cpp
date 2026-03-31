import { LLMClient } from '../llm/llmClient';
import { REVIEWER_SYSTEM_PROMPT, formatReviewerPrompt } from '../prompts/systemRules';

export class ReviewerAgent {
    constructor(private client: LLMClient, private modelName: string) { }

    async reviewPatch(patch: string): Promise<string> {
        const prompt = formatReviewerPrompt(patch);
        const fullPrompt = `${REVIEWER_SYSTEM_PROMPT}\nModel: ${this.modelName}\n\n${prompt}`;
        // Reviewer uses stricter/CPU-safe options
        const options = {
            temperature: 0.1,
            stop: ["[END]"]
        };
        const response = await this.client.generate(fullPrompt, options);
        return response.content;
    }
}
