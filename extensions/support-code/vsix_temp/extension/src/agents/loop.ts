import { UnifiedProcessManager } from '../process/processManager';
import { ConfigManager } from '../core/config';
import { CentralLogger } from '../core/logger';
import { ReviewParser, DeterministicReview } from '../review/parser';
import axios from 'axios';

export class AgentLoopController {
    private logger = CentralLogger.getInstance();
    private maxIterations = 3;

    public async run(task: string): Promise<string> {
        const config = await ConfigManager.getInstance().loadConfig();
        if (!config) throw new Error('Config missing');

        let currentCode = '';
        let iteration = 0;

        while (iteration < this.maxIterations) {
            iteration++;
            this.logger.info('system', `Loop Iteration ${iteration}: Coding...`);

            // 1. Coder Generates Code
            currentCode = await this.generateCode(config, task, currentCode);

            // 2. Reviewer Checks Code
            this.logger.info('system', `Loop Iteration ${iteration}: Reviewing...`);
            const review = await this.getReview(config, currentCode);

            if (!review) {
                this.logger.error('system', 'Reviewer failed to provide deterministic output.');
                break;
            }

            this.logger.info('system', `Review Severity: ${review.severity}. Errors: ${review.issues.length}`);

            if (review.severity === 'minor' && review.issues.length === 0) {
                this.logger.info('system', 'Auto-validation successful. Code meets quality bar.');
                break;
            }

            // 3. Feedback Loop (Inject review into next coder prompt)
            task = `Previous code had issues: ${review.summary}. Please fix: ${review.recommendedChanges.join(', ')}`;
        }

        return currentCode;
    }

    private async generateCode(config: any, prompt: string, context: string): Promise<string> {
        const url = `http://127.0.0.1:${config.coderModel.port}/v1/chat/completions`;
        const resp = await axios.post(url, {
            messages: [
                { role: 'system', content: 'You are a master software engineer. Output ONLY raw code, no talk.' },
                { role: 'user', content: prompt }
            ]
        });
        return resp.data.choices[0].message.content;
    }

    private async getReview(config: any, code: string): Promise<DeterministicReview | null> {
        const url = `http://127.0.0.1:${config.reviewerModel.port}/v1/chat/completions`;
        const systemPrompt = `Review this code. Output ONLY valid JSON matching: { severity: "critical"|"major"|"minor", issues: [{description, severity, line}], recommendedChanges: [string], summary: string }`;

        try {
            const resp = await axios.post(url, {
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: code }
                ]
            });
            return ReviewParser.parse(resp.data.choices[0].message.content);
        } catch (e) {
            return null;
        }
    }
}
