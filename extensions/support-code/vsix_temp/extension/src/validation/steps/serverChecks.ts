import axios from 'axios';
import { GravitasConfig } from '../../core/config';
import { UnifiedProcessManager } from '../../process/processManager';
import { ValidationResult, ValidationStep } from '../validator';

export class ServerPingStep implements ValidationStep {
    constructor(private type: 'coder' | 'reviewer') { }

    get name() { return `Ping ${this.type} server health endpoint`; }

    async execute(config: GravitasConfig): Promise<ValidationResult> {
        const pm = UnifiedProcessManager.getInstance();
        const port = this.type === 'coder' ? config.coderModel.port : config.reviewerModel.port;
        const endpoint = `http://127.0.0.1:${port}/v1/models`;

        // Start server
        if (this.type === 'coder') await pm.startCoder(config);
        else await pm.startReviewer(config);

        // Wait for health check (max 30s)
        for (let i = 0; i < 30; i++) {
            try {
                await axios.get(endpoint, { timeout: 1000 });
                return { success: true, message: `${this.type} server is healthy.` };
            } catch (e) {
                await new Promise(r => setTimeout(r, 1000));
            }
        }

        return { success: false, message: `${this.type} server failed to respond within 30s.` };
    }

    async rollback() {
        await UnifiedProcessManager.getInstance().stopAll();
    }
}

export class PromptTestStep implements ValidationStep {
    constructor(private type: 'coder' | 'reviewer') { }

    get name() { return `Send test ${this.type} prompt`; }

    async execute(config: GravitasConfig): Promise<ValidationResult> {
        const port = this.type === 'coder' ? config.coderModel.port : config.reviewerModel.port;
        const endpoint = `http://127.0.0.1:${port}/v1/chat/completions`;
        const prompt = this.type === 'coder' ? 'print("hello")' : 'How are you?';

        try {
            const resp = await axios.post(endpoint, {
                messages: [{ role: 'user', content: prompt }],
                max_tokens: 10
            }, { timeout: 10000 });

            if (resp.status === 200) {
                return { success: true, message: `${this.type} prompt test passed.` };
            }
            return { success: false, message: `${this.type} returned status ${resp.status}` };
        } catch (e: any) {
            return { success: false, message: `${this.type} prompt test failed: ${e.message}` };
        }
    }

    async rollback() {
        await UnifiedProcessManager.getInstance().stopAll();
    }
}
