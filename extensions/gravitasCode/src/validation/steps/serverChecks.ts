import axios from 'axios';
import { GravitasConfig } from '../../core/config';
import { UnifiedProcessManager } from '../../process/processManager';
import { ValidationResult, ValidationStep } from '../validator';
import { LlamaHttpClient } from '../../llm/llamaHttpClient';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';

export class ServerPingStep implements ValidationStep {
    constructor(private type: 'coder' | 'reviewer') { }

    get name() { return `Ping ${this.type} server health endpoint`; }

    async execute(config: GravitasConfig): Promise<ValidationResult> {
        const pm = UnifiedProcessManager.getInstance();
        
        const sockPath = path.join(os.homedir(), '.gravitas', 'sockets', `${this.type}.sock`);
        const endpoint = fs.existsSync(sockPath) 
            ? `unix://${sockPath}` 
            : `http://${config[this.type].host || '127.0.0.1'}:${config[this.type].port}`;
            
        const client = new LlamaHttpClient(endpoint);

        // Start server
        if (this.type === 'coder') await pm.startCoder(config);
        else await pm.startReviewer(config);

        // Wait for health check (max 90s for CPU models)
        for (let i = 0; i < 90; i++) {
            try {
                await client.get('/v1/models');
                return { success: true, message: `${this.type} server is healthy.` };
            } catch (e) {
                // Check if process crashed
                // Check if process crashed
                const status = pm.getProcessStatus(this.type);
                if (!status.pid) {
                    const lastError = pm.getLastError(this.type);
                    return { success: false, message: `${this.type} server crashed! Logs:\n${lastError}` };
                }
                await new Promise(r => setTimeout(r, 1000));
            }
        }

        return { success: false, message: `${this.type} server failed to respond within 90s.` };
    }

    async rollback() {
        // Don't stop servers - keep them running for user to inspect
        // await UnifiedProcessManager.getInstance().stopAll();
    }
}

export class PromptTestStep implements ValidationStep {
    constructor(private type: 'coder' | 'reviewer') { }

    get name() { return `Send test ${this.type} prompt`; }

    async execute(config: GravitasConfig): Promise<ValidationResult> {
        const sockPath = path.join(os.homedir(), '.gravitas', 'sockets', `${this.type}.sock`);
        const endpoint = fs.existsSync(sockPath) 
            ? `unix://${sockPath}` 
            : `http://${config[this.type].host || '127.0.0.1'}:${config[this.type].port}`;
            
        const client = new LlamaHttpClient(endpoint);
        const prompt = this.type === 'coder' ? 'print("hello")' : 'How are you?';

        // Retry for up to 60s for model load
        for (let i = 0; i < 30; i++) {
            try {
                const resp = await (client as any).client.post('/v1/chat/completions', {
                    messages: [{ role: 'user', content: prompt }],
                    max_tokens: 10
                }, { timeout: 20000 }); // Increased timeout for inference

                if (resp.status === 200) {
                    return { success: true, message: `${this.type} prompt test passed.` };
                }
                return { success: false, message: `${this.type} returned status ${resp.status}` };
            } catch (e: any) {
                // If 503 or connection reset (model loading), wait and retry
                if (e.response?.status === 503 || e.code === 'ECONNRESET') {
                    await new Promise(r => setTimeout(r, 2000));
                    continue;
                }
                // If it's the last attempt, fail
                if (i === 29) {
                    return { success: false, message: `${this.type} prompt test failed: ${e.message}` };
                }
                // Recoverable network error?
                await new Promise(r => setTimeout(r, 1000));
            }
        }
        return { success: false, message: `${this.type} failed to load model within 60s.` };
    }

    async rollback() {
        // Don't stop servers - keep them running for user to inspect
        // await UnifiedProcessManager.getInstance().stopAll();
    }
}
