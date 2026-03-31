import { LlamaProcess } from './llamaProcess';
import { GravitasConfig } from '../core/config';

export class UnifiedProcessManager {
    private static instance: UnifiedProcessManager;
    private coder: LlamaProcess;
    private reviewer: LlamaProcess;

    private constructor() {
        this.coder = new LlamaProcess('Coder Server', 'coder');
        this.reviewer = new LlamaProcess('Reviewer Server', 'reviewer');
    }

    public static getInstance(): UnifiedProcessManager {
        if (!UnifiedProcessManager.instance) {
            UnifiedProcessManager.instance = new UnifiedProcessManager();
        }
        return UnifiedProcessManager.instance;
    }

    public async startCoder(config: GravitasConfig): Promise<boolean> {
        const args = [
            '--batch-size', config.coderModel.batch?.toString() || '512',
            '--ubatch-size', config.coderModel.ubatch?.toString() || '512',
            '--top-p', config.coderModel.topP?.toString() || '0.95'
        ];
        return this.coder.start(config.llamaBinaryPath, config.coderModel, args);
    }

    public async startReviewer(config: GravitasConfig): Promise<boolean> {
        return this.reviewer.start(config.llamaBinaryPath, config.reviewerModel);
    }

    public async stopAll(): Promise<void> {
        await Promise.all([this.coder.stop(), this.reviewer.stop()]);
    }
}
