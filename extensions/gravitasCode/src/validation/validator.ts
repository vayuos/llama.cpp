import * as vscode from 'vscode';
import { GravitasConfig } from '../core/config';

export interface ValidationResult {
    success: boolean;
    message: string;
    logs?: string[];
}

export interface ValidationStep {
    name: string;
    execute: (config: GravitasConfig) => Promise<ValidationResult>;
    rollback?: () => Promise<void>;
}

export class ValidationEngine {
    private steps: ValidationStep[] = [];
    private logs: string[] = [];

    public addStep(step: ValidationStep) {
        this.steps.push(step);
    }

    public async run(config: GravitasConfig): Promise<ValidationResult> {
        const logger = require('../core/logger').CentralLogger.getInstance();
        this.logs = [];
        logger.info('system', 'ValidationEngine: Starting validation pipeline...');

        for (const step of this.steps) {
            this.logs.push(`[STEP] ${step.name}`);
            logger.debug('system', `ValidationEngine: Executing step: ${step.name}`);
            try {
                const result = await step.execute(config);
                if (!result.success) {
                    this.logs.push(`[FAILURE] ${step.name}: ${result.message}`);
                    logger.error('system', `ValidationEngine: Step FAILURE (${step.name}): ${result.message}`);
                    if (step.rollback) {
                        this.logs.push(`[ROLLBACK] Triggered for ${step.name}`);
                        logger.warn('system', `ValidationEngine: Rolling back step ${step.name}...`);
                        await step.rollback();
                    }
                    return { success: false, message: result.message, logs: this.logs };
                }
                this.logs.push(`[SUCCESS] ${step.name}`);
                logger.debug('system', `ValidationEngine: Step SUCCESS: ${step.name}`);
            } catch (e: any) {
                this.logs.push(`[ERROR] ${step.name}: ${e.message}`);
                logger.error('system', `ValidationEngine: Step CRASHED (${step.name}): ${e.message}`);
                return { success: false, message: e.message, logs: this.logs };
            }
        }

        logger.info('system', 'ValidationEngine: All validation steps passed successfully.');
        return { success: true, message: 'All validation steps passed.', logs: this.logs };
    }

    public getLogs(): string[] {
        return this.logs;
    }
}
