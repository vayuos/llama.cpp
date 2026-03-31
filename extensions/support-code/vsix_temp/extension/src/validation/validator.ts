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
        this.logs = [];
        console.log('Starting validation pipeline...');

        for (const step of this.steps) {
            this.logs.push(`[STEP] ${step.name}`);
            try {
                const result = await step.execute(config);
                if (!result.success) {
                    this.logs.push(`[FAILURE] ${step.name}: ${result.message}`);
                    if (step.rollback) {
                        this.logs.push(`[ROLLBACK] Triggered for ${step.name}`);
                        await step.rollback();
                    }
                    return { success: false, message: result.message, logs: this.logs };
                }
                this.logs.push(`[SUCCESS] ${step.name}`);
            } catch (e: any) {
                this.logs.push(`[ERROR] ${step.name}: ${e.message}`);
                return { success: false, message: e.message, logs: this.logs };
            }
        }

        return { success: true, message: 'All validation steps passed.', logs: this.logs };
    }

    public getLogs(): string[] {
        return this.logs;
    }
}
