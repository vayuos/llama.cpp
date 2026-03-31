import * as fs from 'fs';
import { GravitasConfig } from '../../core/config';
import { ValidationResult, ValidationStep } from '../validator';

export class BinaryCheckStep implements ValidationStep {
    name = 'Check llama-server binary exists';
    async execute(config: GravitasConfig): Promise<ValidationResult> {
        if (fs.existsSync(config.llamaBinaryPath)) {
            return { success: true, message: 'Binary found.' };
        }
        return { success: false, message: `llama-server binary not found at: ${config.llamaBinaryPath}` };
    }
}

export class ModelCheckStep implements ValidationStep {
    name = 'Check model files exist';
    async execute(config: GravitasConfig): Promise<ValidationResult> {
        if (!fs.existsSync(config.coderModel.modelPath)) {
            return { success: false, message: `Coder model not found at: ${config.coderModel.modelPath}` };
        }
        if (!fs.existsSync(config.reviewerModel.modelPath)) {
            return { success: false, message: `Reviewer model not found at: ${config.reviewerModel.modelPath}` };
        }
        return { success: true, message: 'Model files found.' };
    }
}
