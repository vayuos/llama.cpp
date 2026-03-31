import * as fs from 'fs';
import { GravitasConfig } from '../../core/config';
import { ValidationResult, ValidationStep } from '../validator';
import { resolveTilde, resolveBinaryPath } from '../../utils/pathUtils';

export class BinaryCheckStep implements ValidationStep {
    name = 'Check llama-server binary exists';
    async execute(config: GravitasConfig): Promise<ValidationResult> {
        const rawPath = config.llamaBinPath || '(empty)';
        const resolvedPath = resolveBinaryPath(config.llamaBinPath || '');

        if (fs.existsSync(resolvedPath)) {
            const stat = fs.statSync(resolvedPath);
            if (stat.isDirectory()) {
                return { success: false, message: `The path provided is a directory, not a binary file. User setting: "${rawPath}", Attempted absolute path: "${resolvedPath}"` };
            }
            return { success: true, message: 'Binary found.' };
        }
        return { success: false, message: `llama-server binary not found. User setting: "${rawPath}", Attempted absolute path: "${resolvedPath}"` };
    }
}

export class ModelCheckStep implements ValidationStep {
    name = 'Check model files exist';
    async execute(config: GravitasConfig): Promise<ValidationResult> {
        const rawCoder = config.coder.modelPath || '(empty)';
        const rawReviewer = config.reviewer.modelPath || '(empty)';
        const coderPath = resolveTilde(config.coder.modelPath || '');
        const reviewerPath = resolveTilde(config.reviewer.modelPath || '');

        if (!fs.existsSync(coderPath)) {
            return { success: false, message: `Coder model not found. User setting: "${rawCoder}", Attempted absolute path: "${coderPath}"` };
        }
        if (!fs.existsSync(reviewerPath)) {
            return { success: false, message: `Reviewer model not found. User setting: "${rawReviewer}", Attempted absolute path: "${reviewerPath}"` };
        }
        return { success: true, message: 'Model files found.' };
    }
}
