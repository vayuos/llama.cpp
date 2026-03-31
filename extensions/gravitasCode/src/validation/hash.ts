import * as crypto from 'crypto';
import { GravitasConfig } from '../core/config';

export function calculateValidationHash(config: GravitasConfig): string {
    const data = JSON.stringify({
        bin: config.llamaBinPath,
        coder: config.coder,
        reviewer: config.reviewer,
        root: config.workspaceRoot
    });
    return crypto.createHash('sha256').update(data).digest('hex');
}
