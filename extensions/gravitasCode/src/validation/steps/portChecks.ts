import * as net from 'net';
import { GravitasConfig } from '../../core/config';
import { ValidationResult, ValidationStep } from '../validator';

export class PortCheckStep implements ValidationStep {
    name = 'Check ports are free';
    async execute(config: GravitasConfig): Promise<ValidationResult> {
        const ports = [config.coder.port, config.reviewer.port];
        for (const port of ports) {
            const isFree = await this.isPortFree(port);
            if (!isFree) {
                return { success: false, message: `Port ${port} is already in use.` };
            }
        }
        return { success: true, message: 'Ports are free.' };
    }

    private isPortFree(port: number): Promise<boolean> {
        return new Promise((resolve) => {
            const server = net.createServer();
            server.once('error', () => resolve(false));
            server.once('listening', () => {
                server.close();
                resolve(true);
            });
            server.listen(port);
        });
    }
}
