import axios from 'axios';
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
        const c = config.coder as any;
        const monorepoBin = '/home/viren/llama/llama.cpp/build_cuda_mmq_moe/bin/llama-server';
        const binPath = c.binPath?.trim() ? c.binPath : (config as any).llamaBinPath || monorepoBin;
        return this.coder.start(binPath, c, []);
    }

    public async startReviewer(config: GravitasConfig): Promise<boolean> {
        const r = config.reviewer as any;
        const monorepoBin = '/home/viren/llama/llama.cpp/build_cuda_mmq_moe/bin/llama-server';
        const binPath = r.binPath?.trim() ? r.binPath : (config as any).llamaBinPath || monorepoBin;
        return this.reviewer.start(binPath, r, []);
    }

    public async stopAll(): Promise<void> {
        await Promise.all([this.coder.stop(), this.reviewer.stop()]);
    }

    public getLastError(type: 'coder' | 'reviewer'): string {
        return type === 'coder' ? this.coder.getLastError() : this.reviewer.getLastError();
    }

    public async getLiveStatus(type: 'coder' | 'reviewer', config: GravitasConfig): Promise<{ running: boolean; external: boolean; pid?: number; telemetry?: string }> {
        const proc = type === 'coder' ? this.coder : this.reviewer;
        const localPid = proc.getPid();
        
        if (localPid) {
            return { running: true, external: false, pid: localPid, telemetry: proc.getTelemetry() };
        }

        // Pulse check for external process
        const modelCfg = type === 'coder' ? config.coder : config.reviewer;
        const url = `http://${modelCfg.host || '127.0.0.1'}:${modelCfg.port}/v1/models`;
        
        try {
            await axios.get(url, { timeout: 1000 });
            return { running: true, external: true, telemetry: 'Live (External)' };
        } catch (e) {
            return { running: false, external: false };
        }
    }

    public getProcessStatus(type: 'coder' | 'reviewer'): { pid?: number; telemetry?: string } {
        const proc = type === 'coder' ? this.coder : this.reviewer;
        return {
            pid: proc.getPid(),
            telemetry: proc.getTelemetry()
        };
    }
}
