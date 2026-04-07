import axios, { AxiosInstance } from 'axios';
import * as http from 'http';

export class LlamaHttpClient {
    private client: AxiosInstance;

    constructor(endpoint: string) {
        const logger = require('../core/logger').CentralLogger.getInstance();
        const isSocket = endpoint.startsWith('unix://');
        
        logger.debug('system', `LlamaHttpClient: Creating client for ${endpoint}. Protocol: ${isSocket ? 'UDS' : 'HTTP'}`);

        // --- ZERO-LATENCY: Persistent Connection Pool ---
        const agent = new http.Agent({ 
            keepAlive: true, 
            maxSockets: 32, // Allow parallel telemetry + inference
            keepAliveMsecs: 1000 
        });

        const config: any = {
            timeout: 300000, 
            httpAgent: agent,
            headers: { 
                'Content-Type': 'application/json',
                'Connection': 'keep-alive' 
            }
        };

        if (isSocket) {
            config.socketPath = endpoint.replace('unix://', '');
            config.baseURL = 'http://localhost';
            logger.debug('system', `LlamaHttpClient: UDS Socket Path resolved to: ${config.socketPath}`);
        } else {
            config.baseURL = endpoint;
        }

        this.client = axios.create(config);
    }

    async post(path: string, data: any, retries = 2): Promise<any> {
        try {
            const response = await this.client.post(path, data);
            return response.data;
        } catch (e: any) {
            const logger = require('../core/logger').CentralLogger.getInstance();
            if (retries > 0 && this.isRetryable(e)) {
                logger.warn('system', `LlamaHttpClient: POST ${path} failed (${e.code}). Retrying... (${retries} left)`);
                await new Promise(r => setTimeout(r, 500));
                return this.post(path, data, retries - 1);
            }
            logger.error('system', `LlamaHttpClient: POST ${path} FATAL ERROR: ${e.message} (Code: ${e.code})`);
            throw e;
        }
    }

    async get(path: string, retries = 2): Promise<any> {
        try {
            const response = await this.client.get(path);
            return response.data;
        } catch (e: any) {
            const logger = require('../core/logger').CentralLogger.getInstance();
            if (retries > 0 && this.isRetryable(e)) {
                logger.warn('system', `LlamaHttpClient: GET ${path} failed (${e.code}). Retrying... (${retries} left)`);
                await new Promise(r => setTimeout(r, 500));
                return this.get(path, retries - 1);
            }
            logger.error('system', `LlamaHttpClient: GET ${path} FATAL ERROR: ${e.message} (Code: ${e.code})`);
            throw e;
        }
    }

    private isRetryable(e: any): boolean {
        const code = e.code;
        return code === 'ECONNREFUSED' || code === 'ECONNRESET' || code === 'ETIMEDOUT';
    }
}
