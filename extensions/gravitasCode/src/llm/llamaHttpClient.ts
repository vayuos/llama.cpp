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
            // endpoint format: unix:///path/to/server.sock/v1
            const rawPath = endpoint.replace('unix://', '');
            const sockIndex = rawPath.indexOf('.sock');
            
            if (sockIndex !== -1) {
                const socketPath = rawPath.substring(0, sockIndex + 5); // include '.sock'
                const apiPrefix = rawPath.substring(sockIndex + 5); // e.g. '/v1'
                
                config.socketPath = socketPath;
                config.baseURL = `http://localhost${apiPrefix}`;
                logger.debug('system', `LlamaHttpClient: UDS Mode - Socket: ${socketPath}, Prefix: ${apiPrefix}`);
            } else {
                config.socketPath = rawPath;
                config.baseURL = 'http://localhost';
                logger.debug('system', `LlamaHttpClient: UDS Mode - Raw Socket: ${rawPath}`);
            }
        } else {
            config.baseURL = endpoint;
        }

        this.client = axios.create(config);
    }

    async post(path: string, data: any, retries = 2, signal?: AbortSignal): Promise<any> {
        try {
            const response = await this.client.post(path, data, { signal });
            return response.data;
        } catch (e: any) {
            if (axios.isCancel(e) || e.name === 'AbortError') {
                const logger = require('../core/logger').CentralLogger.getInstance();
                logger.debug('system', `LlamaHttpClient: POST ${path} aborted via signal.`);
                throw e;
            }
            const logger = require('../core/logger').CentralLogger.getInstance();
            if (retries > 0 && this.isRetryable(e)) {
                logger.warn('system', `LlamaHttpClient: POST ${path} failed (${e.code}). Retrying... (${retries} left)`);
                await new Promise(r => setTimeout(r, 500));
                return this.post(path, data, retries - 1, signal);
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
