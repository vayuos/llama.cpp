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
            timeout: 30000, 
            httpAgent: agent,
            headers: { 
                'Content-Type': 'application/json',
                'Connection': 'keep-alive' 
            }
        };

        if (isSocket) {
            // Remove httpAgent for UDS to avoid connection conflicts in axios
            delete config.httpAgent;
            
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

    async post(path: string, data: any, retries = 2, signal?: AbortSignal, timeout?: number): Promise<any> {
        try {
            const isStream = data.stream === true;
            const response = await this.client.post(path, data, { 
                signal, 
                timeout,
                responseType: isStream ? 'stream' : 'json'
            });
            return isStream ? response : response.data;
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
            if (e.response?.status === 404) {
                logger.debug('system', `LlamaHttpClient: POST ${path} returned 404 (Expected/Optional).`);
            } else {
                logger.error('system', `LlamaHttpClient: POST ${path} FATAL ERROR: ${e.message} (Code: ${e.code})`);
            }
            throw e;
        }
    }

    async get(path: string, retries = 2, timeout?: number): Promise<any> {
        try {
            const response = await this.client.get(path, { timeout });
            return response.data;
        } catch (e: any) {
            const logger = require('../core/logger').CentralLogger.getInstance();
            if (retries > 0 && this.isRetryable(e)) {
                logger.warn('system', `LlamaHttpClient: GET ${path} failed (${e.code}). Retrying... (${retries} left)`);
                await new Promise(r => setTimeout(r, 500));
                return this.get(path, retries - 1);
            }
            if (e.response?.status === 404) {
                logger.debug('system', `LlamaHttpClient: GET ${path} returned 404 (Expected/Optional).`);
            } else {
                logger.error('system', `LlamaHttpClient: GET ${path} FATAL ERROR: ${e.message} (Code: ${e.code})`);
            }
            throw e;
        }
    }

    private isRetryable(e: any): boolean {
        const code = e.code;
        return code === 'ECONNREFUSED' || code === 'ECONNRESET' || code === 'ETIMEDOUT';
    }
}
