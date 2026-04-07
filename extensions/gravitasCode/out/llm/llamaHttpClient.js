"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.LlamaHttpClient = void 0;
const axios_1 = __importDefault(require("axios"));
const http = __importStar(require("http"));
class LlamaHttpClient {
    constructor(endpoint) {
        const logger = require('../core/logger').CentralLogger.getInstance();
        const isSocket = endpoint.startsWith('unix://');
        logger.debug('system', `LlamaHttpClient: Creating client for ${endpoint}. Protocol: ${isSocket ? 'UDS' : 'HTTP'}`);
        // --- ZERO-LATENCY: Persistent Connection Pool ---
        const agent = new http.Agent({
            keepAlive: true,
            maxSockets: 32, // Allow parallel telemetry + inference
            keepAliveMsecs: 1000
        });
        const config = {
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
        }
        else {
            config.baseURL = endpoint;
        }
        this.client = axios_1.default.create(config);
    }
    async post(path, data, retries = 2) {
        try {
            const response = await this.client.post(path, data);
            return response.data;
        }
        catch (e) {
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
    async get(path, retries = 2) {
        try {
            const response = await this.client.get(path);
            return response.data;
        }
        catch (e) {
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
    isRetryable(e) {
        const code = e.code;
        return code === 'ECONNREFUSED' || code === 'ECONNRESET' || code === 'ETIMEDOUT';
    }
}
exports.LlamaHttpClient = LlamaHttpClient;
//# sourceMappingURL=llamaHttpClient.js.map