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
Object.defineProperty(exports, "__esModule", { value: true });
exports.ConfigManager = exports.GravitasConfigSchema = exports.RuntimeConfigSchema = exports.ConnectionConfigSchema = exports.ModelConfigSchema = void 0;
const vscode = __importStar(require("vscode"));
const z = __importStar(require("zod"));
const logger_1 = require("./logger");
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
exports.ModelConfigSchema = z.object({
    enabled: z.boolean().default(true),
    baseUrl: z.string().optional(),
    host: z.string().default('127.0.0.1'),
    port: z.number().default(18080),
    binPath: z.string().optional(),
    contextSize: z.number().default(102400),
    temperature: z.number().default(0.2),
    topP: z.number().optional(),
    topK: z.number().optional(),
    repeatPenalty: z.number().default(1.1),
    modelName: z.string().optional(),
    strictMode: z.boolean().default(true),
    modelPath: z.string().optional()
});
exports.ConnectionConfigSchema = z.object({
    mode: z.enum(['local', 'remote']).default('local'),
    system3Ip: z.string().default('127.0.0.1')
});
exports.RuntimeConfigSchema = z.object({
    autoTestOnStart: z.boolean().default(true),
    showLogs: z.boolean().default(true),
    logLevel: z.enum(['debug', 'info', 'warn', 'error']).default('debug'),
    killSignal: z.enum(['SIGTERM', 'SIGKILL']).default('SIGTERM')
});
exports.GravitasConfigSchema = z.object({
    connection: exports.ConnectionConfigSchema,
    coder: exports.ModelConfigSchema,
    reviewer: exports.ModelConfigSchema,
    vayuforge: z.object({
        ragEndpoint: z.string().default('http://127.0.0.1:18081/retrieve')
    }),
    runtime: exports.RuntimeConfigSchema,
    workspaceRoot: z.string().optional(),
    logDir: z.string().optional(),
    llamaBinPath: z.string().optional()
});
class ConfigManager {
    constructor() {
        this.lastLoggedSync = {};
        this.cachedConfig = null;
    }
    static getInstance() {
        if (!ConfigManager.instance) {
            ConfigManager.instance = new ConfigManager();
        }
        return ConfigManager.instance;
    }
    async loadConfig() {
        const logger = logger_1.CentralLogger.getInstance();
        try {
            // SMART SCOPING
            let bestFolder = vscode.workspace.workspaceFolders?.[0];
            if (vscode.workspace.workspaceFolders) {
                for (const folder of vscode.workspace.workspaceFolders) {
                    const folderConfig = vscode.workspace.getConfiguration('gravitasCode', folder.uri);
                    const inspect = folderConfig.inspect('coder.general.port');
                    if (inspect?.workspaceFolderValue !== undefined || inspect?.workspaceValue !== undefined) {
                        bestFolder = folder;
                        break;
                    }
                }
            }
            const workspaceFolder = bestFolder?.uri;
            const config = vscode.workspace.getConfiguration('gravitasCode', workspaceFolder);
            const connection = {
                mode: config.get('connection.mode') ?? 'local',
                system3Ip: config.get('connection.system3Ip') ?? '127.0.0.1'
            };
            const runtime = {
                autoTestOnStart: config.get('runtime.autoTestOnStart') ?? true,
                showLogs: config.get('runtime.showLogs') ?? true,
                logLevel: config.get('runtime.logLevel') ?? 'debug',
                killSignal: config.get('runtime.killSignal') ?? 'SIGTERM'
            };
            const getModelConfig = (section) => {
                let host = config.get(`${section}.general.host`) || '127.0.0.1';
                let port = config.get(`${section}.general.port`) || (section === 'reviewer' ? 18080 : 8080);
                if (connection.mode === 'remote') {
                    host = connection.system3Ip;
                }
                // --- DISK SYNC (Priority) ---
                const workspacePath = workspaceFolder?.fsPath;
                if (workspacePath) {
                    const settingsPath = path.join(workspacePath, '.vscode', 'settings.json');
                    if (fs.existsSync(settingsPath)) {
                        try {
                            const raw = fs.readFileSync(settingsPath, 'utf8');
                            const settings = JSON.parse(raw);
                            const diskPort = settings[`gravitasCode.${section}.general.port`];
                            if (diskPort && diskPort !== port) {
                                port = diskPort;
                                const logKey = `${section}-${settingsPath}-${port}`;
                                if (this.lastLoggedSync[section] !== logKey) {
                                    logger.info('system', `[Disk Sync] Resolved ${section} port to ${port} via ${settingsPath}`);
                                    this.lastLoggedSync[section] = logKey;
                                }
                            }
                            const diskHost = settings[`gravitasCode.${section}.general.host`];
                            if (diskHost)
                                host = diskHost;
                        }
                        catch (e) { }
                    }
                }
                let defaultBaseUrl = `http://${host}:${port}`;
                // Override with UDS for local Linux
                if (connection.mode === 'local' && process.platform === 'linux') {
                    const socketPath = path.join(require('os').homedir(), '.gravitas', 'sockets', `${section}.sock`);
                    defaultBaseUrl = `unix://${socketPath}`;
                }
                const userBaseUrl = config.get(`${section}.general.baseUrl`);
                let finalBaseUrl = userBaseUrl && userBaseUrl.trim() !== "" ? userBaseUrl : defaultBaseUrl;
                // --- NORMALIZE: Ensure no /v1 suffix for health/metrics compatibility ---
                finalBaseUrl = finalBaseUrl.replace(/\/v1\/?$/, "");
                finalBaseUrl = finalBaseUrl.replace(/\/$/, "");
                return {
                    enabled: config.get(`${section}.general.enabled`) ?? true,
                    baseUrl: finalBaseUrl,
                    host: host,
                    port: port,
                    modelPath: config.get(`${section}.general.modelPath`) || '',
                    contextSize: config.get(`${section}.hardware.contextSize`) || 102400,
                    temperature: config.get(`${section}.sampling.temperature`) ?? 0.2,
                    topP: config.get(`${section}.sampling.topP`),
                    topK: config.get(`${section}.sampling.topK`),
                    repeatPenalty: config.get(`${section}.sampling.repeatPenalty`) ?? 1.1,
                    modelName: config.get(`${section}.general.modelName`),
                    strictMode: config.get(`${section}.general.strictMode`) ?? true
                };
            };
            const resolvedConfig = {
                workspaceRoot: workspaceFolder?.fsPath || '',
                logDir: workspaceFolder ? path.join(workspaceFolder.fsPath, '.gravitas', 'logs') : '',
                connection,
                runtime,
                coder: getModelConfig('coder'),
                reviewer: getModelConfig('reviewer'),
                vayuforge: {
                    ragEndpoint: config.get('vayuforge.ragEndpoint') || `http://${connection.mode === 'remote' ? connection.system3Ip : '127.0.0.1'}:8081/retrieve`
                }
            };
            this.cachedConfig = resolvedConfig;
            return resolvedConfig;
        }
        catch (e) {
            logger.error('system', `Failed to load configuration: ${e.message}`);
            // Return a minimal but COMPLETE fallback config to allow UI to register
            const defaultMode = 'local';
            const defaultCoderUrl = process.platform === 'linux' ? `unix://${path.join(require('os').homedir(), '.gravitas', 'sockets', 'coder.sock')}` : 'http://127.0.0.1:8040';
            const defaultReviewerUrl = process.platform === 'linux' ? `unix://${path.join(require('os').homedir(), '.gravitas', 'sockets', 'reviewer.sock')}` : 'http://127.0.0.1:18080';
            return {
                workspaceRoot: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '',
                logDir: '',
                connection: { mode: defaultMode, system3Ip: '127.0.0.1' },
                runtime: { autoTestOnStart: false, showLogs: true, logLevel: 'debug', killSignal: 'SIGTERM' },
                coder: { enabled: true, host: '127.0.0.1', port: 8040, baseUrl: defaultCoderUrl, contextSize: 102400, temperature: 0.2, repeatPenalty: 1.1, strictMode: true },
                reviewer: { enabled: true, host: '127.0.0.1', port: 18080, baseUrl: defaultReviewerUrl, contextSize: 102400, temperature: 0.0, repeatPenalty: 1.0, strictMode: true },
                vayuforge: { ragEndpoint: 'http://127.0.0.1:18081/retrieve' }
            };
        }
    }
    getCachedConfig() {
        return this.cachedConfig;
    }
}
exports.ConfigManager = ConfigManager;
//# sourceMappingURL=config.js.map