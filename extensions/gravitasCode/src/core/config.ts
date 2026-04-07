import * as vscode from 'vscode';
import * as z from 'zod';
import { CentralLogger } from './logger';
import * as fs from 'fs';
import * as path from 'path';

export const ModelConfigSchema = z.object({
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

export const ConnectionConfigSchema = z.object({
    mode: z.enum(['local', 'remote']).default('local'),
    system3Ip: z.string().default('127.0.0.1')
});

export const RuntimeConfigSchema = z.object({
    autoTestOnStart: z.boolean().default(true),
    showLogs: z.boolean().default(true),
    logLevel: z.enum(['debug', 'info', 'warn', 'error']).default('debug'),
    killSignal: z.enum(['SIGTERM', 'SIGKILL']).default('SIGTERM')
});

export const GravitasConfigSchema = z.object({
    connection: ConnectionConfigSchema,
    coder: ModelConfigSchema,
    reviewer: ModelConfigSchema,
    vayuforge: z.object({
        ragEndpoint: z.string().default('http://127.0.0.1:18081/retrieve')
    }),
    runtime: RuntimeConfigSchema,
    workspaceRoot: z.string().optional(),
    logDir: z.string().optional(),
    llamaBinPath: z.string().optional()
});

export type GravitasConfig = z.infer<typeof GravitasConfigSchema>;
export type ModelConfig = z.infer<typeof ModelConfigSchema>;

export class ConfigManager {
    private static instance: ConfigManager;
    private lastLoggedSync: { [key: string]: string } = {};
    private cachedConfig: GravitasConfig | null = null;

    private constructor() { }

    public static getInstance(): ConfigManager {
        if (!ConfigManager.instance) {
            ConfigManager.instance = new ConfigManager();
        }
        return ConfigManager.instance;
    }

    public async loadConfig(): Promise<GravitasConfig | null> {
        const logger = CentralLogger.getInstance();
        
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
            mode: config.get<'local' | 'remote'>('connection.mode') ?? 'local',
            system3Ip: config.get<string>('connection.system3Ip') ?? '127.0.0.1'
        };

        const runtime = {
            autoTestOnStart: config.get<boolean>('runtime.autoTestOnStart') ?? true,
            showLogs: config.get<boolean>('runtime.showLogs') ?? true,
            logLevel: (config.get<string>('runtime.logLevel') as any) ?? 'debug',
            killSignal: (config.get<string>('runtime.killSignal') as any) ?? 'SIGTERM'
        };

        const getModelConfig = (section: string): ModelConfig => {
            let host = config.get<string>(`${section}.general.host`) || '127.0.0.1';
            let port = config.get<number>(`${section}.general.port`) || (section === 'reviewer' ? 18080 : 8080);
            
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
                        if (diskHost) host = diskHost;
                    } catch (e) {}
                }
            }

            let defaultBaseUrl = `http://${host}:${port}`;
            
            // Override with UDS for local Linux
            if (connection.mode === 'local' && process.platform === 'linux') {
                const socketPath = path.join(require('os').homedir(), '.gravitas', 'sockets', `${section}.sock`);
                defaultBaseUrl = `unix://${socketPath}`;
            }

            const userBaseUrl = config.get<string>(`${section}.general.baseUrl`);
            let finalBaseUrl = userBaseUrl && userBaseUrl.trim() !== "" ? userBaseUrl : defaultBaseUrl;

            // --- NORMALIZE: Ensure no /v1 suffix for health/metrics compatibility ---
            finalBaseUrl = finalBaseUrl.replace(/\/v1\/?$/, "");
            finalBaseUrl = finalBaseUrl.replace(/\/$/, "");

            return {
                enabled: config.get<boolean>(`${section}.general.enabled`) ?? true,
                baseUrl: finalBaseUrl,
                host: host,
                port: port,
                modelPath: config.get<string>(`${section}.general.modelPath`) || '',
                contextSize: config.get<number>(`${section}.hardware.contextSize`) || 102400,
                temperature: config.get<number>(`${section}.sampling.temperature`) ?? 0.2,
                topP: config.get<number>(`${section}.sampling.topP`),
                topK: config.get<number>(`${section}.sampling.topK`),
                repeatPenalty: config.get<number>(`${section}.sampling.repeatPenalty`) ?? 1.1,
                modelName: config.get<string>(`${section}.general.modelName`),
                strictMode: config.get<boolean>(`${section}.general.strictMode`) ?? true
            };
        };

        const resolvedConfig: GravitasConfig = {
            workspaceRoot: workspaceFolder?.fsPath || '',
            logDir: workspaceFolder ? path.join(workspaceFolder.fsPath, '.gravitas', 'logs') : '',
            connection,
            runtime,
            coder: getModelConfig('coder'),
            reviewer: getModelConfig('reviewer'),
            vayuforge: {
                ragEndpoint: config.get<string>('vayuforge.ragEndpoint') || `http://${connection.mode === 'remote' ? connection.system3Ip : '127.0.0.1'}:8081/retrieve`
            }
        };

        this.cachedConfig = resolvedConfig;
        return resolvedConfig;
        } catch (e: any) {
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
            } as any;
        }
    }

    public getCachedConfig(): GravitasConfig | null {
        return this.cachedConfig;
    }
}
