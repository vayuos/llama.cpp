import * as vscode from 'vscode';
import * as z from 'zod';
import { CentralLogger } from './logger';
import * as fs from 'fs';
import * as path from 'path';

export const ModelConfigSchema = z.object({
    enabled: z.boolean().default(true),
    baseUrl: z.string().optional(),
    host: z.string().default('127.0.0.1'),
    port: z.number().default(8080),
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

export const RuntimeConfigSchema = z.object({
    autoTestOnStart: z.boolean().default(true),
    showLogs: z.boolean().default(true),
    logLevel: z.enum(['debug', 'info', 'warn', 'error']).default('debug'),
    killSignal: z.enum(['SIGTERM', 'SIGKILL']).default('SIGTERM')
});

export const GravitasConfigSchema = z.object({
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

        const runtime = {
            autoTestOnStart: config.get<boolean>('runtime.autoTestOnStart') ?? true,
            showLogs: config.get<boolean>('runtime.showLogs') ?? true,
            logLevel: (config.get<string>('runtime.logLevel') as any) ?? 'info',
            killSignal: (config.get<string>('runtime.killSignal') as any) ?? 'SIGTERM'
        };

        const getModelConfig = (section: string): ModelConfig => {
            let host = config.get<string>(`${section}.general.host`) || '127.0.0.1';
            let port = config.get<number>(`${section}.general.port`) || (section === 'reviewer' ? 8011 : 8010);
            
            // --- IRONCLAD FALLBACK ---
            const workspacePath = '/home/viren/runs/full-server/gravitas-code';
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

            const defaultBaseUrl = `http://${host}:${port}/v1`;
            const userBaseUrl = config.get<string>(`${section}.general.baseUrl`);

            return {
                enabled: config.get<boolean>(`${section}.general.enabled`) ?? true,
                baseUrl: userBaseUrl && userBaseUrl.trim() !== "" ? userBaseUrl : defaultBaseUrl,
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
            runtime,
            coder: getModelConfig('coder'),
            reviewer: getModelConfig('reviewer'),
            vayuforge: {
                ragEndpoint: config.get<string>('vayuforge.ragEndpoint') || 'http://127.0.0.1:18081/retrieve'
            }
        };

        this.cachedConfig = resolvedConfig;
        return resolvedConfig;
    }

    public getCachedConfig(): GravitasConfig | null {
        return this.cachedConfig;
    }
}
