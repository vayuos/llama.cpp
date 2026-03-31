import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { z } from 'zod';

export const ModelConfigSchema = z.object({
    modelPath: z.string(),
    port: z.number(),
    gpuLayers: z.number().optional(),
    threads: z.number().optional(),
    ctx: z.number(),
    batch: z.number().optional(),
    ubatch: z.number().optional(),
    temp: z.number(),
    topP: z.number().optional()
});

export const GravitasConfigSchema = z.object({
    llamaBinaryPath: z.string(),
    coderModel: ModelConfigSchema,
    reviewerModel: ModelConfigSchema,
    workspaceRoot: z.string(),
    logDir: z.string()
});

export type GravitasConfig = z.infer<typeof GravitasConfigSchema>;

export class ConfigManager {
    private static instance: ConfigManager;
    private configPath: string;

    private constructor() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        const root = workspaceFolders ? workspaceFolders[0].uri.fsPath : '';
        this.configPath = path.join(root, '.gravitas', 'config.json');
    }

    public static getInstance(): ConfigManager {
        if (!ConfigManager.instance) {
            ConfigManager.instance = new ConfigManager();
        }
        return ConfigManager.instance;
    }

    public async loadConfig(): Promise<GravitasConfig | null> {
        if (!fs.existsSync(this.configPath)) {
            return null;
        }
        try {
            const content = fs.readFileSync(this.configPath, 'utf-8');
            const data = JSON.parse(content);
            return GravitasConfigSchema.parse(data);
        } catch (e) {
            console.error('Failed to load/parse config:', e);
            return null;
        }
    }

    public async saveConfig(config: GravitasConfig): Promise<void> {
        const dir = path.dirname(this.configPath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(this.configPath, JSON.stringify(config, null, 2), 'utf-8');
    }

    public getConfigPath(): string {
        return this.configPath;
    }
}
