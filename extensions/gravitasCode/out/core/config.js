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
exports.ConfigManager = exports.GravitasConfigSchema = exports.RuntimeConfigSchema = exports.ModelConfigSchema = void 0;
const vscode = __importStar(require("vscode"));
const z = __importStar(require("zod"));
const pathUtils_1 = require("../utils/pathUtils");
exports.ModelConfigSchema = z.object({
    enabled: z.boolean().default(true),
    mode: z.enum(['gpu', 'cpu']).default('cpu'),
    modelPath: z.string(),
    host: z.string().default('127.0.0.1'),
    port: z.number(),
    cudaVisibleDevices: z.string().optional(),
    nGpuLayers: z.number().default(0),
    contextSize: z.number().default(8192),
    threads: z.number().default(8),
    threadsBatch: z.number().optional(),
    batchSize: z.number().default(512),
    ubatchSize: z.number().default(512),
    temperature: z.number().default(0.2),
    topP: z.number().optional(),
    topK: z.number().optional(),
    repeatPenalty: z.number().default(1.1),
    noWarmup: z.boolean().default(true),
    numaInterleave: z.boolean().optional(),
    prefixCommand: z.string().optional()
});
exports.RuntimeConfigSchema = z.object({
    autoTestOnStart: z.boolean().default(true),
    showLogs: z.boolean().default(true),
    logLevel: z.enum(['debug', 'info', 'warn', 'error']).default('debug'),
    killSignal: z.enum(['SIGTERM', 'SIGKILL']).default('SIGTERM')
});
exports.GravitasConfigSchema = z.object({
    llamaBinPath: z.string().default('~/llama/llama.cpp/build/bin/'),
    coder: exports.ModelConfigSchema,
    reviewer: exports.ModelConfigSchema,
    runtime: exports.RuntimeConfigSchema,
    // Deprecated but kept for compatibility logic if needed
    workspaceRoot: z.string().optional(),
    logDir: z.string().optional()
});
class ConfigManager {
    constructor() { }
    static getInstance() {
        if (!ConfigManager.instance) {
            ConfigManager.instance = new ConfigManager();
        }
        return ConfigManager.instance;
    }
    /**
     * Load ONLY user-configured values (not defaults from package.json).
     * Returns partial config with only explicitly set values.
     * Used by Setup Wizard to show placeholders for unset fields.
     */
    async loadUserConfig() {
        const config = vscode.workspace.getConfiguration('gravitasCode');
        const userConfig = {};
        // Helper to get effective user-set value (Workspace > Global)
        const getUserValue = (key) => {
            const inspection = config.inspect(key);
            // Correct precedence: Folder > Workspace > Global
            return inspection?.workspaceFolderValue ?? inspection?.workspaceValue ?? inspection?.globalValue;
        };
        // Global settings
        const llamaBinPath = getUserValue('llamaBinPath');
        if (llamaBinPath !== undefined) {
            userConfig.llamaBinPath = llamaBinPath;
        }
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (workspaceRoot) {
            userConfig.workspaceRoot = workspaceRoot;
        }
        // Runtime settings
        const runtime = {};
        const autoTestOnStart = getUserValue('runtime.autoTestOnStart');
        const showLogs = getUserValue('runtime.showLogs');
        const logLevel = getUserValue('runtime.logLevel');
        const killSignal = getUserValue('runtime.killSignal');
        if (autoTestOnStart !== undefined)
            runtime.autoTestOnStart = autoTestOnStart;
        if (showLogs !== undefined)
            runtime.showLogs = showLogs;
        if (logLevel !== undefined)
            runtime.logLevel = logLevel;
        if (killSignal !== undefined)
            runtime.killSignal = killSignal;
        if (Object.keys(runtime).length > 0) {
            userConfig.runtime = runtime;
        }
        // Model configs
        const getModelUserConfig = (section) => {
            const model = {};
            const enabled = getUserValue(`${section}.general.enabled`);
            const mode = getUserValue(`${section}.general.mode`);
            const modelPath = getUserValue(`${section}.general.modelPath`);
            const host = getUserValue(`${section}.general.host`);
            const port = getUserValue(`${section}.general.port`);
            const noWarmup = getUserValue(`${section}.general.noWarmup`);
            if (enabled !== undefined)
                model.enabled = enabled;
            if (mode !== undefined)
                model.mode = mode;
            if (modelPath !== undefined)
                model.modelPath = modelPath;
            if (host !== undefined)
                model.host = host;
            if (port !== undefined)
                model.port = port;
            if (noWarmup !== undefined)
                model.noWarmup = noWarmup;
            const cudaVisibleDevices = getUserValue(`${section}.hardware.cudaVisibleDevices`);
            const nGpuLayers = getUserValue(`${section}.hardware.nGpuLayers`);
            const contextSize = getUserValue(`${section}.hardware.contextSize`);
            const threads = getUserValue(`${section}.hardware.threads`);
            const threadsBatch = getUserValue(`${section}.hardware.threadsBatch`);
            const batchSize = getUserValue(`${section}.hardware.batchSize`);
            const ubatchSize = getUserValue(`${section}.hardware.ubatchSize`);
            const numaInterleave = getUserValue(`${section}.hardware.numaInterleave`);
            const prefixCommand = getUserValue(`${section}.hardware.prefixCommand`);
            if (cudaVisibleDevices !== undefined)
                model.cudaVisibleDevices = cudaVisibleDevices;
            if (nGpuLayers !== undefined)
                model.nGpuLayers = nGpuLayers;
            if (contextSize !== undefined)
                model.contextSize = contextSize;
            if (threads !== undefined)
                model.threads = threads;
            if (threadsBatch !== undefined)
                model.threadsBatch = threadsBatch;
            if (batchSize !== undefined)
                model.batchSize = batchSize;
            if (ubatchSize !== undefined)
                model.ubatchSize = ubatchSize;
            if (numaInterleave !== undefined)
                model.numaInterleave = numaInterleave;
            if (prefixCommand !== undefined)
                model.prefixCommand = prefixCommand;
            const temperature = getUserValue(`${section}.sampling.temperature`);
            const topP = getUserValue(`${section}.sampling.topP`);
            const topK = getUserValue(`${section}.sampling.topK`);
            const repeatPenalty = getUserValue(`${section}.sampling.repeatPenalty`);
            if (temperature !== undefined)
                model.temperature = temperature;
            if (topP !== undefined)
                model.topP = topP;
            if (topK !== undefined)
                model.topK = topK;
            if (repeatPenalty !== undefined)
                model.repeatPenalty = repeatPenalty;
            return Object.keys(model).length > 0 ? model : undefined;
        };
        const coder = getModelUserConfig('coder');
        const reviewer = getModelUserConfig('reviewer');
        if (coder)
            userConfig.coder = coder;
        if (reviewer)
            userConfig.reviewer = reviewer;
        return userConfig;
    }
    async loadConfig() {
        const config = vscode.workspace.getConfiguration('gravitasCode');
        // All defaults come from package.json - no hardcoded fallbacks
        const llamaBinPath = config.get('llamaBinPath');
        const runtime = {
            autoTestOnStart: config.get('runtime.autoTestOnStart'),
            showLogs: config.get('runtime.showLogs'),
            logLevel: config.get('runtime.logLevel'),
            killSignal: config.get('runtime.killSignal')
        };
        // Helper to safely get model configs
        const getModelConfig = (section, defaultPort) => {
            return {
                enabled: config.get(`${section}.general.enabled`),
                mode: config.get(`${section}.general.mode`),
                modelPath: (0, pathUtils_1.resolveTilde)(config.get(`${section}.general.modelPath`)),
                host: config.get(`${section}.general.host`),
                port: config.get(`${section}.general.port`),
                noWarmup: config.get(`${section}.general.noWarmup`),
                cudaVisibleDevices: config.get(`${section}.hardware.cudaVisibleDevices`),
                nGpuLayers: config.get(`${section}.hardware.nGpuLayers`),
                contextSize: config.get(`${section}.hardware.contextSize`),
                threads: config.get(`${section}.hardware.threads`),
                threadsBatch: config.get(`${section}.hardware.threadsBatch`),
                batchSize: config.get(`${section}.hardware.batchSize`),
                ubatchSize: config.get(`${section}.hardware.ubatchSize`),
                numaInterleave: config.get(`${section}.hardware.numaInterleave`),
                prefixCommand: config.get(`${section}.hardware.prefixCommand`),
                temperature: config.get(`${section}.sampling.temperature`),
                topP: config.get(`${section}.sampling.topP`),
                topK: config.get(`${section}.sampling.topK`),
                repeatPenalty: config.get(`${section}.sampling.repeatPenalty`)
            };
        };
        return {
            llamaBinPath,
            workspaceRoot: vscode.workspace.workspaceFolders ? vscode.workspace.workspaceFolders[0].uri.fsPath : '',
            logDir: '',
            runtime,
            coder: getModelConfig('coder', 8010),
            reviewer: getModelConfig('reviewer', 8011)
        };
    }
    async saveConfig(val) {
        const config = vscode.workspace.getConfiguration('gravitasCode');
        // Helper to determine target: Write to Workspace if it's already defined there or overridden, otherwise Global
        // Ideally we write to Global by default, but if Workspace overrides it, we MUST write to Workspace to see effect.
        const getTarget = (key) => {
            const inspect = config.inspect(key);
            if (inspect?.workspaceFolderValue !== undefined)
                return vscode.ConfigurationTarget.WorkspaceFolder;
            if (inspect?.workspaceValue !== undefined)
                return vscode.ConfigurationTarget.Workspace;
            return vscode.ConfigurationTarget.Global;
        };
        // Helper update wrapper
        const update = async (key, value) => {
            await config.update(key, value, getTarget(key));
        };
        // 1. Global / Runtime
        await update('llamaBinPath', val.llamaBinPath);
        const runtime = val.runtime || {};
        // Only update runtime if it's defined in val (avoid wiping it if not sent)
        if (val.runtime) {
            await update('runtime.autoTestOnStart', runtime.autoTestOnStart);
            await update('runtime.showLogs', runtime.showLogs);
            await update('runtime.logLevel', runtime.logLevel);
            await update('runtime.killSignal', runtime.killSignal);
        }
        // Helper to update model configs
        const updateModelConfig = async (section, modelVal) => {
            if (!modelVal)
                return;
            // General
            await update(`${section}.general.enabled`, modelVal.enabled);
            await update(`${section}.general.mode`, modelVal.mode);
            await update(`${section}.general.modelPath`, modelVal.modelPath);
            await update(`${section}.general.host`, modelVal.host);
            await update(`${section}.general.port`, modelVal.port);
            await update(`${section}.general.noWarmup`, modelVal.noWarmup);
            // Hardware
            await update(`${section}.hardware.cudaVisibleDevices`, modelVal.cudaVisibleDevices);
            await update(`${section}.hardware.nGpuLayers`, modelVal.nGpuLayers);
            await update(`${section}.hardware.contextSize`, modelVal.contextSize);
            await update(`${section}.hardware.threads`, modelVal.threads);
            if (modelVal.threadsBatch !== undefined)
                await update(`${section}.hardware.threadsBatch`, modelVal.threadsBatch);
            await update(`${section}.hardware.batchSize`, modelVal.batchSize);
            await update(`${section}.hardware.ubatchSize`, modelVal.ubatchSize);
            if (modelVal.numaInterleave !== undefined)
                await update(`${section}.hardware.numaInterleave`, modelVal.numaInterleave);
            if (modelVal.prefixCommand !== undefined)
                await update(`${section}.hardware.prefixCommand`, modelVal.prefixCommand);
            // Sampling
            await update(`${section}.sampling.temperature`, modelVal.temperature);
            if (modelVal.topP !== undefined)
                await update(`${section}.sampling.topP`, modelVal.topP);
            if (modelVal.topK !== undefined)
                await update(`${section}.sampling.topK`, modelVal.topK);
            await update(`${section}.sampling.repeatPenalty`, modelVal.repeatPenalty);
        };
        // 2. Coder
        if (val.coder)
            await updateModelConfig('coder', val.coder);
        // 3. Reviewer
        if (val.reviewer)
            await updateModelConfig('reviewer', val.reviewer);
    }
}
exports.ConfigManager = ConfigManager;
//# sourceMappingURL=config.js.map