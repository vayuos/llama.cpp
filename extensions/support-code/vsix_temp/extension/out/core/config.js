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
exports.ConfigManager = exports.GravitasConfigSchema = exports.ModelConfigSchema = void 0;
const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const zod_1 = require("zod");
exports.ModelConfigSchema = zod_1.z.object({
    modelPath: zod_1.z.string(),
    port: zod_1.z.number(),
    gpuLayers: zod_1.z.number().optional(),
    threads: zod_1.z.number().optional(),
    ctx: zod_1.z.number(),
    batch: zod_1.z.number().optional(),
    ubatch: zod_1.z.number().optional(),
    temp: zod_1.z.number(),
    topP: zod_1.z.number().optional()
});
exports.GravitasConfigSchema = zod_1.z.object({
    llamaBinaryPath: zod_1.z.string(),
    coderModel: exports.ModelConfigSchema,
    reviewerModel: exports.ModelConfigSchema,
    workspaceRoot: zod_1.z.string(),
    logDir: zod_1.z.string()
});
class ConfigManager {
    constructor() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        const root = workspaceFolders ? workspaceFolders[0].uri.fsPath : '';
        this.configPath = path.join(root, '.gravitas', 'config.json');
    }
    static getInstance() {
        if (!ConfigManager.instance) {
            ConfigManager.instance = new ConfigManager();
        }
        return ConfigManager.instance;
    }
    async loadConfig() {
        if (!fs.existsSync(this.configPath)) {
            return null;
        }
        try {
            const content = fs.readFileSync(this.configPath, 'utf-8');
            const data = JSON.parse(content);
            return exports.GravitasConfigSchema.parse(data);
        }
        catch (e) {
            console.error('Failed to load/parse config:', e);
            return null;
        }
    }
    async saveConfig(config) {
        const dir = path.dirname(this.configPath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(this.configPath, JSON.stringify(config, null, 2), 'utf-8');
    }
    getConfigPath() {
        return this.configPath;
    }
}
exports.ConfigManager = ConfigManager;
//# sourceMappingURL=config.js.map