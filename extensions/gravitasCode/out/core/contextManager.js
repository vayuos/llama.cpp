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
exports.ContextManager = void 0;
const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
class ContextManager {
    constructor() {
        this.excludedPaths = new Set();
        // Default exclusions
        this.defaultExclusions = [
            'node_modules',
            '.git',
            'dist',
            'out',
            '.vscode',
            'package-lock.json',
            'bun.lockb',
            'yarn.lock'
        ];
        this.excludedPaths = new Set(this.defaultExclusions);
    }
    static getInstance() {
        if (!ContextManager.instance) {
            ContextManager.instance = new ContextManager();
        }
        return ContextManager.instance;
    }
    isExcluded(filePath) {
        const relativePath = vscode.workspace.asRelativePath(filePath);
        const segments = relativePath.split(path.sep);
        // Check if any segment of the path is in the exclusion set
        return segments.some(segment => this.excludedPaths.has(segment));
    }
    toggleExclusion(filePath) {
        const relativePath = vscode.workspace.asRelativePath(filePath);
        // If it's already directly excluded, remove it
        if (this.excludedPaths.has(relativePath)) {
            this.excludedPaths.delete(relativePath);
            return;
        }
        // If it's not excluded, exclude it
        this.excludedPaths.add(relativePath);
    }
    getTokenEstimate(filePath) {
        try {
            const stats = fs.statSync(filePath);
            if (stats.isDirectory())
                return 0;
            // Simple heuristic: 4 characters ~= 1 token
            // This is a rough estimation for UI purposes
            return Math.ceil(stats.size / 4);
        }
        catch (e) {
            return 0;
        }
    }
    formatTokenCount(count) {
        if (count >= 1000000)
            return `${(count / 1000000).toFixed(1)}M`;
        if (count >= 1000)
            return `${(count / 1000).toFixed(1)}k`;
        return count.toString();
    }
    async getContextString() {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders)
            return "";
        let context = "";
        for (const folder of workspaceFolders) {
            context += await this.walkDirectory(folder.uri.fsPath);
        }
        return context;
    }
    async walkDirectory(dir) {
        let results = "";
        try {
            const list = fs.readdirSync(dir, { withFileTypes: true });
            for (const dirent of list) {
                const fullPath = path.join(dir, dirent.name);
                if (this.isExcluded(fullPath))
                    continue;
                if (dirent.isDirectory()) {
                    results += await this.walkDirectory(fullPath);
                }
                else {
                    // Collect file content
                    try {
                        const content = fs.readFileSync(fullPath, 'utf8');
                        const relativePath = vscode.workspace.asRelativePath(fullPath);
                        results += `\n--- FILE: ${relativePath} ---\n${content}\n`;
                    }
                    catch (e) {
                        // Skip binary or unreadable files
                    }
                }
            }
        }
        catch (e) {
            // Ignore directory read errors
        }
        return results;
    }
}
exports.ContextManager = ContextManager;
//# sourceMappingURL=contextManager.js.map