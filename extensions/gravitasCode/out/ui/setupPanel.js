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
exports.SetupPanel = void 0;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const config_1 = require("../core/config");
const storageManager_1 = require("../core/storageManager");
class SetupPanel {
    constructor(panel, extensionUri) {
        this._disposables = [];
        this._panel = panel;
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
        this._panel.webview.html = this._getHtmlForWebview(this._panel.webview, extensionUri);
        this._setWebviewMessageListener(this._panel.webview);
    }
    static async createOrShow(extensionUri) {
        const column = vscode.window.activeTextEditor ? vscode.window.activeTextEditor.viewColumn : undefined;
        if (SetupPanel.currentPanel) {
            SetupPanel.currentPanel._panel.reveal(column);
            return;
        }
        const panel = vscode.window.createWebviewPanel('gravitasSetup', 'Gravitas Setup Wizard', column || vscode.ViewColumn.One, { enableScripts: true, localResourceRoots: [extensionUri] });
        SetupPanel.currentPanel = new SetupPanel(panel, extensionUri);
        // Load ONLY user-configured values (not defaults) to show placeholders
        const userConfig = await config_1.ConfigManager.getInstance().loadUserConfig();
        SetupPanel.currentPanel._panel.webview.postMessage({ command: 'loadConfig', config: userConfig });
    }
    _setWebviewMessageListener(webview) {
        webview.onDidReceiveMessage(async (message) => {
            switch (message.command) {
                case 'saveConfig':
                    const config = message.config;
                    // Inject workspace root
                    const workspaceFolder = vscode.workspace.workspaceFolders?.[0].uri.fsPath || '';
                    config.workspaceRoot = workspaceFolder;
                    await config_1.ConfigManager.getInstance().saveConfig(config);
                    vscode.window.showInformationMessage('Gravitas: Configuration saved.');
                    // Trigger validation (which starts servers and keeps them running)
                    await vscode.commands.executeCommand('gravitas.setup.validate');
                    this._panel.dispose();
                    return;
                case 'clearStorage':
                    const choice = await vscode.window.showWarningMessage('Clear Gravitas Storage?', { modal: true }, 'Clear Logs Only', 'Clear Settings (Reset to Defaults)', 'Clear Everything (Logs + Settings + Validation)');
                    if (!choice)
                        return;
                    const storageManager = storageManager_1.StorageManager.getInstance();
                    const configManager = config_1.ConfigManager.getInstance();
                    let result;
                    if (choice === 'Clear Logs Only') {
                        result = storageManager.clearLogs();
                    }
                    else if (choice === 'Clear Settings (Reset to Defaults)') {
                        // Clear all VS Code settings for gravitasCode
                        const config = vscode.workspace.getConfiguration('gravitasCode');
                        const allKeys = [
                            'llamaBinPath',
                            'runtime.autoTestOnStart', 'runtime.showLogs', 'runtime.logLevel', 'runtime.killSignal',
                            'coder.general.enabled', 'coder.general.mode', 'coder.general.modelPath', 'coder.general.host', 'coder.general.port', 'coder.general.noWarmup',
                            'coder.hardware.cudaVisibleDevices', 'coder.hardware.nGpuLayers', 'coder.hardware.contextSize', 'coder.hardware.threads', 'coder.hardware.threadsBatch', 'coder.hardware.batchSize', 'coder.hardware.ubatchSize', 'coder.hardware.numaInterleave', 'coder.hardware.prefixCommand',
                            'coder.sampling.temperature', 'coder.sampling.topP', 'coder.sampling.topK', 'coder.sampling.repeatPenalty',
                            'reviewer.general.enabled', 'reviewer.general.mode', 'reviewer.general.modelPath', 'reviewer.general.host', 'reviewer.general.port', 'reviewer.general.noWarmup',
                            'reviewer.hardware.cudaVisibleDevices', 'reviewer.hardware.nGpuLayers', 'reviewer.hardware.contextSize', 'reviewer.hardware.threads', 'reviewer.hardware.threadsBatch', 'reviewer.hardware.batchSize', 'reviewer.hardware.ubatchSize', 'reviewer.hardware.numaInterleave', 'reviewer.hardware.prefixCommand',
                            'reviewer.sampling.temperature', 'reviewer.sampling.topP', 'reviewer.sampling.topK', 'reviewer.sampling.repeatPenalty'
                        ];
                        for (const key of allKeys) {
                            await config.update(key, undefined, vscode.ConfigurationTarget.Global);
                        }
                        result = { success: true, message: 'All settings cleared. Extension reset to defaults.' };
                        // Reload the wizard with empty fields
                        const userConfig = await configManager.loadUserConfig();
                        this._panel.webview.postMessage({ command: 'loadConfig', config: userConfig });
                    }
                    else {
                        // Clear everything
                        result = storageManager.clearAll();
                        // Also clear settings
                        const config = vscode.workspace.getConfiguration('gravitasCode');
                        const allKeys = [
                            'llamaBinPath',
                            'runtime.autoTestOnStart', 'runtime.showLogs', 'runtime.logLevel', 'runtime.killSignal',
                            'coder.general.enabled', 'coder.general.mode', 'coder.general.modelPath', 'coder.general.host', 'coder.general.port', 'coder.general.noWarmup',
                            'coder.hardware.cudaVisibleDevices', 'coder.hardware.nGpuLayers', 'coder.hardware.contextSize', 'coder.hardware.threads', 'coder.hardware.threadsBatch', 'coder.hardware.batchSize', 'coder.hardware.ubatchSize', 'coder.hardware.numaInterleave', 'coder.hardware.prefixCommand',
                            'coder.sampling.temperature', 'coder.sampling.topP', 'coder.sampling.topK', 'coder.sampling.repeatPenalty',
                            'reviewer.general.enabled', 'reviewer.general.mode', 'reviewer.general.modelPath', 'reviewer.general.host', 'reviewer.general.port', 'reviewer.general.noWarmup',
                            'reviewer.hardware.cudaVisibleDevices', 'reviewer.hardware.nGpuLayers', 'reviewer.hardware.contextSize', 'reviewer.hardware.threads', 'reviewer.hardware.threadsBatch', 'reviewer.hardware.batchSize', 'reviewer.hardware.ubatchSize', 'reviewer.hardware.numaInterleave', 'reviewer.hardware.prefixCommand',
                            'reviewer.sampling.temperature', 'reviewer.sampling.topP', 'reviewer.sampling.topK', 'reviewer.sampling.repeatPenalty'
                        ];
                        for (const key of allKeys) {
                            await config.update(key, undefined, vscode.ConfigurationTarget.Global);
                        }
                        // Reload the wizard with empty fields
                        const userConfig = await configManager.loadUserConfig();
                        this._panel.webview.postMessage({ command: 'loadConfig', config: userConfig });
                    }
                    if (result.success) {
                        vscode.window.showInformationMessage(result.message);
                    }
                    else {
                        vscode.window.showErrorMessage(result.message);
                    }
                    return;
            }
        }, undefined, this._disposables);
    }
    _getHtmlForWebview(webview, extensionUri) {
        const htmlPath = path.join(extensionUri.fsPath, 'media', 'setup.html');
        let html = fs.readFileSync(htmlPath, 'utf-8');
        // Resource path conversion if needed (not strictly needed for simple HTML, but good practice)
        return html;
    }
    dispose() {
        SetupPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) {
                x.dispose();
            }
        }
    }
}
exports.SetupPanel = SetupPanel;
//# sourceMappingURL=setupPanel.js.map