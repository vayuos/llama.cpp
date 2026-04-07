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
exports.RuntimeTreeProvider = void 0;
const vscode = __importStar(require("vscode"));
const processManager_1 = require("../process/processManager");
const config_1 = require("../core/config");
const telemetry_1 = require("../llm/telemetry");
class RuntimeTreeProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.config = null;
        this.refresh();
        // Poll status every 2.5 seconds for a "live" feel
        setInterval(() => this.refresh(), 2500);
    }
    refresh() {
        const mgr = config_1.ConfigManager.getInstance();
        // Prefer cached config for status checks if available
        const cached = mgr.getCachedConfig();
        if (cached) {
            this.config = cached;
            this._onDidChangeTreeData.fire();
        }
        else {
            mgr.loadConfig().then(c => {
                this.config = c;
                this._onDidChangeTreeData.fire();
            });
        }
    }
    getTreeItem(element) {
        return element;
    }
    async getChildren(element) {
        if (!element) {
            return [
                new RuntimeItem('Coder Model', 'coder', vscode.TreeItemCollapsibleState.Expanded),
                new RuntimeItem('Reviewer Model', 'reviewer', vscode.TreeItemCollapsibleState.Expanded)
            ];
        }
        else {
            const pm = processManager_1.UnifiedProcessManager.getInstance();
            const type = element.modelType;
            if (!type || !this.config)
                return [];
            const status = await pm.getLiveStatus(type, this.config);
            const pid = status.pid ? status.pid.toString() : 'N/A';
            const isRunning = status.running;
            const statusLabel = isRunning ? (status.external ? 'Online (Remote)' : 'Online (Local)') : 'Offline';
            const telemetry = telemetry_1.TelemetryService.getInstance().getTelemetry(type);
            return [
                new RuntimeItem(`Status: ${statusLabel}`, undefined, vscode.TreeItemCollapsibleState.None, isRunning ? 'running' : 'stopped'),
                new RuntimeItem(`PID: ${pid}`, undefined, vscode.TreeItemCollapsibleState.None, 'pid'),
                new RuntimeItem(`VRAM: ${telemetry.vram}`, undefined, vscode.TreeItemCollapsibleState.None, 'metrics'),
                new RuntimeItem(`Perf: ${telemetry.tps}`, undefined, vscode.TreeItemCollapsibleState.None, 'metrics'),
                new RuntimeItem(`Usage: ${telemetry.slots}`, undefined, vscode.TreeItemCollapsibleState.None, 'metrics'),
                new RuntimeItem(`Ping: ${telemetry.latency}`, undefined, vscode.TreeItemCollapsibleState.None, 'metrics')
            ];
        }
    }
    getStatus(type) {
        return processManager_1.UnifiedProcessManager.getInstance().getProcessStatus(type);
    }
}
exports.RuntimeTreeProvider = RuntimeTreeProvider;
class RuntimeItem extends vscode.TreeItem {
    constructor(label, modelType, collapsibleState = vscode.TreeItemCollapsibleState.None, contextValue) {
        super(label, collapsibleState);
        this.label = label;
        this.modelType = modelType;
        this.collapsibleState = collapsibleState;
        this.contextValue = contextValue;
        if (modelType) {
            this.iconPath = new vscode.ThemeIcon('server');
        }
        else if (contextValue === 'running') {
            this.iconPath = new vscode.ThemeIcon('pass', new vscode.ThemeColor('testing.iconPassed'));
        }
        else if (contextValue === 'stopped') {
            this.iconPath = new vscode.ThemeIcon('circle-slash', new vscode.ThemeColor('testing.iconFailed'));
        }
        else if (contextValue === 'metrics') {
            this.iconPath = new vscode.ThemeIcon('pulse');
        }
        else {
            this.iconPath = new vscode.ThemeIcon('symbol-property');
        }
    }
}
//# sourceMappingURL=runtimeTreeProvider.js.map