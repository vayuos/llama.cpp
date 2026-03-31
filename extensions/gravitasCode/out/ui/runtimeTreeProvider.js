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
class RuntimeTreeProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this.config = null;
        this.refresh();
        // Poll status every 2 seconds
        setInterval(() => this.refresh(), 2000);
    }
    refresh() {
        config_1.ConfigManager.getInstance().loadConfig().then(c => {
            this.config = c;
            this._onDidChangeTreeData.fire();
        });
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (!element) {
            // Root items: Coder and Reviewer
            return Promise.resolve([
                new RuntimeItem('Coder Model', 'coder', vscode.TreeItemCollapsibleState.Expanded),
                new RuntimeItem('Reviewer Model', 'reviewer', vscode.TreeItemCollapsibleState.Expanded)
            ]);
        }
        else {
            // Children: Status details
            const pm = processManager_1.UnifiedProcessManager.getInstance();
            const type = element.modelType;
            if (!type || !this.config)
                return Promise.resolve([]);
            // We need a way to check if running. For now, we'll implement a basic check.
            // Since LlamaProcess doesn't expose public PID/State cleanly yet, we might need to update it.
            // For this iteration, we will assume UnifiedProcessManager can give us basic status text or we infer it.
            // Let's add a `getStatus(type)` method to UnifiedProcessManager ideally, but for now we'll stub it or use pings if we had them.
            // A better way for V1 is strict config read + generic status.
            // Since we can't easily get live PID without modifying LlamaProcess, we will list config details for now 
            // and assume "Unknown" status until we wire up state tracking deeper.
            // Actually, let's extend this later. For now simple properties.
            const cfg = type === 'coder' ? this.config.coder : this.config.reviewer;
            const status = this.getStatus(type);
            const pid = status.pid ? status.pid.toString() : 'N/A';
            const telemetry = status.telemetry || 'Idle';
            const isRunning = !!status.pid;
            // Simple status string logic
            const statusLabel = isRunning ? 'Running' : 'Stopped';
            return Promise.resolve([
                new RuntimeItem(`Status: ${statusLabel}`, undefined, vscode.TreeItemCollapsibleState.None, isRunning ? 'running' : 'stopped'),
                new RuntimeItem(`Port: ${cfg.port}`, undefined, vscode.TreeItemCollapsibleState.None, 'port'),
                new RuntimeItem(`PID: ${pid}`, undefined, vscode.TreeItemCollapsibleState.None, 'pid'),
                new RuntimeItem(`Metrics: ${telemetry}`, undefined, vscode.TreeItemCollapsibleState.None, 'metrics'),
                new RuntimeItem(`Mode: ${cfg.mode.toUpperCase()}`, undefined, vscode.TreeItemCollapsibleState.None, 'mode')
            ]);
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