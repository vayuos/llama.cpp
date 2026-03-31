import * as vscode from 'vscode';
import { UnifiedProcessManager } from '../process/processManager';
import { GravitasConfig, ConfigManager } from '../core/config';
import { TelemetryService } from '../llm/telemetry';

export class RuntimeTreeProvider implements vscode.TreeDataProvider<RuntimeItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<RuntimeItem | undefined | null | void> = new vscode.EventEmitter<RuntimeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<RuntimeItem | undefined | null | void> = this._onDidChangeTreeData.event;

    private config: GravitasConfig | null = null;

    constructor() {
        this.refresh();
        // Poll status every 2.5 seconds for a "live" feel
        setInterval(() => this.refresh(), 2500);
    }

    refresh(): void {
        const mgr = ConfigManager.getInstance();
        // Prefer cached config for status checks if available
        const cached = mgr.getCachedConfig();
        if (cached) {
            this.config = cached;
            this._onDidChangeTreeData.fire();
        } else {
            mgr.loadConfig().then(c => {
                this.config = c;
                this._onDidChangeTreeData.fire();
            });
        }
    }

    getTreeItem(element: RuntimeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: RuntimeItem): Promise<RuntimeItem[]> {
        if (!element) {
            return [
                new RuntimeItem('Coder Model', 'coder', vscode.TreeItemCollapsibleState.Expanded),
                new RuntimeItem('Reviewer Model', 'reviewer', vscode.TreeItemCollapsibleState.Expanded)
            ];
        } else {
            const pm = UnifiedProcessManager.getInstance();
            const type = element.modelType;
            if (!type || !this.config) return [];

            const status = await pm.getLiveStatus(type, this.config);
            const pid = status.pid ? status.pid.toString() : 'N/A';
            const isRunning = status.running;

            const statusLabel = isRunning ? (status.external ? 'Online (Remote)' : 'Online (Local)') : 'Offline';

            const telemetry = TelemetryService.getInstance().getTelemetry(type);

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

    private getStatus(type: 'coder' | 'reviewer'): { pid?: number; telemetry?: string } {
        return UnifiedProcessManager.getInstance().getProcessStatus(type);
    }
}

class RuntimeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly modelType?: 'coder' | 'reviewer',
        public readonly collapsibleState: vscode.TreeItemCollapsibleState = vscode.TreeItemCollapsibleState.None,
        public readonly contextValue?: string
    ) {
        super(label, collapsibleState);
        if (modelType) {
            this.iconPath = new vscode.ThemeIcon('server');
        } else if (contextValue === 'running') {
            this.iconPath = new vscode.ThemeIcon('pass', new vscode.ThemeColor('testing.iconPassed'));
        } else if (contextValue === 'stopped') {
            this.iconPath = new vscode.ThemeIcon('circle-slash', new vscode.ThemeColor('testing.iconFailed'));
        } else if (contextValue === 'metrics') {
            this.iconPath = new vscode.ThemeIcon('pulse');
        } else {
            this.iconPath = new vscode.ThemeIcon('symbol-property');
        }
    }
}
