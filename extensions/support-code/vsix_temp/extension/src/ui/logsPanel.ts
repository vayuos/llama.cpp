import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { CentralLogger, LogEntry } from '../core/logger';

export class LogsPanel {
    public static currentPanel: LogsPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private _disposables: vscode.Disposable[] = [];

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
        this._panel = panel;
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
        this._panel.webview.html = this._getHtmlForWebview(this._panel.webview, extensionUri);

        // Listen to logs
        CentralLogger.getInstance().onDidLog(entry => {
            this._panel.webview.postMessage({ command: 'addLog', entry });
        }, null, this._disposables);
    }

    public static createOrShow(extensionUri: vscode.Uri) {
        if (LogsPanel.currentPanel) {
            LogsPanel.currentPanel._panel.reveal(vscode.ViewColumn.Two);
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'gravitasLogs',
            'Gravitas Logs',
            vscode.ViewColumn.Two,
            { enableScripts: true, localResourceRoots: [extensionUri] }
        );

        LogsPanel.currentPanel = new LogsPanel(panel, extensionUri);
    }

    private _getHtmlForWebview(webview: vscode.Webview, extensionUri: vscode.Uri) {
        const htmlPath = path.join(extensionUri.fsPath, 'media', 'logs.html');
        return fs.readFileSync(htmlPath, 'utf-8');
    }

    public dispose() {
        LogsPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) { x.dispose(); }
        }
    }
}
