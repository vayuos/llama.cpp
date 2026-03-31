import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { ConfigManager, GravitasConfig } from '../core/config';
import { GravitasState } from '../core/state';

export class SetupPanel {
    public static currentPanel: SetupPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private _disposables: vscode.Disposable[] = [];

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
        this._panel = panel;
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
        this._panel.webview.html = this._getHtmlForWebview(this._panel.webview, extensionUri);
        this._setWebviewMessageListener(this._panel.webview);
    }

    public static createOrShow(extensionUri: vscode.Uri) {
        const column = vscode.window.activeTextEditor ? vscode.window.activeTextEditor.viewColumn : undefined;

        if (SetupPanel.currentPanel) {
            SetupPanel.currentPanel._panel.reveal(column);
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'gravitasSetup',
            'Gravitas Setup Wizard',
            column || vscode.ViewColumn.One,
            { enableScripts: true, localResourceRoots: [extensionUri] }
        );

        SetupPanel.currentPanel = new SetupPanel(panel, extensionUri);
    }

    private _setWebviewMessageListener(webview: vscode.Webview) {
        webview.onDidReceiveMessage(
            async (message) => {
                switch (message.command) {
                    case 'saveConfig':
                        const config: GravitasConfig = message.config;
                        // Inject workspace root
                        const workspaceFolder = vscode.workspace.workspaceFolders?.[0].uri.fsPath || '';
                        config.workspaceRoot = workspaceFolder;

                        await ConfigManager.getInstance().saveConfig(config);
                        vscode.window.showInformationMessage('Gravitas: Configuration saved.');

                        // Trigger validation immediately
                        vscode.commands.executeCommand('gravitas.setup.validate');
                        this._panel.dispose();
                        return;
                }
            },
            undefined,
            this._disposables
        );
    }

    private _getHtmlForWebview(webview: vscode.Webview, extensionUri: vscode.Uri) {
        const htmlPath = path.join(extensionUri.fsPath, 'media', 'setup.html');
        let html = fs.readFileSync(htmlPath, 'utf-8');

        // Resource path conversion if needed (not strictly needed for simple HTML, but good practice)
        return html;
    }

    public dispose() {
        SetupPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) { x.dispose(); }
        }
    }
}
