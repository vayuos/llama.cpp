import * as vscode from 'vscode';
import { GravitasPanel } from './gravitasPanel';
import { ProcessManager } from '../runtime/processManager';

export class PanelProvider implements vscode.WebviewViewProvider {
    private readonly _processManager = new ProcessManager();

    constructor(private readonly _extensionUri: vscode.Uri) { }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        const panel = new GravitasPanel(this._extensionUri);
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = panel.getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(data => {
            switch (data.command) {
                case 'runPipeline':
                    vscode.commands.executeCommand('gravitas.pipeline.run', data.prompt);
                    break;
                case 'startAll':
                    this._processManager.startAll();
                    break;
                case 'stopAll':
                    this._processManager.stopAll();
                    break;
                case 'restartAll':
                    this._processManager.restartAll();
                    break;
                case 'pollStatus':
                    webviewView.webview.postMessage({
                        command: 'updateStatus',
                        status: this._processManager.getStatus()
                    });
                    break;
                case 'openSettings':
                    vscode.commands.executeCommand('workbench.action.openSettings', 'gravitas');
                    break;
            }
        });
    }
}
