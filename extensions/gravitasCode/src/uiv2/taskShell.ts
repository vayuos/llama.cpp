import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { TaskManager } from './taskManager';
import { Task } from './types';

export class TaskShellPanel {
    public static currentPanel: TaskShellPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];
    private _currentTaskId: string | undefined;

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri, taskId: string) {
        this._panel = panel;
        this._extensionUri = extensionUri;
        this._currentTaskId = taskId;

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
        this._panel.webview.html = this._getHtmlForWebview(this._panel.webview);

        this._setWebviewMessageListener(this._panel.webview);

        // 1. Task State Stream: Synchronizes full task snapshots when the reducer fires
        TaskManager.getInstance().onDidTaskUpdate((task: Task) => {
            if (task.id === this._currentTaskId) {
                this._panel.webview.postMessage({ type: 'updateTask', task });
            }
        });

        // 2. Incremental Event Stream: Live sync of telemetry (Thoughts, Tools, Artifacts)
        TaskManager.getInstance().onDidEmitEvent(({ taskId, event }: { taskId: string; event: any }) => {
            if (taskId === this._currentTaskId || !this._currentTaskId) {
                this._panel.webview.postMessage({ type: 'event', taskId, event });
            }
        });
    }

    public static createOrShow(extensionUri: vscode.Uri, taskId: string) {
        const column = vscode.window.activeTextEditor ? vscode.window.activeTextEditor.viewColumn : undefined;

        // If we already have a panel, show it if it matches the task?
        // Spec says "Every user command creates exactly one Task Shell".
        // So we likely want a new panel or replace the content.
        // For strict "Shell" behavior, we might want one main window that REPLACES content.
        // But for now, let's assume we spawn a new panel.

        const panel = vscode.window.createWebviewPanel(
            'gravitasTaskShell',
            'Task Shell',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')]
            }
        );

        TaskShellPanel.currentPanel = new TaskShellPanel(panel, extensionUri, taskId);

        // Initial Load - Full Snapshot derived from Ledger (Gap 5)
        const tm = TaskManager.getInstance();
        const task = tm.getTask(taskId);
        if (task) {
            panel.webview.postMessage({ type: 'loadSnapshot', taskId, task });
        }
    }

    private _setWebviewMessageListener(webview: vscode.Webview) {
        webview.onDidReceiveMessage(
            async (message) => {
                switch (message.command) {
                    case 'spawnTask':
                        // Create NEW task from Footer input or Replay control
                        const pId = message.parentId;
                        const rType = message.lineage; // 'REPLAY' etc
                        const newTask = TaskManager.getInstance().createTask(message.text, 'user', pId, rType);

                        // Dispose current and open new? Or navigate?
                        // "Submitting creates a NEW Task Shell"
                        this.dispose();
                        TaskShellPanel.createOrShow(this._extensionUri, newTask.id);
                        return;

                    case 'openArtifact':
                        if (message.path) {
                            const uri = vscode.Uri.file(message.path);
                            // Safety check? User rule: "Opening artifacts must respect VS Code safety policies"
                            // vscode.commands.executeCommand('vscode.open', uri);
                            // Check strict safety policies if needed, but for now standard open.
                            vscode.window.showTextDocument(uri, { preview: true });
                        }
                        return;

                    case 'revealArtifact':
                        if (message.path) {
                            const uri = vscode.Uri.file(message.path);
                            vscode.commands.executeCommand('revealInExplorer', uri);
                        }
                        return;

                    case 'abortTask':
                        TaskManager.getInstance().abortTask(this._currentTaskId!);
                        return;

                    case 'recordVisibilityOverride':
                        TaskManager.getInstance().emitEvent(this._currentTaskId!, {
                            type: 'UserOverrideVisibility',
                            elementId: message.elementId,
                            action: message.action,
                            actor: 'user'
                        } as any);
                        return;
                }
            },
            undefined,
            this._disposables
        );
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'taskShell.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'taskShell.css'));
        const toolkitUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'toolkit.js'));

        const htmlPath = path.join(this._extensionUri.fsPath, 'media', 'taskShell.html');
        let html = fs.readFileSync(htmlPath, 'utf-8');

        const nonce = getNonce();

        // Replace placeholders
        html = html.replace('${styleUri}', styleUri.toString());
        html = html.replace('${scriptUri}', scriptUri.toString());
        html = html.replace('${toolkitUri}', toolkitUri.toString());
        html = html.replace('${nonce}', nonce);
        html = html.replace('${webview.cspSource}', webview.cspSource);
        
        // Re-inject the toolkit and script with nonce
        html = html.replace('</body>', `    <script nonce="${nonce}" type="module" src="${toolkitUri}"></script>\n    <script nonce="${nonce}" src="${scriptUri}"></script>\n</body>`);

        return html;
    }

    public dispose() {
        TaskShellPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) { x.dispose(); }
        }
    }
}

function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
