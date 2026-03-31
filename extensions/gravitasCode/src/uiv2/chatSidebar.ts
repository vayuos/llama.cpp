import * as vscode from 'vscode';
import { TaskManager } from './taskManager';
import { CentralLogger } from '../core/logger';
import { TelemetryService } from '../llm/telemetry';

/**
 * High-fidelity Sidebar Chat Provider (Gravitas chat Infrastructure).
 * Implements a robust Init-Ready handshake and event buffering.
 */
export class ChatSidebarProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'gravitas.chat';
    private _view?: vscode.WebviewView;
    private _disposables: vscode.Disposable[] = [];
    
    // Handshake & Buffering State
    private _isReady = false;
    private _messageBuffer: any[] = [];
    private _watchdogTimer?: NodeJS.Timeout;
    private _isSubscribed = false;

    constructor(private readonly _extensionUri: vscode.Uri) {
        this.subscribeToEvents();
    }

    private subscribeToEvents() {
        if (this._isSubscribed) return;
        
        TaskManager.getInstance().onDidEmitEvent(({ taskId, event }) => {
            this.postOrBuffer({ type: 'event', taskId, event });
        });

        TaskManager.getInstance().onDidTaskUpdate((task) => {
            this.postOrBuffer({ type: 'updateTask', task });
        });

        TelemetryService.getInstance().onDidUpdate(() => {
            const coder = TelemetryService.getInstance().getTelemetry('coder');
            const reviewer = TelemetryService.getInstance().getTelemetry('reviewer');
            this.postOrBuffer({ type: 'telemetry', coder, reviewer });
        });
        
        this._isSubscribed = true;
    }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;
        this._isReady = false;
        // DO NOT CLEAR _messageBuffer here - preserve commands like 'reset'

        webviewView.webview.options = { 
            enableScripts: true, 
            localResourceRoots: [this._extensionUri] 
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        this._setWebviewMessageListener(webviewView.webview);

        // 🛡️ Handshake Watchdog
        this._watchdogTimer = setTimeout(() => {
            if (!this._isReady) {
                CentralLogger.getInstance().info('system', 'Gravitas Chat: Handshake timeout. Sidebar may be unresponsive or blocked by CSP.');
            }
        }, 5000);

        // 🛡️ Lifecycle Cleanup
        webviewView.onDidDispose(() => {
            this._view = undefined;
            this._isReady = false;
            if (this._watchdogTimer) clearTimeout(this._watchdogTimer);
            // We don't dispose the global TaskManager listeners here as they are part of the provider instance
        }, null, this._disposables);
    }

    private postOrBuffer(message: any) {
        if (this._isReady && this._view) {
            this._view.webview.postMessage(message);
        } else {
            this._messageBuffer.push(message);
        }
    }

    private clearDisposables() {
        this._disposables.forEach(d => d.dispose());
        this._disposables = [];
    }

    public showTask(taskId: string) {
        let task;
        if (taskId === 'last') {
            task = TaskManager.getInstance().getLastTask();
        } else {
            task = TaskManager.getInstance().getTask(taskId);
        }

        this._view?.show?.(true); // Bring sidebar into focus
        if (task) {
            this.postOrBuffer({ type: 'loadSnapshot', task });
        }
    }

    public reset() {
        this.postOrBuffer({ type: 'reset' });
    }

    public focus() {
        this.postOrBuffer({ type: 'focus' });
    }

    private _setWebviewMessageListener(webview: vscode.Webview) {
        webview.onDidReceiveMessage(async (message) => {
            switch (message.type) {
                case 'ready': {
                    CentralLogger.getInstance().info('system', 'Gravitas Chat: Handshake received from frontend.');
                    this._isReady = true;
                    if (this._watchdogTimer) clearTimeout(this._watchdogTimer);
                    
                    // 1. Flush Buffer FIRST (Prioritizes any 'reset' or state-changing commands)
                    if (this._messageBuffer.length > 0) {
                        CentralLogger.getInstance().info('system', `Gravitas Chat: Flushing ${this._messageBuffer.length} buffered events.`);
                        this._messageBuffer.forEach(msg => webview.postMessage(msg));
                        this._messageBuffer = [];
                    }

                    // 2. Initial State Sync
                    const lastTask = TaskManager.getInstance().getLastTask();
                    if (lastTask) {
                        webview.postMessage({ type: 'loadSnapshot', task: lastTask });
                    }
                    break;
                }
                case 'submitPrompt': {
                    await vscode.commands.executeCommand('gravitas.task.spawn', message.text);
                    break;
                }
                case 'abortTask': {
                    if (message.taskId) {
                        TaskManager.getInstance().abortTask(message.taskId);
                    }
                    break;
                }
                case 'error': {
                    CentralLogger.getInstance().error('system', `Gravitas Chat Frontend Error: ${message.message}\n${message.stack}`);
                    break;
                }
            }
        });
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'taskShell.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'taskShell.css'));
        const toolkitUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'toolkit.js'));

        const nonce = getNonce();

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src ${webview.cspSource} 'nonce-${nonce}';">
    <title>Gravitas chat</title>
    <link href="${styleUri}" rel="stylesheet">
    <script nonce="${nonce}" src="${toolkitUri}"></script>
    <style>
        body { padding: 0; display: flex; flex-direction: column; height: 100vh; background: #000; color: #fff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        #taskFeed { flex: 1; padding: 12px; gap: 24px; overflow-y: auto; }
        .task-footer { padding: 16px; background: rgba(10, 10, 12, 0.95); border-top: 1px solid rgba(255,255,255,0.05); }
        .welcome-screen { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80%; text-align: center; color: var(--text-secondary); padding: 40px; }
        .welcome-logo { font-size: 32px; margin-bottom: 24px; filter: drop-shadow(0 0 10px var(--accent-primary)); }
        .welcome-title { font-size: 16px; font-weight: 700; color: #fff; margin-bottom: 8px; }
        .welcome-hint { font-size: 11px; opacity: 0.6; }
    </style>
</head>
<body class="sidebar-mode">
    <div class="telemetry-wrapper">
        <div class="dashboard-section" id="coderDash">
            <div class="section-header">
                <div id="coderStatus" class="status-dot"></div>
                <span class="section-label">CODER.ACE</span>
                <span id="coderLoad" class="load-tag">IDLE</span>
            </div>
            <div class="metrics-grid">
                <div class="metric-item"><span class="label">VRAM</span><span id="coderVram" class="value">0%</span></div>
                <div class="metric-item highlight"><span class="label">GEN</span><span id="coderTps" class="value">0.0</span></div>
                <div class="metric-item"><span class="label">KV</span><span id="coderKv" class="value">0%</span></div>
            </div>
        </div>
        <div class="dashboard-section" id="reviewerDash">
            <div class="section-header">
                <div id="reviewerStatus" class="status-dot"></div>
                <span class="section-label">REVIEWER.ACE</span>
                <span id="reviewerLoad" class="load-tag">IDLE</span>
            </div>
            <div class="metrics-grid">
                <div class="metric-item"><span class="label">VRAM</span><span id="reviewerVram" class="value">0%</span></div>
                <div class="metric-item highlight"><span class="label">GEN</span><span id="reviewerTps" class="value">0.0</span></div>
                <div class="metric-item"><span class="label">KV</span><span id="reviewerKv" class="value">0%</span></div>
            </div>
        </div>
    </div>
    <div id="taskFeed" class="task-feed">
        <div class="welcome-screen" id="welcomeScreen">
            <div class="welcome-logo">🛡️</div>
            <div class="welcome-title">Gravitas chat</div>
            <div class="welcome-hint">Initialize a session to begin autonomous engineering.</div>
        </div>
    </div>
    <div id="debugOverlay" style="position:fixed;bottom:80px;left:5px;right:5px;background:rgba(0,0,0,0.8);color:#0f0;font-family:monospace;font-size:9px;padding:4px;z-index:9999;border-top:1px solid #333;pointer-events:none;display:none;"></div>
    <div class="task-footer">
        <div class="input-container">
            <textarea id="commandInput" class="command-input" placeholder="Ask anything..." autocomplete="off" rows="1"></textarea>
            <div id="submitBtn" class="submit-btn">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M1 8L15 1M1 8L15 15M1 8H15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
        </div>
    </div>
    <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
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
