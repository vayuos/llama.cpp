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
exports.GravitasPanel = void 0;
const vscode = __importStar(require("vscode"));
class GravitasPanel {
    constructor(_extensionUri) {
        this._extensionUri = _extensionUri;
    }
    getHtmlForWebview(webview) {
        const toolkitUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'node_modules', '@vscode', 'webview-ui-toolkit', 'dist', 'toolkit.js'));
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script type="module" src="${toolkitUri}"></script>
    <title>Gravitas Control</title>
    <style>
        body { padding: 10px; display: flex; flex-direction: column; gap: 15px; font-family: var(--vscode-font-family); }
        .section { display: flex; flex-direction: column; gap: 8px; }
        .divider { border-top: 1px solid var(--vscode-panel-border); margin: 5px 0; }
        .status-container { display: flex; flex-direction: column; gap: 5px; font-size: 0.9em; }
        .status-item { display: flex; align-items: center; gap: 8px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--vscode-testing-iconUnsetColor); }
        .dot.running { background: var(--vscode-testing-iconPassedColor); }
        .dot.stopped { background: var(--vscode-testing-iconFailedColor); }
        h3 { margin: 0; font-size: 1.1em; color: var(--vscode-foreground); }
        .btn-group { display: flex; flex-direction: column; gap: 8px; }
    </style>
</head>
<body>
    <div class="section">
        <h3>[ Configuration ]</h3>
        <vscode-button appearance="secondary" id="open-settings-btn" style="width: 100%">Open Gravitas Settings</vscode-button>
    </div>

    <div class="divider"></div>

    <div class="section">
        <h3>[ Runtime ]</h3>
        <div class="btn-group">
            <vscode-button id="start-btn" style="width: 100%">▶ Start All</vscode-button>
            <vscode-button id="restart-btn" style="width: 100%">⟳ Restart All</vscode-button>
            <vscode-button id="stop-btn" style="width: 100%">■ Stop All</vscode-button>
        </div>
    </div>

    <div class="divider"></div>

    <div class="section">
        <h3>[ Status ]</h3>
        <div class="status-container">
            <div class="status-item">
                <div id="coder-dot" class="dot"></div>
                <span>Coder: <span id="coder-status">STOPPED</span></span>
            </div>
            <div class="status-item">
                <div id="reviewer-dot" class="dot"></div>
                <span>Reviewer: <span id="reviewer-status">STOPPED</span></span>
            </div>
        </div>
    </div>

    <div class="divider"></div>

    <div class="section">
        <h3>[ Commands ]</h3>
        <vscode-text-area resize="vertical" placeholder="Enter task for Coder..." id="prompt-input" style="width: 100%"></vscode-text-area>
        <vscode-button id="run-pipeline-btn">Run Pipeline</vscode-button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        document.getElementById('open-settings-btn').onclick = () => vscode.postMessage({ command: 'openSettings' });
        document.getElementById('start-btn').onclick = () => vscode.postMessage({ command: 'startAll' });
        document.getElementById('restart-btn').onclick = () => vscode.postMessage({ command: 'restartAll' });
        document.getElementById('stop-btn').onclick = () => vscode.postMessage({ command: 'stopAll' });
        document.getElementById('run-pipeline-btn').onclick = () => {
            const prompt = document.getElementById('prompt-input').value;
            if (prompt) vscode.postMessage({ command: 'runPipeline', prompt });
        };

        window.addEventListener('message', event => {
            const message = event.data;
            if (message.command === 'updateStatus') {
                const { coder, reviewer } = message.status;
                
                document.getElementById('coder-status').innerText = coder;
                document.getElementById('coder-dot').className = 'dot ' + coder.toLowerCase();
                
                document.getElementById('reviewer-status').innerText = reviewer;
                document.getElementById('reviewer-dot').className = 'dot ' + reviewer.toLowerCase();
            }
        });

        // Poll for status every second
        setInterval(() => vscode.postMessage({ command: 'pollStatus' }), 1000);
    </script>
</body>
</html>`;
    }
}
exports.GravitasPanel = GravitasPanel;
GravitasPanel.viewType = 'gravitas.chatView';
//# sourceMappingURL=gravitasPanel.js.map