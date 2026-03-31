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
exports.ChatViewProvider = void 0;
const vscode = __importStar(require("vscode"));
class ChatViewProvider {
    constructor(_extensionUri) {
        this._extensionUri = _extensionUri;
    }
    resolveWebviewView(webviewView, context, _token) {
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
        webviewView.webview.onDidReceiveMessage(data => {
            switch (data.type) {
                case 'runPipeline': {
                    vscode.commands.executeCommand('gravitas.pipeline.run', data.value);
                    break;
                }
            }
        });
    }
    _getHtmlForWebview(webview) {
        const toolkitUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'node_modules', '@vscode', 'webview-ui-toolkit', 'dist', 'toolkit.min.js'));
        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Gravitas Controller</title>
                <script type="module" src="${toolkitUri}"></script>
                <style>
                    body { padding: 10px; display: flex; flex-direction: column; gap: 10px; }
                    .panel { display: none; }
                    .panel.active { display: block; }
                    .diff-container { background: #1e1e1e; font-family: monospace; padding: 5px; border-radius: 4px; overflow-x: auto; }
                    .diff-added { color: #4ec9b0; }
                    .diff-removed { color: #f44747; }
                </style>
            </head>
            <body>
                <vscode-panels activeid="tab-coder">
                    <vscode-panel-tab id="tab-coder">CODER</vscode-panel-tab>
                    <vscode-panel-tab id="tab-review">REVIEW</vscode-panel-tab>
                    <vscode-panel-view id="view-coder">
                        <vscode-text-area id="prompt" placeholder="Enter instructions..." style="width: 100%;"></vscode-text-area>
                        <vscode-button id="run" style="margin-top: 10px;">Run Pipeline</vscode-button>
                    </vscode-panel-view>
                    <vscode-panel-view id="view-review">
                        <div id="violations">No violations found.</div>
                    </vscode-panel-view>
                </vscode-panels>

                <script>
                    const vscode = acquireVsCodeApi();
                    document.getElementById('run').addEventListener('click', () => {
                        const prompt = document.getElementById('prompt').value;
                        vscode.postMessage({ type: 'runPipeline', value: prompt });
                    });
                </script>
            </body>
            </html>`;
    }
}
exports.ChatViewProvider = ChatViewProvider;
ChatViewProvider.viewType = 'gravitas.chatView';
//# sourceMappingURL=chatView.js.map