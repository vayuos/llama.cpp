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
exports.PanelProvider = void 0;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const taskManager_1 = require("../uiv2/taskManager");
class PanelProvider {
    constructor(_extensionUri) {
        this._extensionUri = _extensionUri;
    }
    resolveWebviewView(webviewView, context, _token) {
        this._view = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.joinPath(this._extensionUri, 'media')]
        };
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
        this._setWebviewMessageListener(webviewView.webview);
        // Listen for task updates
        taskManager_1.TaskManager.getInstance().onDidTaskUpdate((task) => {
            if (this._view) {
                const anyTask = task;
                if (anyTask._type === 'log') {
                    // Streaming Log Chunk
                    this._view.webview.postMessage({
                        type: 'addTerminalChunk',
                        chunk: {
                            taskId: task.id,
                            data: anyTask._chunk
                        }
                    });
                }
                else {
                    // Full Task Update
                    this._view.webview.postMessage({ type: 'updateTask', task });
                    if (task.attempts && task.attempts.length > 0) {
                        const lastAttempt = task.attempts[task.attempts.length - 1];
                        this._view.webview.postMessage({ type: 'addAttempt', attempt: lastAttempt });
                    }
                }
            }
        });
    }
    _setWebviewMessageListener(webview) {
        webview.onDidReceiveMessage(async (message) => {
            switch (message.command) {
                case 'spawnTask':
                    // "Submitting input always creates a new Task Shell"
                    // Trigger the pipeline runner which creates the task and starts execution
                    vscode.commands.executeCommand('gravitas.pipeline.run', message.text);
                    return;
            }
        });
    }
    _getHtmlForWebview(webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'taskShell.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'taskShell.css'));
        const toolkitUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'toolkit.js'));
        const htmlPath = path.join(this._extensionUri.fsPath, 'media', 'taskShell.html');
        let html = fs.readFileSync(htmlPath, 'utf-8');
        // Replace placeholders
        html = html.replace('${styleUri}', styleUri.toString());
        html = html.replace('${scriptUri}', scriptUri.toString());
        html = html.replace('${toolkitUri}', toolkitUri.toString());
        return html;
    }
}
exports.PanelProvider = PanelProvider;
PanelProvider.viewType = 'gravitas.agentConsole';
//# sourceMappingURL=panelProvider.js.map