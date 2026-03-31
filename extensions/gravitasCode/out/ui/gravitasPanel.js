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
        const toolkitUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'toolkit.js'));
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script type="module" src="${toolkitUri}"></script>
    <title>Agent Console</title>
    <style>
        body { 
            padding: 0; 
            margin: 0;
            display: flex; 
            flex-direction: column; 
            height: 100vh;
            font-family: var(--vscode-font-family);
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
        }

        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            display: flex;
            flex-direction: column;
            gap: 5px;
            max-width: 90%;
            animation: fadeIn 0.1s ease-in-out;
        }

        .message.user {
            align-self: flex-end;
            align-items: flex-end;
        }

        .message.coder, .message.reviewer, .message.system {
            align-self: flex-start;
            align-items: flex-start;
        }

        .bubble {
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 13px;
            line-height: 1.4;
            position: relative;
        }

        .user .bubble {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border-bottom-right-radius: 2px;
        }

        .coder .bubble {
            background-color: var(--vscode-editor-inactiveSelectionBackground);
            border: 1px solid var(--vscode-editor-lineHighlightBorder);
            border-bottom-left-radius: 2px;
        }

        .reviewer .bubble {
            background-color: var(--vscode-inputValidation-warningBackground);
            border: 1px solid var(--vscode-inputValidation-warningBorder);
            color: var(--vscode-input-foreground);
            border-bottom-left-radius: 2px;
        }

        .system .bubble {
            background-color: var(--vscode-editorWidget-background);
            border: 1px solid var(--vscode-panel-border);
            font-style: italic;
            font-size: 12px;
            width: 100%;
            text-align: center;
        }

        .badge {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 2px;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .user .badge { display: none; }
        .coder .badge { color: var(--vscode-textLink-foreground); }
        .reviewer .badge { color: var(--vscode-inputValidation-warningForeground); }

        .meta {
            font-size: 10px;
            opacity: 0.6;
            margin-top: 2px;
            display: flex;
            gap: 8px;
        }

        #input-area {
            padding: 15px;
            border-top: 1px solid var(--vscode-panel-border);
            background-color: var(--vscode-sideBar-background);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .actions {
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding-bottom: 5px;
        }
        
        .action-chip {
            background: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            cursor: pointer;
            white-space: nowrap;
            border: none;
        }
        
        .action-chip:hover { opacity: 0.8; }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(2px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div id="chat-container">
        <div class="message system">
            <div class="bubble">Gravitas System Initialized. Runtime Ready.</div>
        </div>
    </div>

    <div id="input-area">
        <div class="actions">
            <button class="action-chip" onclick="quickAction('Apply Patch')">Apply Patch</button>
            <button class="action-chip" onclick="quickAction('Ask Reviewer')">Ask Reviewer</button>
            <button class="action-chip" onclick="quickAction('Explain')">Explain</button>
        </div>
        <div style="display: flex; gap: 8px;">
            <vscode-text-area id="prompt-input" placeholder="Ask Coder or Reviewer..." style="flex: 1;"></vscode-text-area>
            <vscode-button id="send-btn" appearance="primary">Send</vscode-button>
        </div>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        const container = document.getElementById('chat-container');
        const input = document.getElementById('prompt-input');

        // Restore state
        const state = vscode.getState() || { history: [] };
        if (state.history.length > 0) {
            state.history.forEach(msg => appendMessage(msg));
        }

        let streamingDiv = null;

        function appendMessage(msg) {
            const div = document.createElement('div');
            div.className = 'message ' + msg.role;
            
            let html = '';
            if (msg.role !== 'user' && msg.role !== 'system') {
                const icon = msg.role === 'coder' ? '🟦' : '🟨';
                html += '<div class="badge">' + icon + ' ' + msg.role.toUpperCase() + '</div>';
            }
            
            html += '<div class="bubble">' + msg.text + '</div>';
            
            if (msg.meta) {
                html += '<div class="meta"><span>' + msg.meta.time + '</span><span>' + (msg.meta.tokens || '') + '</span></div>';
            }

            div.innerHTML = html;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return div;
        }

        function saveMessage(msg) {
            state.history.push(msg);
            vscode.setState(state);
            appendMessage(msg);
        }

        // Send Handler
        document.getElementById('send-btn').onclick = () => {
            const text = input.value;
            if (text) {
                input.value = '';
                saveMessage({ role: 'user', text, meta: { time: new Date().toLocaleTimeString() } });
                vscode.postMessage({ command: 'userMessage', text });
            }
        };

        // Quick Action Handler
        window.quickAction = (action) => {
            vscode.postMessage({ command: 'action', action });
        };

        // Incoming Messages
        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.command) {
                case 'addMessage':
                    if (streamingDiv) {
                        streamingDiv.remove();
                        streamingDiv = null;
                    }
                    saveMessage(message.data);
                    break;
                case 'updateStreamingMessage':
                    if (!streamingDiv) {
                        streamingDiv = appendMessage({ ...message.data, text: '...' });
                    }
                    streamingDiv.querySelector('.bubble').textContent = message.data.text;
                    container.scrollTop = container.scrollHeight;
                    break;
                case 'updateStatus':
                    break;
            }
        });
    </script>
</body>
</html>`;
    }
}
exports.GravitasPanel = GravitasPanel;
GravitasPanel.viewType = 'gravitas.agentConsole';
//# sourceMappingURL=gravitasPanel.js.map