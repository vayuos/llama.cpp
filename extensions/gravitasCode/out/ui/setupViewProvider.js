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
exports.SetupViewProvider = void 0;
const vscode = __importStar(require("vscode"));
const state_1 = require("../core/state");
class SetupViewProvider {
    constructor(_extensionUri) {
        this._extensionUri = _extensionUri;
    }
    resolveWebviewView(webviewView, _context, _token) {
        this._view = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);
        webviewView.webview.onDidReceiveMessage(data => {
            switch (data.command) {
                case 'startSetup':
                    vscode.commands.executeCommand('gravitas.setup.open');
                    break;
                case 'runValidation':
                    vscode.commands.executeCommand('gravitas.setup.validate');
                    break;
                case 'reconfigure':
                    vscode.commands.executeCommand('gravitas.setup.open');
                    break;
                case 'startServices':
                    vscode.commands.executeCommand('gravitas.runtime.start');
                    break;
                case 'stopServices':
                    vscode.commands.executeCommand('gravitas.runtime.stop');
                    break;
            }
        });
        // Listen for state changes to refresh the view
        const refreshInterval = setInterval(() => {
            if (this._view) {
                this._view.webview.html = this._getHtmlForWebview(this._view.webview);
            }
        }, 2000);
        webviewView.onDidDispose(() => {
            clearInterval(refreshInterval);
        });
    }
    _getHtmlForWebview(webview) {
        const state = state_1.GravitasState.getInstance().state;
        const isConfigured = state.configLoaded;
        const isValidated = state.validated;
        if (!isConfigured) {
            // Not configured - show setup button
            return this._getSetupRequiredHtml();
        }
        else {
            // Configured - check if services are running
            const servicesRunning = this._areServicesRunning();
            return this._getConfiguredHtml(isValidated, servicesRunning);
        }
    }
    _areServicesRunning() {
        try {
            const { UnifiedProcessManager } = require('../process/processManager');
            const pm = UnifiedProcessManager.getInstance();
            const coderStatus = pm.getProcessStatus('coder');
            const reviewerStatus = pm.getProcessStatus('reviewer');
            // Services are running if either has a PID
            return !!(coderStatus.pid || reviewerStatus.pid);
        }
        catch {
            return false;
        }
    }
    _getSetupRequiredHtml() {
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    ${this._getCommonStyles()}
                </style>
            </head>
            <body>
                <div class="icon">🛡️</div>
                <h2>Setup Required</h2>
                <p>Gravitas Core requires configuration before use.</p>
                <button onclick="startSetup()">Configure System</button>
                <script>
                    const vscode = acquireVsCodeApi();
                    function startSetup() {
                        vscode.postMessage({ command: 'startSetup' });
                    }
                </script>
            </body>
            </html>
        `;
    }
    _getConfiguredHtml(isValidated, servicesRunning) {
        const statusIcon = isValidated ? '✅' : '⚙️';
        const statusText = isValidated ? 'System Validated' : 'Ready to Validate';
        const statusDesc = isValidated
            ? 'Services are configured and validated. You can re-run validation anytime.'
            : 'Configuration loaded. Run validation to start services.';
        // Determine which service control button to show
        const serviceButton = servicesRunning
            ? '<button class="stop-button" onclick="stopServices()">⏹️ Stop Services</button>'
            : '<button class="start-button" onclick="startServices()">▶️ Start Services</button>';
        return `
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    ${this._getCommonStyles()}
                    .button-group {
                        display: flex;
                        flex-direction: column;
                        gap: 10px;
                        width: 100%;
                        max-width: 200px;
                    }
                    .secondary-button {
                        background-color: var(--vscode-button-secondaryBackground);
                        color: var(--vscode-button-secondaryForeground);
                    }
                    .secondary-button:hover {
                        background-color: var(--vscode-button-secondaryHoverBackground);
                    }
                    .start-button {
                        background-color: #4caf50;
                        color: white;
                        width: 100%;
                        padding: 10px 16px;
                    }
                    .start-button:hover {
                        background-color: #45a049;
                    }
                    .stop-button {
                        background-color: #f44336;
                        color: white;
                        width: 100%;
                        padding: 10px 16px;
                    }
                    .stop-button:hover {
                        background-color: #da190b;
                    }
                    .divider {
                        height: 1px;
                        background-color: var(--vscode-widget-border);
                        margin: 10px 0;
                        opacity: 0.3;
                    }
                </style>
            </head>
            <body>
                <div class="icon">${statusIcon}</div>
                <h2>${statusText}</h2>
                <p>${statusDesc}</p>
                <div class="button-group">
                    <button onclick="runValidation()">🚀 Run Validation</button>
                    <div class="divider"></div>
                    ${serviceButton}
                    <button class="secondary-button" onclick="reconfigure()">⚙️ Reconfigure</button>
                </div>
                <script>
                    const vscode = acquireVsCodeApi();
                    function runValidation() {
                        vscode.postMessage({ command: 'runValidation' });
                    }
                    function startServices() {
                        vscode.postMessage({ command: 'startServices' });
                    }
                    function stopServices() {
                        vscode.postMessage({ command: 'stopServices' });
                    }
                    function reconfigure() {
                        vscode.postMessage({ command: 'reconfigure' });
                    }
                </script>
            </body>
            </html>
        `;
    }
    _getCommonStyles() {
        return `
            body {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                padding: 20px;
                color: var(--vscode-foreground);
                font-family: var(--vscode-font-family);
                text-align: center;
            }
            .icon {
                font-size: 48px;
                margin-bottom: 20px;
            }
            h2 { 
                margin-bottom: 10px;
                font-size: 18px;
            }
            p { 
                margin-bottom: 20px;
                opacity: 0.8;
                font-size: 13px;
                line-height: 1.5;
            }
            button {
                background-color: var(--vscode-button-background);
                color: var(--vscode-button-foreground);
                border: none;
                padding: 10px 16px;
                cursor: pointer;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
                width: 100%;
            }
            button:hover {
                background-color: var(--vscode-button-hoverBackground);
            }
        `;
    }
}
exports.SetupViewProvider = SetupViewProvider;
//# sourceMappingURL=setupViewProvider.js.map