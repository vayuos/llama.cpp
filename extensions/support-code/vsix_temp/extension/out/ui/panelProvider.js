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
const gravitasPanel_1 = require("./gravitasPanel");
const processManager_1 = require("../runtime/processManager");
class PanelProvider {
    constructor(_extensionUri) {
        this._extensionUri = _extensionUri;
        this._processManager = new processManager_1.ProcessManager();
    }
    resolveWebviewView(webviewView, _context, _token) {
        const panel = new gravitasPanel_1.GravitasPanel(this._extensionUri);
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
exports.PanelProvider = PanelProvider;
//# sourceMappingURL=panelProvider.js.map