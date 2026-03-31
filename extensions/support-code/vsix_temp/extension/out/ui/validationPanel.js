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
exports.ValidationPanel = void 0;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const validator_1 = require("../validation/validator");
const fileChecks_1 = require("../validation/steps/fileChecks");
const portChecks_1 = require("../validation/steps/portChecks");
const serverChecks_1 = require("../validation/steps/serverChecks");
const state_1 = require("../core/state");
const hash_1 = require("../validation/hash");
const processManager_1 = require("../process/processManager");
class ValidationPanel {
    constructor(panel, extensionUri) {
        this._disposables = [];
        this._panel = panel;
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
        this._panel.webview.html = this._getHtmlForWebview(this._panel.webview, extensionUri);
    }
    static async showAndRun(extensionUri, config) {
        const column = vscode.ViewColumn.Beside;
        if (ValidationPanel.currentPanel) {
            ValidationPanel.currentPanel._panel.reveal(column);
        }
        else {
            const panel = vscode.window.createWebviewPanel('gravitasValidation', 'Gravitas Validation', column, { enableScripts: true, localResourceRoots: [extensionUri] });
            ValidationPanel.currentPanel = new ValidationPanel(panel, extensionUri);
        }
        await ValidationPanel.currentPanel._runValidation(config);
    }
    async _runValidation(config) {
        const engine = new validator_1.ValidationEngine();
        engine.addStep(new fileChecks_1.BinaryCheckStep());
        engine.addStep(new fileChecks_1.ModelCheckStep());
        engine.addStep(new portChecks_1.PortCheckStep());
        engine.addStep(new serverChecks_1.ServerPingStep('reviewer'));
        engine.addStep(new serverChecks_1.PromptTestStep('reviewer'));
        engine.addStep(new serverChecks_1.ServerPingStep('coder'));
        engine.addStep(new serverChecks_1.PromptTestStep('coder'));
        // Mocking the engine.run to stream logs
        const originalEngineRun = engine.run.bind(engine);
        // Wrap execution to send logs to webview
        const runWithStreaming = async () => {
            for (const step of engine.steps) {
                this._panel.webview.postMessage({ command: 'addLog', text: `[STEP] ${step.name}` });
                try {
                    const result = await step.execute(config);
                    if (result.success) {
                        this._panel.webview.postMessage({ command: 'addLog', text: `[SUCCESS] ${step.name}` });
                    }
                    else {
                        this._panel.webview.postMessage({ command: 'addLog', text: `[FAILURE] ${step.name}: ${result.message}` });
                        if (step.rollback) {
                            this._panel.webview.postMessage({ command: 'addLog', text: `[ROLLBACK] Triggered for ${step.name}` });
                            await step.rollback();
                        }
                        this._panel.webview.postMessage({ command: 'setResult', success: false });
                        return false;
                    }
                }
                catch (e) {
                    this._panel.webview.postMessage({ command: 'addLog', text: `[ERROR] ${step.name}: ${e.message}` });
                    this._panel.webview.postMessage({ command: 'setResult', success: false });
                    return false;
                }
            }
            this._panel.webview.postMessage({ command: 'setResult', success: true });
            return true;
        };
        const success = await runWithStreaming();
        if (success) {
            const hash = (0, hash_1.calculateValidationHash)(config);
            state_1.GravitasState.getInstance().updateState({
                validated: true,
                validationHash: hash
            });
            vscode.window.showInformationMessage('Gravitas: System validation complete. Chat unlocked.');
        }
        else {
            state_1.GravitasState.getInstance().updateState({ validated: false });
        }
        // Cleanup servers after validation
        await processManager_1.UnifiedProcessManager.getInstance().stopAll();
    }
    _getHtmlForWebview(webview, extensionUri) {
        const htmlPath = path.join(extensionUri.fsPath, 'media', 'validation.html');
        return fs.readFileSync(htmlPath, 'utf-8');
    }
    dispose() {
        ValidationPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) {
                x.dispose();
            }
        }
    }
}
exports.ValidationPanel = ValidationPanel;
//# sourceMappingURL=validationPanel.js.map