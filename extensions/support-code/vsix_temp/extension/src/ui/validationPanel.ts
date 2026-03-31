import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { ValidationEngine } from '../validation/validator';
import { GravitasConfig } from '../core/config';
import { BinaryCheckStep, ModelCheckStep } from '../validation/steps/fileChecks';
import { PortCheckStep } from '../validation/steps/portChecks';
import { ServerPingStep, PromptTestStep } from '../validation/steps/serverChecks';
import { GravitasState } from '../core/state';
import { calculateValidationHash } from '../validation/hash';
import { UnifiedProcessManager } from '../process/processManager';

export class ValidationPanel {
    public static currentPanel: ValidationPanel | undefined;
    private readonly _panel: vscode.WebviewPanel;
    private _disposables: vscode.Disposable[] = [];

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
        this._panel = panel;
        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
        this._panel.webview.html = this._getHtmlForWebview(this._panel.webview, extensionUri);
    }

    public static async showAndRun(extensionUri: vscode.Uri, config: GravitasConfig) {
        const column = vscode.ViewColumn.Beside;

        if (ValidationPanel.currentPanel) {
            ValidationPanel.currentPanel._panel.reveal(column);
        } else {
            const panel = vscode.window.createWebviewPanel(
                'gravitasValidation',
                'Gravitas Validation',
                column,
                { enableScripts: true, localResourceRoots: [extensionUri] }
            );
            ValidationPanel.currentPanel = new ValidationPanel(panel, extensionUri);
        }

        await ValidationPanel.currentPanel._runValidation(config);
    }

    private async _runValidation(config: GravitasConfig) {
        const engine = new ValidationEngine();
        engine.addStep(new BinaryCheckStep());
        engine.addStep(new ModelCheckStep());
        engine.addStep(new PortCheckStep());
        engine.addStep(new ServerPingStep('reviewer'));
        engine.addStep(new PromptTestStep('reviewer'));
        engine.addStep(new ServerPingStep('coder'));
        engine.addStep(new PromptTestStep('coder'));

        // Mocking the engine.run to stream logs
        const originalEngineRun = engine.run.bind(engine);

        // Wrap execution to send logs to webview
        const runWithStreaming = async () => {
            for (const step of (engine as any).steps) {
                this._panel.webview.postMessage({ command: 'addLog', text: `[STEP] ${step.name}` });
                try {
                    const result = await step.execute(config);
                    if (result.success) {
                        this._panel.webview.postMessage({ command: 'addLog', text: `[SUCCESS] ${step.name}` });
                    } else {
                        this._panel.webview.postMessage({ command: 'addLog', text: `[FAILURE] ${step.name}: ${result.message}` });
                        if (step.rollback) {
                            this._panel.webview.postMessage({ command: 'addLog', text: `[ROLLBACK] Triggered for ${step.name}` });
                            await step.rollback();
                        }
                        this._panel.webview.postMessage({ command: 'setResult', success: false });
                        return false;
                    }
                } catch (e: any) {
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
            const hash = calculateValidationHash(config);
            GravitasState.getInstance().updateState({
                validated: true,
                validationHash: hash
            });
            vscode.window.showInformationMessage('Gravitas: System validation complete. Chat unlocked.');
        } else {
            GravitasState.getInstance().updateState({ validated: false });
        }

        // Cleanup servers after validation
        await UnifiedProcessManager.getInstance().stopAll();
    }

    private _getHtmlForWebview(webview: vscode.Webview, extensionUri: vscode.Uri) {
        const htmlPath = path.join(extensionUri.fsPath, 'media', 'validation.html');
        return fs.readFileSync(htmlPath, 'utf-8');
    }

    public dispose() {
        ValidationPanel.currentPanel = undefined;
        this._panel.dispose();
        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) { x.dispose(); }
        }
    }
}
