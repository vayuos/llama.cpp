import * as vscode from 'vscode';
import { StateManager } from './state/sessionState';
import { GravitasState } from './core/state';
import { ConfigManager } from './core/config';
import { SetupPanel } from './ui/setupPanel';
import { ValidationPanel } from './ui/validationPanel';
import { UnifiedProcessManager } from './process/processManager';
import { runPipeline } from './commands/pipelineRun';
import { PanelProvider } from './ui/panelProvider';
import { CentralLogger } from './core/logger';
import { GravitasStatusBar } from './ui/statusBar';
import { registerCleanup } from './process/cleanup';
import { registerWatchers } from './core/watchers';
import { LogsPanel } from './ui/logsPanel';

export class ActivationManager {
    private state = new StateManager();
    private gravitasState = GravitasState.getInstance();
    private processManager = UnifiedProcessManager.getInstance();

    async activate(context: vscode.ExtensionContext) {
        console.log('Gravitas Code: Initializing infrastructure control...');

        // 1. Core Services
        const logger = CentralLogger.getInstance();
        const statusBar = GravitasStatusBar.getInstance();

        // 2. State & Config
        this.gravitasState.syncToContext();
        const config = await ConfigManager.getInstance().loadConfig();

        if (config) {
            logger.setLogDir(config.logDir);
            this.gravitasState.updateState({ configLoaded: true });
            logger.info('system', 'Configuration loaded successfully.');
        } else {
            this.gravitasState.updateState({ configLoaded: false, validated: false });
            logger.warn('system', 'No configuration found. Awaiting setup.');
            vscode.window.showInformationMessage('Gravitas: Professional setup required.', 'Start Setup').then(s => {
                if (s === 'Start Setup') vscode.commands.executeCommand('gravitas.setup.open');
            });
        }

        // 3. Register Hooks
        registerCleanup(context);
        registerWatchers(context);

        // 4. UI Providers
        const panelProvider = new PanelProvider(context.extensionUri);
        context.subscriptions.push(
            vscode.window.registerWebviewViewProvider('gravitas.chatView', panelProvider)
        );

        // 5. Commands
        context.subscriptions.push(
            vscode.commands.registerCommand('gravitas.setup.open', () => SetupPanel.createOrShow(context.extensionUri)),
            vscode.commands.registerCommand('gravitas.setup.validate', async () => {
                const cfg = await ConfigManager.getInstance().loadConfig();
                if (!cfg) return;
                await ValidationPanel.showAndRun(context.extensionUri, cfg);
                statusBar.update();
            }),
            vscode.commands.registerCommand('gravitas.logs.open', () => LogsPanel.createOrShow(context.extensionUri)),
            vscode.commands.registerCommand('gravitas.pipeline.run', async (prompt?: string) => {
                if (!this.gravitasState.state.validated) {
                    vscode.window.showErrorMessage('Gravitas: Invariant Violation: System must be validated.');
                    return;
                }
                const input = prompt || await vscode.window.showInputBox({ prompt: 'Task for Dual-Agent Loop' });
                if (input) await runPipeline(input, this.state);
            }),
            vscode.commands.registerCommand('gravitas.runtime.start', async () => {
                if (!this.gravitasState.state.validated) return;
                const cfg = await ConfigManager.getInstance().loadConfig();
                if (cfg) {
                    await this.processManager.startReviewer(cfg);
                    await this.processManager.startCoder(cfg);
                    logger.info('system', 'LLM Servers started.');
                }
            }),
            vscode.commands.registerCommand('gravitas.runtime.stop', () => {
                this.processManager.stopAll();
                logger.info('system', 'LLM Servers stopped.');
            })
        );

        // Update status bar on any state change
        // In a more complex app, we'd use an event emitter, but direct call suffices here.
        const originalUpdate = this.gravitasState.updateState.bind(this.gravitasState);
        this.gravitasState.updateState = (s) => {
            originalUpdate(s);
            statusBar.update();
        };
    }
}
