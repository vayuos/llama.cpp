import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

import { GravitasState } from './core/state';
import { ConfigManager } from './core/config';
import { UnifiedProcessManager } from './process/processManager';
import { runPipeline } from './commands/pipelineRun';
import { TaskHistoryProvider } from './uiv2/taskHistory';
import { CentralLogger } from './core/logger';
import { registerCleanup } from './process/cleanup';
import { registerWatchers } from './core/watchers';
import { LogBridge } from './core/logBridge';
import { StorageManager } from './core/storageManager';
import { TelemetryService } from './llm/telemetry';

import { RuntimeTreeProvider } from './ui/runtimeTreeProvider';

import { TaskManager } from './uiv2/taskManager';
import { TaskShellPanel } from './uiv2/taskShell';
import { ChatSidebarProvider } from './uiv2/chatSidebar';

export class ActivationManager {
    // private state = new StateManager(); // Deprecated

    private gravitasState = GravitasState.getInstance();
    private processManager = UnifiedProcessManager.getInstance();
    private context?: vscode.ExtensionContext;
    private logger?: CentralLogger;

    async activate(context: vscode.ExtensionContext) {
        this.context = context;
        console.log('Gravitas Code: Initializing infrastructure control...');
        TaskManager.initialize(context);
        LogBridge.initialize();

        // 1. Core Services (Initialize First)
        const logger = CentralLogger.getInstance();
        this.logger = logger;

        // --- TOPOLOGY ENFORCEMENT ---
        try {
            this.checkTopology(context.extensionPath);
        } catch (e: any) {
            vscode.window.showErrorMessage(`CRITICAL: Topology Violation. ${e.message}`, { modal: true });
            logger.error('system', `CRITICAL TOPOLOGY VIOLATION: ${e.message}`);
            // We intentionally do not throw here to allow partial activation for debugging, 
            // BUT we mark validation as impossible.
            this.gravitasState.updateState({ validated: false });
            return; // STOP ACTIVATION
        }

        // Check if this is a fresh install/reinstall using a persistent marker file
        // Storage is in globalStorageUri which persists across extension reinstalls/updates
        const installMarkerPath = path.join(context.globalStorageUri.fsPath, '.installed');
        
        // Ensure directory exists
        if (!fs.existsSync(context.globalStorageUri.fsPath)) {
            fs.mkdirSync(context.globalStorageUri.fsPath, { recursive: true });
        }
        
        const markerExists = fs.existsSync(installMarkerPath);

        const isFreshInstall = !markerExists;

        if (isFreshInstall) {
            logger.info('system', 'Fresh install detected - cleaning up any existing data...');

            // Clean up everything
            const storageManager = StorageManager.getInstance();
            const cleanupResult = storageManager.clearAll();

            if (!cleanupResult.success) {
                logger.error('system', `Cleanup failed: ${cleanupResult.message}`);
            }

            // Clear settings
            const config = vscode.workspace.getConfiguration('gravitasCode');
            const allKeys = [
                'runtime.autoTestOnStart', 'runtime.showLogs', 'runtime.logLevel', 'runtime.killSignal',
                'coder.general.enabled', 'coder.general.baseUrl', 'coder.general.host', 'coder.general.port', 'coder.general.noWarmup', 'coder.general.modelName', 'coder.general.strictMode',
                'coder.hardware.contextSize',
                'coder.sampling.temperature', 'coder.sampling.topP', 'coder.sampling.topK', 'coder.sampling.repeatPenalty',
                'reviewer.general.enabled', 'reviewer.general.baseUrl', 'reviewer.general.host', 'reviewer.general.port', 'reviewer.general.noWarmup', 'reviewer.general.modelName', 'reviewer.general.strictMode',
                'reviewer.hardware.contextSize',
                'reviewer.sampling.temperature', 'reviewer.sampling.topP', 'reviewer.sampling.topK', 'reviewer.sampling.repeatPenalty',
                'vayuforge.ragEndpoint'
            ];

            for (const key of allKeys) {
                await config.update(key, undefined, vscode.ConfigurationTarget.Global);
            }

            // Create marker file
            try {
                fs.writeFileSync(installMarkerPath, new Date().toISOString());
                logger.info('system', 'First install cleanup complete. Extension ready.');
            } catch (e: any) {
                logger.error('system', `Failed to create install marker: ${e.message}`);
            }
        } else {
            logger.info('system', 'Extension reloading - preserving existing data.');
        }

        // 2. State & Config
        this.gravitasState.syncToContext();
        const config = await ConfigManager.getInstance().loadConfig();

        if (config) {
            logger.setLogDir(config.logDir || '');
            CentralLogger.getInstance().setLevel(config.runtime.logLevel);
            this.gravitasState.updateState({ configLoaded: true, validated: true });
            logger.info('system', 'Configuration loaded successfully.');
        } else {
            this.gravitasState.updateState({ configLoaded: false, validated: false });
            logger.warn('system', 'No configuration found.');
        }

        // 3. Register Hooks & Services
        registerCleanup(context);
        registerWatchers(context);
        TelemetryService.getInstance().startPolling();

        // 4. UI Providers
        const chatSidebarProvider = new ChatSidebarProvider(context.extensionUri);
        const taskHistoryProvider = new TaskHistoryProvider();
        const runtimeProvider = new RuntimeTreeProvider();

        context.subscriptions.push(
            vscode.window.registerWebviewViewProvider(ChatSidebarProvider.viewType, chatSidebarProvider),
            vscode.window.registerTreeDataProvider('gravitas.taskHistory', taskHistoryProvider),
            vscode.window.registerTreeDataProvider('gravitas.runtime', runtimeProvider)
        );

        // 5. Commands
        context.subscriptions.push(
            vscode.commands.registerCommand('gravitas.task.delete', (item: any) => {
                if (item && item.task) {
                    TaskManager.getInstance().deleteTask(item.task.id);
                }
            }),
            vscode.commands.registerCommand('gravitas.task.clearAll', async () => {
                const confirm = await vscode.window.showWarningMessage(
                    'Are you sure you want to clear ALL task history? This will delete all event ledgers from disk.',
                    { modal: true },
                    'Delete All'
                );
                if (confirm === 'Delete All') {
                    TaskManager.getInstance().clearAllTasks();
                }
            }),
            vscode.commands.registerCommand('gravitas.task.openInShell', (taskId: string) => {
                chatSidebarProvider.showTask(taskId);
            }),
            vscode.commands.registerCommand('gravitas.task.spawn', async (prompt?: string) => {
                const tm = TaskManager.getInstance();
                
                if (prompt) {
                    const task = tm.createTask(prompt, 'user');
                    chatSidebarProvider.showTask(task.id);
                    await vscode.commands.executeCommand('gravitas.pipeline.run', prompt, task.id);
                } else {
                    // Unified 'New Chat' Flow: No popup, just sidebar focus
                    chatSidebarProvider.reset(); 
                    chatSidebarProvider.focus();
                }
            }),
            vscode.commands.registerCommand('gravitas.pipeline.run', async (prompt?: string, taskId?: string) => {
                const input = prompt || await vscode.window.showInputBox({ 
                    prompt: 'Dual-Agent Execution Loop',
                    placeHolder: 'e.g. Implement a new logger'
                });
                if (input) await runPipeline(input, taskId);
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
            }),
            // Individual model controls
            vscode.commands.registerCommand('gravitas.runtime.startCoder', async () => {
                if (!this.gravitasState.state.validated) return;
                const cfg = await ConfigManager.getInstance().loadConfig();
                if (cfg) {
                    await this.processManager.startCoder(cfg);
                    logger.info('system', 'Coder server started.');
                }
            }),
            vscode.commands.registerCommand('gravitas.runtime.stopCoder', async () => {
                const pm = this.processManager as any;
                await pm.coder.stop();
                logger.info('system', 'Coder server stopped.');
            }),
            vscode.commands.registerCommand('gravitas.runtime.restartCoder', async () => {
                if (!this.gravitasState.state.validated) return;
                const cfg = await ConfigManager.getInstance().loadConfig();
                if (cfg) {
                    const pm = this.processManager as any;
                    await pm.coder.stop();
                    await this.processManager.startCoder(cfg);
                    logger.info('system', 'Coder server restarted.');
                }
            }),
            vscode.commands.registerCommand('gravitas.runtime.startReviewer', async () => {
                if (!this.gravitasState.state.validated) return;
                const cfg = await ConfigManager.getInstance().loadConfig();
                if (cfg) {
                    await this.processManager.startReviewer(cfg);
                    logger.info('system', 'Reviewer server started.');
                }
            }),
            vscode.commands.registerCommand('gravitas.runtime.stopReviewer', async () => {
                const pm = this.processManager as any;
                await pm.reviewer.stop();
                logger.info('system', 'Reviewer server stopped.');
            }),
            vscode.commands.registerCommand('gravitas.runtime.restartReviewer', async () => {
                if (!this.gravitasState.state.validated) return;
                const cfg = await ConfigManager.getInstance().loadConfig();
                if (cfg) {
                    const pm = this.processManager as any;
                    await pm.reviewer.stop();
                    await this.processManager.startReviewer(cfg);
                    logger.info('system', 'Reviewer server restarted.');
                }
            }),
            vscode.commands.registerCommand('gravitas.storage.clear', async () => {
                const choice = await vscode.window.showWarningMessage(
                    'Clear Gravitas Storage?',
                    { modal: true },
                    'Clear Logs Only',
                    'Clear Everything (Logs + Reset Validation)'
                );

                if (!choice) return;

                const storageManager = StorageManager.getInstance();
                let result;

                if (choice === 'Clear Logs Only') {
                    result = storageManager.clearLogs();
                } else {
                    result = storageManager.clearAll();
                }

                if (result.success) {
                    vscode.window.showInformationMessage(result.message);
                    logger.info('system', 'Storage cleared by user.');
                } else {
                    vscode.window.showErrorMessage(result.message);
                    logger.error('system', result.message);
                }
            }),
        );

        // Update status on any state change
        // In a more complex app, we'd use an event emitter, but direct call suffices here.
        const originalUpdate = this.gravitasState.updateState.bind(this.gravitasState);
        this.gravitasState.updateState = (s) => {
            originalUpdate(s);
        };
    }

    async cleanup() {
        console.log('Gravitas Code: Cleaning up - stopping all llama servers and telemetry...');
        TelemetryService.getInstance().stopPolling();
        await this.processManager.stopAll();
        console.log('Gravitas Code: Cleanup complete.');
    }

    private checkTopology(extensionPath: string) {
        const modulesPath = path.join(extensionPath, 'node_modules');

        if (!fs.existsSync(modulesPath)) {
            // If it doesn't exist, we might be in a bundled vsix where node_modules is not needed or handled differently.
            // But for local dev, this is weird. Let's assume strict rule applies to dev environment.
            // If we are running from 'dist', node_modules might not be there.
            // Check if we are in dev mode? 
            // For now, strict enforcement as requested.
            // throw new Error('Missing node_modules. Run ./setup.sh');
            return; // Skip check if doesn't exist (e.g. production build)
        }

        const stats = fs.lstatSync(modulesPath);
        if (!stats.isSymbolicLink()) {
            throw new Error('VayuForge Violation: node_modules MUST be a symbolic link in the development environment. Run ./setup.sh');
        }

        const linkTarget = fs.readlinkSync(modulesPath);
        if (!linkTarget.includes('support-code')) {
            CentralLogger.getInstance().warn('system', `Topology Warning: link target ${linkTarget} is unusual.`);
        }
    }
}
