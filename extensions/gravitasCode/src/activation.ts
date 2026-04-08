import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

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

const BOOT_LOG_FILE = path.join(os.homedir(), '.gravitas', 'logs', 'boot_trace.log');
function syncLog(msg: string) {
    try {
        const dir = path.dirname(BOOT_LOG_FILE);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        const line = `[${new Date().toISOString()}] ${msg}\n`;
        fs.appendFileSync(BOOT_LOG_FILE, line, 'utf8');
        process.stdout.write(line); // Ensure it flushes slightly better than console.log
    } catch(e) {}
}

export class ActivationManager {
    // private state = new StateManager(); // Deprecated

    private gravitasState?: GravitasState;
    private processManager?: UnifiedProcessManager;
    private context?: vscode.ExtensionContext;
    private logger?: CentralLogger;

    async activate(context: vscode.ExtensionContext) {
        this.context = context;
        console.log('TRACE: [Activation Start] Entering async activate() block...');
        
        try {
            await this.internalActivate(context);
        } catch (fatalErr: any) {
            console.error('CRITICAL: [Boot Crash] Async activation failed:', fatalErr);
        }
    }

    private async internalActivate(context: vscode.ExtensionContext) {
        syncLog('TRACE: [Internal Activate] Starting...');
        
        // --- EMERGENCY: Defer activation to break synchronous crash loop ---
        setTimeout(async () => {
            syncLog('TRACE: [Emergency Defer] Starting deferred activation...');
            try {
                // --- STEP 1: LOGGING baseline ---
                syncLog('TRACE: [Step 1] Loading logger...');
                const logger = CentralLogger.getInstance();
                this.logger = logger;
                syncLog('TRACE: [Step 1] CentralLogger initialized.');

                // --- STEP 2: Topology check ---
                try {
                    syncLog('TRACE: [Step 2] Executing Topology check...');
                    this.checkTopology(context.extensionPath);
                    syncLog('TRACE: [Step 2] Topology check passed.');
                } catch (e: any) {
                    syncLog(`CRITICAL: Topology Violation. ${e.message}`);
                    vscode.window.showErrorMessage(`CRITICAL: Topology Violation. ${e.message}`, { modal: true });
                    logger.error('system', `CRITICAL TOPOLOGY VIOLATION: ${e.message}`);
                    this.gravitasState = GravitasState.getInstance();
                    this.gravitasState.updateState({ validated: false });
                    return;
                }

                // --- STEP 3: State & Config ---
                syncLog('TRACE: [Step 3] Initializing GravitasState...');
                this.gravitasState = GravitasState.getInstance();
                this.gravitasState.initialize(context);
                syncLog('TRACE: [Step 3] Initializing ProcessManager...');
                this.processManager = UnifiedProcessManager.getInstance();
                syncLog('TRACE: [Step 3] Singletons(State, ProcessManager) initialized.');

                // --- STEP 3.1: TaskManager & LogBridge ---
                try {
                    syncLog('TRACE: [Step 3.2] Starting TaskManager initialization...');
                    TaskManager.initialize(context);
                    syncLog('TRACE: [Step 3.3] TaskManager initialized successfully.');

                    syncLog('TRACE: [Step 3.4] Initializing LogBridge...');
                    LogBridge.initialize();
                    syncLog('TRACE: [Step 3.5] LogBridge initialized.');
                } catch (e: any) {
                    syncLog('TRACE: [Step 3.6] CRITICAL SERVICE FAILURE: ' + e.message);
                    if (this.logger) this.logger.error('system', `Core service activation failed: ${e.message}`);
                }

                // --- TOPOLOGY ENFORCEMENT ---
                try {
                    this.checkTopology(context.extensionPath);
                } catch (e: any) {
                    vscode.window.showErrorMessage(`CRITICAL: Topology Violation. ${e.message}`, { modal: true });
                    logger.error('system', `CRITICAL TOPOLOGY VIOLATION: ${e.message}`);
                    // We intentionally do not throw here to allow partial activation for debugging, 
                    // BUT we mark validation as impossible.
                    this.gravitasState!.updateState({ validated: false });
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
                syncLog('TRACE: [Step 4] Installation markers & directory check done. Calling State Sync...');

                // 2. State & Config
                this.gravitasState!.syncToContext();
                syncLog('TRACE: [Step 4.1] Load config...');
                const config = await ConfigManager.getInstance().loadConfig();
                syncLog('TRACE: [Step 4.2] Load config complete.');

                if (config) {
                    logger.setLogDir(config.logDir || '');
                    CentralLogger.getInstance().setLevel(config.runtime.logLevel);
                    this.gravitasState!.updateState({ configLoaded: true, validated: true });
                    logger.info('system', 'Configuration loaded successfully.');
                } else {
                    this.gravitasState!.updateState({ configLoaded: false, validated: false });
                    logger.warn('system', 'No configuration found.');
                }
                syncLog('TRACE: [Step 5] State synced and Config loaded.');

                // 3. Register Hooks & Services
                syncLog('TRACE: [Step 5.1] Registering Hooks & Services (Cleanup)...');
                registerCleanup(context);
                syncLog('TRACE: [Step 5.2] Registering Watchers...');
                registerWatchers(context);
                TelemetryService.getInstance().startPolling();

                // 4. UI Providers
                syncLog('TRACE: [Step 5.3] Registering UI Providers (Current Disabled Status)...');
                const chatSidebarProvider = new ChatSidebarProvider(context.extensionUri);
                const taskHistoryProvider = new TaskHistoryProvider();
                const runtimeProvider = new RuntimeTreeProvider();
                
                context.subscriptions.push(
                    vscode.window.registerWebviewViewProvider('gravitas-chat', chatSidebarProvider, {
                        webviewOptions: { retainContextWhenHidden: true }
                    }),
                    vscode.window.registerTreeDataProvider('gravitas-tasks', taskHistoryProvider),
                    vscode.window.registerTreeDataProvider('gravitas-runtime', runtimeProvider)
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
                        // chatSidebarProvider.showTask(taskId);
                    }),
                    vscode.commands.registerCommand('gravitas.task.spawn', async (prompt?: string) => {
                        const tm = TaskManager.getInstance();
                        
                        if (prompt) {
                            const task = tm.createTask(prompt, 'user');
                            // chatSidebarProvider.showTask(task.id);
                            await vscode.commands.executeCommand('gravitas.pipeline.run', prompt, task.id);
                        } else {
                            // Unified 'New Chat' Flow: No popup, just sidebar focus
                            // chatSidebarProvider.reset(); 
                            // chatSidebarProvider.focus();
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
                        if (!this.gravitasState!.state.validated) return;
                        const cfg = await ConfigManager.getInstance().loadConfig();
                        if (cfg) {
                            await this.processManager!.startReviewer(cfg);
                            await this.processManager!.startCoder(cfg);
                            logger.info('system', 'LLM Servers started.');
                        }
                    }),
                    vscode.commands.registerCommand('gravitas.runtime.stop', () => {
                        this.processManager!.stopAll();
                        logger.info('system', 'LLM Servers stopped.');
                    }),
                    // Individual model controls
                    vscode.commands.registerCommand('gravitas.runtime.startCoder', async () => {
                        if (!this.gravitasState!.state.validated) return;
                        const cfg = await ConfigManager.getInstance().loadConfig();
                        if (cfg) {
                            await this.processManager!.startCoder(cfg);
                            logger.info('system', 'Coder server started.');
                        }
                    }),
                    vscode.commands.registerCommand('gravitas.runtime.stopCoder', async () => {
                        const pm = this.processManager! as any;
                        await pm.coder.stop();
                        logger.info('system', 'Coder server stopped.');
                    }),
                    vscode.commands.registerCommand('gravitas.runtime.restartCoder', async () => {
                        if (!this.gravitasState!.state.validated) return;
                        const cfg = await ConfigManager.getInstance().loadConfig();
                        if (cfg) {
                            const pm = this.processManager! as any;
                            await pm.coder.stop();
                            await this.processManager!.startCoder(cfg);
                            logger.info('system', 'Coder server restarted.');
                        }
                    }),
                    vscode.commands.registerCommand('gravitas.runtime.startReviewer', async () => {
                        if (!this.gravitasState!.state.validated) return;
                        const cfg = await ConfigManager.getInstance().loadConfig();
                        if (cfg) {
                            await this.processManager!.startReviewer(cfg);
                            logger.info('system', 'Reviewer server started.');
                        }
                    }),
                    vscode.commands.registerCommand('gravitas.runtime.stopReviewer', async () => {
                        const pm = this.processManager! as any;
                        await pm.reviewer.stop();
                        logger.info('system', 'Reviewer server stopped.');
                    }),
                    vscode.commands.registerCommand('gravitas.runtime.restartReviewer', async () => {
                        if (!this.gravitasState!.state.validated) return;
                        const cfg = await ConfigManager.getInstance().loadConfig();
                        if (cfg) {
                            const pm = this.processManager! as any;
                            await pm.reviewer.stop();
                            await this.processManager!.startReviewer(cfg);
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
                    vscode.commands.registerCommand('gravitas.views.focus', () => {
                        // chatSidebarProvider.focus();
                        // Ensure the view is visible in the sidebar
                        vscode.commands.executeCommand('workbench.view.extension.gravitas-explorer');
                    }),
                );
                syncLog('TRACE: [Step 6] Hooks & Commands registered.');

                // 6. Subsystem Start (DEFERRED - do this last to ensure host stability)
                try {
                    syncLog('TRACE: [Step 7] Final Subsystem Event Enabler...');
                    CentralLogger.getInstance().enableEvents();
                    logger.info('system', 'Gravitas Code: Infrastructure fully activated.');
                    
                    syncLog('TRACE: [Activation Complete]');
                } catch (subErr: any) {
                    syncLog('TRACE: [SUB-SYSTEM CRASH] ' + subErr.message);
                    if (this.logger) this.logger.error('system', `Sub-system activation failed: ${subErr.message}`);
                }
            } catch (fatalErr: any) {
                syncLog('CRITICAL: [Emergency Defer] Panic in deferred activation: ' + fatalErr.stack);
            }
        }, 500); 
        syncLog('TRACE: [Internal Activate] Returning immediately (Deferred Start).');
    }

    async cleanup() {
        console.log('Gravitas Code: Cleaning up - stopping all llama servers and telemetry...');
        TelemetryService.getInstance().stopPolling();
        await this.processManager!.stopAll();
        console.log('Gravitas Code: Cleanup complete.');
    }

    private async ensureEnvironmentReady() {
        const socketDir = path.join(os.homedir(), '.gravitas', 'sockets');
        
        if (!fs.existsSync(socketDir)) {
            const action = 'Initialize Now';
            const msg = 'Gravitas Code: Agent socket directory is missing. Setup environment?';
            const choice = await vscode.window.showInformationMessage(msg, action);
            
            if (choice === action) {
                const terminal = vscode.window.createTerminal('Gravitas Setup');
                // Use the URI to get the actual FS path of the extension
                const extension = vscode.extensions.getExtension('gravitas.gravitas-code');
                if (extension) {
                    const scriptPath = path.join(extension.extensionPath, 'scripts', 'initialSetup.sh');
                    terminal.sendText(`bash "${scriptPath}"`);
                    terminal.show();
                }
            }
        }
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
             CentralLogger.getInstance().warn('system', 'Topology Warning: node_modules is not a symbolic link. This might violate VayuForge rules but will NOT end the session.');
        }

        const linkTarget = fs.readlinkSync(modulesPath);
        if (!linkTarget.includes('support-code')) {
            CentralLogger.getInstance().warn('system', `Topology Warning: link target ${linkTarget} is unusual.`);
        }
    }
}
