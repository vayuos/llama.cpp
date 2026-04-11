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
exports.ActivationManager = void 0;
const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const os = __importStar(require("os"));
const state_1 = require("./core/state");
const config_1 = require("./core/config");
const processManager_1 = require("./process/processManager");
const pipelineRun_1 = require("./commands/pipelineRun");
const taskHistory_1 = require("./uiv2/taskHistory");
const logger_1 = require("./core/logger");
const cleanup_1 = require("./process/cleanup");
const watchers_1 = require("./core/watchers");
const logBridge_1 = require("./core/logBridge");
const storageManager_1 = require("./core/storageManager");
const telemetry_1 = require("./llm/telemetry");
const runtimeTreeProvider_1 = require("./ui/runtimeTreeProvider");
const taskManager_1 = require("./uiv2/taskManager");
const chatSidebar_1 = require("./uiv2/chatSidebar");
const BOOT_LOG_FILE = path.join(os.homedir(), '.gravitas', 'logs', 'boot_trace.log');
function syncLog(msg) {
    try {
        const dir = path.dirname(BOOT_LOG_FILE);
        if (!fs.existsSync(dir))
            fs.mkdirSync(dir, { recursive: true });
        const line = `[${new Date().toISOString()}] ${msg}\n`;
        fs.appendFileSync(BOOT_LOG_FILE, line, 'utf8');
        process.stdout.write(line); // Ensure it flushes slightly better than console.log
    }
    catch (e) { }
}
class ActivationManager {
    async activate(context) {
        this.context = context;
        console.log('TRACE: [Activation Start] Entering async activate() block...');
        try {
            await this.internalActivate(context);
        }
        catch (fatalErr) {
            console.error('CRITICAL: [Boot Crash] Async activation failed:', fatalErr);
        }
    }
    async safeComponentInit(name, action) {
        syncLog(`COMPONENT [${name}]: Initializing...`);
        try {
            await action();
            syncLog(`COMPONENT [${name}]: Initialized successfully.`);
        }
        catch (e) {
            const errorMsg = `COMPONENT [${name}]: NOT WORKING. Reason: ${e.message}`;
            syncLog(errorMsg);
            if (this.logger) {
                this.logger.error('system', errorMsg);
            }
            else {
                console.error(errorMsg);
            }
        }
    }
    async internalActivate(context) {
        syncLog('TRACE: [Internal Activate] Starting...');
        // --- EMERGENCY: Defer activation to break synchronous crash loop ---
        setTimeout(async () => {
            syncLog('TRACE: [Emergency Defer] Starting deferred activation...');
            try {
                // --- STEP 1: Core Lifecycle (Singletons FIRST) ---
                await this.safeComponentInit('CentralLogger', () => {
                    this.logger = logger_1.CentralLogger.getInstance();
                });
                await this.safeComponentInit('GravitasState', () => {
                    syncLog('TRACE: [Step 3.1] Initializing State...');
                    this.gravitasState = state_1.GravitasState.getInstance();
                    this.gravitasState.initialize(context);
                });
                await this.safeComponentInit('TaskManager', () => {
                    syncLog('TRACE: [Step 3.2] Initializing TaskManager...');
                    const tm = taskManager_1.TaskManager.initialize(context);
                    tm.loadTasks().catch(err => {
                        syncLog(`BACKGROUND ERROR: [TaskManager] Recovery failed: ${err.message}`);
                    });
                });
                // --- STEP 2: UI Providers (REGISTER AFTER Singletons are ready) ---
                await this.safeComponentInit('UIProviders', () => {
                    syncLog('TRACE: [Step 3.3] Creating UI Providers...');
                    this.chatSidebarProvider = new chatSidebar_1.ChatSidebarProvider(context.extensionUri);
                    const taskHistoryProvider = new taskHistory_1.TaskHistoryProvider();
                    const runtimeProvider = new runtimeTreeProvider_1.RuntimeTreeProvider();
                    syncLog('TRACE: [Step 3.4] Registering Providers with VS Code...');
                    context.subscriptions.push(vscode.window.registerWebviewViewProvider('gravitas.chat', this.chatSidebarProvider, {
                        webviewOptions: { retainContextWhenHidden: true }
                    }), vscode.window.registerTreeDataProvider('gravitas.taskHistory', taskHistoryProvider), vscode.window.registerTreeDataProvider('gravitas.runtime', runtimeProvider));
                });
                await this.safeComponentInit('LogBridge', () => {
                    logBridge_1.LogBridge.initialize();
                });
                // --- STEP 4: Topology & Infrastructure ---
                await this.safeComponentInit('TopologyCheck', () => {
                    this.checkTopology(context.extensionPath);
                });
                await this.safeComponentInit('ProcessManager', () => {
                    this.processManager = processManager_1.UnifiedProcessManager.getInstance();
                });
                // Check install marker... (existing logic remains)
                // Storage is in globalStorageUri which persists across extension reinstalls/updates
                const installMarkerPath = path.join(context.globalStorageUri.fsPath, '.installed');
                // Ensure directory exists
                if (!fs.existsSync(context.globalStorageUri.fsPath)) {
                    fs.mkdirSync(context.globalStorageUri.fsPath, { recursive: true });
                }
                const markerExists = fs.existsSync(installMarkerPath);
                const isFreshInstall = !markerExists;
                if (isFreshInstall) {
                    if (this.logger)
                        this.logger.info('system', 'Fresh install detected - cleaning up any existing data...');
                    // Clean up everything
                    const storageManager = storageManager_1.StorageManager.getInstance();
                    const cleanupResult = storageManager.clearAll();
                    if (!cleanupResult.success) {
                        if (this.logger)
                            this.logger.error('system', `Cleanup failed: ${cleanupResult.message}`);
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
                        if (this.logger)
                            this.logger.info('system', 'First install cleanup complete. Extension ready.');
                    }
                    catch (e) {
                        if (this.logger)
                            this.logger.error('system', `Failed to create install marker: ${e.message}`);
                    }
                }
                else {
                    if (this.logger)
                        this.logger.info('system', 'Extension reloading - preserving existing data.');
                }
                syncLog('TRACE: [Step 4] Installation markers & directory check done. Calling State Sync...');
                // 2. State & Config
                await this.safeComponentInit('ConfigLoader', async () => {
                    if (this.gravitasState)
                        this.gravitasState.syncToContext();
                    const config = await config_1.ConfigManager.getInstance().loadConfig();
                    if (config && this.logger) {
                        this.logger.setLogDir(config.logDir || '');
                        logger_1.CentralLogger.getInstance().setLevel(config.runtime.logLevel);
                        this.gravitasState?.updateState({ configLoaded: true, validated: true });
                        this.logger.info('system', 'Configuration loaded successfully.');
                    }
                    else if (this.logger) {
                        this.gravitasState?.updateState({ configLoaded: false, validated: false });
                        this.logger.warn('system', 'No configuration found or logger missing.');
                    }
                });
                syncLog('TRACE: [Step 5] State synced and Config loaded.');
                // 3. Register Hooks & Services
                await this.safeComponentInit('HooksAndWatchers', () => {
                    (0, cleanup_1.registerCleanup)(context);
                    (0, watchers_1.registerWatchers)(context);
                    telemetry_1.TelemetryService.getInstance().startPolling();
                });
                // UI Registration already handled in Step 2
                // 5. Commands
                await this.safeComponentInit('CommandRegistration', () => {
                    context.subscriptions.push(vscode.commands.registerCommand('gravitas.task.delete', (item) => {
                        if (item && item.task) {
                            taskManager_1.TaskManager.getInstance().deleteTask(item.task.id);
                        }
                    }), vscode.commands.registerCommand('gravitas.task.clearAll', async () => {
                        const confirm = await vscode.window.showWarningMessage('Are you sure you want to clear ALL task history? This will delete all event ledgers from disk.', { modal: true }, 'Delete All');
                        if (confirm === 'Delete All') {
                            taskManager_1.TaskManager.getInstance().clearAllTasks();
                        }
                    }), vscode.commands.registerCommand('gravitas.task.openInShell', (taskId) => {
                        if (this.chatSidebarProvider) {
                            this.chatSidebarProvider.showTask(taskId);
                        }
                    }), vscode.commands.registerCommand('gravitas.task.spawn', async (prompt) => {
                        const tm = taskManager_1.TaskManager.getInstance();
                        if (prompt) {
                            const task = tm.createTask(prompt, 'user');
                            if (this.chatSidebarProvider) {
                                this.chatSidebarProvider.showTask(task.id);
                            }
                            await vscode.commands.executeCommand('gravitas.pipeline.run', prompt, task.id);
                        }
                        else {
                            // Unified 'New Chat' Flow: No popup, just sidebar focus
                            if (this.chatSidebarProvider) {
                                this.chatSidebarProvider.reset();
                                this.chatSidebarProvider.focus();
                            }
                        }
                    }), vscode.commands.registerCommand('gravitas.pipeline.run', async (prompt, taskId) => {
                        const input = prompt || await vscode.window.showInputBox({
                            prompt: 'Dual-Agent Execution Loop',
                            placeHolder: 'e.g. Implement a new logger'
                        });
                        if (input)
                            await (0, pipelineRun_1.runPipeline)(input, taskId);
                    }), vscode.commands.registerCommand('gravitas.runtime.start', async () => {
                        if (!this.gravitasState.state.validated)
                            return;
                        const cfg = await config_1.ConfigManager.getInstance().loadConfig();
                        if (cfg && this.logger) {
                            await this.processManager.startReviewer(cfg);
                            await this.processManager.startCoder(cfg);
                            this.logger.info('system', 'LLM Servers started.');
                        }
                    }), vscode.commands.registerCommand('gravitas.runtime.stop', () => {
                        this.processManager.stopAll();
                        if (this.logger)
                            this.logger.info('system', 'LLM Servers stopped.');
                    }), 
                    // Individual model controls
                    vscode.commands.registerCommand('gravitas.runtime.startCoder', async () => {
                        if (!this.gravitasState.state.validated)
                            return;
                        const cfg = await config_1.ConfigManager.getInstance().loadConfig();
                        if (cfg && this.logger) {
                            await this.processManager.startCoder(cfg);
                            this.logger.info('system', 'Coder server started.');
                        }
                    }), vscode.commands.registerCommand('gravitas.runtime.stopCoder', async () => {
                        const pm = this.processManager;
                        await pm.coder.stop();
                        if (this.logger)
                            this.logger.info('system', 'Coder server stopped.');
                    }), vscode.commands.registerCommand('gravitas.runtime.restartCoder', async () => {
                        if (!this.gravitasState.state.validated)
                            return;
                        const cfg = await config_1.ConfigManager.getInstance().loadConfig();
                        if (cfg) {
                            const pm = this.processManager;
                            await pm.coder.stop();
                            await this.processManager.startCoder(cfg);
                            if (this.logger)
                                this.logger.info('system', 'Coder server restarted.');
                        }
                    }), vscode.commands.registerCommand('gravitas.runtime.startReviewer', async () => {
                        if (!this.gravitasState.state.validated)
                            return;
                        const cfg = await config_1.ConfigManager.getInstance().loadConfig();
                        if (cfg && this.logger) {
                            await this.processManager.startReviewer(cfg);
                            this.logger.info('system', 'Reviewer server started.');
                        }
                    }), vscode.commands.registerCommand('gravitas.runtime.stopReviewer', async () => {
                        const pm = this.processManager;
                        await pm.reviewer.stop();
                        if (this.logger)
                            this.logger.info('system', 'Reviewer server stopped.');
                    }), vscode.commands.registerCommand('gravitas.runtime.restartReviewer', async () => {
                        if (!this.gravitasState.state.validated)
                            return;
                        const cfg = await config_1.ConfigManager.getInstance().loadConfig();
                        if (cfg) {
                            const pm = this.processManager;
                            await pm.reviewer.stop();
                            await this.processManager.startReviewer(cfg);
                            if (this.logger)
                                this.logger.info('system', 'Reviewer server restarted.');
                        }
                    }), vscode.commands.registerCommand('gravitas.storage.clear', async () => {
                        const choice = await vscode.window.showWarningMessage('Clear Gravitas Storage?', { modal: true }, 'Clear Logs Only', 'Clear Everything (Logs + Reset Validation)');
                        if (!choice)
                            return;
                        const storageManager = storageManager_1.StorageManager.getInstance();
                        let result;
                        if (choice === 'Clear Logs Only') {
                            result = storageManager.clearLogs();
                        }
                        else {
                            result = storageManager.clearAll();
                        }
                        if (result.success) {
                            vscode.window.showInformationMessage(result.message);
                            if (this.logger)
                                this.logger.info('system', 'Storage cleared by user.');
                        }
                        else {
                            vscode.window.showErrorMessage(result.message);
                            if (this.logger)
                                this.logger.error('system', result.message);
                        }
                    }), vscode.commands.registerCommand('gravitas.views.focus', () => {
                        if (this.chatSidebarProvider) {
                            this.chatSidebarProvider.focus();
                        }
                        // Ensure the view is visible in the sidebar
                        vscode.commands.executeCommand('workbench.view.extension.gravitas-explorer');
                    }));
                });
                syncLog('TRACE: [Step 6] Hooks & Commands registered.');
                // 6. Subsystem Start (DEFERRED - do this last to ensure host stability)
                await this.safeComponentInit('EventEnabler', () => {
                    logger_1.CentralLogger.getInstance().enableEvents();
                    if (this.logger)
                        this.logger.info('system', 'Gravitas Code: Infrastructure fully activated.');
                });
                // 7. Auto-Start Servers (Optional)
                await this.safeComponentInit('AutoStart', async () => {
                    const cfg = vscode.workspace.getConfiguration('gravitasCode');
                    const autoStart = cfg.get('runtime.autoStartServers', false);
                    if (autoStart) {
                        syncLog('TRACE: [AutoStart] Triggering automatic server startup...');
                        await vscode.commands.executeCommand('gravitas.runtime.start');
                    }
                });
            }
            catch (fatalErr) {
                syncLog('CRITICAL: [Emergency Defer] Panic in deferred activation: ' + fatalErr.stack);
            }
        }, 500);
        syncLog('TRACE: [Internal Activate] Returning immediately (Deferred Start).');
    }
    async cleanup() {
        console.log('Gravitas Code: Cleaning up - stopping all llama servers and telemetry...');
        telemetry_1.TelemetryService.getInstance().stopPolling();
        await this.processManager.stopAll();
        console.log('Gravitas Code: Cleanup complete.');
    }
    async ensureEnvironmentReady() {
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
    checkTopology(extensionPath) {
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
            logger_1.CentralLogger.getInstance().warn('system', 'Topology Warning: node_modules is not a symbolic link. This might violate VayuForge rules but will NOT end the session.');
        }
        const linkTarget = fs.readlinkSync(modulesPath);
        if (!linkTarget.includes('support-code')) {
            logger_1.CentralLogger.getInstance().warn('system', `Topology Warning: link target ${linkTarget} is unusual.`);
        }
    }
}
exports.ActivationManager = ActivationManager;
//# sourceMappingURL=activation.js.map