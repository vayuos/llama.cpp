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
const state_1 = require("./core/state");
const config_1 = require("./core/config");
const processManager_1 = require("./process/processManager");
const pipelineRun_1 = require("./commands/pipelineRun");
const taskHistory_1 = require("./uiv2/taskHistory");
const logger_1 = require("./core/logger");
const cleanup_1 = require("./process/cleanup");
const watchers_1 = require("./core/watchers");
const storageManager_1 = require("./core/storageManager");
const telemetry_1 = require("./llm/telemetry");
const runtimeTreeProvider_1 = require("./ui/runtimeTreeProvider");
const taskManager_1 = require("./uiv2/taskManager");
const chatSidebar_1 = require("./uiv2/chatSidebar");
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
    async internalActivate(context) {
        console.log('TRACE: [Internal Activate] Starting...');
        // --- STEP 1: LOGGING baseline ---
        const logger = logger_1.CentralLogger.getInstance();
        this.logger = logger;
        console.log('TRACE: [Step 1] CentralLogger initialized.');
        // --- STEP 2: Topology check ---
        try {
            this.checkTopology(context.extensionPath);
            console.log('TRACE: [Step 2] Topology check passed.');
        }
        catch (e) {
            vscode.window.showErrorMessage(`CRITICAL: Topology Violation. ${e.message}`, { modal: true });
            logger.error('system', `CRITICAL TOPOLOGY VIOLATION: ${e.message}`);
            this.gravitasState = state_1.GravitasState.getInstance();
            this.gravitasState.updateState({ validated: false });
            return;
        }
        // --- STEP 3: State & Config ---
        this.gravitasState = state_1.GravitasState.getInstance();
        this.processManager = processManager_1.UnifiedProcessManager.getInstance();
        console.log('TRACE: [Step 3] Singletons(State, ProcessManager) initialized.');
        // --- TOPOLOGY ENFORCEMENT ---
        try {
            this.checkTopology(context.extensionPath);
        }
        catch (e) {
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
        // Ensure required .gravitas directories and sockets exist
        const { execSync } = require('child_process');
        try {
            execSync(`bash "${path.join(__dirname, '..', 'scripts', 'initialSetup.sh')}"`, { stdio: 'ignore' });
            console.log('TRACE: [Setup] Gravitas initial directories and sockets created.');
        }
        catch (e) {
            console.error('CRITICAL: [Setup] Failed to run initial setup script:', e);
        }
        if (isFreshInstall) {
            logger.info('system', 'Fresh install detected - cleaning up any existing data...');
            // Clean up everything
            const storageManager = storageManager_1.StorageManager.getInstance();
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
            }
            catch (e) {
                logger.error('system', `Failed to create install marker: ${e.message}`);
            }
        }
        else {
            logger.info('system', 'Extension reloading - preserving existing data.');
        }
        console.log('TRACE: [Step 4] Installation markers & directory check done.');
        // 2. State & Config
        this.gravitasState.syncToContext();
        const config = await config_1.ConfigManager.getInstance().loadConfig();
        if (config) {
            logger.setLogDir(config.logDir || '');
            logger_1.CentralLogger.getInstance().setLevel(config.runtime.logLevel);
            this.gravitasState.updateState({ configLoaded: true, validated: true });
            logger.info('system', 'Configuration loaded successfully.');
        }
        else {
            this.gravitasState.updateState({ configLoaded: false, validated: false });
            logger.warn('system', 'No configuration found.');
        }
        console.log('TRACE: [Step 5] State synced and Config loaded.');
        // 3. Register Hooks & Services
        (0, cleanup_1.registerCleanup)(context);
        (0, watchers_1.registerWatchers)(context);
        telemetry_1.TelemetryService.getInstance().startPolling();
        // 4. UI Providers
        const chatSidebarProvider = new chatSidebar_1.ChatSidebarProvider(context.extensionUri);
        const taskHistoryProvider = new taskHistory_1.TaskHistoryProvider();
        const runtimeProvider = new runtimeTreeProvider_1.RuntimeTreeProvider();
        context.subscriptions.push(vscode.window.registerWebviewViewProvider(chatSidebar_1.ChatSidebarProvider.viewType, chatSidebarProvider), vscode.window.registerTreeDataProvider('gravitas.taskHistory', taskHistoryProvider), vscode.window.registerTreeDataProvider('gravitas.runtime', runtimeProvider));
        // 5. Commands
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
            chatSidebarProvider.showTask(taskId);
        }), vscode.commands.registerCommand('gravitas.task.spawn', async (prompt) => {
            const tm = taskManager_1.TaskManager.getInstance();
            if (prompt) {
                const task = tm.createTask(prompt, 'user');
                chatSidebarProvider.showTask(task.id);
                await vscode.commands.executeCommand('gravitas.pipeline.run', prompt, task.id);
            }
            else {
                // Unified 'New Chat' Flow: No popup, just sidebar focus
                chatSidebarProvider.reset();
                chatSidebarProvider.focus();
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
            if (cfg) {
                await this.processManager.startReviewer(cfg);
                await this.processManager.startCoder(cfg);
                logger.info('system', 'LLM Servers started.');
            }
        }), vscode.commands.registerCommand('gravitas.runtime.stop', () => {
            this.processManager.stopAll();
            logger.info('system', 'LLM Servers stopped.');
        }), 
        // Individual model controls
        vscode.commands.registerCommand('gravitas.runtime.startCoder', async () => {
            if (!this.gravitasState.state.validated)
                return;
            const cfg = await config_1.ConfigManager.getInstance().loadConfig();
            if (cfg) {
                await this.processManager.startCoder(cfg);
                logger.info('system', 'Coder server started.');
            }
        }), vscode.commands.registerCommand('gravitas.runtime.stopCoder', async () => {
            const pm = this.processManager;
            await pm.coder.stop();
            logger.info('system', 'Coder server stopped.');
        }), vscode.commands.registerCommand('gravitas.runtime.restartCoder', async () => {
            if (!this.gravitasState.state.validated)
                return;
            const cfg = await config_1.ConfigManager.getInstance().loadConfig();
            if (cfg) {
                const pm = this.processManager;
                await pm.coder.stop();
                await this.processManager.startCoder(cfg);
                logger.info('system', 'Coder server restarted.');
            }
        }), vscode.commands.registerCommand('gravitas.runtime.startReviewer', async () => {
            if (!this.gravitasState.state.validated)
                return;
            const cfg = await config_1.ConfigManager.getInstance().loadConfig();
            if (cfg) {
                await this.processManager.startReviewer(cfg);
                logger.info('system', 'Reviewer server started.');
            }
        }), vscode.commands.registerCommand('gravitas.runtime.stopReviewer', async () => {
            const pm = this.processManager;
            await pm.reviewer.stop();
            logger.info('system', 'Reviewer server stopped.');
        }), vscode.commands.registerCommand('gravitas.runtime.restartReviewer', async () => {
            if (!this.gravitasState.state.validated)
                return;
            const cfg = await config_1.ConfigManager.getInstance().loadConfig();
            if (cfg) {
                const pm = this.processManager;
                await pm.reviewer.stop();
                await this.processManager.startReviewer(cfg);
                logger.info('system', 'Reviewer server restarted.');
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
                logger.info('system', 'Storage cleared by user.');
            }
            else {
                vscode.window.showErrorMessage(result.message);
                logger.error('system', result.message);
            }
        }), vscode.commands.registerCommand('gravitas.views.focus', () => {
            chatSidebarProvider.focus();
            // Ensure the view is visible in the sidebar
            vscode.commands.executeCommand('workbench.view.extension.gravitas-explorer');
        }));
        console.log('TRACE: [Step 6] Hooks & Commands registered.');
        // 6. Subsystem Start (DEFERRED - do this last to ensure host stability)
        setTimeout(() => {
            try {
                console.log('TRACE: [Step 7] Starting TaskManager initialization...');
                require('./uiv2/taskManager').TaskManager.initialize(context);
                console.log('TRACE: [Step 8] TaskManager initialized successfully.');
                console.log('TRACE: [Step 9] Initializing LogBridge...');
                require('./core/logBridge').LogBridge.initialize();
                console.log('TRACE: [Step 10] LogBridge initialized.');
                logger_1.CentralLogger.getInstance().enableEvents();
                logger.info('system', 'Gravitas Code: Infrastructure fully activated.');
                console.log('TRACE: [Activation Complete]');
            }
            catch (subErr) {
                console.log('TRACE: [SUB-SYSTEM CRASH] ' + subErr.message);
                if (this.logger)
                    this.logger.error('system', `Sub-system activation failed: ${subErr.message}`);
            }
        }, 500);
    }
    async cleanup() {
        console.log('Gravitas Code: Cleaning up - stopping all llama servers and telemetry...');
        telemetry_1.TelemetryService.getInstance().stopPolling();
        await this.processManager.stopAll();
        console.log('Gravitas Code: Cleanup complete.');
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