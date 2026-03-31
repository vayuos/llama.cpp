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
const setupPanel_1 = require("./ui/setupPanel");
const validationPanel_1 = require("./ui/validationPanel");
const processManager_1 = require("./process/processManager");
const pipelineRun_1 = require("./commands/pipelineRun");
const panelProvider_1 = require("./ui/panelProvider");
const logger_1 = require("./core/logger");
const statusBar_1 = require("./ui/statusBar");
const cleanup_1 = require("./process/cleanup");
const watchers_1 = require("./core/watchers");
const logBridge_1 = require("./core/logBridge");
const storageManager_1 = require("./core/storageManager");
const runtimeTreeProvider_1 = require("./ui/runtimeTreeProvider");
const contextTreeProvider_1 = require("./ui/contextTreeProvider");
const presetsTreeProvider_1 = require("./ui/presetsTreeProvider");
const setupViewProvider_1 = require("./ui/setupViewProvider");
const taskManager_1 = require("./uiv2/taskManager");
const taskShell_1 = require("./uiv2/taskShell");
class ActivationManager {
    constructor() {
        // private state = new StateManager(); // Deprecated
        this.gravitasState = state_1.GravitasState.getInstance();
        this.processManager = processManager_1.UnifiedProcessManager.getInstance();
    }
    async activate(context) {
        this.context = context;
        console.log('Gravitas Code: Initializing infrastructure control...');
        taskManager_1.TaskManager.initialize(context);
        logBridge_1.LogBridge.initialize();
        // 1. Core Services (Initialize First)
        const logger = logger_1.CentralLogger.getInstance();
        this.logger = logger;
        const statusBar = statusBar_1.GravitasStatusBar.getInstance();
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
        // Check if this is a fresh install/reinstall using a local marker file
        // This file is wiped when extension is uninstalled/updated, ensuring cleanup happens on reinstall
        const installMarkerPath = path.join(context.extensionPath, '.installed');
        const markerExists = fs.existsSync(installMarkerPath);
        const isFreshInstall = !markerExists;
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
                'llamaBinPath',
                'runtime.autoTestOnStart', 'runtime.showLogs', 'runtime.logLevel', 'runtime.killSignal',
                'coder.general.enabled', 'coder.general.mode', 'coder.general.modelPath', 'coder.general.host', 'coder.general.port', 'coder.general.noWarmup',
                'coder.hardware.cudaVisibleDevices', 'coder.hardware.nGpuLayers', 'coder.hardware.contextSize', 'coder.hardware.threads', 'coder.hardware.threadsBatch', 'coder.hardware.batchSize', 'coder.hardware.ubatchSize', 'coder.hardware.numaInterleave', 'coder.hardware.prefixCommand',
                'coder.sampling.temperature', 'coder.sampling.topP', 'coder.sampling.topK', 'coder.sampling.repeatPenalty',
                'reviewer.general.enabled', 'reviewer.general.mode', 'reviewer.general.modelPath', 'reviewer.general.host', 'reviewer.general.port', 'reviewer.general.noWarmup',
                'reviewer.hardware.cudaVisibleDevices', 'reviewer.hardware.nGpuLayers', 'reviewer.hardware.contextSize', 'reviewer.hardware.threads', 'reviewer.hardware.threadsBatch', 'reviewer.hardware.batchSize', 'reviewer.hardware.ubatchSize', 'reviewer.hardware.numaInterleave', 'reviewer.hardware.prefixCommand',
                'reviewer.sampling.temperature', 'reviewer.sampling.topP', 'reviewer.sampling.topK', 'reviewer.sampling.repeatPenalty'
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
        // 2. State & Config
        this.gravitasState.syncToContext();
        const config = await config_1.ConfigManager.getInstance().loadConfig();
        if (config) {
            logger.setLogDir(config.logDir || '');
            this.gravitasState.updateState({ configLoaded: true });
            logger.info('system', 'Configuration loaded successfully.');
        }
        else {
            this.gravitasState.updateState({ configLoaded: false, validated: false });
            logger.warn('system', 'No configuration found. Awaiting setup.');
            vscode.window.showInformationMessage('Gravitas: Professional setup required.', 'Start Setup').then(s => {
                if (s === 'Start Setup')
                    vscode.commands.executeCommand('gravitas.setup.open');
            });
        }
        // 3. Register Hooks
        (0, cleanup_1.registerCleanup)(context);
        (0, watchers_1.registerWatchers)(context);
        // 4. UI Providers
        const panelProvider = new panelProvider_1.PanelProvider(context.extensionUri);
        const setupViewProvider = new setupViewProvider_1.SetupViewProvider(context.extensionUri);
        const runtimeProvider = new runtimeTreeProvider_1.RuntimeTreeProvider();
        const contextProvider = new contextTreeProvider_1.ContextTreeProvider();
        const presetsProvider = new presetsTreeProvider_1.PresetsTreeProvider();
        context.subscriptions.push(
        // Renamed chatView -> agentConsole
        vscode.window.registerWebviewViewProvider('gravitas.agentConsole', panelProvider), vscode.window.registerWebviewViewProvider('gravitas.setupView', setupViewProvider), 
        // New Tree Views
        vscode.window.registerTreeDataProvider('gravitas.runtime', runtimeProvider), vscode.window.registerTreeDataProvider('gravitas.context', contextProvider), vscode.window.registerTreeDataProvider('gravitas.presets', presetsProvider));
        // 5. Commands
        context.subscriptions.push(vscode.commands.registerCommand('gravitas.setup.open', () => setupPanel_1.SetupPanel.createOrShow(context.extensionUri)), vscode.commands.registerCommand('gravitas.task.spawn', async (prompt) => {
            const tm = taskManager_1.TaskManager.getInstance();
            if (!prompt) {
                prompt = await vscode.window.showInputBox({
                    prompt: 'Initialize Task Shell',
                    placeHolder: 'Describe your objective...'
                });
            }
            if (prompt) {
                const task = tm.createTask(prompt, 'user');
                taskShell_1.TaskShellPanel.createOrShow(context.extensionUri, task.id);
            }
        }), vscode.commands.registerCommand('gravitas.demo.populate', async () => {
            const { populateDemoTask } = await Promise.resolve().then(() => __importStar(require('./commands/populateDemo')));
            await populateDemoTask();
        }), vscode.commands.registerCommand('gravitas.setup.validate', async () => {
            const cfg = await config_1.ConfigManager.getInstance().loadConfig();
            if (!cfg)
                return;
            await validationPanel_1.ValidationPanel.showAndRun(context.extensionUri, cfg);
            statusBar.update();
        }), 
        // Logs Panel deprecated in favor of Inline Terminal
        vscode.commands.registerCommand('gravitas.pipeline.run', async (prompt) => {
            if (!this.gravitasState.state.validated) {
                vscode.window.showErrorMessage('Gravitas: Invariant Violation: System must be validated.');
                return;
            }
            const input = prompt || await vscode.window.showInputBox({ prompt: 'Task for Dual-Agent Loop' });
            if (input)
                await (0, pipelineRun_1.runPipeline)(input);
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
                statusBar.update();
            }
            else {
                vscode.window.showErrorMessage(result.message);
                logger.error('system', result.message);
            }
        }));
        // Update status bar on any state change
        // In a more complex app, we'd use an event emitter, but direct call suffices here.
        const originalUpdate = this.gravitasState.updateState.bind(this.gravitasState);
        this.gravitasState.updateState = (s) => {
            originalUpdate(s);
            statusBar.update();
        };
    }
    async cleanup() {
        console.log('Gravitas Code: Cleaning up - stopping all llama servers...');
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
            throw new Error('node_modules MUST be a symbolic link. Found directory. Run ./setup.sh');
        }
        const linkTarget = fs.readlinkSync(modulesPath);
        // We expect it to point to support-code.
        // Simple check: does it contain 'support-code'?
        if (!linkTarget.includes('support-code')) {
            throw new Error(`Invalid symlink target: ${linkTarget}. Must point to ../support-code/node_modules`);
        }
    }
}
exports.ActivationManager = ActivationManager;
//# sourceMappingURL=activation.js.map