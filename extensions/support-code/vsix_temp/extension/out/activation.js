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
const sessionState_1 = require("./state/sessionState");
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
const logsPanel_1 = require("./ui/logsPanel");
class ActivationManager {
    constructor() {
        this.state = new sessionState_1.StateManager();
        this.gravitasState = state_1.GravitasState.getInstance();
        this.processManager = processManager_1.UnifiedProcessManager.getInstance();
    }
    async activate(context) {
        console.log('Gravitas Code: Initializing infrastructure control...');
        // 1. Core Services
        const logger = logger_1.CentralLogger.getInstance();
        const statusBar = statusBar_1.GravitasStatusBar.getInstance();
        // 2. State & Config
        this.gravitasState.syncToContext();
        const config = await config_1.ConfigManager.getInstance().loadConfig();
        if (config) {
            logger.setLogDir(config.logDir);
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
        context.subscriptions.push(vscode.window.registerWebviewViewProvider('gravitas.chatView', panelProvider));
        // 5. Commands
        context.subscriptions.push(vscode.commands.registerCommand('gravitas.setup.open', () => setupPanel_1.SetupPanel.createOrShow(context.extensionUri)), vscode.commands.registerCommand('gravitas.setup.validate', async () => {
            const cfg = await config_1.ConfigManager.getInstance().loadConfig();
            if (!cfg)
                return;
            await validationPanel_1.ValidationPanel.showAndRun(context.extensionUri, cfg);
            statusBar.update();
        }), vscode.commands.registerCommand('gravitas.logs.open', () => logsPanel_1.LogsPanel.createOrShow(context.extensionUri)), vscode.commands.registerCommand('gravitas.pipeline.run', async (prompt) => {
            if (!this.gravitasState.state.validated) {
                vscode.window.showErrorMessage('Gravitas: Invariant Violation: System must be validated.');
                return;
            }
            const input = prompt || await vscode.window.showInputBox({ prompt: 'Task for Dual-Agent Loop' });
            if (input)
                await (0, pipelineRun_1.runPipeline)(input, this.state);
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
        }));
        // Update status bar on any state change
        // In a more complex app, we'd use an event emitter, but direct call suffices here.
        const originalUpdate = this.gravitasState.updateState.bind(this.gravitasState);
        this.gravitasState.updateState = (s) => {
            originalUpdate(s);
            statusBar.update();
        };
    }
}
exports.ActivationManager = ActivationManager;
//# sourceMappingURL=activation.js.map