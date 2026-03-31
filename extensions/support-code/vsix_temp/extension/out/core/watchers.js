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
exports.registerWatchers = registerWatchers;
const vscode = __importStar(require("vscode"));
const state_1 = require("./state");
const config_1 = require("./config");
const logger_1 = require("./logger");
function registerWatchers(context) {
    const logger = logger_1.CentralLogger.getInstance();
    const state = state_1.GravitasState.getInstance();
    // Watch for VS Code configuration changes (if any)
    context.subscriptions.push(vscode.workspace.onDidChangeConfiguration(e => {
        if (e.affectsConfiguration('gravitas')) {
            logger.warn('system', 'Configuration change detected. Invalidating validation.');
            state.updateState({ validated: false });
        }
    }));
    // Watch for manual edits to .gravitas/config.json
    const configPath = config_1.ConfigManager.getInstance().getConfigPath();
    const watcher = vscode.workspace.createFileSystemWatcher(configPath);
    watcher.onDidChange(() => {
        logger.warn('system', 'Manual config.json edit detected. Invalidating validation.');
        state.updateState({ validated: false });
    });
    context.subscriptions.push(watcher);
}
//# sourceMappingURL=watchers.js.map