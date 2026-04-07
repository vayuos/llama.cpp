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
exports.CentralLogger = void 0;
const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const os = __importStar(require("os"));
const LEVEL_WEIGHTS = {
    'debug': 0,
    'info': 1,
    'warn': 2,
    'error': 3
};
/**
 * Centralized Logging Engine for Gravitas.
 * Supports granular levels and dynamic filtering based on user settings.
 */
class CentralLogger {
    constructor() {
        this.outputChannel = null;
        this.logFile = null;
        this.minLevelWeight = 0; // Default to debug until initialized
        this._onDidLog = new vscode.EventEmitter();
        this.onDidLog = this._onDidLog.event;
        this.isLogging = false;
        this.isEnabled = false; // 🛡️ Safe Boot: Events disabled until activate() completes
        // 🛡️ Zero-API Constructor: No VS Code calls here to avoid early boot crashes.
        // Initial setup for early boot logs (pure path logic)
        this.setLogDir(path.join(os.homedir(), '.gravitas', 'logs'));
    }
    getOutputChannel() {
        if (!this.outputChannel) {
            this.outputChannel = vscode.window.createOutputChannel('Gravitas Logs');
        }
        return this.outputChannel;
    }
    static getInstance() {
        if (!CentralLogger.instance) {
            CentralLogger.instance = new CentralLogger();
        }
        return CentralLogger.instance;
    }
    /**
     * Sets the minimum log level to display.
     */
    setLevel(level) {
        this.minLevelWeight = LEVEL_WEIGHTS[level] ?? 0;
    }
    /**
     * Safety: Enables firing of onDidLog events. Should only be called
     * at the end of the activation sequence.
     */
    enableEvents() {
        this.isEnabled = true;
    }
    setLogDir(logDir) {
        if (!fs.existsSync(logDir)) {
            try {
                fs.mkdirSync(logDir, { recursive: true });
            }
            catch (e) {
                console.error('Failed to create log dir:', e);
                return;
            }
        }
        this.logFile = path.join(logDir, `gravitas-${new Date().toISOString().split('T')[0]}.log`);
    }
    log(source, message, level = 'info') {
        const weight = LEVEL_WEIGHTS[level] ?? 1;
        // 🛡️ Loop Prevention: Immediate return if we are already in a logging call
        if (this.isLogging) {
            return;
        }
        // 🛡️ Filtering: Only output if level meets minimum priority
        if (weight < this.minLevelWeight) {
            return;
        }
        this.isLogging = true;
        try {
            const entry = {
                timestamp: new Date().toISOString(),
                source,
                message,
                level
            };
            const formatted = `[${entry.timestamp}] [${entry.source.toUpperCase()}] [${entry.level.toUpperCase()}] ${entry.message}`;
            this.getOutputChannel().appendLine(formatted);
            if (this.logFile) {
                try {
                    fs.appendFileSync(this.logFile, formatted + '\n');
                }
                catch (e) { }
            }
            if (this.isEnabled) {
                this._onDidLog.fire(entry);
            }
        }
        finally {
            this.isLogging = false;
        }
    }
    debug(source, message) { this.log(source, message, 'debug'); }
    info(source, message) { this.log(source, message, 'info'); }
    warn(source, message) { this.log(source, message, 'warn'); }
    error(source, message) { this.log(source, message, 'error'); }
}
exports.CentralLogger = CentralLogger;
//# sourceMappingURL=logger.js.map