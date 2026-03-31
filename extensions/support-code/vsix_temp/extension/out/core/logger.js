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
class CentralLogger {
    constructor() {
        this.logFile = null;
        this._onDidLog = new vscode.EventEmitter();
        this.onDidLog = this._onDidLog.event;
        this.outputChannel = vscode.window.createOutputChannel('Gravitas Logs');
    }
    static getInstance() {
        if (!CentralLogger.instance) {
            CentralLogger.instance = new CentralLogger();
        }
        return CentralLogger.instance;
    }
    setLogDir(logDir) {
        if (!fs.existsSync(logDir)) {
            fs.mkdirSync(logDir, { recursive: true });
        }
        this.logFile = path.join(logDir, `gravitas-${new Date().toISOString().split('T')[0]}.log`);
    }
    log(source, message, level = 'info') {
        const entry = {
            timestamp: new Date().toISOString(),
            source,
            message,
            level
        };
        const formatted = `[${entry.timestamp}] [${entry.source.toUpperCase()}] [${entry.level.toUpperCase()}] ${entry.message}`;
        this.outputChannel.appendLine(formatted);
        if (this.logFile) {
            fs.appendFileSync(this.logFile, formatted + '\n');
        }
        this._onDidLog.fire(entry);
    }
    info(source, message) { this.log(source, message, 'info'); }
    warn(source, message) { this.log(source, message, 'warn'); }
    error(source, message) { this.log(source, message, 'error'); }
}
exports.CentralLogger = CentralLogger;
//# sourceMappingURL=logger.js.map