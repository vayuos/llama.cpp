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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ProcessManager = exports.ProcessStatus = void 0;
const vscode = __importStar(require("vscode"));
const net = __importStar(require("net"));
const axios_1 = __importDefault(require("axios"));
const serverLauncher_1 = require("./serverLauncher");
const loadConfig_1 = require("../config/loadConfig");
const validateConfig_1 = require("../config/validateConfig");
var ProcessStatus;
(function (ProcessStatus) {
    ProcessStatus["STOPPED"] = "STOPPED";
    ProcessStatus["STARTING"] = "STARTING";
    ProcessStatus["RUNNING"] = "RUNNING";
    ProcessStatus["ERROR"] = "ERROR";
    ProcessStatus["OFFLINE"] = "OFFLINE";
})(ProcessStatus || (exports.ProcessStatus = ProcessStatus = {}));
class ProcessManager {
    constructor() {
        this.statuses = { coder: ProcessStatus.STOPPED, reviewer: ProcessStatus.STOPPED };
    }
    async startAll() {
        const config = (0, loadConfig_1.loadConfig)();
        try {
            (0, validateConfig_1.validateConfig)(config);
        }
        catch (e) {
            vscode.window.showErrorMessage(e.message);
            return;
        }
        const coderPort = parseInt(config.coder.endpoint.split(':')[2]);
        const reviewerPort = parseInt(config.reviewer.endpoint.split(':')[2]);
        if (!(await this.checkPort(coderPort))) {
            vscode.window.showErrorMessage(`Gravitas: Coder port ${coderPort} is already in use.`);
            return;
        }
        if (!(await this.checkPort(reviewerPort))) {
            vscode.window.showErrorMessage(`Gravitas: Reviewer port ${reviewerPort} is already in use.`);
            return;
        }
        this.statuses.coder = ProcessStatus.STARTING;
        this.statuses.reviewer = ProcessStatus.STARTING;
        this.reviewerTerminal = serverLauncher_1.ServerLauncher.launchReviewer(config);
        this.coderTerminal = serverLauncher_1.ServerLauncher.launchCoder(config);
        // Start health polling
        this.pollHealth(config.coder.endpoint, 'coder');
        this.pollHealth(config.reviewer.endpoint, 'reviewer');
    }
    async checkPort(port) {
        return new Promise((resolve) => {
            const server = net.createServer();
            server.once('error', () => resolve(false));
            server.once('listening', () => {
                server.close();
                resolve(true);
            });
            server.listen(port);
        });
    }
    async pollHealth(endpoint, type) {
        const interval = setInterval(async () => {
            if (this.statuses[type] === ProcessStatus.STOPPED) {
                clearInterval(interval);
                return;
            }
            try {
                await axios_1.default.get(`${endpoint}/v1/models`, { timeout: 2000 });
                this.statuses[type] = ProcessStatus.RUNNING;
            }
            catch (e) {
                // If the terminal still exists, it's starting or failed
                const term = type === 'coder' ? this.coderTerminal : this.reviewerTerminal;
                if (term) {
                    this.statuses[type] = ProcessStatus.STARTING;
                }
                else {
                    this.statuses[type] = ProcessStatus.OFFLINE;
                    clearInterval(interval);
                }
            }
        }, 5000);
    }
    stopAll() {
        if (this.coderTerminal) {
            this.coderTerminal.dispose();
            this.coderTerminal = undefined;
        }
        if (this.reviewerTerminal) {
            this.reviewerTerminal.dispose();
            this.reviewerTerminal = undefined;
        }
        this.statuses.coder = ProcessStatus.STOPPED;
        this.statuses.reviewer = ProcessStatus.STOPPED;
    }
    restartAll() {
        this.stopAll();
        setTimeout(() => this.startAll(), 1000);
    }
    handleTerminalClosed(terminal) {
        if (terminal === this.coderTerminal) {
            this.coderTerminal = undefined;
            this.statuses.coder = ProcessStatus.STOPPED;
        }
        else if (terminal === this.reviewerTerminal) {
            this.reviewerTerminal = undefined;
            this.statuses.reviewer = ProcessStatus.STOPPED;
        }
    }
    getStatus() {
        return this.statuses;
    }
}
exports.ProcessManager = ProcessManager;
//# sourceMappingURL=processManager.js.map