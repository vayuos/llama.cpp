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
exports.StorageManager = void 0;
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const os = __importStar(require("os"));
const state_1 = require("./state");
class StorageManager {
    constructor() { }
    static getInstance() {
        if (!StorageManager.instance) {
            StorageManager.instance = new StorageManager();
        }
        return StorageManager.instance;
    }
    /**
     * Clear all log files in ~/.gravitas/logs/
     */
    clearLogs() {
        try {
            const logDir = path.join(os.homedir(), '.gravitas', 'logs');
            if (!fs.existsSync(logDir)) {
                return { success: true, message: 'No logs directory found (already clean).' };
            }
            const files = fs.readdirSync(logDir);
            let deletedCount = 0;
            for (const file of files) {
                const filePath = path.join(logDir, file);
                if (fs.statSync(filePath).isFile()) {
                    fs.unlinkSync(filePath);
                    deletedCount++;
                }
            }
            return {
                success: true,
                message: `Cleared ${deletedCount} log file(s) from ${logDir}`
            };
        }
        catch (error) {
            return {
                success: false,
                message: `Failed to clear logs: ${error}`
            };
        }
    }
    /**
     * Reset validation state
     */
    resetValidation() {
        try {
            const state = state_1.GravitasState.getInstance();
            state.updateState({
                validated: false,
                validationHash: null,
                coderStatus: 'stopped',
                reviewerStatus: 'stopped'
            });
            return {
                success: true,
                message: 'Validation state reset successfully.'
            };
        }
        catch (error) {
            return {
                success: false,
                message: `Failed to reset validation: ${error}`
            };
        }
    }
    /**
     * Clear everything: logs + validation state
     */
    clearAll() {
        const logsResult = this.clearLogs();
        const validationResult = this.resetValidation();
        if (logsResult.success && validationResult.success) {
            return {
                success: true,
                message: `Storage cleared successfully.\n${logsResult.message}\n${validationResult.message}`
            };
        }
        else {
            const errors = [];
            if (!logsResult.success)
                errors.push(logsResult.message);
            if (!validationResult.success)
                errors.push(validationResult.message);
            return {
                success: false,
                message: `Partial failure:\n${errors.join('\n')}`
            };
        }
    }
}
exports.StorageManager = StorageManager;
//# sourceMappingURL=storageManager.js.map