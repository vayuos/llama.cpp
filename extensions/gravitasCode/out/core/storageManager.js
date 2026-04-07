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
const logger_1 = require("./logger");
const taskManager_1 = require("../uiv2/taskManager");
class StorageManager {
    constructor() { }
    static getInstance() {
        if (!StorageManager.instance) {
            StorageManager.instance = new StorageManager();
        }
        return StorageManager.instance;
    }
    /**
     * Clear all log files in current log directory
     */
    clearLogs() {
        try {
            const logger = logger_1.CentralLogger.getInstance();
            const logDir = path.join(os.homedir(), '.gravitas', 'logs');
            const wipeDir = (dir) => {
                if (!fs.existsSync(dir)) {
                    logger.debug('system', `StorageManager: Directory ${dir} does not exist, skipping wipe.`);
                    return 0;
                }
                const files = fs.readdirSync(dir);
                let count = 0;
                for (const file of files) {
                    const filePath = path.join(dir, file);
                    if (fs.statSync(filePath).isFile()) {
                        fs.unlinkSync(filePath);
                        logger.debug('system', `StorageManager: Deleted file ${filePath}`);
                        count++;
                    }
                }
                return count;
            };
            const deletedLegacy = wipeDir(path.join(os.homedir(), '.gravitas', 'logs'));
            return {
                success: true,
                message: `Cleared ${deletedLegacy} legacy log file(s).`
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
     * Reset validation state and global state
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
            logger_1.CentralLogger.getInstance().info('system', 'StorageManager: Validation state and hardware status reset.');
            return {
                success: true,
                message: 'State reset successfully.'
            };
        }
        catch (error) {
            return {
                success: false,
                message: `Failed to reset state: ${error}`
            };
        }
    }
    /**
     * Clear everything: logs + validation state + Task Ledgers
     */
    clearAll() {
        try {
            // 1. Clear System Logs
            const logsResult = this.clearLogs();
            // 2. Clear Validation State
            const validationResult = this.resetValidation();
            // 3. Clear Task Ledgers (The "Dummy Chats")
            taskManager_1.TaskManager.getInstance().clearAllTasks();
            logger_1.CentralLogger.getInstance().info('system', 'StorageManager: Deep Wipe (logs, state, tasks) completed.');
            return {
                success: true,
                message: `ABSOLUTE WIPE COMPLETE.\n- Logs: ${logsResult.message}\n- State: ${validationResult.message}\n- Tasks: All ledgers and task history deleted physically.`
            };
        }
        catch (error) {
            return {
                success: false,
                message: `Deep Wipe failed: ${error}`
            };
        }
    }
}
exports.StorageManager = StorageManager;
//# sourceMappingURL=storageManager.js.map