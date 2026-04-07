import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { GravitasState } from './state';
import { CentralLogger } from './logger';
import { TaskManager } from '../uiv2/taskManager';

export class StorageManager {
    private static instance: StorageManager;

    private constructor() { }

    public static getInstance(): StorageManager {
        if (!StorageManager.instance) {
            StorageManager.instance = new StorageManager();
        }
        return StorageManager.instance;
    }

    /**
     * Clear all log files in current log directory
     */
    public clearLogs(): { success: boolean; message: string } {
        try {
            const logger = CentralLogger.getInstance();
            const logDir = path.join(os.homedir(), '.gravitas', 'logs'); 
            
            const wipeDir = (dir: string) => {
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
        } catch (error) {
            return {
                success: false,
                message: `Failed to clear logs: ${error}`
            };
        }
    }

    /**
     * Reset validation state and global state
     */
    public resetValidation(): { success: boolean; message: string } {
        try {
            const state = GravitasState.getInstance();
            state.updateState({
                validated: false,
                validationHash: null,
                coderStatus: 'stopped',
                reviewerStatus: 'stopped'
            });

            CentralLogger.getInstance().info('system', 'StorageManager: Validation state and hardware status reset.');

            return {
                success: true,
                message: 'State reset successfully.'
            };
        } catch (error) {
            return {
                success: false,
                message: `Failed to reset state: ${error}`
            };
        }
    }

    /**
     * Clear everything: logs + validation state + Task Ledgers
     */
    public clearAll(): { success: boolean; message: string } {
        try {
            // 1. Clear System Logs
            const logsResult = this.clearLogs();
            
            // 2. Clear Validation State
            const validationResult = this.resetValidation();

            // 3. Clear Task Ledgers (The "Dummy Chats")
            TaskManager.getInstance().clearAllTasks();

            CentralLogger.getInstance().info('system', 'StorageManager: Deep Wipe (logs, state, tasks) completed.');

            return {
                success: true,
                message: `ABSOLUTE WIPE COMPLETE.\n- Logs: ${logsResult.message}\n- State: ${validationResult.message}\n- Tasks: All ledgers and task history deleted physically.`
            };
        } catch (error) {
            return {
                success: false,
                message: `Deep Wipe failed: ${error}`
            };
        }
    }
}
