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
            // Get log dir from CentralLogger or fallback
            const logDir = path.join(os.homedir(), '.gravitas', 'logs'); 
            // Better: We should use the one from config, but for now let's wipe both
            
            const wipeDir = (dir: string) => {
                if (!fs.existsSync(dir)) return 0;
                const files = fs.readdirSync(dir);
                let count = 0;
                for (const file of files) {
                    const filePath = path.join(dir, file);
                    if (fs.statSync(filePath).isFile()) {
                        fs.unlinkSync(filePath);
                        count++;
                    }
                }
                return count;
            };

            const deletedLegacy = wipeDir(path.join(os.homedir(), '.gravitas', 'logs'));
            // Also wipe global storage logs if any
            
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
