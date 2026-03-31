import * as vscode from 'vscode';
import { UnifiedProcessManager } from './processManager';
import { CentralLogger } from '../core/logger';

export function registerCleanup(context: vscode.ExtensionContext) {
    const logger = CentralLogger.getInstance();

    // Kill processes on VS Code exit
    context.subscriptions.push({
        dispose: () => {
            logger.info('system', 'Extension deactivating, killing LLM processes...');
            UnifiedProcessManager.getInstance().stopAll();
        }
    });

    // Handle terminal closures (if we were using terminals, but we use spawn now)
    // However, if we ever scale to terminals, this is where it goes.
}
