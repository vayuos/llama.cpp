import * as vscode from 'vscode';
import { GravitasState } from './state';
import { ConfigManager } from './config';
import { CentralLogger } from './logger';

export function registerWatchers(context: vscode.ExtensionContext) {
    const logger = CentralLogger.getInstance();
    const state = GravitasState.getInstance();

    // Watch for VS Code configuration changes (if any)
    context.subscriptions.push(vscode.workspace.onDidChangeConfiguration(e => {
        if (e.affectsConfiguration('gravitas')) {
            logger.warn('system', 'Configuration change detected. Invalidating validation.');
            state.updateState({ validated: false });
        }
    }));

    // Watch for manual edits to .gravitas/config.json
    const configPath = ConfigManager.getInstance().getConfigPath();
    const watcher = vscode.workspace.createFileSystemWatcher(configPath);

    watcher.onDidChange(() => {
        logger.warn('system', 'Manual config.json edit detected. Invalidating validation.');
        state.updateState({ validated: false });
    });

    context.subscriptions.push(watcher);
}
