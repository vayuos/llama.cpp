import * as vscode from 'vscode';
import { GravitasState } from './state';
import { ConfigManager } from './config';
import { CentralLogger } from './logger';

export function registerWatchers(context: vscode.ExtensionContext) {
    const logger = CentralLogger.getInstance();
    const state = GravitasState.getInstance();

    // Watch for VS Code configuration changes (if any)
    context.subscriptions.push(vscode.workspace.onDidChangeConfiguration(async e => {
        if (e.affectsConfiguration('gravitasCode')) {
            if (e.affectsConfiguration('gravitasCode.runtime.logLevel')) {
                const config = await ConfigManager.getInstance().loadConfig();
                if (config) {
                    logger.setLevel(config.runtime.logLevel);
                    logger.info('system', `Log level updated to: ${config.runtime.logLevel}`);
                }
            }

            const hasCriticalChange = 
                e.affectsConfiguration('gravitasCode.coder.general.port') ||
                e.affectsConfiguration('gravitasCode.coder.general.host') ||
                e.affectsConfiguration('gravitasCode.reviewer.general.port') ||
                e.affectsConfiguration('gravitasCode.reviewer.general.host') ||
                e.affectsConfiguration('gravitasCode.coder.general.modelPath') ||
                e.affectsConfiguration('gravitasCode.reviewer.general.modelPath');

            if (hasCriticalChange) {
                logger.warn('system', 'Critical configuration change detected. Invalidating validation.');
                state.updateState({ validated: false });
            }
        }
    }));

}
