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
            logger.debug('system', 'watchers: VS Code configuration "gravitasCode" changed.');
            
            if (e.affectsConfiguration('gravitasCode.runtime.logLevel')) {
                const config = await ConfigManager.getInstance().loadConfig();
                if (config) {
                    logger.setLevel(config.runtime.logLevel);
                    logger.info('system', `watchers: Log level updated to: ${config.runtime.logLevel}`);
                }
            }

            const criticalKeys = [
                'gravitasCode.coder.general.port',
                'gravitasCode.coder.general.host',
                'gravitasCode.reviewer.general.port',
                'gravitasCode.reviewer.general.host',
                'gravitasCode.coder.general.modelPath',
                'gravitasCode.reviewer.general.modelPath'
            ];

            const changedCritical = criticalKeys.filter(k => e.affectsConfiguration(k));

            if (changedCritical.length > 0) {
                logger.warn('system', `watchers: Critical changes detected in: ${changedCritical.join(', ')}. Invalidating validation.`);
                state.updateState({ validated: false });
            }
        }
    }));

}
