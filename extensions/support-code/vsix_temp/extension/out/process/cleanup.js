"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.registerCleanup = registerCleanup;
const processManager_1 = require("./processManager");
const logger_1 = require("../core/logger");
function registerCleanup(context) {
    const logger = logger_1.CentralLogger.getInstance();
    // Kill processes on VS Code exit
    context.subscriptions.push({
        dispose: () => {
            logger.info('system', 'Extension deactivating, killing LLM processes...');
            processManager_1.UnifiedProcessManager.getInstance().stopAll();
        }
    });
    // Handle terminal closures (if we were using terminals, but we use spawn now)
    // However, if we ever scale to terminals, this is where it goes.
}
//# sourceMappingURL=cleanup.js.map