"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.registerCleanup = registerCleanup;
const logger_1 = require("../core/logger");
function registerCleanup(context) {
    const logger = logger_1.CentralLogger.getInstance();
    // Kill processes on VS Code exit
    context.subscriptions.push({
        dispose: () => {
            // logger.info('system', 'Extension deactivating, killing LLM processes...');
            // UnifiedProcessManager.getInstance().stopAll();
            // Cleanup is now handled in extension.ts deactivate() with validation checks
        }
    });
    // Handle terminal closures (if we were using terminals, but we use spawn now)
    // However, if we ever scale to terminals, this is where it goes.
}
//# sourceMappingURL=cleanup.js.map