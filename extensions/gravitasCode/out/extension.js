"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
console.log('BOOT: [Bundle Evaluation Success] - Entering extension.ts');
const activation_1 = require("./activation");
let manager;
async function activate(context) {
    manager = new activation_1.ActivationManager();
    await manager.activate(context);
}
async function deactivate() {
    console.log('Gravitas Code deactivating - stopping all LLM servers...');
    try {
        if (manager) {
            // Always stop servers when VS Code closes
            // This prevents orphaned llama-server processes
            await manager.cleanup();
            console.log('Gravitas Code: All servers stopped successfully.');
        }
        else {
            console.log('Gravitas Code: Manager not initialized. Skipping cleanup.');
        }
    }
    catch (e) {
        console.error('Gravitas Code: Error during cleanup:', e.message);
    }
}
//# sourceMappingURL=extension.js.map