"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const activation_1 = require("./activation");
const manager = new activation_1.ActivationManager();
async function activate(context) {
    await manager.activate(context);
}
async function deactivate() {
    console.log('Gravitas Code deactivating - stopping all LLM servers...');
    try {
        // Always stop servers when VS Code closes
        // This prevents orphaned llama-server processes
        await manager.cleanup();
        console.log('Gravitas Code: All servers stopped successfully.');
    }
    catch (e) {
        console.error('Gravitas Code: Error during cleanup:', e.message);
    }
}
//# sourceMappingURL=extension.js.map