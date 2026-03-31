"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const activation_1 = require("./activation");
const manager = new activation_1.ActivationManager();
async function activate(context) {
    await manager.activate(context);
}
function deactivate() {
    console.log('Gravitas Code is now deactivated.');
}
//# sourceMappingURL=extension.js.map