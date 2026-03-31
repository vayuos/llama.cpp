"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.coderRun = coderRun;
async function coderRun(state) {
    const input = await vscode.window.showInputBox({ prompt: 'Coder Task' });
    if (input) {
        state.updateStatus(SessionStatus.CODER_RUNNING);
        vscode.window.showInformationMessage('Gravitas Coder: Manual run started.');
        // Implementation logic
    }
}
//# sourceMappingURL=coderRun.js.map