"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.reviewRun = reviewRun;
async function reviewRun(state) {
    state.updateStatus(SessionStatus.REVIEWER_RUNNING);
    vscode.window.showInformationMessage('Gravitas Reviewer: Manual review started.');
    // Implementation logic
}
//# sourceMappingURL=reviewRun.js.map