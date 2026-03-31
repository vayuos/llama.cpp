import * as vscode from 'vscode';
import { StateManager, SessionStatus } from '../state/sessionState';

export async function reviewRun(state: StateManager) {
    state.updateStatus(SessionStatus.REVIEWER_RUNNING);
    vscode.window.showInformationMessage('Gravitas Reviewer: Manual review started.');
    // Implementation logic
}
