import * as vscode from 'vscode';
import { StateManager, SessionStatus } from '../state/sessionState';

export async function coderRun(state: StateManager) {
    const input = await vscode.window.showInputBox({ prompt: 'Coder Task' });
    if (input) {
        state.updateStatus(SessionStatus.CODER_RUNNING);
        vscode.window.showInformationMessage('Gravitas Coder: Manual run started.');
        // Implementation logic
    }
}
