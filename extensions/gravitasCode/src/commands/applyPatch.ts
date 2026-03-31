import * as vscode from 'vscode';

export async function applyPatch(_patch: string) {
    vscode.window.showInformationMessage('Gravitas: Applying approved patch...');
    // patchApplier logic here
}
