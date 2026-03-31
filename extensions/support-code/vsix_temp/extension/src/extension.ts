import * as vscode from 'vscode';
import { ActivationManager } from './activation';

const manager = new ActivationManager();

export async function activate(context: vscode.ExtensionContext) {
    await manager.activate(context);
}

export function deactivate() {
    console.log('Gravitas Code is now deactivated.');
}
