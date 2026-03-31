import * as vscode from 'vscode';
import { ActivationManager } from './activation';

const manager = new ActivationManager();

export async function activate(context: vscode.ExtensionContext) {
    await manager.activate(context);
}

export async function deactivate() {
    console.log('Gravitas Code deactivating - stopping all LLM servers...');

    try {
        // Always stop servers when VS Code closes
        // This prevents orphaned llama-server processes
        await manager.cleanup();
        console.log('Gravitas Code: All servers stopped successfully.');
    } catch (e: any) {
        console.error('Gravitas Code: Error during cleanup:', e.message);
    }
}
