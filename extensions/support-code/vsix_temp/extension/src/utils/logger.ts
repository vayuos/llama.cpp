import * as vscode from 'vscode';

export class Logger {
    private static output = vscode.window.createOutputChannel('Gravitas');

    static info(message: string) {
        this.output.appendLine(`[INFO] ${message}`);
    }

    static warn(message: string) {
        this.output.appendLine(`[WARN] ${message}`);
    }

    static error(message: string) {
        this.output.appendLine(`[ERROR] ${message}`);
    }
}
