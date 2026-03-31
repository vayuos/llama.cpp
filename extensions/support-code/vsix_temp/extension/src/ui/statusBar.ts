import * as vscode from 'vscode';
import { GravitasState } from '../core/state';

export class GravitasStatusBar {
    private static instance: GravitasStatusBar;
    private item: vscode.StatusBarItem;

    private constructor() {
        this.item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.update();
        this.item.show();
    }

    public static getInstance(): GravitasStatusBar {
        if (!GravitasStatusBar.instance) {
            GravitasStatusBar.instance = new GravitasStatusBar();
        }
        return GravitasStatusBar.instance;
    }

    public update() {
        const state = GravitasState.getInstance().state;

        if (!state.configLoaded) {
            this.item.text = "$(settings) Gravitas: Setup Needed";
            this.item.color = new vscode.ThemeColor('errorForeground');
            this.item.command = 'gravitas.setup.open';
        } else if (!state.validated) {
            this.item.text = "$(shield) Gravitas: Not Validated";
            this.item.color = "#ebcb8b"; // Warning yellow
            this.item.command = 'gravitas.setup.validate';
        } else {
            this.item.text = "$(check) Gravitas: Ready";
            this.item.color = "#a3be8c"; // Success green
            this.item.command = 'gravitas.setup.open'; // Or logs
        }
    }
}
