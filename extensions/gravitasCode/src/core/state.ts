import * as vscode from 'vscode';

export type ProcessStatus = 'stopped' | 'running' | 'failed' | 'starting';

export interface IGravitasState {
    configLoaded: boolean;
    validated: boolean;
    validationHash: string | null;
    coderStatus: ProcessStatus;
    reviewerStatus: ProcessStatus;
}

export class GravitasState {
    private static instance: GravitasState;
    private _state: IGravitasState = {
        configLoaded: false,
        validated: false,
        validationHash: null,
        coderStatus: 'stopped',
        reviewerStatus: 'stopped'
    };

    private constructor() { }

    public static getInstance(): GravitasState {
        if (!GravitasState.instance) {
            GravitasState.instance = new GravitasState();
        }
        return GravitasState.instance;
    }

    public get state(): IGravitasState {
        return { ...this._state };
    }

    public updateState(newState: Partial<IGravitasState>) {
        this._state = { ...this._state, ...newState };
        this.syncToContext();
    }

    public syncToContext() {
        vscode.commands.executeCommand('setContext', 'gravitas.configLoaded', this._state.configLoaded);
        vscode.commands.executeCommand('setContext', 'gravitas.validated', this._state.validated);
        vscode.commands.executeCommand('setContext', 'gravitas.coderStatus', this._state.coderStatus);
        vscode.commands.executeCommand('setContext', 'gravitas.reviewerStatus', this._state.reviewerStatus);
    }
}
