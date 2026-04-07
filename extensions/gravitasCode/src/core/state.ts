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
    private context: vscode.ExtensionContext | undefined;
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

    public initialize(context: vscode.ExtensionContext) {
        this.context = context;
        // Restore persistent state
        const val = this.context.globalState.get<boolean>('gravitas.validated', false);
        const hash = this.context.globalState.get<string | null>('gravitas.validationHash', null);
        
        this._state.validated = val;
        this._state.validationHash = hash;
        
        this.syncToContext();
    }

    public get state(): IGravitasState {
        return { ...this._state };
    }

    public updateState(newState: Partial<IGravitasState>) {
        const logger = require('./logger').CentralLogger.getInstance();
        const prev = { ...this._state };
        this._state = { ...this._state, ...newState };
        
        // Persist critical flags
        if (this.context) {
            if (newState.validated !== undefined) {
                this.context.globalState.update('gravitas.validated', newState.validated);
            }
            if (newState.validationHash !== undefined) {
                this.context.globalState.update('gravitas.validationHash', newState.validationHash);
            }
        }

        // Detailed diff logging
        const changes = Object.keys(newState).map(k => `${k}: ${prev[k as keyof IGravitasState]} -> ${newState[k as keyof IGravitasState]}`);
        logger.debug('system', `GravitasState: Updated. Changes: [${changes.join(', ')}]`);
        
        this.syncToContext();
    }

    public syncToContext() {
        vscode.commands.executeCommand('setContext', 'gravitas.configLoaded', this._state.configLoaded);
        vscode.commands.executeCommand('setContext', 'gravitas.validated', this._state.validated);
        vscode.commands.executeCommand('setContext', 'gravitas.coderStatus', this._state.coderStatus);
        vscode.commands.executeCommand('setContext', 'gravitas.reviewerStatus', this._state.reviewerStatus);
    }
}
