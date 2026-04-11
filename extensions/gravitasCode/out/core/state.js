"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.GravitasState = void 0;
const vscode = __importStar(require("vscode"));
class GravitasState {
    constructor() {
        this._state = {
            configLoaded: false,
            validated: false,
            validationHash: null,
            coderStatus: 'stopped',
            reviewerStatus: 'stopped'
        };
    }
    static getInstance() {
        if (!GravitasState.instance) {
            GravitasState.instance = new GravitasState();
        }
        return GravitasState.instance;
    }
    initialize(context) {
        this.context = context;
        // Restore persistent state
        const val = this.context.globalState.get('gravitas.validated', false);
        const hash = this.context.globalState.get('gravitas.validationHash', null);
        this._state.validated = val;
        this._state.validationHash = hash;
        this.syncToContext();
    }
    get state() {
        return { ...this._state };
    }
    updateState(newState) {
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
        const changes = Object.keys(newState).map(k => `${k}: ${prev[k]} -> ${newState[k]}`);
        logger.debug('system', `GravitasState: Updated. Changes: [${changes.join(', ')}]`);
        this.syncToContext();
    }
    syncToContext() {
        vscode.commands.executeCommand('setContext', 'gravitas.configLoaded', this._state.configLoaded);
        vscode.commands.executeCommand('setContext', 'gravitas.validated', this._state.validated);
        vscode.commands.executeCommand('setContext', 'gravitas.coderStatus', this._state.coderStatus);
        vscode.commands.executeCommand('setContext', 'gravitas.reviewerStatus', this._state.reviewerStatus);
    }
}
exports.GravitasState = GravitasState;
//# sourceMappingURL=state.js.map