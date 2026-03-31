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
exports.ServerLauncher = void 0;
const vscode = __importStar(require("vscode"));
class ServerLauncher {
    static launchCoder(config) {
        const name = 'Gravitas: Coder';
        let terminal = vscode.window.terminals.find(t => t.name === name);
        if (terminal)
            terminal.dispose();
        terminal = vscode.window.createTerminal(name);
        const c = config.coder;
        const cmd = [
            `CUDA_VISIBLE_DEVICES=${c.cudaDevices}`,
            c.binaryPath,
            `-m ${c.modelPath}`,
            `--host ${c.endpoint.split(':')[1].replace('//', '')}`,
            `--port ${c.endpoint.split(':')[2]}`,
            `-ngl ${c.gpuLayers}`,
            `-c ${c.contextSize}`,
            `--threads ${c.threads}`
        ].join(' ');
        terminal.show();
        terminal.sendText(cmd);
        return terminal;
    }
    static launchReviewer(config) {
        const name = 'Gravitas: Reviewer';
        let terminal = vscode.window.terminals.find(t => t.name === name);
        if (terminal)
            terminal.dispose();
        terminal = vscode.window.createTerminal(name);
        const r = config.reviewer;
        const cmd = [
            r.binaryPath,
            `-m ${r.modelPath}`,
            `--host ${r.endpoint.split(':')[1].replace('//', '')}`,
            `--port ${r.endpoint.split(':')[2]}`,
            `-ngl 0`,
            `-c 8192`,
            `--threads ${r.threads}`,
            `--temp 0.0`
        ].join(' ');
        terminal.show();
        terminal.sendText(cmd);
        return terminal;
    }
}
exports.ServerLauncher = ServerLauncher;
//# sourceMappingURL=serverLauncher.js.map