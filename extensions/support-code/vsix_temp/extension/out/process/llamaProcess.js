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
exports.LlamaProcess = void 0;
const vscode = __importStar(require("vscode"));
const child_process_1 = require("child_process");
const state_1 = require("../core/state");
class LlamaProcess {
    constructor(name, type) {
        this.name = name;
        this.type = type;
        this.process = null;
        this.outputChannel = vscode.window.createOutputChannel(`Gravitas: ${name}`);
    }
    async start(binaryPath, modelConfig, additionalArgs = []) {
        if (this.process) {
            await this.stop();
        }
        const args = [
            '-m', modelConfig.modelPath,
            '--port', modelConfig.port.toString(),
            '-c', modelConfig.ctx.toString(),
            '--temp', modelConfig.temp.toString(),
            ...additionalArgs
        ];
        if (modelConfig.gpuLayers !== undefined) {
            args.push('-ngl', modelConfig.gpuLayers.toString());
        }
        if (modelConfig.threads !== undefined) {
            args.push('-t', modelConfig.threads.toString());
        }
        this.outputChannel.appendLine(`Starting ${this.name} with command: ${binaryPath} ${args.join(' ')}`);
        this.process = (0, child_process_1.spawn)(binaryPath, args, {
            env: { ...process.env, CUDA_VISIBLE_DEVICES: '0' }, // Default for now
            detached: false
        });
        this.process.stdout?.on('data', (data) => this.outputChannel.append(data.toString()));
        this.process.stderr?.on('data', (data) => this.outputChannel.append(data.toString()));
        this.process.on('close', (code) => {
            this.outputChannel.appendLine(`Process ${this.name} exited with code ${code}`);
            this.process = null;
            this.updateStatus('stopped');
        });
        this.updateStatus('starting');
        return true;
    }
    async stop() {
        if (this.process) {
            this.process.kill();
            this.process = null;
        }
        this.updateStatus('stopped');
    }
    updateStatus(status) {
        const state = state_1.GravitasState.getInstance();
        if (this.type === 'coder') {
            state.updateState({ coderStatus: status });
        }
        else {
            state.updateState({ reviewerStatus: status });
        }
    }
    getName() { return this.name; }
}
exports.LlamaProcess = LlamaProcess;
//# sourceMappingURL=llamaProcess.js.map