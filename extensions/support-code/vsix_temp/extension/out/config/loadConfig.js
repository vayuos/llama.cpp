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
exports.loadConfig = loadConfig;
const vscode = __importStar(require("vscode"));
function loadConfig() {
    const cfg = vscode.workspace.getConfiguration("gravitas");
    return {
        codebaseRoot: cfg.get("codebaseRoot", ""),
        coder: {
            binaryPath: cfg.get("coder.binaryPath", ""),
            modelPath: cfg.get("coder.modelPath", ""),
            cudaDevices: cfg.get("coder.cudaDevices", "0"),
            endpoint: cfg.get("coder.endpoint", ""),
            gpuLayers: cfg.get("coder.gpuLayers", 33),
            threads: cfg.get("coder.threads", 8),
            contextSize: cfg.get("coder.contextSize", 8192)
        },
        reviewer: {
            binaryPath: cfg.get("reviewer.binaryPath", ""),
            modelPath: cfg.get("reviewer.modelPath", ""),
            endpoint: cfg.get("reviewer.endpoint", ""),
            modelName: cfg.get("reviewer.modelName", ""),
            threads: cfg.get("reviewer.threads", 16),
            strictMode: cfg.get("reviewer.strictMode", true)
        }
    };
}
//# sourceMappingURL=loadConfig.js.map