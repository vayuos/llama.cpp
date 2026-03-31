import * as vscode from "vscode";
import { GravitasConfig } from "./gravitasConfig";

export function loadConfig(): GravitasConfig {
    const cfg = vscode.workspace.getConfiguration("gravitas");

    return {
        codebaseRoot: cfg.get<string>("codebaseRoot", ""),
        coder: {
            binaryPath: cfg.get<string>("coder.binaryPath", ""),
            modelPath: cfg.get<string>("coder.modelPath", ""),
            cudaDevices: cfg.get<string>("coder.cudaDevices", "0"),
            endpoint: cfg.get<string>("coder.endpoint", ""),
            gpuLayers: cfg.get<number>("coder.gpuLayers", 33),
            threads: cfg.get<number>("coder.threads", 8),
            contextSize: cfg.get<number>("coder.contextSize", 8192)
        },
        reviewer: {
            binaryPath: cfg.get<string>("reviewer.binaryPath", ""),
            modelPath: cfg.get<string>("reviewer.modelPath", ""),
            endpoint: cfg.get<string>("reviewer.endpoint", ""),
            modelName: cfg.get<string>("reviewer.modelName", ""),
            threads: cfg.get<number>("reviewer.threads", 16),
            strictMode: cfg.get<boolean>("reviewer.strictMode", true)
        }
    };
}
