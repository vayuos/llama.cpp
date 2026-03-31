import * as fs from "fs";
import { GravitasConfig } from "./gravitasConfig";

export function validateConfig(cfg: GravitasConfig): void {
    if (!cfg.codebaseRoot) {
        throw new Error("Gravitas: codebaseRoot not set");
    }

    // Coder Validation
    if (!cfg.coder.binaryPath) {
        throw new Error("Gravitas: Coder binary path not set");
    }
    if (!fs.existsSync(cfg.coder.binaryPath)) {
        throw new Error(`Gravitas: Coder binary does not exist at ${cfg.coder.binaryPath}`);
    }
    if (cfg.coder.modelPath && !fs.existsSync(cfg.coder.modelPath)) {
        throw new Error(`Gravitas: Coder model file does not exist at ${cfg.coder.modelPath}`);
    }

    // Reviewer Validation
    if (!cfg.reviewer.binaryPath) {
        throw new Error("Gravitas: Reviewer binary path not set");
    }
    if (!fs.existsSync(cfg.reviewer.binaryPath)) {
        throw new Error(`Gravitas: Reviewer binary does not exist at ${cfg.reviewer.binaryPath}`);
    }
    if (cfg.reviewer.modelPath && !fs.existsSync(cfg.reviewer.modelPath)) {
        throw new Error(`Gravitas: Reviewer model file does not exist at ${cfg.reviewer.modelPath}`);
    }

    if (!cfg.coder.endpoint) {
        throw new Error("Gravitas: Coder endpoint missing");
    }
    if (!cfg.reviewer.endpoint) {
        throw new Error("Gravitas: Reviewer endpoint missing");
    }
}
