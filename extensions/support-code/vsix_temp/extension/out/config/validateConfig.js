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
exports.validateConfig = validateConfig;
const fs = __importStar(require("fs"));
function validateConfig(cfg) {
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
//# sourceMappingURL=validateConfig.js.map