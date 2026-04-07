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
exports.ModelCheckStep = exports.BinaryCheckStep = void 0;
const fs = __importStar(require("fs"));
const pathUtils_1 = require("../../utils/pathUtils");
class BinaryCheckStep {
    constructor() {
        this.name = 'Check llama-server binary exists';
    }
    async execute(config) {
        const rawPath = config.llamaBinPath || '(empty)';
        const resolvedPath = (0, pathUtils_1.resolveBinaryPath)(config.llamaBinPath || '');
        if (fs.existsSync(resolvedPath)) {
            const stat = fs.statSync(resolvedPath);
            if (stat.isDirectory()) {
                return { success: false, message: `The path provided is a directory, not a binary file. User setting: "${rawPath}", Attempted absolute path: "${resolvedPath}"` };
            }
            return { success: true, message: 'Binary found.' };
        }
        return { success: false, message: `llama-server binary not found. User setting: "${rawPath}", Attempted absolute path: "${resolvedPath}"` };
    }
}
exports.BinaryCheckStep = BinaryCheckStep;
class ModelCheckStep {
    constructor() {
        this.name = 'Check model files exist';
    }
    async execute(config) {
        const rawCoder = config.coder.modelPath || '(empty)';
        const rawReviewer = config.reviewer.modelPath || '(empty)';
        const coderPath = (0, pathUtils_1.resolveTilde)(config.coder.modelPath || '');
        const reviewerPath = (0, pathUtils_1.resolveTilde)(config.reviewer.modelPath || '');
        if (!fs.existsSync(coderPath)) {
            return { success: false, message: `Coder model not found. User setting: "${rawCoder}", Attempted absolute path: "${coderPath}"` };
        }
        if (!fs.existsSync(reviewerPath)) {
            return { success: false, message: `Reviewer model not found. User setting: "${rawReviewer}", Attempted absolute path: "${reviewerPath}"` };
        }
        return { success: true, message: 'Model files found.' };
    }
}
exports.ModelCheckStep = ModelCheckStep;
//# sourceMappingURL=fileChecks.js.map