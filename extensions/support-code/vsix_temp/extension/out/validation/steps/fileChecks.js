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
class BinaryCheckStep {
    constructor() {
        this.name = 'Check llama-server binary exists';
    }
    async execute(config) {
        if (fs.existsSync(config.llamaBinaryPath)) {
            return { success: true, message: 'Binary found.' };
        }
        return { success: false, message: `llama-server binary not found at: ${config.llamaBinaryPath}` };
    }
}
exports.BinaryCheckStep = BinaryCheckStep;
class ModelCheckStep {
    constructor() {
        this.name = 'Check model files exist';
    }
    async execute(config) {
        if (!fs.existsSync(config.coderModel.modelPath)) {
            return { success: false, message: `Coder model not found at: ${config.coderModel.modelPath}` };
        }
        if (!fs.existsSync(config.reviewerModel.modelPath)) {
            return { success: false, message: `Reviewer model not found at: ${config.reviewerModel.modelPath}` };
        }
        return { success: true, message: 'Model files found.' };
    }
}
exports.ModelCheckStep = ModelCheckStep;
//# sourceMappingURL=fileChecks.js.map