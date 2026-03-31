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
exports.Pipeline = void 0;
const vscode = __importStar(require("vscode"));
const llmClient_1 = require("./llmClient");
const reviewerValidator_1 = require("./reviewerValidator");
const safety_1 = require("./safety");
const configManager_1 = require("./configManager");
const diffEngine_1 = require("./diffEngine");
const sanitizer_1 = require("./sanitizer");
class Pipeline {
    async init() {
        const globalConfig = vscode.workspace.getConfiguration('gravitas');
        const localConfig = await configManager_1.ConfigManager.getLocalConfig();
        const coderEndpoint = localConfig.coder?.endpoint || globalConfig.get('coder.endpoint', 'http://127.0.0.1:8000');
        const reviewerEndpoint = localConfig.reviewer?.endpoint || globalConfig.get('reviewer.endpoint', 'http://127.0.0.1:8001');
        this.coderClient = new llmClient_1.LLMClient(coderEndpoint);
        this.reviewerClient = new llmClient_1.LLMClient(reviewerEndpoint);
    }
    async runPipeline(prompt) {
        let currentPrompt = prompt;
        let iterations = 0;
        const maxIterations = 3;
        while (iterations < maxIterations) {
            iterations++;
            vscode.window.showInformationMessage(`Iteration ${iterations}: Running Coder (GPU)...`);
            // 1. Coder Phase
            const coderResponse = await this.coderClient.generate(currentPrompt);
            const diff = diffEngine_1.DiffEngine.normalize(coderResponse.content);
            if (!diffEngine_1.DiffEngine.isValidUnifiedDiff(diff)) {
                vscode.window.showWarningMessage('Coder produced an invalid diff. Normalizing...');
            }
            // 2. Reviewer Phase
            vscode.window.showInformationMessage(`Iteration ${iterations}: Running Reviewer (CPU)...`);
            const reviewPrompt = `[PROPOSED PATCH]\n${diff}\n\n[INSTRUCTION]\nReview this patch and output JSON only.`;
            const reviewerResponse = await this.reviewerClient.generate(reviewPrompt, safety_1.Safety.getReviewerOptions());
            const sanitizedReview = sanitizer_1.JSONSanitizer.sanitize(reviewerResponse.content);
            let review;
            try {
                review = reviewerValidator_1.ReviewValidator.validate(sanitizedReview);
            }
            catch (e) {
                vscode.window.showErrorMessage(`Protocol Error: Reviewer output was not valid JSON. ${e.message}`);
                return;
            }
            if (review.status === 'approve') {
                vscode.window.showInformationMessage('Pipeline Successful: Patch Approved.');
                return;
            }
            else if (review.status === 'reject') {
                vscode.window.showErrorMessage('Pipeline Failed: Reviewer rejected the patch.');
                return;
            }
            else {
                vscode.window.showInformationMessage('Reviewer requested revisions. Re-running Coder...');
                currentPrompt = `${prompt}\n\n[PREVIOUS ATTEMPT DIARY]\nCoder generated: ${diff}\nReviewer Feedback: ${JSON.stringify(review.issues)}`;
            }
        }
        vscode.window.showErrorMessage('Pipeline Failed: Max iterations reached.');
    }
}
exports.Pipeline = Pipeline;
//# sourceMappingURL=pipeline.js.map