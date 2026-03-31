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
exports.runPipeline = runPipeline;
const vscode = __importStar(require("vscode"));
const coderAgent_1 = require("../agents/coderAgent");
const reviewerAgent_1 = require("../agents/reviewerAgent");
const llmClient_1 = require("../llm/llmClient");
const loadConfig_1 = require("../config/loadConfig");
const validateConfig_1 = require("../config/validateConfig");
const diffNormalizer_1 = require("../diff/diffNormalizer");
const reviewParser_1 = require("../review/reviewParser");
const reviewValidator_1 = require("../review/reviewValidator");
const sessionState_1 = require("../state/sessionState");
async function runPipeline(prompt, state) {
    const config = (0, loadConfig_1.loadConfig)();
    try {
        (0, validateConfig_1.validateConfig)(config);
    }
    catch (e) {
        vscode.window.showErrorMessage(e.message);
        return;
    }
    const coderClient = new llmClient_1.LLMClient(config.coder.endpoint);
    const reviewerClient = new llmClient_1.LLMClient(config.reviewer.endpoint);
    const coder = new coderAgent_1.CoderAgent(coderClient);
    const reviewer = new reviewerAgent_1.ReviewerAgent(reviewerClient, config.reviewer.modelName);
    state.startSession();
    let currentPrompt = prompt;
    const maxIterations = 3;
    for (let i = 0; i < maxIterations; i++) {
        state.incrementIteration();
        state.updateStatus(sessionState_1.SessionStatus.CODER_RUNNING);
        vscode.window.showInformationMessage(`Iteration ${i + 1}: Running Coder...`);
        const rawPatch = await coder.generatePatch(currentPrompt);
        const patch = diffNormalizer_1.DiffNormalizer.normalize(rawPatch);
        state.updateStatus(sessionState_1.SessionStatus.REVIEWER_RUNNING);
        vscode.window.showInformationMessage(`Iteration ${i + 1}: Running Reviewer...`);
        const rawReview = await reviewer.reviewPatch(patch);
        const sanitizedReview = reviewParser_1.ReviewParser.sanitize(rawReview, config.reviewer.strictMode);
        const review = reviewValidator_1.ReviewValidator.validate(sanitizedReview);
        if (review.status === 'approve') {
            state.updateStatus(sessionState_1.SessionStatus.COMPLETED);
            vscode.window.showInformationMessage('Pipeline Successful: Patch Approved.');
            return;
        }
        else if (review.status === 'reject') {
            state.updateStatus(sessionState_1.SessionStatus.FAILED);
            vscode.window.showErrorMessage('Pipeline Failed: Reviewer rejected the patch.');
            return;
        }
        else {
            vscode.window.showInformationMessage('Reviewer requested revisions.');
            currentPrompt = `${prompt}\n\n[FEEDBACK]\n${JSON.stringify(review.issues)}`;
        }
    }
    state.updateStatus(sessionState_1.SessionStatus.FAILED);
    vscode.window.showErrorMessage('Pipeline Failed: Max iterations reached.');
}
//# sourceMappingURL=pipelineRun.js.map