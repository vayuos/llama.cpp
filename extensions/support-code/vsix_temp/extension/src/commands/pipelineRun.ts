import * as vscode from 'vscode';
import { CoderAgent } from '../agents/coderAgent';
import { ReviewerAgent } from '../agents/reviewerAgent';
import { LLMClient } from '../llm/llmClient';
import { loadConfig } from '../config/loadConfig';
import { validateConfig } from '../config/validateConfig';
import { DiffNormalizer } from '../diff/diffNormalizer';
import { ReviewParser } from '../review/reviewParser';
import { ReviewValidator } from '../review/reviewValidator';
import { StateManager, SessionStatus } from '../state/sessionState';

export async function runPipeline(prompt: string, state: StateManager) {
    const config = loadConfig();
    try {
        validateConfig(config);
    } catch (e: any) {
        vscode.window.showErrorMessage(e.message);
        return;
    }

    const coderClient = new LLMClient(config.coder.endpoint);
    const reviewerClient = new LLMClient(config.reviewer.endpoint);

    const coder = new CoderAgent(coderClient);
    const reviewer = new ReviewerAgent(reviewerClient, config.reviewer.modelName);

    state.startSession();
    let currentPrompt = prompt;
    const maxIterations = 3;

    for (let i = 0; i < maxIterations; i++) {
        state.incrementIteration();
        state.updateStatus(SessionStatus.CODER_RUNNING);
        vscode.window.showInformationMessage(`Iteration ${i + 1}: Running Coder...`);

        const rawPatch = await coder.generatePatch(currentPrompt);
        const patch = DiffNormalizer.normalize(rawPatch);

        state.updateStatus(SessionStatus.REVIEWER_RUNNING);
        vscode.window.showInformationMessage(`Iteration ${i + 1}: Running Reviewer...`);

        const rawReview = await reviewer.reviewPatch(patch);
        const sanitizedReview = ReviewParser.sanitize(rawReview, config.reviewer.strictMode);
        const review = ReviewValidator.validate(sanitizedReview);

        if (review.status === 'approve') {
            state.updateStatus(SessionStatus.COMPLETED);
            vscode.window.showInformationMessage('Pipeline Successful: Patch Approved.');
            return;
        } else if (review.status === 'reject') {
            state.updateStatus(SessionStatus.FAILED);
            vscode.window.showErrorMessage('Pipeline Failed: Reviewer rejected the patch.');
            return;
        } else {
            vscode.window.showInformationMessage('Reviewer requested revisions.');
            currentPrompt = `${prompt}\n\n[FEEDBACK]\n${JSON.stringify(review.issues)}`;
        }
    }

    state.updateStatus(SessionStatus.FAILED);
    vscode.window.showErrorMessage('Pipeline Failed: Max iterations reached.');
}
