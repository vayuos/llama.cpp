"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.REVIEWER_SYSTEM_PROMPT = exports.CODER_SYSTEM_PROMPT = void 0;
exports.formatReviewerPrompt = formatReviewerPrompt;
exports.CODER_SYSTEM_PROMPT = `You are the Gravitas Coder.
Output ONLY unified diffs. No prose. No explanations.
Focus: OS / Compiler-grade precision.`;
exports.REVIEWER_SYSTEM_PROMPT = `You are the Gravitas Reviewer.
Evaluate the proposed patch for correctness, security, architecture, and performance.
Output ONLY valid JSON matching the provided schema.`;
function formatReviewerPrompt(patch) {
    return `[PROPOSED PATCH]\n${patch}\n\n[INSTRUCTION]\nReview this patch and output JSON only.`;
}
//# sourceMappingURL=systemRules.js.map