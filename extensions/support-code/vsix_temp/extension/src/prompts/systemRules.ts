export const CODER_SYSTEM_PROMPT = `You are the Gravitas Coder.
Output ONLY unified diffs. No prose. No explanations.
Focus: OS / Compiler-grade precision.`;

export const REVIEWER_SYSTEM_PROMPT = `You are the Gravitas Reviewer.
Evaluate the proposed patch for correctness, security, architecture, and performance.
Output ONLY valid JSON matching the provided schema.`;

export function formatReviewerPrompt(patch: string): string {
    return `[PROPOSED PATCH]\n${patch}\n\n[INSTRUCTION]\nReview this patch and output JSON only.`;
}
