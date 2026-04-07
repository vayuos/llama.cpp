import { z } from 'zod';

export const ReviewIssueSchema = z.object({
    description: z.string().default('Missing description'),
    line: z.number().optional(),
    severity: z.preprocess((val) => {
        if (typeof val !== 'string') return 'minor';
        const s = val.toLowerCase();
        if (s.includes('crit') || s.includes('fatal')) return 'critical';
        if (s.includes('maj') || s.includes('error') || s.includes('fail')) return 'major';
        return 'minor';
    }, z.enum(['critical', 'major', 'minor'])),
    suggestion: z.string().optional()
});

export const DeterministicReviewSchema = z.object({
    severity: z.preprocess((val) => {
        if (typeof val !== 'string') return 'minor';
        const s = val.toLowerCase();
        if (s.includes('crit') || s.includes('fatal')) return 'critical';
        if (s.includes('maj') || s.includes('error') || s.includes('fail')) return 'major';
        if (s.includes('pass') || s.includes('lgtm') || s.includes('ok')) return 'minor';
        return 'minor';
    }, z.enum(['critical', 'major', 'minor'])),
    issues: z.array(ReviewIssueSchema).default([]),
    recommendedChanges: z.array(z.string()).default([]),
    summary: z.string().default('No summary provided')
});

export type DeterministicReview = z.infer<typeof DeterministicReviewSchema>;

export class ReviewParser {
    public static parse(content: string): DeterministicReview | null {
        const logger = require('../core/logger').CentralLogger.getInstance();
        logger.debug('system', `ReviewParser: Parsing raw content (${content.length} chars)`);

        try {
            const rawJson = this.extractJson(content);
            if (!rawJson) throw new Error('No valid JSON block found in content.');
            
            logger.debug('system', `ReviewParser: Isolated JSON part (${rawJson.length} chars)`);
            const data = JSON.parse(rawJson);
            
            // Apply schema with lenient preprocessing
            return DeterministicReviewSchema.parse(data);
        } catch (e: any) {
            logger.warn('system', `ReviewParser: Failed to parse or validate review. Error: ${e.message}`);
            // Fallback for extremely malformed content that still has some info
            if (content.toLowerCase().includes('success') || content.toLowerCase().includes('pass') || content.toLowerCase().includes('lgtm')) {
                return { severity: 'minor', issues: [], recommendedChanges: [], summary: 'Review parsed as PASS via heuristic fallback.' };
            }
            return null;
        }
    }

    private static extractJson(content: string): string | null {
        // 1. Try Markdown block first for cleaner isolation
        const markdownMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
        if (markdownMatch) return markdownMatch[1];

        // 2. String-Aware Balanced-Brace Extraction
        let openBraces = 0;
        let startIndex = -1;
        let inString = false;
        let escaped = false;

        for (let i = 0; i < content.length; i++) {
            const char = content[i];

            if (inString) {
                if (escaped) {
                    escaped = false;
                } else if (char === '\\') {
                    escaped = true;
                } else if (char === '"') {
                    inString = false;
                }
                continue;
            }

            if (char === '"') {
                inString = true;
                continue;
            }

            if (char === '{') {
                if (startIndex === -1) startIndex = i;
                openBraces++;
            } else if (char === '}') {
                if (startIndex !== -1) {
                    openBraces--;
                    if (openBraces === 0) {
                        return content.substring(startIndex, i + 1);
                    }
                }
            }
        }

        return null;
    }
}
