import { z } from 'zod';

export const ReviewIssueSchema = z.object({
    description: z.string().default('Missing description'),
    line: z.number().optional(),
    severity: z.preprocess((val) => {
        if (typeof val !== 'string') return 'minor';
        const s = val.toLowerCase();
        if (s.includes('crit')) return 'critical';
        if (s.includes('maj') || s.includes('error')) return 'major';
        return 'minor';
    }, z.enum(['critical', 'major', 'minor'])),
    suggestion: z.string().optional()
});

export const DeterministicReviewSchema = z.object({
    severity: z.preprocess((val) => {
        if (typeof val !== 'string') return 'minor';
        const s = val.toLowerCase();
        if (s.includes('crit')) return 'critical';
        if (s.includes('maj') || s.includes('err')) return 'major';
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
            // Attempt to find JSON block if the model wrapped it in markdown
            const jsonMatch = content.match(/(\{[\s\S]*\})/);
            const rawJson = jsonMatch ? jsonMatch[1] : content;
            
            logger.debug('system', `ReviewParser: Isolated JSON part (${rawJson.length} chars)`);
            const data = JSON.parse(rawJson);
            
            // Apply schema with lenient preprocessing
            const result = DeterministicReviewSchema.parse(data);
            
            logger.debug('system', `ReviewParser: Successfully parsed review with severity: ${result.severity}`);
            return result;
        } catch (e: any) {
            logger.warn('system', `ReviewParser: Failed to parse or validate review. Error: ${e.message}`);
            // Fallback for extremely malformed content that still has some info
            if (content.toLowerCase().includes('success') || content.toLowerCase().includes('pass')) {
                return { severity: 'minor', issues: [], recommendedChanges: [], summary: 'Review parsed as PASS via heuristic fallback.' };
            }
            return null;
        }
    }
}
