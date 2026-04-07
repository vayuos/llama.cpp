import { z } from 'zod';

export const ReviewIssueSchema = z.object({
    description: z.string(),
    line: z.number().optional(),
    severity: z.enum(['critical', 'major', 'minor']),
    suggestion: z.string().optional()
});

export const DeterministicReviewSchema = z.object({
    severity: z.enum(['critical', 'major', 'minor']),
    issues: z.array(ReviewIssueSchema),
    recommendedChanges: z.array(z.string()),
    summary: z.string()
});

export type DeterministicReview = z.infer<typeof DeterministicReviewSchema>;

export class ReviewParser {
    public static parse(content: string): DeterministicReview | null {
        const logger = require('../core/logger').CentralLogger.getInstance();
        logger.debug('system', `ReviewParser: Parsing raw content (${content.length} chars)`);

        try {
            // Attempt to find JSON block if the model wrapped it in markdown
            const jsonPart = content.match(/\{[\s\S]*\}/);
            const rawJson = jsonPart ? jsonPart[0] : content;
            
            logger.debug('system', `ReviewParser: Isolated JSON part (${rawJson.length} chars)`);
            const data = JSON.parse(rawJson);
            const result = DeterministicReviewSchema.parse(data);
            
            logger.debug('system', `ReviewParser: Successfully parsed review with severity: ${result.severity}`);
            return result;
        } catch (e: any) {
            logger.warn('system', `ReviewParser: Failed to parse or validate review. Error: ${e.message}`);
            logger.debug('system', `ReviewParser: Full invalid content for debugging: ${content}`);
            return null;
        }
    }
}
