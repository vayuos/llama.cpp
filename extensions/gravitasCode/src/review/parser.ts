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
        try {
            // Attempt to find JSON block if the model wrapped it in markdown
            const jsonPart = content.match(/\{[\s\S]*\}/);
            const rawJson = jsonPart ? jsonPart[0] : content;
            const data = JSON.parse(rawJson);
            return DeterministicReviewSchema.parse(data);
        } catch (e) {
            console.error('Failed to parse deterministic review:', e);
            return null;
        }
    }
}
