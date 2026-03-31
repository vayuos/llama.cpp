"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ReviewParser = exports.DeterministicReviewSchema = exports.ReviewIssueSchema = void 0;
const zod_1 = require("zod");
exports.ReviewIssueSchema = zod_1.z.object({
    description: zod_1.z.string(),
    line: zod_1.z.number().optional(),
    severity: zod_1.z.enum(['critical', 'major', 'minor']),
    suggestion: zod_1.z.string().optional()
});
exports.DeterministicReviewSchema = zod_1.z.object({
    severity: zod_1.z.enum(['critical', 'major', 'minor']),
    issues: zod_1.z.array(exports.ReviewIssueSchema),
    recommendedChanges: zod_1.z.array(zod_1.z.string()),
    summary: zod_1.z.string()
});
class ReviewParser {
    static parse(content) {
        try {
            // Attempt to find JSON block if the model wrapped it in markdown
            const jsonPart = content.match(/\{[\s\S]*\}/);
            const rawJson = jsonPart ? jsonPart[0] : content;
            const data = JSON.parse(rawJson);
            return exports.DeterministicReviewSchema.parse(data);
        }
        catch (e) {
            console.error('Failed to parse deterministic review:', e);
            return null;
        }
    }
}
exports.ReviewParser = ReviewParser;
//# sourceMappingURL=parser.js.map