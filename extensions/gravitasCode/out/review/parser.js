"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ReviewParser = exports.DeterministicReviewSchema = exports.ReviewIssueSchema = void 0;
const zod_1 = require("zod");
exports.ReviewIssueSchema = zod_1.z.object({
    description: zod_1.z.string().default('Missing description'),
    line: zod_1.z.number().optional(),
    severity: zod_1.z.preprocess((val) => {
        if (typeof val !== 'string')
            return 'minor';
        const s = val.toLowerCase();
        if (s.includes('crit'))
            return 'critical';
        if (s.includes('maj') || s.includes('error'))
            return 'major';
        return 'minor';
    }, zod_1.z.enum(['critical', 'major', 'minor'])),
    suggestion: zod_1.z.string().optional()
});
exports.DeterministicReviewSchema = zod_1.z.object({
    severity: zod_1.z.preprocess((val) => {
        if (typeof val !== 'string')
            return 'minor';
        const s = val.toLowerCase();
        if (s.includes('crit'))
            return 'critical';
        if (s.includes('maj') || s.includes('err'))
            return 'major';
        return 'minor';
    }, zod_1.z.enum(['critical', 'major', 'minor'])),
    issues: zod_1.z.array(exports.ReviewIssueSchema).default([]),
    recommendedChanges: zod_1.z.array(zod_1.z.string()).default([]),
    summary: zod_1.z.string().default('No summary provided')
});
class ReviewParser {
    static parse(content) {
        const logger = require('../core/logger').CentralLogger.getInstance();
        logger.debug('system', `ReviewParser: Parsing raw content (${content.length} chars)`);
        try {
            // Attempt to find JSON block if the model wrapped it in markdown
            const jsonMatch = content.match(/(\{[\s\S]*\})/);
            const rawJson = jsonMatch ? jsonMatch[1] : content;
            logger.debug('system', `ReviewParser: Isolated JSON part (${rawJson.length} chars)`);
            const data = JSON.parse(rawJson);
            // Apply schema with lenient preprocessing
            const result = exports.DeterministicReviewSchema.parse(data);
            logger.debug('system', `ReviewParser: Successfully parsed review with severity: ${result.severity}`);
            return result;
        }
        catch (e) {
            logger.warn('system', `ReviewParser: Failed to parse or validate review. Error: ${e.message}`);
            // Fallback for extremely malformed content that still has some info
            if (content.toLowerCase().includes('success') || content.toLowerCase().includes('pass')) {
                return { severity: 'minor', issues: [], recommendedChanges: [], summary: 'Review parsed as PASS via heuristic fallback.' };
            }
            return null;
        }
    }
}
exports.ReviewParser = ReviewParser;
//# sourceMappingURL=parser.js.map