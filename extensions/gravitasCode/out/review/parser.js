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
        if (s.includes('crit') || s.includes('fatal'))
            return 'critical';
        if (s.includes('maj') || s.includes('error') || s.includes('fail'))
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
        if (s.includes('crit') || s.includes('fatal'))
            return 'critical';
        if (s.includes('maj') || s.includes('error') || s.includes('fail'))
            return 'major';
        if (s.includes('pass') || s.includes('lgtm') || s.includes('ok'))
            return 'minor';
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
            const rawJson = this.extractJson(content);
            if (!rawJson)
                throw new Error('No valid JSON block found in content.');
            logger.debug('system', `ReviewParser: Isolated JSON part (${rawJson.length} chars)`);
            const data = JSON.parse(rawJson);
            // Apply schema with lenient preprocessing
            return exports.DeterministicReviewSchema.parse(data);
        }
        catch (e) {
            logger.warn('system', `ReviewParser: Failed to parse or validate review. Error: ${e.message}`);
            // Fallback for extremely malformed content that still has some info
            if (content.toLowerCase().includes('success') || content.toLowerCase().includes('pass') || content.toLowerCase().includes('lgtm')) {
                return { severity: 'minor', issues: [], recommendedChanges: [], summary: 'Review parsed as PASS via heuristic fallback.' };
            }
            return null;
        }
    }
    static extractJson(content) {
        // 1. Try Markdown block first for cleaner isolation
        const markdownMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
        if (markdownMatch)
            return markdownMatch[1];
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
                }
                else if (char === '\\') {
                    escaped = true;
                }
                else if (char === '"') {
                    inString = false;
                }
                continue;
            }
            if (char === '"') {
                inString = true;
                continue;
            }
            if (char === '{') {
                if (startIndex === -1)
                    startIndex = i;
                openBraces++;
            }
            else if (char === '}') {
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
exports.ReviewParser = ReviewParser;
//# sourceMappingURL=parser.js.map