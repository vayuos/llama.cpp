"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ReviewParser = void 0;
class ReviewParser {
    static sanitize(raw, strict = true) {
        const start = raw.indexOf('{');
        const end = raw.lastIndexOf('}');
        if (start === -1 || end === -1 || start >= end) {
            if (strict)
                throw new Error("Gravitas Review: Failed to find valid JSON in strict mode.");
            return raw;
        }
        return raw.substring(start, end + 1);
    }
}
exports.ReviewParser = ReviewParser;
//# sourceMappingURL=reviewParser.js.map