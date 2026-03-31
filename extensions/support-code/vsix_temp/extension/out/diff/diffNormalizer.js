"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DiffNormalizer = void 0;
class DiffNormalizer {
    static normalize(raw) {
        const lines = raw.split('\n');
        const start = lines.findIndex(l => l.startsWith('--- ') || l.startsWith('@@ '));
        if (start === -1)
            return raw;
        return lines.slice(start).join('\n');
    }
    static isValidUnifiedDiff(diff) {
        return diff.includes('@@ ') && (diff.includes('--- ') || diff.includes('+++ '));
    }
}
exports.DiffNormalizer = DiffNormalizer;
//# sourceMappingURL=diffNormalizer.js.map