"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DiffEngine = void 0;
class DiffEngine {
    /**
     * Normalizes a diff to ensure it follows a strict unified format.
     * Trims trailing whitespace and ensures consistent line endings.
     */
    static normalize(rawDiff) {
        return rawDiff
            .split('\n')
            .map(line => line.trimEnd())
            .filter(line => line.length > 0 || line === '')
            .join('\n');
    }
    /**
     * Validates if a string is a valid unified diff fragment.
     */
    static isValidUnifiedDiff(diff) {
        const lines = diff.split('\n');
        return lines.some(line => line.startsWith('--- ')) &&
            lines.some(line => line.startsWith('+++ ')) &&
            lines.some(line => line.startsWith('@@ '));
    }
}
exports.DiffEngine = DiffEngine;
//# sourceMappingURL=diffEngine.js.map