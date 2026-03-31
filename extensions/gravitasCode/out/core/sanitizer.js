"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.JSONSanitizer = void 0;
class JSONSanitizer {
    /**
     * Sanitizes a string that is supposed to be JSON.
     * Removes leading/trailing prose often added by models.
     */
    static sanitize(raw) {
        const start = raw.indexOf('{');
        const end = raw.lastIndexOf('}');
        if (start === -1 || end === -1 || start >= end) {
            return raw;
        }
        return raw.substring(start, end + 1);
    }
}
exports.JSONSanitizer = JSONSanitizer;
//# sourceMappingURL=sanitizer.js.map