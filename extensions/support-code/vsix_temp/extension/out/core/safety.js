"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Safety = void 0;
class Safety {
    static validateDiffSize(diff) {
        const lineCount = diff.split('\n').length;
        return lineCount < 1000; // Limit to 1000 lines per patch
    }
    static isPotentialHallucination(output) {
        // Detect common hallucination patterns like markdown artifacts in non-md mode
        if (output.includes('```') && !output.includes('diff')) {
            return true;
        }
        return false;
    }
    static getReviewerOptions() {
        return {
            temperature: 0.0, // Strict determinism
            n_predict: 1024
        };
    }
}
exports.Safety = Safety;
//# sourceMappingURL=safety.js.map