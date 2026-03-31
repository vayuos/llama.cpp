export class ReviewParser {
    static sanitize(raw: string, strict = true): string {
        const start = raw.indexOf('{');
        const end = raw.lastIndexOf('}');

        if (start === -1 || end === -1 || start >= end) {
            if (strict) throw new Error("Gravitas Review: Failed to find valid JSON in strict mode.");
            return raw;
        }

        return raw.substring(start, end + 1);
    }
}
