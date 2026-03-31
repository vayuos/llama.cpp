export class DiffNormalizer {
    static normalize(raw: string): string {
        const lines = raw.split('\n');
        const start = lines.findIndex(l => l.startsWith('--- ') || l.startsWith('@@ '));
        if (start === -1) return raw;

        return lines.slice(start).join('\n');
    }

    static isValidUnifiedDiff(diff: string): boolean {
        return diff.includes('@@ ') && (diff.includes('--- ') || diff.includes('+++ '));
    }
}
