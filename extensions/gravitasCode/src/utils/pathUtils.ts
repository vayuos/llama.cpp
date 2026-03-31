import * as os from 'os';
import * as path from 'path';

/**
 * Resolves paths starting with '~~/' to the user's home directory.
 * Also normalizes the path for the current platform.
 */
export function resolveTilde(p: string): string {
    if (!p) return '';

    if (p.startsWith('~/')) {
        return path.normalize(path.join(os.homedir(), p.slice(2)));
    }

    // Support just '~' as home as well (optional but common)
    if (p === '~') {
        return os.homedir();
    }

    return path.normalize(p);
}

/**
 * Resolves a directory or file path to the exact llama-server binary.
 * If the path ends in a directory, appends 'llama-server'.
 */
export function resolveBinaryPath(p: string): string {
    // If empty, use the standard default directory
    let resolved = (p || '~/llama/llama.cpp/build/bin/').trim();
    // If it's a directory (ends in /) or just doesn't end in the binary name, append it
    if (resolved.endsWith('/') || resolved.endsWith('\\') || !resolved.toLowerCase().endsWith('llama-server')) {
        const separator = (resolved.endsWith('/') || resolved.endsWith('\\')) ? '' : path.sep;
        resolved = `${resolved}${separator}llama-server`;
    }

    return resolveTilde(resolved);
}
