import * as crypto from 'crypto';

export interface IntegrityResult {
    verified: boolean;
    hash: string;
    driftDetails?: string;
}

/**
 * Calculates SHA-256 hash of a string.
 */
export function calculateHash(content: string): string {
    return crypto.createHash('sha256').update(content).digest('hex');
}

/**
 * Verifies if the hash matches the last known hash in the event stream.
 * @param content The full JSONL content
 */
export function verifyContentIntegrity(content: string): IntegrityResult {
    const lines = content.split('\n').filter(line => line.trim());
    if (lines.length === 0) return { verified: true, hash: '', driftDetails: 'Empty ledger' };

    // Find the last ReplayVerified event
    let verifiedIdx = -1;
    let expectedHash = '';
    for (let i = lines.length - 1; i >= 0; i--) {
        try {
            const event = JSON.parse(lines[i]);
            if (event.type === 'ReplayVerified') {
                verifiedIdx = i;
                expectedHash = event.streamHash;
                break;
            }
        } catch (e) {
            // Ignore malformed lines if any
        }
    }

    if (verifiedIdx === -1) {
        // No baseline yet. Hash the whole thing.
        const currentHash = calculateHash(content);
        return { verified: true, hash: currentHash, driftDetails: 'Baseline established (No previous verification found)' };
    }

    // Hash everything UP TO (but not including) the last ReplayVerified event
    // If verifiedIdx is 0, it means the first line is ReplayVerified, so precedingContent should be empty.
    const precedingContent = lines.slice(0, verifiedIdx).join('\n') + (verifiedIdx > 0 && lines.slice(0, verifiedIdx).length > 0 ? '\n' : '');
    const currentHash = calculateHash(precedingContent);

    if (currentHash === expectedHash) {
        if (lines.length > verifiedIdx + 1) {
            return {
                verified: true,
                hash: currentHash,
                driftDetails: `Verified up to line ${verifiedIdx + 1}. ${lines.length - verifiedIdx - 1} unverified trailing events.`
            };
        }
        return { verified: true, hash: currentHash };
    } else {
        return {
            verified: false,
            hash: currentHash,
            driftDetails: `Drift detected in verified history. Expected ${expectedHash}, found ${currentHash}`
        };
    }
}
