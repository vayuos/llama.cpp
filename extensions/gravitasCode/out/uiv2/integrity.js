"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.calculateHash = calculateHash;
exports.verifyContentIntegrity = verifyContentIntegrity;
const crypto = __importStar(require("crypto"));
/**
 * Calculates SHA-256 hash of a string.
 */
function calculateHash(content) {
    return crypto.createHash('sha256').update(content).digest('hex');
}
/**
 * Verifies if the hash matches the last known hash in the event stream.
 * @param content The full JSONL content
 */
function verifyContentIntegrity(content) {
    const lines = content.split('\n').filter(line => line.trim());
    if (lines.length === 0)
        return { verified: true, hash: '', driftDetails: 'Empty ledger' };
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
        }
        catch (e) {
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
    }
    else {
        return {
            verified: false,
            hash: currentHash,
            driftDetails: `Drift detected in verified history. Expected ${expectedHash}, found ${currentHash}`
        };
    }
}
//# sourceMappingURL=integrity.js.map