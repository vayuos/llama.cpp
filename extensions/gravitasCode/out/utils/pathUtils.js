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
exports.resolveTilde = resolveTilde;
exports.resolveBinaryPath = resolveBinaryPath;
const os = __importStar(require("os"));
const path = __importStar(require("path"));
/**
 * Resolves paths starting with '~~/' to the user's home directory.
 * Also normalizes the path for the current platform.
 */
function resolveTilde(p) {
    if (!p)
        return '';
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
function resolveBinaryPath(p) {
    // If empty, use the standard default directory
    let resolved = (p || '~/llama/llama.cpp/build/bin/').trim();
    // If it's a directory (ends in /) or just doesn't end in the binary name, append it
    if (resolved.endsWith('/') || resolved.endsWith('\\') || !resolved.toLowerCase().endsWith('llama-server')) {
        const separator = (resolved.endsWith('/') || resolved.endsWith('\\')) ? '' : path.sep;
        resolved = `${resolved}${separator}llama-server`;
    }
    return resolveTilde(resolved);
}
//# sourceMappingURL=pathUtils.js.map