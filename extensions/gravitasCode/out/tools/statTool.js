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
exports.CodebaseStatTool = void 0;
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
/**
 * Implementation of a production-grade analytical tool.
 * (Gap 6: Hardened Native Tools)
 */
class CodebaseStatTool {
    constructor() {
        this.name = 'codebase_stats';
        this.description = 'Generates a JSON summary of file counts and lines of code across the src directory.';
    }
    /**
     * Executes the line-count analysis recursively.
     */
    async execute(args = {}) {
        const rootPath = args.path || path.join(process.cwd(), 'src');
        let fileCount = 0;
        let totalLoc = 0;
        const extensions = ['.ts', '.js', '.json', '.css'];
        const walk = (currentDir) => {
            if (!fs.existsSync(currentDir))
                return;
            const entries = fs.readdirSync(currentDir, { withFileTypes: true });
            for (const entry of entries) {
                const fullPath = path.join(currentDir, entry.name);
                if (entry.isDirectory()) {
                    // Skip hidden dirs and node_modules
                    if (entry.name !== 'node_modules' && !entry.name.startsWith('.')) {
                        walk(fullPath);
                    }
                }
                else {
                    const ext = path.extname(entry.name);
                    if (extensions.includes(ext)) {
                        fileCount++;
                        try {
                            const content = fs.readFileSync(fullPath, 'utf8');
                            totalLoc += content.split('\n').length;
                        }
                        catch (e) { }
                    }
                }
            }
        };
        walk(rootPath);
        return {
            root: rootPath,
            fileCount,
            totalLoc,
            averageLocPerFile: fileCount > 0 ? Math.round(totalLoc / fileCount) : 0,
            timestamp: new Date().toISOString()
        };
    }
}
exports.CodebaseStatTool = CodebaseStatTool;
//# sourceMappingURL=statTool.js.map