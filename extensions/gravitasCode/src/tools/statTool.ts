import * as fs from 'fs';
import * as path from 'path';
import { NativeTool } from './types';

/**
 * Implementation of a production-grade analytical tool.
 * (Gap 6: Hardened Native Tools)
 */
export class CodebaseStatTool implements NativeTool {
    public readonly name = 'codebase_stats';
    public readonly description = 'Generates a JSON summary of file counts and lines of code across the src directory.';

    /**
     * Executes the line-count analysis recursively.
     */
    async execute(args: { path?: string } = {}): Promise<any> {
        const rootPath = args.path || path.join(process.cwd(), 'src');
        
        let fileCount = 0;
        let totalLoc = 0;
        const extensions = ['.ts', '.js', '.json', '.css'];

        const walk = (currentDir: string) => {
            if (!fs.existsSync(currentDir)) return;
            
            const entries = fs.readdirSync(currentDir, { withFileTypes: true });
            for (const entry of entries) {
                const fullPath = path.join(currentDir, entry.name);
                
                if (entry.isDirectory()) {
                    // Skip hidden dirs and node_modules
                    if (entry.name !== 'node_modules' && !entry.name.startsWith('.')) {
                        walk(fullPath);
                    }
                } else {
                    const ext = path.extname(entry.name);
                    if (extensions.includes(ext)) {
                        fileCount++;
                        try {
                            const content = fs.readFileSync(fullPath, 'utf8');
                            totalLoc += content.split('\n').length;
                        } catch (e) {}
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
