import { NativeTool } from './types';
import { CodebaseStatTool } from './statTool';

/**
 * Central registry for all Native (TypeScript) tools.
 * (Gap 6: Hardened Registry)
 */
export class NativeToolRegistry {
    private static instance: NativeToolRegistry;
    private tools: Map<string, NativeTool> = new Map();

    private constructor() {
        this.register(new CodebaseStatTool());
        // Add future native tools here
    }

    public static getInstance(): NativeToolRegistry {
        if (!NativeToolRegistry.instance) {
            NativeToolRegistry.instance = new NativeToolRegistry();
        }
        return NativeToolRegistry.instance;
    }

    public register(tool: NativeTool) {
        this.tools.set(tool.name, tool);
    }

    public getTool(name: string): NativeTool | undefined {
        return this.tools.get(name);
    }

    public hasTool(name: string): boolean {
        return this.tools.has(name);
    }

    public getAllTools(): NativeTool[] {
        return Array.from(this.tools.values());
    }
}
