/**
 * Define the structure for Native (TypeScript-based) tools.
 */
export interface NativeTool {
    /**
     * Unique identifier for the tool (e.g., 'codebase_stats').
     */
    readonly name: string;

    /**
     * Human-readable description for model context.
     */
    readonly description: string;

    /**
     * Main execution logic.
     * Returns a JSON-serializable result.
     */
    execute(args: any): Promise<any>;
}
