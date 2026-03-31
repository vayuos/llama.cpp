import * as vscode from 'vscode';
import { ToolWrapper } from '../uiv2/toolWrapper';
import { TaskManager } from '../uiv2/taskManager';

/**
 * Verification Script: Codebase Statistics Tool
 * Demonstrates the newly hardened hybrid tool execution model.
 */
export async function runStatsDemo() {
    const tm = TaskManager.getInstance();
    const task = tm.createTask('Demonstrate Codebase Statistics Tool', 'system');
    
    console.log(`[Demo] Created Task: ${task.id}`);
    
    const wrapper = new ToolWrapper();
    const result = await wrapper.execute(task.id, 'codebase_stats', [], process.cwd(), 'Demo: Stats Analysis');
    
    console.log('[Demo] Execution Result:');
    console.log(result.output);
    
    if (result.exitCode === 0) {
        console.log('[Demo] SUCCESS: Native tool executed with full telemetry.');
    } else {
        console.log('[Demo] FAILURE: Tool execution failed.');
    }
}
