"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.runStatsDemo = runStatsDemo;
const toolWrapper_1 = require("../uiv2/toolWrapper");
const taskManager_1 = require("../uiv2/taskManager");
/**
 * Verification Script: Codebase Statistics Tool
 * Demonstrates the newly hardened hybrid tool execution model.
 */
async function runStatsDemo() {
    const tm = taskManager_1.TaskManager.getInstance();
    const task = tm.createTask('Demonstrate Codebase Statistics Tool', 'system');
    console.log(`[Demo] Created Task: ${task.id}`);
    const wrapper = new toolWrapper_1.ToolWrapper();
    const result = await wrapper.execute(task.id, 'codebase_stats', [], process.cwd(), 'Demo: Stats Analysis');
    console.log('[Demo] Execution Result:');
    console.log(result.output);
    if (result.exitCode === 0) {
        console.log('[Demo] SUCCESS: Native tool executed with full telemetry.');
    }
    else {
        console.log('[Demo] FAILURE: Tool execution failed.');
    }
}
//# sourceMappingURL=demo_stats.js.map