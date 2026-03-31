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
const taskManager_1 = require("../src/uiv2/taskManager");
const types_1 = require("../src/uiv2/types");
const assert = __importStar(require("assert"));
// Mock VS Code Context
const mockContext = {
    globalState: {
        data: {},
        get: (key, def) => mockContext.globalState.data[key] || def,
        update: async (key, value) => { mockContext.globalState.data[key] = value; }
    }
};
async function testTaskManager() {
    console.log('Testing TaskManager...');
    // Initialize
    taskManager_1.TaskManager.initialize(mockContext);
    const tm = taskManager_1.TaskManager.getInstance();
    // 1. Create Task
    const task = tm.createTask('Test Command');
    assert.strictEqual(task.command, 'Test Command');
    assert.strictEqual(task.status, types_1.TaskState.CREATED);
    assert.ok(task.id);
    console.log('✅ Create Task passed');
    // 2. Start Task
    tm.updateTaskState(task.id, types_1.TaskState.RUNNING);
    assert.strictEqual(tm.getTask(task.id)?.status, types_1.TaskState.RUNNING);
    console.log('✅ Update State passed');
    // 3. Add Attempt
    tm.addAttempt(task.id, { thought: 'Thinking...' });
    const updatedTask = tm.getTask(task.id);
    assert.strictEqual(updatedTask?.attempts.length, 1);
    assert.strictEqual(updatedTask?.attempts[0].data.thought, 'Thinking...');
    console.log('✅ Add Attempt passed');
    // 4. Persistence Check
    // Re-initialize manager to simulate reload
    // In a real scenario we'd need to clear the static instance, but for this quick check
    // we just verify the mock store has the data
    const stored = mockContext.globalState.data['gravitas_tasks_v2'];
    assert.ok(stored[task.id]);
    assert.strictEqual(stored[task.id].command, 'Test Command');
    console.log('✅ Persistence Check passed');
    console.log('All tests passed!');
}
testTaskManager().catch(console.error);
//# sourceMappingURL=taskManagerTest.js.map