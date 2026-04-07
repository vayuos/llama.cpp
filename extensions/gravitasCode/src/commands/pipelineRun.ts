import * as vscode from 'vscode';
import { AgentLoopController } from '../agents/loop';
import { TaskManager } from '../uiv2/taskManager';
import { TaskId } from '../uiv2/types';
import { TaskShellPanel } from '../uiv2/taskShell';

/**
 * Command: Gravitas: Run Pipeline
 * Entry point for the autonomous implementation loop.
 */
export async function runPipeline(prompt: string, existingTaskId?: TaskId) {
    const tm = TaskManager.getInstance();
    const logger = require('../core/logger').CentralLogger.getInstance();
    
    // 1. Get or Initialize Task Container
    const taskId = existingTaskId || tm.createTask(prompt, 'user').id;
    const task = tm.getTask(taskId)!;
    
    logger.info('system', `runPipeline: Entering command. Prompt: "${prompt.substring(0, 100)}..." (TaskId: ${taskId})`);

    try {
        // 2. Delegate to Unified Agentic Engine
        logger.debug('system', `runPipeline: Initializing AgentLoopController for task ${taskId}`);
        const controller = new AgentLoopController();
        
        // 3. Start the loop directly. 
        // Operational status ('Thinking...', 'Reviewing...') is managed by the sidebar events.
        await controller.run(task.id, prompt);

    } catch (error: any) {
        logger.error('system', `runPipeline: COMMAND CRASHED for task ${task.id}. Reason: ${error.message}`);
        tm.failTask(task.id, error.message || 'Pipeline crashed before completion.');
        vscode.window.showErrorMessage(`Gravitas Pipeline Error: ${error.message}`);
    }
}
