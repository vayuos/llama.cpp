import { ConfigManager, GravitasConfig } from '../core/config';
import { CentralLogger } from '../core/logger';
import { ReviewParser } from '../review/parser';
import { LLMClient, LLMOptions } from '../llm/llmClient';
import { CODER_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT, formatMasterInstructionPrompt, formatReviewerPrompt } from '../prompts/systemRules';
import { TaskManager } from '../uiv2/taskManager';
import { TaskId } from '../uiv2/types';
import { DiffNormalizer } from '../diff/diffNormalizer';
import { ContextCollector } from '../context/contextCollector';
import { v4 as uuidv4 } from 'uuid';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';
import * as vscode from 'vscode';

/**
 * The Unified Agentic Engine for Gravitas.
 * Orchestrates Coder and Reviewer agents via TaskManager telemetry.
 */
export class AgentLoopController {
    private static instance: AgentLoopController;
    private logger = CentralLogger.getInstance();
    private maxIterations = 100; // 🛡️ Infinite-capable cap for autonomous reasoning.

    private constructor() {}

    public static getInstance(): AgentLoopController {
        if (!AgentLoopController.instance) {
            AgentLoopController.instance = new AgentLoopController();
        }
        return AgentLoopController.instance;
    }

    /**
     * Executes the autonomous implement-and-review loop.
     */
    public async run(taskId: TaskId, prompt: string): Promise<string> {
        const config: GravitasConfig | null = await ConfigManager.getInstance().loadConfig();
        if (!config) throw new Error('Config missing');

        const tm = TaskManager.getInstance();
        
        const coderSockPath = path.join(os.homedir(), '.gravitas', 'sockets', 'coder.sock');
        let defaultCoderUrl = fs.existsSync(coderSockPath) ? `unix://${coderSockPath}/v1` : `http://${config.coder.host || '127.0.0.1'}:${config.coder.port}/v1`;
        const coderClient = new LLMClient(config.coder.baseUrl || defaultCoderUrl);
        
        const revSockPath = path.join(os.homedir(), '.gravitas', 'sockets', 'reviewer.sock');
        let defaultRevUrl = fs.existsSync(revSockPath) ? `unix://${revSockPath}/v1` : `http://${config.reviewer.host || '127.0.0.1'}:${config.reviewer.port}/v1`;
        const reviewerClient = new LLMClient(config.reviewer.baseUrl || defaultRevUrl);

        const coderOpts: LLMOptions = {
            temperature: config.coder.temperature,
            top_p: config.coder.topP,
            top_k: config.coder.topK,
            repeat_penalty: config.coder.repeatPenalty
        };

        const reviewerOpts: LLMOptions = {
            temperature: config.reviewer.temperature ?? 0.0,
            top_p: config.reviewer.topP,
            top_k: config.reviewer.topK ?? 1,
            repeat_penalty: config.reviewer.repeatPenalty
        };

        let currentCode = '';
        let iteration = 0;
        let masterInstructions = '';

        // 🧪 Context Collection
        const contextCollector = new ContextCollector();
        const workspaceContext = await contextCollector.retrieve(prompt, config);

        const abortController = new AbortController();
        const onTaskUpdate = tm.onDidTaskUpdate((updatedTask) => {
            if (updatedTask.id === taskId && updatedTask.status === 'ABORTED') {
                abortController.abort();
            }
        });

        const { applyPatch } = require('../commands/applyPatch');

        try {
            while (iteration < 5) { // 🛡️ Cap Master-Slave cycles
                iteration++;
                this.checkCancellation(taskId);
                this.logger.info('system', `Master-Slave Cycle ${iteration} Starting...`);
                
                const attempt = tm.startNextAttempt(taskId);

                // PHASE 1: MASTER PLANNING
                tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: 'Master Planning...' });
                const reviewerPhaseId = tm.startPhase(taskId, 'reviewer', `Iteration ${iteration}: Master Planning`);
                tm.bindAgent(taskId, reviewerPhaseId, 'reviewer-master-architect', config.reviewer.modelName || 'default-reviewer');
                
                const masterStart = Date.now();
                const masterPrompt = formatMasterInstructionPrompt(prompt, workspaceContext + (currentCode ? `\n\nExisting Changes:\n${currentCode}` : ''));
                
                const masterResp = await reviewerClient.generate([
                    { role: 'system', content: REVIEWER_SYSTEM_PROMPT },
                    { role: 'user', content: masterPrompt }
                ], reviewerOpts, undefined, abortController.signal);
                
                masterInstructions = masterResp.content;
                tm.recordPhaseMetrics(taskId, reviewerPhaseId, Date.now() - masterStart);

                // PHASE 2: SLAVE EXECUTION (CODER)
                this.checkCancellation(taskId);
                tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: 'Slave Executing...' });
                const coderPhaseId = tm.startPhase(taskId, 'coder', `Iteration ${iteration}: Slave Implementation`);
                tm.bindAgent(taskId, coderPhaseId, 'coder-implementation-slave', config.coder.modelName || 'default-coder');
                
                let executionStepPrompt = `MASTER INSTRUCTIONS:\n${masterInstructions}`;
                let executionDone = false;
                let slaveIteration = 0;

                while (!executionDone && slaveIteration < 10) {
                    slaveIteration++;
                    const coderStart = Date.now();
                    const thoughtId = uuidv4();
                    tm.emitEvent(taskId, { type: 'ThoughtStarted', attemptNo: attempt.attemptNo, phaseId: coderPhaseId, thoughtId, startedAt: new Date().toISOString() });

                    const coderResp = await coderClient.generate([
                        { role: 'system', content: CODER_SYSTEM_PROMPT },
                        { role: 'user', content: executionStepPrompt }
                    ], coderOpts, (chunk) => tm.emitStreamingChunk(taskId, chunk, 'implementation'), abortController.signal);

                    const rawContent = coderResp.content;
                    
                    // 1.1 Process Tool Actions
                    const toolMatch = rawContent.match(/\[TOOL:\s*(\w+)\((.*?)\)\]/);
                    if (toolMatch) {
                        const toolName = toolMatch[1];
                        const toolArgs = toolMatch[2];
                        let toolResult = '';
                        try {
                            if (toolName === 'list_dir') {
                                const d = toolArgs.match(/path\s*=\s*(?:"([^"]*)"|'([^']*)')/);
                                const dirPath = d ? (d[1] || d[2]) : '.';
                                toolResult = fs.readdirSync(path.resolve(dirPath)).join('\n');
                            } else if (toolName === 'view_file') {
                                const f = toolArgs.match(/path\s*=\s*(?:"([^"]*)"|'([^']*)')/);
                                const filePath = f ? (f[1] || f[2]) : '';
                                toolResult = fs.readFileSync(path.resolve(filePath), 'utf8');
                            } else if (toolName === 'write_file') {
                                const p = toolArgs.match(/path\s*=\s*(?:"([^"]*)"|'([^']*)')/);
                                const c = toolArgs.match(/content\s*=\s*(?:"([\s\S]*?)"|'([\s\S]*?)')/);
                                if (p && c) {
                                    fs.writeFileSync(path.resolve(p[1] || p[2]), c[1] || c[2], 'utf8');
                                    toolResult = `Successfully wrote file.`;
                                }
                            } else if (toolName === 'run_command') {
                                const c = toolArgs.match(/command\s*=\s*(?:"([^"]*)"|'([^']*)')/);
                                if (c) toolResult = require('child_process').execSync(c[1] || c[2], { encoding: 'utf8' });
                            }
                        } catch (e: any) {
                            toolResult = `Error: ${e.message}`;
                        }

                        // Truncate tool results to prevent socket hang up
                        const MAX_TOOL_OUTPUT = 8000;
                        if (toolResult.length > MAX_TOOL_OUTPUT) {
                            toolResult = toolResult.substring(0, MAX_TOOL_OUTPUT) + `\n\n... (Result truncated from ${toolResult.length} chars)`;
                        }

                        executionStepPrompt = `Tool Result (${toolName}):\n${toolResult}\n\nBased on this, proceed with [PATCH] or another [TOOL].`;
                        tm.recordPhaseMetrics(taskId, coderPhaseId, Date.now() - coderStart);
                        continue;
                    }

                    // 1.2 Extract Thought & Patch
                    const thoughtMatch = rawContent.match(/<thought>([\s\S]*?)<\/thought>/i);
                    const thoughtContent = thoughtMatch ? thoughtMatch[1].trim() : 'Executing master instructions.';
                    const finalOutput = rawContent.replace(/<thought>[\s\S]*?<\/thought>/i, '').trim();

                    const patchMatch = finalOutput.match(/\[PATCH\]\s*([\s\S]*)/);
                    if (patchMatch) {
                        currentCode = DiffNormalizer.normalize(patchMatch[1]);
                        this.logger.info('system', 'Autonomous Patch detected. Applying...');
                        try {
                            await applyPatch(currentCode);
                            tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: 'Patch applied successfully.' });
                        } catch (e: any) {
                            this.logger.error('system', `Patch failed: ${e.message}`);
                        }
                    }

                    tm.emitEvent(taskId, { type: 'ThoughtCompleted', attemptNo: attempt.attemptNo, phaseId: coderPhaseId, thoughtId, content: thoughtContent, endedAt: new Date().toISOString(), durationMs: Date.now() - coderStart });
                    tm.recordPhaseMetrics(taskId, coderPhaseId, Date.now() - coderStart);
                    executionDone = true;
                }

                // PHASE 4: MASTER SYNC & VERIFICATION
                this.checkCancellation(taskId);
                tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: 'Master Verifying...' });
                const verifyPhaseId = tm.startPhase(taskId, 'reviewer', `Iteration ${iteration}: Master Verification`);
                
                const verifyStart = Date.now();
                const reviewerResp = await reviewerClient.generate([
                    { role: 'system', content: REVIEWER_SYSTEM_PROMPT },
                    { role: 'user', content: formatReviewerPrompt(currentCode) }
                ], reviewerOpts, undefined, abortController.signal);

                const review = ReviewParser.parse(reviewerResp.content);
                tm.recordPhaseMetrics(taskId, verifyPhaseId, Date.now() - verifyStart);

                if (review && review.severity === 'minor' && review.issues.length === 0) {
                    this.logger.info('system', 'Master approved.');
                    const response = review.finalUserResponse || review.summary || 'Task completed.';
                    tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: 'Success' });
                    tm.completeTask(taskId, response);
                    break;
                }
            }
        } finally {
            onTaskUpdate.dispose();
        }

        return currentCode;
    }

    private checkCancellation(taskId: TaskId) {
        const task = TaskManager.getInstance().getTask(taskId);
        if (task && task.status === 'ABORTED') {
            this.logger.info('system', `Agent Loop: Cancellation signal detected for task ${taskId}. Aborting loop.`);
            throw new Error('TASK_ABORTED_BY_USER');
        }
    }
}
