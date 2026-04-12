import { ConfigManager, GravitasConfig } from '../core/config';
import { CentralLogger } from '../core/logger';
import { ReviewParser } from '../review/parser';
import { LLMClient, LLMOptions } from '../llm/llmClient';
import { CODER_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT } from '../prompts/systemRules';
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
        let currentPrompt = prompt;

        // 🧪 Context Throttling & Absolute Isolation
        let workspaceContext = '';
        let systemPromptEnv = CODER_SYSTEM_PROMPT;

        if (prompt.length > 10 || /implement|fix|refactor|add|create|show|explain|clean|delete|remove|run|execute/i.test(prompt)) {
            const contextCollector = new ContextCollector();
            workspaceContext = await contextCollector.retrieve(prompt, config);
            
            if (workspaceContext.trim()) {
                this.logger.debug('system', `Hybrid Workspace context retrieved: ${workspaceContext.length} chars.`);
                systemPromptEnv = `${CODER_SYSTEM_PROMPT}\n\n--- CURRENT WORKSPACE CONTEXT ---\n${workspaceContext.trim()}\n--- END CONTEXT ---\n\nInstructions: Use the above context to guide your implementation. If the context includes file contents, prioritize them. If it only includes a folder map, use it to locate relevant files.`;
            } else {
                this.logger.debug('system', 'Workspace context is still empty after local fallback.');
                systemPromptEnv = `${CODER_SYSTEM_PROMPT}\n\nNote: The current workspace appears to be empty or inaccessible. Proceed with high caution and ask for clarification if file paths are unknown.`;
            }
        }
        // REMOVED: Generic assistant fallback that stripped agent authority.


        this.logger.debug('system', `System Prompt Initialized: ${systemPromptEnv.substring(0, 200)}...`);

        const abortController = new AbortController();
        const onTaskUpdate = tm.onDidTaskUpdate((updatedTask) => {
            if (updatedTask.id === taskId && updatedTask.status === 'ABORTED') {
                this.logger.info('system', `AgentLoop: Task ${taskId} Aborted by user. Triggering AbortSignal.`);
                abortController.abort();
            }
        });

        try {
            while (iteration < this.maxIterations) {
            iteration++;
            this.checkCancellation(taskId);
            this.logger.info('system', `Loop Iteration ${iteration}: Starting Implementation...`);
            
            const attempt = tm.startNextAttempt(taskId);
            const coderPhaseId = tm.startPhase(taskId, 'coder', `Iteration ${iteration}: Implementation`);
            tm.bindAgent(taskId, coderPhaseId, 'coder-v2-hyperion', config.coder.modelName || 'default-coder');
            
            const phaseStart = Date.now();

            // 1. Coder Generates Code
            tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: 'Thinking...' });
            const thoughtId = uuidv4();
            
            this.checkCancellation(taskId);
            const promptContent = currentPrompt + (currentCode ? `\n\nPrevious Code Attempt:\n${currentCode}` : '');
            this.logger.info('system', `Gravitas Loop [Iteration ${iteration}]: Sending Coder Request (Prompt length: ${promptContent.length} chars)`);
            this.logger.debug('system', `Full Coder Prompt: ${promptContent}`);

            let coderResp;
            try {
                coderResp = await coderClient.generate([
                    { role: 'system', content: systemPromptEnv },
                    { role: 'user', content: promptContent }
                ], coderOpts, (chunk) => {
                    tm.emitStreamingChunk(taskId, chunk, 'implementation');
                }, abortController.signal);
                this.logger.info('system', `Gravitas Loop: Coder Response received (${coderResp.content.length} chars)`);
            } catch (e: any) {
                this.logger.error('system', `Gravitas Loop: Coder LLM Call Failed! Error: ${e.message}`);
                tm.failTask(taskId, `Coder LLM Error: ${e.message}`);
                throw e;
            }
            
            tm.emitEvent(taskId, {
                type: 'ThoughtEmitted',
                content: (coderResp as any).thought || 'System: Analyzing next steps...'
            });
            this.logger.info('system', `AgentLoopController: Coder Thought: ${(coderResp as any).thought || '[No thought provided]'}`);

            tm.emitEvent(taskId, {
                type: 'ThoughtStarted',
                attemptNo: attempt.attemptNo,
                phaseId: coderPhaseId,
                thoughtId,
                startedAt: new Date().toISOString()
            });
            
            const rawContent = coderResp.content;
            let finalOutput = rawContent;
            let thoughtContent = 'Refining implementation strategy based on workspace context.';

            // 1.1 Process Tool Actions (High-Fidelity Autonomy)
            const toolMatch = rawContent.match(/\[TOOL:\s*(\w+)\((.*?)\)\]/);
            if (toolMatch) {
                const toolName = toolMatch[1];
                const toolArgs = toolMatch[2];
                this.logger.info('system', `AgentLoop: Coder requested tool: ${toolName}(${toolArgs})`);

                tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: `🔧 Executing ${toolName}...` });
                tm.emitEvent(taskId, { 
                    type: 'ToolCallEmitted', 
                    tool: toolName, 
                    args: toolArgs 
                });
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
                            const targetPath = p[1] || p[2];
                            const content = c[1] || c[2];
                            const filePath = path.resolve(targetPath);
                            if (!filePath.startsWith(os.homedir())) throw new Error('Security: Cannot write outside home directory.');
                            fs.writeFileSync(filePath, content, 'utf8');
                            toolResult = `Successfully wrote to ${targetPath}`;
                        } else {
                            throw new Error('Invalid args for write_file. Expected path="..." and content="..."');
                        }
                    } else if (toolName === 'delete_file') {
                        const p = toolArgs.match(/path\s*=\s*(?:"([^"]*)"|'([^']*)')/);
                        if (p) {
                            const targetPath = p[1] || p[2];
                            const filePath = path.resolve(targetPath);
                            if (!filePath.startsWith(os.homedir())) throw new Error('Security: Cannot delete outside home directory.');
                            if (!fs.existsSync(filePath)) throw new Error(`File not found: ${targetPath}`);
                            fs.unlinkSync(filePath);
                            toolResult = `Successfully deleted ${targetPath}`;
                        } else {
                            throw new Error('Invalid args for delete_file. Expected path="..."');
                        }
                    } else if (toolName === 'run_command') {
                        const c = toolArgs.match(/command\s*=\s*(?:"([^"]*)"|'([^']*)')/);
                        if (c) {
                            const command = c[1] || c[2];
                            const { execSync } = require('child_process');
                            try {
                                const stdout = execSync(command, { 
                                    encoding: 'utf8', 
                                    timeout: 30000,
                                    maxBuffer: 1024 * 1024 
                                });
                                toolResult = stdout || 'Command executed successfully.';
                            } catch (e: any) {
                                toolResult = `Command failed: ${e.message}`;
                            }
                        } else {
                            throw new Error('Invalid args for run_command. Expected command="..."');
                        }
                    } else if (toolName === 'grep_search') {
                        const q = toolArgs.match(/query\s*=\s*(?:"([^"]*)"|'([^']*)')/);
                        const query = q ? (q[1] || q[2]) : '';
                        if (!query) throw new Error('Search query is empty.');
                        
                        this.logger.info('system', `AgentLoop: Executing rg search for "${query}"`);
                        const { execSync } = require('child_process');
                        try {
                            const stdout = execSync(`rg --max-count 50 --fixed-strings --line-number --column "${query}" .`, { 
                                encoding: 'utf8', 
                                timeout: 5000,
                                maxBuffer: 1024 * 1024 
                            });
                            toolResult = stdout || 'No matches found.';
                        } catch (e: any) {
                            toolResult = e.stdout || 'No matches found or search error.';
                        }
                    }
                } catch (e: any) {
                    toolResult = `Error executing tool: ${e.message}`;
                }

                currentPrompt = `Tool Result (${toolName}):\n${toolResult}\n\nBased on this, proceed with [PATCH] or another [TOOL].`;
                tm.recordPhaseMetrics(taskId, coderPhaseId, Date.now() - phaseStart);
                continue; // 🚀 RECURSIVE REASONING LOOP
            }

            // 1.2 Extract <thought> if present
            const thoughtMatch = rawContent.match(/<thought>([\s\S]*?)<\/thought>/i);
            if (thoughtMatch) {
                thoughtContent = thoughtMatch[1].trim();
                finalOutput = rawContent.replace(/<thought>[\s\S]*?<\/thought>/i, '').trim();
            }

            const patchMatch = finalOutput.match(/\[PATCH\]\s*([\s\S]*)/);
            const normalizedCode = DiffNormalizer.normalize(patchMatch ? patchMatch[1] : finalOutput);
            currentCode = normalizedCode;

            tm.emitEvent(taskId, {
                type: 'CoderResultEmitted',
                attemptNo: attempt.attemptNo,
                phaseId: coderPhaseId,
                emittedAt: new Date().toISOString(),
                content: currentCode,
                file: 'impl.ts' 
            });

            tm.emitEvent(taskId, {
                type: 'ThoughtCompleted',
                attemptNo: attempt.attemptNo,
                phaseId: coderPhaseId,
                thoughtId,
                content: thoughtContent,
                endedAt: new Date().toISOString(),
                durationMs: Date.now() - phaseStart
            });

            tm.recordPhaseMetrics(taskId, coderPhaseId, Date.now() - phaseStart);

            // 2. Reviewer (Master Architect) Checks Code
            this.checkCancellation(taskId);
            tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: 'Master Architect Reviewing...' });
            this.logger.info('system', `Loop Iteration ${iteration}: Reviewing...`);
            const reviewerPhaseId = tm.startPhase(taskId, 'reviewer', `Iteration ${iteration}: Master Architect Review`);
            tm.bindAgent(taskId, reviewerPhaseId, 'reviewer-v2-stochastic', config.reviewer.modelName || 'default-reviewer');
            
            const revPhaseStart = Date.now();
            
            this.logger.info('system', `Gravitas Loop [Iteration ${iteration}]: Sending Reviewer Request (Content length: ${currentCode.length} chars)`);
            this.logger.debug('system', `Full Reviewer Prompt: ${currentCode}`);
            const reviewerResp = await reviewerClient.generate([
                { role: 'system', content: REVIEWER_SYSTEM_PROMPT },
                { role: 'user', content: currentCode }
            ], reviewerOpts, (chunk) => {
                tm.emitStreamingChunk(taskId, chunk, 'review');
            }, abortController.signal);
            this.logger.info('system', `Gravitas Loop [Iteration ${iteration}]: Reviewer Response received (${reviewerResp.content.length} chars)`);
            this.logger.debug('system', `Full Reviewer Response: ${reviewerResp.content}`);

            const review = ReviewParser.parse(reviewerResp.content);

            if (!review) {
                this.logger.error('system', `AgentLoopController [Iteration ${iteration}]: Reviewer failed to provide a valid deterministic JSON review.`);
                tm.failTask(taskId, 'Master Architect output was non-deterministic.');
                break;
            }

            this.logger.info('system', `AgentLoopController [Iteration ${iteration}]: Reviewer Verdict: ${review.severity.toUpperCase()}. Summary: ${review.summary}`);
            if (review.issues.length > 0) {
                this.logger.debug('system', `AgentLoopController [Iteration ${iteration}]: Reviewer Issues: ${JSON.stringify(review.issues)}`);
            }

            tm.emitEvent(taskId, {
                type: 'ReviewerResultEmitted',
                attemptNo: attempt.attemptNo,
                phaseId: reviewerPhaseId,
                emittedAt: new Date().toISOString(),
                verdict: review.severity === 'critical' ? 'FAIL' : 'PASS',
                summary: review.summary,
                issues: review.issues.map(i => ({
                    type: 'correctness',
                    file: 'impl.ts',
                    line: i.line ?? 0,
                    message: i.description,
                    severity: i.severity === 'critical' ? 'error' : 'warning'
                }))
            });

            tm.recordPhaseMetrics(taskId, reviewerPhaseId, Date.now() - revPhaseStart);

            if (review.severity !== 'critical') {
                this.logger.info('system', 'Master Architect approved the implementation.');
                tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: 'Success' });
                
                // The Master (Reviewer) speaks to the user.
                const finalResponse = review.finalUserResponse || review.summary || `Implementation finalized after ${iteration} iterations.`;
                tm.completeTask(taskId, finalResponse);
                break;
            }

            // 3. Feedback Loop (Slave Fixes it)
            this.logger.info('system', `AgentLoopController: Master Architect REJECTED logic. Slave moving to iteration ${iteration + 1} with feedback.`);
            
            // 🛡️ User Awareness: Alert if loop is going high
            if (iteration > 10) {
                tm.emitEvent(taskId, { 
                    type: 'TaskStatusEmitted', 
                    status: `Autonomy Warning: Reasoning for ${iteration} iterations... (Consider manual intervention?)` 
                });
            }

            currentPrompt = `${prompt}\n\n[MASTER ARCHITECT FEEDBACK]\nThe previous proposal was rejected with the following summary: ${review.summary}\n\nPlease address these specific instructions from the Master Architect:\n- ${review.recommendedChanges.join('\n- ')}\n\nIMPORTANT: Provide the COMPLETE updated implementation suggestion.`;
            
            if (iteration >= this.maxIterations) {
                this.logger.error('system', `AgentLoopController: Maximum iterations (${this.maxIterations}) reached without Master approval.`);
                tm.failTask(taskId, `Maximum autonomous iterations (${this.maxIterations}) reached.`);
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
