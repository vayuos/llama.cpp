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

/**
 * The Unified Agentic Engine for Gravitas.
 * Orchestrates Coder and Reviewer agents via TaskManager telemetry.
 */
export class AgentLoopController {
    private logger = CentralLogger.getInstance();
    private maxIterations = 3;

    /**
     * Executes the autonomous implement-and-review loop.
     */
    public async run(taskId: TaskId, prompt: string): Promise<string> {
        const config: GravitasConfig | null = await ConfigManager.getInstance().loadConfig();
        if (!config) throw new Error('Config missing');

        const tm = TaskManager.getInstance();
        const coderClient = new LLMClient(config.coder.baseUrl || `http://127.0.0.1:${config.coder.port}/v1`);
        const reviewerClient = new LLMClient(config.reviewer.baseUrl || `http://127.0.0.1:${config.reviewer.port}/v1`);

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

        if (prompt.length > 20 || /implement|fix|refactor|add|create/i.test(prompt)) {
            const contextCollector = new ContextCollector();
            workspaceContext = await contextCollector.retrieve(prompt, config);
            
            if (workspaceContext.trim()) {
                systemPromptEnv = `${CODER_SYSTEM_PROMPT}\n\nYou are context-aware. Current Local Workspace State:\n${workspaceContext.trim()}`;
            } else {
                systemPromptEnv = `${CODER_SYSTEM_PROMPT}\n\nStrict Rule: The current project workspace is EMPTY. Do not assume any existing architecture or system components. Act only on the prompt provided.`;
            }
        } else {
            systemPromptEnv = 'You are a helpful assistant. The current workspace is empty. Keep greetings brief and do NOT dump any architectural summaries unless requested.';
        }

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
            
            tm.emitEvent(taskId, {
                type: 'ThoughtStarted',
                attemptNo: attempt.attemptNo,
                phaseId: coderPhaseId,
                thoughtId,
                startedAt: new Date().toISOString()
            });

            this.checkCancellation(taskId);
            const promptContent = currentPrompt + (currentCode ? `\n\nPrevious Code Attempt:\n${currentCode}` : '');
            this.logger.info('system', `Gravitas Loop: Sending Coder Request (Prompt length: ${promptContent.length} chars)`);

            let coderResp;
            try {
                coderResp = await coderClient.generate([
                    { role: 'system', content: systemPromptEnv },
                    { role: 'user', content: promptContent }
                ], coderOpts, (chunk) => {
                    tm.emitStreamingChunk(taskId, chunk, 'implementation');
                    // Intermittent hardware check (every few chunks)
                    if (Math.random() > 0.9) tm.pollHardwareMetrics(taskId);
                });
                this.logger.info('system', `Gravitas Loop: Coder Response received (${coderResp.content.length} chars)`);
            } catch (e: any) {
                this.logger.error('system', `Gravitas Loop: Coder LLM Call Failed! Error: ${e.message}`);
                tm.failTask(taskId, `Coder LLM Error: ${e.message}`);
                throw e;
            }
            
            const rawContent = coderResp.content;
            let thoughtContent = 'Refining implementation strategy based on workspace context.';
            let finalOutput = rawContent;

            // Extract <thought> if present
            const thoughtMatch = rawContent.match(/<thought>([\s\S]*?)<\/thought>/i);
            if (thoughtMatch) {
                thoughtContent = thoughtMatch[1].trim();
                finalOutput = rawContent.replace(/<thought>[\s\S]*?<\/thought>/i, '').trim();
            }

            const normalizedCode = DiffNormalizer.normalize(finalOutput);
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

            // 🧪 Short-Circuit: Skip Reviewer if this is just a greeting/question (no implementation)
            const codeMarkers = ['```', 'function', 'class', 'const ', 'def ', 'import ', 'export '];
            const hasImplementation = codeMarkers.some(m => currentCode.includes(m)) || /<implementation>|impl\.ts/i.test(currentCode);
            
            if (!hasImplementation) {
                this.logger.info('system', 'Conversational response detected. Skipping review phase.');
                tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: 'Success' });
                tm.completeTask(taskId, 'Task completed via conversational response.');
                break; // DONE
            }

            tm.recordPhaseMetrics(taskId, coderPhaseId, Date.now() - phaseStart);

            // 2. Reviewer Checks Code
            this.checkCancellation(taskId);
            tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: 'Reviewing code...' });
            this.logger.info('system', `Loop Iteration ${iteration}: Reviewing...`);
            const reviewerPhaseId = tm.startPhase(taskId, 'reviewer', `Iteration ${iteration}: Code Review`);
            tm.bindAgent(taskId, reviewerPhaseId, 'reviewer-v2-stochastic', config.reviewer.modelName || 'default-reviewer');
            
            const revPhaseStart = Date.now();
            
            this.logger.info('system', `Gravitas Loop: Sending Reviewer Request (Content length: ${currentCode.length} chars)`);
            const reviewerResp = await reviewerClient.generate([
                { role: 'system', content: REVIEWER_SYSTEM_PROMPT },
                { role: 'user', content: currentCode }
            ], reviewerOpts);
            this.logger.info('system', `Gravitas Loop: Reviewer Response received (${reviewerResp.content.length} chars)`);

            const review = ReviewParser.parse(reviewerResp.content);

            if (!review) {
                this.logger.error('system', 'Reviewer failed to provide deterministic output.');
                tm.failTask(taskId, 'Reviewer output was non-deterministic.');
                break;
            }

            tm.emitEvent(taskId, {
                type: 'ReviewerResultEmitted',
                attemptNo: attempt.attemptNo,
                phaseId: reviewerPhaseId,
                emittedAt: new Date().toISOString(),
                verdict: review.severity === 'critical' ? 'FAIL' : 'PASS',
                issues: review.issues.map(i => ({
                    type: 'correctness',
                    file: 'impl.ts',
                    line: i.line,
                    message: i.description,
                    severity: i.severity === 'critical' ? 'error' : 'warning'
                }))
            });

            tm.recordPhaseMetrics(taskId, reviewerPhaseId, Date.now() - revPhaseStart);

            if (review.severity === 'minor' && review.issues.length === 0) {
                this.logger.info('system', 'Auto-validation successful. Code meets quality bar.');
                tm.emitEvent(taskId, { type: 'TaskStatusEmitted', status: 'Success' });
                tm.completeTask(taskId, `Implementation finalized after ${iteration} iterations.`);
                break;
            }

            // 3. Feedback Loop
            currentPrompt = `Previous code implementation had issues: ${review.summary}. Please fix the following: ${review.recommendedChanges.join(', ')}`;
            
            if (iteration === this.maxIterations) {
                tm.failTask(taskId, 'Maximum iterations reached without successful review.');
            }
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
