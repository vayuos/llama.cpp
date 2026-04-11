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
exports.AgentLoopController = void 0;
const config_1 = require("../core/config");
const logger_1 = require("../core/logger");
const parser_1 = require("../review/parser");
const llmClient_1 = require("../llm/llmClient");
const systemRules_1 = require("../prompts/systemRules");
const taskManager_1 = require("../uiv2/taskManager");
const diffNormalizer_1 = require("../diff/diffNormalizer");
const contextCollector_1 = require("../context/contextCollector");
const uuid_1 = require("uuid");
const path = __importStar(require("path"));
const os = __importStar(require("os"));
const fs = __importStar(require("fs"));
/**
 * The Unified Agentic Engine for Gravitas.
 * Orchestrates Coder and Reviewer agents via TaskManager telemetry.
 */
class AgentLoopController {
    constructor() {
        this.logger = logger_1.CentralLogger.getInstance();
        this.maxIterations = 100; // 🛡️ Infinite-capable cap for autonomous reasoning.
    }
    static getInstance() {
        if (!AgentLoopController.instance) {
            AgentLoopController.instance = new AgentLoopController();
        }
        return AgentLoopController.instance;
    }
    /**
     * Executes the autonomous implement-and-review loop.
     */
    async run(taskId, prompt) {
        const config = await config_1.ConfigManager.getInstance().loadConfig();
        if (!config)
            throw new Error('Config missing');
        const tm = taskManager_1.TaskManager.getInstance();
        const coderSockPath = path.join(os.homedir(), '.gravitas', 'sockets', 'coder.sock');
        let defaultCoderUrl = fs.existsSync(coderSockPath) ? `unix://${coderSockPath}/v1` : `http://${config.coder.host || '127.0.0.1'}:${config.coder.port}/v1`;
        const coderClient = new llmClient_1.LLMClient(config.coder.baseUrl || defaultCoderUrl);
        const revSockPath = path.join(os.homedir(), '.gravitas', 'sockets', 'reviewer.sock');
        let defaultRevUrl = fs.existsSync(revSockPath) ? `unix://${revSockPath}/v1` : `http://${config.reviewer.host || '127.0.0.1'}:${config.reviewer.port}/v1`;
        const reviewerClient = new llmClient_1.LLMClient(config.reviewer.baseUrl || defaultRevUrl);
        const coderOpts = {
            temperature: config.coder.temperature,
            top_p: config.coder.topP,
            top_k: config.coder.topK,
            repeat_penalty: config.coder.repeatPenalty
        };
        const reviewerOpts = {
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
        let systemPromptEnv = systemRules_1.CODER_SYSTEM_PROMPT;
        if (prompt.length > 10 || /implement|fix|refactor|add|create|show|explain/i.test(prompt)) {
            const contextCollector = new contextCollector_1.ContextCollector();
            workspaceContext = await contextCollector.retrieve(prompt, config);
            if (workspaceContext.trim()) {
                this.logger.debug('system', `Hybrid Workspace context retrieved: ${workspaceContext.length} chars.`);
                systemPromptEnv = `${systemRules_1.CODER_SYSTEM_PROMPT}\n\n--- CURRENT WORKSPACE CONTEXT ---\n${workspaceContext.trim()}\n--- END CONTEXT ---\n\nInstructions: Use the above context to guide your implementation. If the context includes file contents, prioritize them. If it only includes a folder map, use it to locate relevant files.`;
            }
            else {
                this.logger.debug('system', 'Workspace context is still empty after local fallback.');
                systemPromptEnv = `${systemRules_1.CODER_SYSTEM_PROMPT}\n\nNote: The current workspace appears to be empty or inaccessible. Proceed with high caution and ask for clarification if file paths are unknown.`;
            }
        }
        else {
            this.logger.debug('system', 'Skipping context retrieval for short/non-implementation prompt.');
            systemPromptEnv = 'You are a helpful assistant. The current workspace is empty. Keep greetings brief and do NOT dump any architectural summaries unless requested.';
        }
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
                const thoughtId = (0, uuid_1.v4)();
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
                }
                catch (e) {
                    this.logger.error('system', `Gravitas Loop: Coder LLM Call Failed! Error: ${e.message}`);
                    tm.failTask(taskId, `Coder LLM Error: ${e.message}`);
                    throw e;
                }
                tm.emitEvent(taskId, {
                    type: 'ThoughtEmitted',
                    content: coderResp.thought || 'System: Analyzing next steps...'
                });
                this.logger.info('system', `AgentLoopController: Coder Thought: ${coderResp.thought || '[No thought provided]'}`);
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
                            const d = toolArgs.match(/path=["'](.*?)["']/);
                            toolResult = fs.readdirSync(d ? d[1] : '.').join('\n');
                        }
                        else if (toolName === 'view_file') {
                            const f = toolArgs.match(/path=["'](.*?)["']/);
                            toolResult = fs.readFileSync(f ? f[1] : '', 'utf8');
                        }
                        else if (toolName === 'grep_search') {
                            const q = toolArgs.match(/query=["'](.*?)["']/);
                            const query = q ? q[1] : '';
                            if (!query)
                                throw new Error('Search query is empty.');
                            this.logger.info('system', `AgentLoop: Executing rg search for "${query}"`);
                            // Use rg (ripgrep) with 50 matches limit for performance/token safety
                            const { execSync } = require('child_process');
                            try {
                                const stdout = execSync(`rg --max-count 50 --fixed-strings --line-number --column "${query}" .`, {
                                    encoding: 'utf8',
                                    timeout: 5000,
                                    maxBuffer: 1024 * 1024
                                });
                                toolResult = stdout || 'No matches found.';
                            }
                            catch (e) {
                                // execSync throws on non-zero exit (common for 0 matches in rg)
                                toolResult = e.stdout || 'No matches found or search error.';
                            }
                        }
                    }
                    catch (e) {
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
                const normalizedCode = diffNormalizer_1.DiffNormalizer.normalize(patchMatch ? patchMatch[1] : finalOutput);
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
                this.logger.info('system', `Gravitas Loop [Iteration ${iteration}]: Sending Reviewer Request (Content length: ${currentCode.length} chars)`);
                this.logger.debug('system', `Full Reviewer Prompt: ${currentCode}`);
                const reviewerResp = await reviewerClient.generate([
                    { role: 'system', content: systemRules_1.REVIEWER_SYSTEM_PROMPT },
                    { role: 'user', content: currentCode }
                ], reviewerOpts, (chunk) => {
                    tm.emitStreamingChunk(taskId, chunk, 'review');
                }, abortController.signal);
                this.logger.info('system', `Gravitas Loop [Iteration ${iteration}]: Reviewer Response received (${reviewerResp.content.length} chars)`);
                this.logger.debug('system', `Full Reviewer Response: ${reviewerResp.content}`);
                const review = parser_1.ReviewParser.parse(reviewerResp.content);
                if (!review) {
                    this.logger.error('system', `AgentLoopController [Iteration ${iteration}]: Reviewer failed to provide a valid deterministic JSON review.`);
                    tm.failTask(taskId, 'Reviewer output was non-deterministic.');
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
                this.logger.info('system', `AgentLoopController: Review FAILED. Moving to iteration ${iteration + 1} with feedback.`);
                // 🛡️ User Awareness: Alert if loop is going high
                if (iteration > 10) {
                    tm.emitEvent(taskId, {
                        type: 'TaskStatusEmitted',
                        status: `Autonomy Warning: Reasoning for ${iteration} iterations... (Consider manual intervention?)`
                    });
                }
                currentPrompt = `${prompt}\n\n[ITERATION ${iteration} FEEDBACK]\nThe previous implementation failed review with the following summary: ${review.summary}\n\nPlease address these specific issues:\n- ${review.recommendedChanges.join('\n- ')}\n\nIMPORTANT: Provide the COMPLETE updated implementation.`;
                if (iteration >= this.maxIterations) {
                    this.logger.error('system', `AgentLoopController: Maximum iterations (${this.maxIterations}) reached without passing review.`);
                    tm.failTask(taskId, `Maximum autonomous iterations (${this.maxIterations}) reached.`);
                }
            }
        }
        finally {
            onTaskUpdate.dispose();
        }
        return currentCode;
    }
    checkCancellation(taskId) {
        const task = taskManager_1.TaskManager.getInstance().getTask(taskId);
        if (task && task.status === 'ABORTED') {
            this.logger.info('system', `Agent Loop: Cancellation signal detected for task ${taskId}. Aborting loop.`);
            throw new Error('TASK_ABORTED_BY_USER');
        }
    }
}
exports.AgentLoopController = AgentLoopController;
//# sourceMappingURL=loop.js.map