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
exports.runPipeline = runPipeline;
const vscode = __importStar(require("vscode"));
const uuid_1 = require("uuid");
const coderAgent_1 = require("../agents/coderAgent");
const reviewerAgent_1 = require("../agents/reviewerAgent");
const llmClient_1 = require("../llm/llmClient");
const loadConfig_1 = require("../config/loadConfig");
const validateConfig_1 = require("../config/validateConfig");
const diffNormalizer_1 = require("../diff/diffNormalizer");
const reviewParser_1 = require("../review/reviewParser");
const reviewValidator_1 = require("../review/reviewValidator");
const taskManager_1 = require("../uiv2/taskManager");
const types_1 = require("../uiv2/types");
const taskShell_1 = require("../uiv2/taskShell");
const toolWrapper_1 = require("../uiv2/toolWrapper");
// Renamed from 'state' to 'taskManager' conceptual usages
async function runPipeline(prompt, legacyState) {
    const config = (0, loadConfig_1.loadConfig)();
    try {
        (0, validateConfig_1.validateConfig)(config);
    }
    catch (e) {
        vscode.window.showErrorMessage(e.message);
        return;
    }
    const tm = taskManager_1.TaskManager.getInstance();
    const task = tm.createTask(prompt, 'user');
    const ext = vscode.extensions.getExtension('VayuOS.gravitas-code');
    if (ext) {
        taskShell_1.TaskShellPanel.createOrShow(ext.extensionUri, task.id);
    }
    tm.updateTaskState(task.id, types_1.TaskState.RUNNING);
    const coderClient = new llmClient_1.LLMClient(config.coder.endpoint);
    const reviewerClient = new llmClient_1.LLMClient(config.reviewer.endpoint);
    const coder = new coderAgent_1.CoderAgent(coderClient);
    const reviewer = new reviewerAgent_1.ReviewerAgent(reviewerClient, config.reviewer.modelName);
    let currentPrompt = prompt;
    const maxIterations = 3;
    for (let i = 0; i < maxIterations; i++) {
        tm.sampleResources(task.id);
        // New Attempt (Starts next monotonic container)
        const attemptNo = tm.startNextAttempt(task.id).attemptNo;
        // --- CODER PHASE ---
        const coderPhaseId = tm.startPhase(task.id, 'coder', `Iteration ${attemptNo}: Implementation`);
        tm.bindAgent(task.id, coderPhaseId, 'coder-agent-v1', config.coder.modelPath || 'unknown-model');
        const phaseStart = Date.now();
        // Start Thought
        const thoughtId = (0, uuid_1.v4)();
        tm.emitEvent(task.id, {
            type: 'ThoughtStarted',
            attemptNo,
            phaseId: coderPhaseId,
            thoughtId,
            startedAt: new Date().toISOString()
        });
        // Measure Reasoning Time
        const thoughtStart = Date.now();
        const rawPatch = await coder.generatePatch(currentPrompt);
        const durationMs = Date.now() - thoughtStart;
        // Complete Thought
        tm.emitEvent(task.id, {
            type: 'ThoughtCompleted',
            attemptNo,
            phaseId: coderPhaseId,
            thoughtId,
            endedAt: new Date().toISOString(),
            durationMs,
            content: `Generated patch plan for "${currentPrompt.substring(0, 30)}..."`
        });
        const patch = diffNormalizer_1.DiffNormalizer.normalize(rawPatch);
        // GAP 6: Tool Execution Wrapping
        const tool = new toolWrapper_1.ToolWrapper();
        await tool.execute(task.id, 'apply_patch --force', './', 'Patch Application');
        // GAP 9: Artifact Detection
        tm.recordArtifact(task.id, 'patch.diff', 'patch', { size: patch.length });
        tm.recordPhaseMetrics(task.id, coderPhaseId, Date.now() - phaseStart);
        // --- REVIEWER PHASE ---
        const reviewerPhaseId = tm.startPhase(task.id, 'reviewer', `Iteration ${attemptNo}: Code Review`);
        tm.bindAgent(task.id, reviewerPhaseId, 'reviewer-agent-v1', config.reviewer.modelName || 'default-model');
        const revPhaseStart = Date.now();
        const rawReview = await reviewer.reviewPatch(patch);
        const sanitizedReview = reviewParser_1.ReviewParser.sanitize(rawReview, config.reviewer.strictMode);
        const review = reviewValidator_1.ReviewValidator.validate(sanitizedReview);
        if (review.status === 'approve') {
            tm.emitEvent(task.id, {
                type: 'ReviewerResultEmitted',
                attemptNo,
                phaseId: reviewerPhaseId,
                emittedAt: new Date().toISOString(),
                verdict: 'PASS',
                issues: []
            });
            tm.recordPhaseMetrics(task.id, reviewerPhaseId, Date.now() - revPhaseStart);
            // GAP 8: Termination Sequencing
            tm.completeTask(task.id, 'Pipeline Successful: Patch Approved.');
            emitFinalSummary(task.id, 'SUCCESS', i + 1);
            return;
        }
        else {
            tm.recordPhaseMetrics(task.id, reviewerPhaseId, Date.now() - revPhaseStart);
            tm.emitEvent(task.id, {
                type: 'ReviewerResultEmitted',
                attemptNo,
                phaseId: reviewerPhaseId,
                emittedAt: new Date().toISOString(),
                verdict: 'FAIL',
                issues: review.issues || []
            });
            tm.completeAttempt(task.id, 'FAIL');
            // Emit Regeneration Trigger if retrying
            if (i < maxIterations - 1) {
                tm.emitEvent(task.id, {
                    type: 'RegenerationTriggered',
                    fromAttemptNo: attemptNo,
                    triggeredAt: new Date().toISOString(),
                    reasonCode: 'REVIEWER_FAILURE',
                    details: `Attempt ${attemptNo} failed review. Regenerating...`
                });
            }
            else {
                tm.recordPolicyDecision(task.id, 'RetryPolicy', 'DENY', 'Max iterations reached.');
            }
        }
        // Allow next iteration
        currentPrompt = `${prompt}\n\n[FEEDBACK]\n${JSON.stringify(review.issues)}`;
    }
    tm.failTask(task.id, 'Pipeline Failed: Max iterations reached.');
    emitFinalSummary(task.id, 'FAILED', maxIterations);
}
function emitFinalSummary(taskId, outcome, attemptCount) {
    const tm = taskManager_1.TaskManager.getInstance();
    tm.emitEvent(taskId, {
        type: 'TaskTerminated',
        terminatedAt: new Date().toISOString(),
        terminationType: outcome === 'ABORTED' ? 'USER_ABORT' : (outcome === 'SUCCESS' ? 'SYSTEM_ABORT' : 'MAX_ATTEMPTS'),
        humanMessage: `Task ${outcome.toLowerCase()}`
    });
    tm.emitEvent(taskId, {
        type: 'FinalSummaryEmitted',
        emittedAt: new Date().toISOString(),
        outcome,
        attemptCount
    });
}
//# sourceMappingURL=pipelineRun.js.map