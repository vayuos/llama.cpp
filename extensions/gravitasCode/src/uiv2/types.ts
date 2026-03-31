import * as vscode from 'vscode';

/**
 * Global unique Task ID (UUID v4)
 */
export type TaskId = string;

/**
 * Monotonic lifecycle states for a Task Shell.
 * Transitions must follow the order defined here.
 */
export enum TaskState {
    CREATED = 'CREATED',
    RUNNING = 'RUNNING',
    FINALIZING = 'FINALIZING',
    COMPLETED = 'COMPLETED',
    FAILED = 'FAILED',
    ABORTED = 'ABORTED'
}

/**
 * Represents a single execution attempt within a Task.
 * Logic/Reasoning happens here.
 */
export interface TaskStore {
    [key: string]: Task;
}

/**
 * The Root Execution Container.
 * Immutable Identity. Determine State.
 */
export interface Task {
    readonly id: TaskId;
    readonly createdAt: number;
    readonly origin: 'user' | 'system';
    readonly command: string; // The immutable intent

    status: TaskState;
    updatedAt: number;

    // Parent linkage for regeneration
    parentTaskId?: TaskId;
    regenerationType?: 'REPLAY' | 'REGENERATE_SAME' | 'REGENERATE_MODIFIED';

    attempts: TaskAttempt[];

    // Final result summary (markdown)
    summary?: string;
}

// --- Event Stream Model ---

export type TaskEventType =
    | 'attempt-start' | 'attempt-complete' // Boundary
    | 'phase-start' | 'phase-complete'     // Agent Execution
    | 'thought'                            // Internal Reasoning
    | 'tool-use'                           // Concrete Actions
    | 'tool-complete'                      // Action Result
    | 'progress-step'
    | 'terminal'                           // Raw Output
    | 'verdict'                            // Reviewer Decision
    | 'review-result'                      // Detailed Review Artifact
    | 'regeneration-trigger'               // Retry Causality
    | 'system';                            // Control Messages

// ...


export interface BaseTaskEvent {
    eventId: string;
    type: string;
    timestamp: string; // ISO-8601
    taskId: string;
}

export interface TaskCreatedEvent extends BaseTaskEvent {
    type: 'TaskCreated';
    createdAt: string;
    origin: 'user' | 'system';
    command: string;
    metadata?: any;
}

export interface AttemptStartEvent extends BaseTaskEvent {
    type: 'AttemptStarted';
    attemptNo: number;
    startedAt: string;
    initiator: 'system' | 'coder' | 'user';
    contextSummary?: string;
}

export interface ThoughtStartedEvent extends BaseTaskEvent {
    type: 'ThoughtStarted';
    attemptNo: number;
    phaseId: string;
    thoughtId: string;
    startedAt: string;
    visibility?: 'VISIBLE' | 'COLLAPSED_BY_DEFAULT';
}

export interface ThoughtCompletedEvent extends BaseTaskEvent {
    type: 'ThoughtCompleted';
    attemptNo: number;
    phaseId: string;
    thoughtId: string;
    endedAt: string;
    durationMs: number;
    content: string;
}

export interface ToolExecutionStartedEvent extends BaseTaskEvent {
    type: 'ToolExecutionStarted';
    attemptNo: number;
    phaseId: string;
    toolExecId: string;
    commandLine: string;
    workingDirectory: string;
    startedAt: string;
    commandLabel?: string;
    env?: Record<string, string>;
}

export interface ToolExecutionOutputEvent extends BaseTaskEvent {
    type: 'ToolExecutionOutput';
    attemptNo: number;
    phaseId: string;
    toolExecId: string;
    stream: 'stdout' | 'stderr';
    text: string;
}

export interface ToolExecutionCompletedEvent extends BaseTaskEvent {
    type: 'ToolExecutionCompleted';
    attemptNo: number;
    phaseId: string;
    toolExecId: string;
    endedAt: string;
    exitCode: number;
    status: 'RUNNING' | 'SUCCESS' | 'FAILURE';
    outputSummary?: string;
}

export interface ReviewIssue {
    type: 'correctness' | 'security' | 'architecture' | 'performance';
    file: string;
    line: number;
    message: string;
    severity?: 'error' | 'warning' | 'info'; // UI enhancement
}

export interface ReviewerResultEmittedEvent extends BaseTaskEvent {
    type: 'ReviewerResultEmitted';
    attemptNo: number;
    phaseId: string;
    emittedAt: string;
    verdict: 'PASS' | 'FAIL';
    issues: ReviewIssue[]; 
}

export interface CoderResultEmittedEvent extends BaseTaskEvent {
    type: 'CoderResultEmitted';
    attemptNo: number;
    phaseId: string;
    emittedAt: string;
    content: string;
    file?: string;
}

export interface StreamingChunkEmittedEvent extends BaseTaskEvent {
    type: 'StreamingChunkEmitted';
    attemptNo: number;
    phaseId: string;
    chunk: string;
    stage: 'thought' | 'implementation' | 'review';
}

export interface ArtifactProducedEvent extends BaseTaskEvent {
    type: 'ArtifactProduced';
    artifactId: string;
    name: string;
    artifactType: 'file' | 'package' | 'diff' | 'report' | 'other';
    path: string;
    producedByAttempt: number;
    producedAt: string;
    metadata?: any;
}

export interface AttemptClosedEvent extends BaseTaskEvent {
    type: 'AttemptClosed';
    attemptNo: number;
    closedAt: string;
    verdict: AttemptVerdict;
    summary: string;
}

export interface PhaseStartedEvent extends BaseTaskEvent {
    type: 'PhaseStarted';
    attemptNo: number;
    phaseId: string;
    actor: 'coder' | 'reviewer' | 'system';
    title: string;
    startedAt: string;
}

export interface PhaseCompletedEvent extends BaseTaskEvent {
    type: 'PhaseCompleted';
    attemptNo: number;
    phaseId: string;
    endedAt: string;
    status: PhaseState;
}

export interface AbortTriggeredEvent extends BaseTaskEvent {
    type: 'AbortTriggered';
    triggeredAt: string;
    triggeredBy: 'user' | 'system';
    reasonCode: string;
    humanMessage: string;
}

export interface TaskTerminatedEvent extends BaseTaskEvent {
    type: 'TaskTerminated';
    terminatedAt: string;
    terminationType: 'USER_ABORT' | 'MAX_ATTEMPTS' | 'SYSTEM_ABORT';
    terminationReasonCode?: string;
    humanMessage?: string;
    details?: any;
}

export interface FinalSummaryEmittedEvent extends BaseTaskEvent {
    type: 'FinalSummaryEmitted';
    emittedAt: string;
    outcome: 'SUCCESS' | 'FAILED' | 'ABORTED' | 'INCOMPLETE';
    attemptCount: number;
    failedAttempts?: number;
    artifactCount?: number;
    artifactsSummary?: any[];
    nextAction?: string;
}

export interface MaxAttemptsReachedEvent extends BaseTaskEvent {
    type: 'MaxAttemptsReached';
    timestamp: string;
    maxAttempts: number;
    attemptCount: number;
    policy?: any;
}

export interface SystemFatalErrorEvent extends BaseTaskEvent {
    type: 'SystemFatalError';
    timestamp: string;
    errorCode: string;
    message: string;
    trace?: string;
}

export interface RegenerationTriggeredEvent extends BaseTaskEvent {
    type: 'RegenerationTriggered';
    fromAttemptNo: number;
    triggeredAt: string;
    reasonCode: string;
    details: string;
}

export interface TaskStateChangedEvent extends BaseTaskEvent {
    type: 'TaskStateChanged';
    previousState: TaskState;
    newState: TaskState;
    reason?: string;
}

export interface ResourceUsageSampledEvent extends BaseTaskEvent {
    type: 'ResourceUsageSampled';
    resources: {
        ramMb: number;
        vramMb: number;
        cpuPercent: number;
        diskIo?: any;
    };
}

export interface ResourceLimitExceededEvent extends BaseTaskEvent {
    type: 'ResourceLimitExceeded';
    resourceType: 'RAM' | 'VRAM' | 'CPU' | 'DISK' | 'TIME';
    limitValue: number;
    actualValue: number;
    severity: 'WARNING' | 'CRITICAL';
}

export interface PolicyEvaluatedEvent extends BaseTaskEvent {
    type: 'PolicyEvaluated';
    policyName: string;
    decision: 'ALLOW' | 'DENY' | 'ABORT' | 'OVERRIDE';
    reasoning: string;
    context?: any;
}

export interface UserActionPerformedEvent extends BaseTaskEvent {
    type: 'UserActionPerformed';
    actionType: 'REGENERATE_CLICKED' | 'ARTIFACT_VIEWED' | 'ABORT_CLICKED' | 'SETTINGS_CHANGED';
    targetId?: string;
    metadata?: any;
}

export interface AgentBoundToPhaseEvent extends BaseTaskEvent {
    type: 'AgentBoundToPhase';
    phaseId: string;
    agentId: string;
    modelId: string;
    configFingerprint?: string;
    parameters?: any;
}

export interface ArtifactValidatedEvent extends BaseTaskEvent {
    type: 'ArtifactValidated';
    artifactId: string;
    validatorId: string;
    status: 'PASS' | 'FAIL' | 'PARTIAL';
    message?: string;
    details?: any;
}

export interface ArtifactConsumedEvent extends BaseTaskEvent {
    type: 'ArtifactConsumed';
    artifactId: string;
    consumerId: string;
    context?: any;
}

export interface ArtifactRejectedEvent extends BaseTaskEvent {
    type: 'ArtifactRejected';
    artifactId: string;
    rejectedBy: string;
    reason: string;
    details?: any;
}

export interface ReplayVerifiedEvent extends BaseTaskEvent {
    type: 'ReplayVerified';
    streamHash: string;
    status: 'VERIFIED' | 'DRIFT_DETECTED';
    driftDetails?: string;
}

export interface PhaseMetricsReportedEvent extends BaseTaskEvent {
    type: 'PhaseMetricsReported';
    phaseId: string;
    durationMs: number;
    tokenCount?: number;
    costEstimate?: number;
    efficiencyScore?: number;
}

export type TaskEvent =
    | TaskCreatedEvent
    | AttemptStartEvent
    | AttemptClosedEvent
    | PhaseStartedEvent
    | PhaseCompletedEvent
    | ThoughtStartedEvent
    | ThoughtCompletedEvent
    | ToolExecutionStartedEvent
    | ToolExecutionOutputEvent
    | ToolExecutionCompletedEvent
    | ReviewerResultEmittedEvent
    | CoderResultEmittedEvent
    | StreamingChunkEmittedEvent
    | AbortTriggeredEvent
    | RegenerationTriggeredEvent
    | TaskStateChangedEvent
    | ResourceUsageSampledEvent
    | ResourceLimitExceededEvent
    | PolicyEvaluatedEvent
    | UserActionPerformedEvent
    | AgentBoundToPhaseEvent
    | ArtifactValidatedEvent
    | ArtifactConsumedEvent
    | ArtifactRejectedEvent
    | ArtifactProducedEvent
    | ReplayVerifiedEvent
    | PhaseMetricsReportedEvent
    | TaskTerminatedEvent
    | FinalSummaryEmittedEvent
    | MaxAttemptsReachedEvent
    | SystemFatalErrorEvent
    | BaseTaskEvent;


/**
 * Represents a single execution attempt within a Task.
 * Logic/Reasoning happens here.
 * Now acts as a container for related events.
 */
export type AttemptVerdict = 'PASS' | 'FAIL' | 'INCOMPLETE';
export type AttemptState = 'OPEN' | 'CLOSED';
export type PhaseActor = 'coder' | 'reviewer' | 'system';
export type PhaseState = 'CREATED' | 'RUNNING' | 'COMPLETED' | 'FAILED';

export interface TaskPhase {
    id: string;
    actor: PhaseActor;
    title: string;
    startedAt: number;
    endedAt?: number;
    status: PhaseState;
    events: TaskEvent[]; // Events are now strictly contained in Phases
}

export interface TaskAttempt {
    id: string; // Internal UUID
    attemptNo: number; // 1-based index
    startedAt: number;
    endedAt?: number;
    state: AttemptState;
    verdict?: AttemptVerdict;
    phases: TaskPhase[]; // Hierarchical structure
}
