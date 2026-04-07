import { Task, TaskEvent, TaskState, TaskAttempt, TaskPhase } from './types';
import { v4 as uuidv4 } from 'uuid';

/**
 * Derives the complete hierarchical Task state from a flat stream of events.
 * (Gap 3: Pure Reducer)
 */
export function reduceTask(taskId: string, events: TaskEvent[]): Task {
    // Initial skeleton
    let task: Task = {
        id: taskId,
        createdAt: 0,
        origin: 'user',
        command: 'unknown',
        status: TaskState.CREATED,
        updatedAt: 0,
        attempts: []
    };

    // Sort events by timestamp just in case
    const sortedEvents = [...events].sort((a, b) =>
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    for (const event of sortedEvents) {
        task = applyEvent(task, event);
    }

    return task;
}

export function applyEvent(task: Task, event: any): Task {
    const timestamp = new Date(event.timestamp).getTime();
    if (timestamp > task.updatedAt) {
        task.updatedAt = timestamp;
    }

    switch (event.type) {
        case 'TaskCreated':
            (task as any).createdAt = timestamp;
            (task as any).origin = event.origin;
            (task as any).command = event.command || 'unknown';
            break;

        case 'TaskStateChanged':
            task.status = event.newState;
            break;
        
        case 'TaskStatusEmitted':
            (task as any).operationalStatus = event.status; // Real-time feedback
            break;

        case 'AttemptStarted':
            task.attempts.push({
                id: uuidv4(), // Internal UI ID, not stored in event
                attemptNo: event.attemptNo,
                startedAt: new Date(event.startedAt).getTime(),
                state: 'OPEN',
                phases: []
            });
            break;

        case 'AttemptClosed':
            const attempt = findAttempt(task, event.attemptNo);
            if (attempt) {
                attempt.state = 'CLOSED';
                attempt.endedAt = new Date(event.closedAt).getTime();
                attempt.verdict = event.verdict;
            }
            break;

        case 'PhaseStarted':
            const att = findAttempt(task, event.attemptNo);
            if (att) {
                att.phases.push({
                    id: event.phaseId,
                    actor: event.actor,
                    title: event.title,
                    startedAt: new Date(event.startedAt).getTime(),
                    status: 'RUNNING',
                    events: []
                });
            }
            break;

        case 'PhaseCompleted':
            const phase = findPhase(task, event.attemptNo, event.phaseId);
            if (phase) {
                phase.status = 'COMPLETED';
                phase.endedAt = new Date(event.endedAt).getTime();
            }
            break;

        case 'PhaseFailed':
            const fPhase = findPhase(task, event.attemptNo, event.phaseId);
            if (fPhase) {
                fPhase.status = 'FAILED';
                fPhase.endedAt = new Date(event.failedAt).getTime();
            }
            break;

        case 'TaskTerminated':
            task.status = mapTerminationToState(event.terminationType);
            break;

        case 'AbortTriggered':
            task.status = TaskState.ABORTED;
            task.summary = 'Aborted by User';
            break;

        case 'FinalSummaryEmitted':
            task.summary = event.outcome === 'SUCCESS' ? 'Success' : 'Failed';
            break;

        // Routing for specific execution events (Thoughts, Tools, Logs)
        default:
            routeExecutionEvent(task, event);
            break;
    }

    return task;
}

function findAttempt(task: Task, attemptNo: number): TaskAttempt | undefined {
    return task.attempts.find(a => a.attemptNo === attemptNo);
}

function findPhase(task: Task, attemptNo: number, phaseId: string): TaskPhase | undefined {
    const attempt = findAttempt(task, attemptNo);
    return attempt?.phases.find(p => p.id === phaseId);
}

function routeExecutionEvent(task: Task, event: any) {
    if (!event.attemptNo) return;

    const attempt = findAttempt(task, event.attemptNo);
    if (!attempt) return;

    // Determine target phase
    let targetPhase: TaskPhase | undefined;
    if (event.phaseId) {
        targetPhase = attempt.phases.find(p => p.id === event.phaseId);
    } else {
        // Fallback to last active phase
        targetPhase = attempt.phases[attempt.phases.length - 1];
    }

    if (targetPhase) {
        targetPhase.events.push(event as TaskEvent);
    }
}

function mapTerminationToState(type: string): TaskState {
    switch (type) {
        case 'USER_ABORT': return TaskState.ABORTED;
        case 'MAX_ATTEMPTS': return TaskState.FAILED;
        case 'SYSTEM_ABORT': return TaskState.FAILED;
        default: return TaskState.FAILED;
    }
}
