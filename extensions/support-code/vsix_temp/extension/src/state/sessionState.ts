export enum SessionStatus {
    IDLE = 'idle',
    CODER_RUNNING = 'coder_running',
    REVIEWER_RUNNING = 'reviewer_running',
    COMPLETED = 'completed',
    FAILED = 'failed'
}

export interface SessionState {
    id: string;
    status: SessionStatus;
    iterations: number;
    startTime: number;
}

export class StateManager {
    private currentSession: SessionState | null = null;

    startSession() {
        this.currentSession = {
            id: Math.random().toString(36).substring(7),
            status: SessionStatus.IDLE,
            iterations: 0,
            startTime: Date.now()
        };
        return this.currentSession;
    }

    updateStatus(status: SessionStatus) {
        if (this.currentSession) {
            this.currentSession.status = status;
        }
    }

    incrementIteration() {
        if (this.currentSession) {
            this.currentSession.iterations++;
        }
    }

    getSession() {
        return this.currentSession;
    }
}
