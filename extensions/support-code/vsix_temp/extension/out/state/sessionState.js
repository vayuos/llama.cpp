"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.StateManager = exports.SessionStatus = void 0;
var SessionStatus;
(function (SessionStatus) {
    SessionStatus["IDLE"] = "idle";
    SessionStatus["CODER_RUNNING"] = "coder_running";
    SessionStatus["REVIEWER_RUNNING"] = "reviewer_running";
    SessionStatus["COMPLETED"] = "completed";
    SessionStatus["FAILED"] = "failed";
})(SessionStatus || (exports.SessionStatus = SessionStatus = {}));
class StateManager {
    constructor() {
        this.currentSession = null;
    }
    startSession() {
        this.currentSession = {
            id: Math.random().toString(36).substring(7),
            status: SessionStatus.IDLE,
            iterations: 0,
            startTime: Date.now()
        };
        return this.currentSession;
    }
    updateStatus(status) {
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
exports.StateManager = StateManager;
//# sourceMappingURL=sessionState.js.map