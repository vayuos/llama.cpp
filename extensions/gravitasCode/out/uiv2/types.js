"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TaskState = void 0;
/**
 * Monotonic lifecycle states for a Task Shell.
 * Transitions must follow the order defined here.
 */
var TaskState;
(function (TaskState) {
    TaskState["CREATED"] = "CREATED";
    TaskState["RUNNING"] = "RUNNING";
    TaskState["FINALIZING"] = "FINALIZING";
    TaskState["COMPLETED"] = "COMPLETED";
    TaskState["FAILED"] = "FAILED";
    TaskState["ABORTED"] = "ABORTED";
})(TaskState || (exports.TaskState = TaskState = {}));
//# sourceMappingURL=types.js.map