"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ValidationEngine = void 0;
class ValidationEngine {
    constructor() {
        this.steps = [];
        this.logs = [];
    }
    addStep(step) {
        this.steps.push(step);
    }
    async run(config) {
        this.logs = [];
        console.log('Starting validation pipeline...');
        for (const step of this.steps) {
            this.logs.push(`[STEP] ${step.name}`);
            try {
                const result = await step.execute(config);
                if (!result.success) {
                    this.logs.push(`[FAILURE] ${step.name}: ${result.message}`);
                    if (step.rollback) {
                        this.logs.push(`[ROLLBACK] Triggered for ${step.name}`);
                        await step.rollback();
                    }
                    return { success: false, message: result.message, logs: this.logs };
                }
                this.logs.push(`[SUCCESS] ${step.name}`);
            }
            catch (e) {
                this.logs.push(`[ERROR] ${step.name}: ${e.message}`);
                return { success: false, message: e.message, logs: this.logs };
            }
        }
        return { success: true, message: 'All validation steps passed.', logs: this.logs };
    }
    getLogs() {
        return this.logs;
    }
}
exports.ValidationEngine = ValidationEngine;
//# sourceMappingURL=validator.js.map