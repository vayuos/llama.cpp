"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NativeToolRegistry = void 0;
const statTool_1 = require("./statTool");
/**
 * Central registry for all Native (TypeScript) tools.
 * (Gap 6: Hardened Registry)
 */
class NativeToolRegistry {
    constructor() {
        this.tools = new Map();
        this.register(new statTool_1.CodebaseStatTool());
        // Add future native tools here
    }
    static getInstance() {
        if (!NativeToolRegistry.instance) {
            NativeToolRegistry.instance = new NativeToolRegistry();
        }
        return NativeToolRegistry.instance;
    }
    register(tool) {
        this.tools.set(tool.name, tool);
    }
    getTool(name) {
        return this.tools.get(name);
    }
    hasTool(name) {
        return this.tools.has(name);
    }
    getAllTools() {
        return Array.from(this.tools.values());
    }
}
exports.NativeToolRegistry = NativeToolRegistry;
//# sourceMappingURL=registry.js.map