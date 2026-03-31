"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.GravitasConfigSchema = exports.ReviewerConfigSchema = exports.CoderConfigSchema = void 0;
const zod_1 = require("zod");
exports.CoderConfigSchema = zod_1.z.object({
    endpoint: zod_1.z.string().url(),
    model: zod_1.z.string().optional()
});
exports.ReviewerConfigSchema = zod_1.z.object({
    endpoint: zod_1.z.string().url(),
    model: zod_1.z.string().optional()
});
exports.GravitasConfigSchema = zod_1.z.object({
    coder: exports.CoderConfigSchema,
    reviewer: exports.ReviewerConfigSchema,
    safety: zod_1.z.object({
        maxDiffLines: zod_1.z.number().default(1000),
        requireApproval: zod_1.z.boolean().default(true)
    }).optional()
});
//# sourceMappingURL=configSchema.js.map