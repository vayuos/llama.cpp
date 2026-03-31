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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ReviewValidator = void 0;
const ajv_1 = __importDefault(require("ajv"));
const schema = __importStar(require("../prompts/reviewSchema.json"));
class ReviewValidator {
    static validate(rawData) {
        try {
            const parsed = JSON.parse(rawData);
            const valid = ReviewValidator.validateFn(parsed);
            if (!valid) {
                const errors = ReviewValidator.validateFn.errors?.map((e) => e.message).join(', ');
                throw new Error(`Schema mismatch: ${errors}`);
            }
            return parsed;
        }
        catch (error) {
            throw new Error(`Reviewer output validation failed: ${error.message}`);
        }
    }
}
exports.ReviewValidator = ReviewValidator;
ReviewValidator.ajv = new ajv_1.default();
ReviewValidator.validateFn = ReviewValidator.ajv.compile(schema);
//# sourceMappingURL=reviewValidator.js.map