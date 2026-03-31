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
exports.EventValidator = void 0;
const ajv_1 = __importDefault(require("ajv"));
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
class EventValidator {
    constructor() {
        this.schemas = new Map();
        this.ajv = new ajv_1.default({
            allErrors: true,
            strict: false,
            // formats: { 'date-time': true } // Usually needs ajv-formats
        });
        this.loadSchemas();
    }
    static getInstance() {
        if (!EventValidator.instance) {
            EventValidator.instance = new EventValidator();
        }
        return EventValidator.instance;
    }
    loadSchemas() {
        const registryPath = path.join(__dirname, '..', 'schemas', 'events.json');
        try {
            if (fs.existsSync(registryPath)) {
                const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
                for (const [key, schema] of Object.entries(registry)) {
                    this.schemas.set(key, schema);
                    this.ajv.addSchema(schema, key);
                }
            }
            else {
                console.error('Event registry not found at:', registryPath);
            }
        }
        catch (err) {
            console.error('Failed to load schemas:', err);
        }
    }
    /**
     * Validate an event against its corresponding schema.
     * @param type The event type (e.g. 'TaskCreated')
     * @param event The event object
     */
    validate(type, event) {
        // Schema keys now match event types exactly (no .v1)
        const schema = this.schemas.get(type);
        if (!schema) {
            return { valid: false, errors: [`No schema found for event type: ${type}`] };
        }
        const validateFn = this.ajv.getSchema(type); // We added it with this.ajv.addSchema(schema, key)
        if (!validateFn) {
            return { valid: false, errors: [`Failed to compile schema for: ${type}`] };
        }
        const valid = validateFn(event);
        if (!valid) {
            return {
                valid: false,
                errors: validateFn.errors?.map(e => `${e.instancePath} ${e.message}`) || ['Unknown validation error']
            };
        }
        return { valid: true };
    }
}
exports.EventValidator = EventValidator;
//# sourceMappingURL=eventValidator.js.map