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
Object.defineProperty(exports, "__esModule", { value: true });
const eventValidator_1 = require("../uiv2/eventValidator");
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
/**
 * Schema Validation Test: Verifies all schemas in events.json.
 */
function testSchemaValidation() {
    const validator = eventValidator_1.EventValidator.getInstance();
    const registryPath = path.join(__dirname, '..', 'schemas', 'events.json');
    const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
    const types = Object.keys(registry);
    console.log(`--- Testing ${types.length} Event Schemas ---`);
    let passedCount = 0;
    let failedCount = 0;
    for (const type of types) {
        // Test 1: Minimal Valid Payload (Mock)
        const validPayload = generateMinimalValidPayload(type, registry[type]);
        const result = validator.validate(type, validPayload);
        if (result.valid) {
            console.log(`✅ ${type}: Minimal Valid Payload PASSED`);
            passedCount++;
        }
        else {
            console.error(`❌ ${type}: Minimal Valid Payload FAILED`, result.errors);
            failedCount++;
        }
        // Test 2: Invalid Payload (Missing eventId)
        const invalidPayload = { ...validPayload };
        delete invalidPayload.eventId;
        const resultInvalid = validator.validate(type, invalidPayload);
        if (!resultInvalid.valid) {
            console.log(`✅ ${type}: Invalid Payload (Missing eventId) correctly REJECTED`);
        }
        else {
            console.error(`❌ ${type}: Invalid Payload (Missing eventId) was incorrectly ACCEPTED`);
            failedCount++;
        }
    }
    console.log(`--- Final Results: ${passedCount} Passed, ${failedCount} Failed ---`);
    if (failedCount > 0) {
        process.exit(1);
    }
}
/**
 * Helper to generate a dummy valid payload matching the schema.
 */
function generateMinimalValidPayload(type, schema, depth = 0) {
    if (depth > 3)
        return {}; // Prevent infinite recursion
    const payload = {};
    if (depth === 0) {
        payload.eventId = 'test-uuid';
        payload.taskId = 'task-uuid';
        payload.timestamp = new Date().toISOString();
        payload.type = type;
    }
    if (schema.required) {
        schema.required.forEach((prop) => {
            if (payload[prop] !== undefined)
                return;
            const propSchema = schema.properties?.[prop];
            if (!propSchema)
                return;
            if (propSchema.enum) {
                payload[prop] = propSchema.enum[0];
            }
            else if (propSchema.type === 'string') {
                payload[prop] = prop === 'timestamp' || prop.endsWith('At') ? new Date().toISOString() : 'test-value';
            }
            else if (propSchema.type === 'number' || propSchema.type === 'integer') {
                payload[prop] = 1;
            }
            else if (propSchema.type === 'boolean') {
                payload[prop] = true;
            }
            else if (propSchema.type === 'object') {
                payload[prop] = generateMinimalValidPayload('nested', propSchema, depth + 1);
            }
            else if (propSchema.type === 'array') {
                payload[prop] = [generateMinimalValidPayload('item', propSchema.items, depth + 1)];
            }
        });
    }
    return payload;
}
testSchemaValidation();
//# sourceMappingURL=schemaValidation.js.map