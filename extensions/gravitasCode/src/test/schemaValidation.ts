import { EventValidator } from '../uiv2/eventValidator';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Schema Validation Test: Verifies all schemas in events.json.
 */
function testSchemaValidation() {
    const validator = EventValidator.getInstance();
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
        } else {
            console.error(`❌ ${type}: Minimal Valid Payload FAILED`, result.errors);
            failedCount++;
        }

        // Test 2: Invalid Payload (Missing eventId)
        const invalidPayload = { ...validPayload };
        delete (invalidPayload as any).eventId;
        const resultInvalid = validator.validate(type, invalidPayload);
        if (!resultInvalid.valid) {
            console.log(`✅ ${type}: Invalid Payload (Missing eventId) correctly REJECTED`);
        } else {
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
function generateMinimalValidPayload(type: string, schema: any, depth: number = 0): any {
    if (depth > 3) return {}; // Prevent infinite recursion

    const payload: any = {};
    if (depth === 0) {
        payload.eventId = 'test-uuid';
        payload.taskId = 'task-uuid';
        payload.timestamp = new Date().toISOString();
        payload.type = type;
    }

    if (schema.required) {
        schema.required.forEach((prop: string) => {
            if (payload[prop] !== undefined) return;

            const propSchema = schema.properties?.[prop];
            if (!propSchema) return;

            if (propSchema.enum) {
                payload[prop] = propSchema.enum[0];
            } else if (propSchema.type === 'string') {
                payload[prop] = prop === 'timestamp' || prop.endsWith('At') ? new Date().toISOString() : 'test-value';
            } else if (propSchema.type === 'number' || propSchema.type === 'integer') {
                payload[prop] = 1;
            } else if (propSchema.type === 'boolean') {
                payload[prop] = true;
            } else if (propSchema.type === 'object') {
                payload[prop] = generateMinimalValidPayload('nested', propSchema, depth + 1);
            } else if (propSchema.type === 'array') {
                payload[prop] = [generateMinimalValidPayload('item', propSchema.items, depth + 1)];
            }
        });
    }

    return payload;
}

testSchemaValidation();
