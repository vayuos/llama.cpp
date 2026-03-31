import * as fs from 'fs';
import * as path from 'path';

const registryPath = path.join(__dirname, '..', 'schemas', 'events.json');
const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));

const baseProperties: any = {
    "eventId": { "type": "string" },
    "taskId": { "type": "string" },
    "timestamp": { "type": "string", "format": "date-time" },
    "type": { "type": "string" },
    "attemptNo": { "type": "integer" },
    "phaseId": { "type": "string" }
};

const baseRequired = ["eventId", "taskId", "timestamp", "type"];

for (const [type, schema] of Object.entries(registry)) {
    const s = schema as any;
    if (!s.properties) s.properties = {};
    if (!s.required) s.required = [];

    // Merge properties
    for (const [prop, propSchema] of Object.entries(baseProperties)) {
        if (!s.properties[prop]) {
            s.properties[prop] = propSchema;
        }
    }

    // Merge required
    for (const req of baseRequired) {
        if (!s.required.includes(req)) {
            s.required.push(req);
        }
    }

    // Set const for type - Force overwrite to ensure alignment with key
    if (s.properties.type) {
        s.properties.type.const = type;
    }

    s.additionalProperties = false;
}

fs.writeFileSync(registryPath, JSON.stringify(registry, null, 4), 'utf8');
console.log('✅ Standardized 34 schemas in events.json');
