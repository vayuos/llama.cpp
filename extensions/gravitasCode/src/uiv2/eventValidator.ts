import Ajv from 'ajv';
import * as fs from 'fs';
import * as path from 'path';

export class EventValidator {
    private static instance: EventValidator;
    private ajv: Ajv;
    private schemas: Map<string, any> = new Map();

    private constructor() {
        this.ajv = new Ajv({
            allErrors: true,
            strict: false,
            // formats: { 'date-time': true } // Usually needs ajv-formats
        });
        this.loadSchemas();
    }

    public static getInstance(): EventValidator {
        if (!EventValidator.instance) {
            EventValidator.instance = new EventValidator();
        }
        return EventValidator.instance;
    }

    private loadSchemas() {
        const registryPath = path.join(__dirname, '..', 'schemas', 'events.json');

        try {
            if (fs.existsSync(registryPath)) {
                const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
                for (const [key, schema] of Object.entries(registry)) {
                    this.schemas.set(key, schema);
                    this.ajv.addSchema(schema as any, key);
                }
            } else {
                console.error('Event registry not found at:', registryPath);
            }
        } catch (err) {
            console.error('Failed to load schemas:', err);
        }
    }

    /**
     * Validate an event against its corresponding schema.
     * @param type The event type (e.g. 'TaskCreated')
     * @param event The event object
     */
    public validate(type: string, event: any): { valid: boolean; errors?: string[] } {
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
