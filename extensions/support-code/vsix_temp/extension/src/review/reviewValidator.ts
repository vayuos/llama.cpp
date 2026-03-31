import Ajv from 'ajv';
import * as schema from '../prompts/reviewSchema.json';
import { ReviewerOutput } from './reviewTypes';

export class ReviewValidator {
    private static ajv = new Ajv();
    private static validateFn = ReviewValidator.ajv.compile(schema);

    static validate(rawData: string): ReviewerOutput {
        try {
            const parsed = JSON.parse(rawData);
            const valid = ReviewValidator.validateFn(parsed);

            if (!valid) {
                const errors = ReviewValidator.validateFn.errors?.map((e: any) => e.message).join(', ');
                throw new Error(`Schema mismatch: ${errors}`);
            }

            return parsed as unknown as ReviewerOutput;
        } catch (error: any) {
            throw new Error(`Reviewer output validation failed: ${error.message}`);
        }
    }
}
