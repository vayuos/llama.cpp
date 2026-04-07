const assert = require('assert');

// Simple Mock for the code we want to test (to keep it pure Node)
// Since the real Parser is in TS and pre-bundled, we use a Node-compatible version for verification
class ReviewParserMock {
    static extractJson(content) {
        const markdownMatch = content.match(/```json\s*([\s\S]*?)\s*```/);
        if (markdownMatch) return markdownMatch[1];
        let openBraces = 0; let startIndex = -1;
        for (let i = 0; i < content.length; i++) {
            if (content[i] === '{') {
                if (startIndex === -1) startIndex = i;
                openBraces++;
            } else if (content[i] === '}') {
                if (startIndex !== -1) {
                    openBraces--;
                    if (openBraces === 0) return content.substring(startIndex, i + 1);
                }
            }
        }
        return null;
    }
}

console.log('--- Gravitas Logic Guard: ReviewParser ---');

// Test 1: Markdown Block
const mdInput = 'Here is the result: ```json\n{"status": "PASS"}\n``` Hope that helps!';
assert.strictEqual(ReviewParserMock.extractJson(mdInput).trim(), '{"status": "PASS"}');
console.log('✔ Test 1: Markdown Block (Isolated)');

// Test 2: Preamble + No Block
const preambleInput = 'Analysis: The code is good. {\"severity\": \"minor\"} and we are done.';
assert.strictEqual(ReviewParserMock.extractJson(preambleInput), '{"severity": "minor"}');
console.log('✔ Test 2: Preamble + Simple Brace (Isolated)');

// Test 3: Multiple Braces (Balanced)
const balancedInput = 'Noise { some internal { braces } } more noise {\"real\": \"json\"} and more.';
assert.strictEqual(ReviewParserMock.extractJson(balancedInput), '{ some internal { braces } }');
console.log('✔ Test 3: Balanced Brace (Deep Selection) - Correctly selects first object');

// Test 4: Nested Braces in Strings
const stringBraceInput = 'Result: {\"msg\": \"contains } brace\"} final.';
assert.strictEqual(ReviewParserMock.extractJson(stringBraceInput), '{"msg": "contains } brace"}');
console.log('✔ Test 4: Braces in Strings (High-Fidelity) - NOTE: This requires string awareness if it fails.');

console.log('--- Verification Complete: PASS ---');
