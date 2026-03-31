import { verifyContentIntegrity, calculateHash } from '../uiv2/integrity';

/**
 * Integrity Test: Verify that drift detection works and hash parity is maintained.
 */
function testIntegrity() {
    console.log('--- Phase 1: Establish Baseline ---');
    const content = '{"type":"TaskCreated","command":"test"}\n';
    const result1 = verifyContentIntegrity(content);
    console.log('Result 1 (Baseline):', result1.verified ? 'VERIFIED' : 'FAILED', result1.hash);
    if (!result1.verified || result1.driftDetails !== 'Baseline established (No previous verification found)') {
        console.error('❌ Expected baseline establishment');
        process.exit(1);
    }

    console.log('--- Phase 2: Verify with Embedded Hash ---');
    const contentWithHash = content + JSON.stringify({
        type: 'ReplayVerified',
        streamHash: result1.hash,
        status: 'VERIFIED'
    }) + '\n';

    const result2 = verifyContentIntegrity(contentWithHash);
    console.log('Result 2 (Unchanged):', result2.verified ? 'VERIFIED' : 'FAILED');
    if (!result2.verified) {
        console.error('❌ Expected verified status for unchanged ledger');
        process.exit(1);
    }

    console.log('--- Phase 3: Add Trailing Events (Growth) ---');
    const grownContent = contentWithHash + '{"type":"ThoughtStarted","content":"New thought"}\n';
    const result3 = verifyContentIntegrity(grownContent);
    console.log('Result 3 (Grown):', result3.verified ? 'VERIFIED_WITH_TAIL' : 'FAILED');
    if (!result3.verified || !result3.driftDetails?.includes('unverified trailing events')) {
        console.error('❌ Expected warning for unverified trailing events');
        process.exit(1);
    }

    console.log('--- Phase 4: Induce Drift in Verified History ---');
    // Modify a line before the ReplayVerified event
    const tamperedContent = '{"type":"TaskCreated","command":"TAMPERED"}\n' + contentWithHash.split('\n')[1] + '\n';
    const result4 = verifyContentIntegrity(tamperedContent);
    console.log('Result 4 (Tampered History):', result4.verified ? 'VERIFIED' : 'DRIFT_DETECTED');
    console.log('Drift Details:', result4.driftDetails);

    if (result4.verified) {
        console.error('❌ Expected drift detection for tampered history');
        process.exit(1);
    } else {
        console.log('✅ Integrity Test PASSED');
    }
}

testIntegrity();
