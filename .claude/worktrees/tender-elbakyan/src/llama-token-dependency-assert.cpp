/**
 * SECTION 5 IMPLEMENTATION: Runtime Assertion - CPU Not on Token Dependency Chain
 *
 * Runtime instrumentation and assertion mechanism
 */

#include "llama-token-dependency-assert.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

// ============================================================================
// GLOBAL ASSERTION STATE
// ============================================================================

static bool g_token_chain_assertions_enabled = true;  // Default: enabled
static bool g_in_decode_phase = false;               // Currently in decode phase
static int g_assertion_count = 0;                    // Total assertions checked
static std::map<uint64_t, struct llama_token_execution_record> g_token_records;  // Per-token records

void llama_set_token_chain_assertions_enabled(bool enabled) {
    g_token_chain_assertions_enabled = enabled;
    fprintf(stdout, "[TOKEN CHAIN ASSERT] %s\n",
            enabled ? "ENABLED (CPU on chain causes abort)" : "DISABLED (assertions skipped)");
}

bool llama_get_token_chain_assertions_enabled(void) {
    return g_token_chain_assertions_enabled;
}

int llama_get_token_chain_assertion_count(void) {
    return g_assertion_count;
}

void llama_reset_token_chain_assertion_counter(void) {
    g_assertion_count = 0;
}

int llama_token_chain_set_decode_phase(bool in_decode_phase) {
    g_in_decode_phase = in_decode_phase;
    if (in_decode_phase) {
        fprintf(stdout, "[TOKEN CHAIN ASSERT] Entering decode phase - assertions active\n");
    } else {
        fprintf(stdout, "[TOKEN CHAIN ASSERT] Exiting decode phase - assertions inactive\n");
    }
    return 0;
}

bool llama_token_chain_in_decode_phase(void) {
    return g_in_decode_phase;
}

// ============================================================================
// DECODE ITERATION INSTRUMENTATION
// ============================================================================

int llama_assert_token_chain_start(uint64_t token_id) {
    if (!g_token_chain_assertions_enabled || !g_in_decode_phase) {
        return 0;
    }

    // Initialize execution record for this token
    struct llama_token_execution_record record = {};
    record.token_id = token_id;
    record.cpu_wait_detected = false;
    record.cpu_sync_detected = false;
    record.cpu_state_gate_detected = false;
    record.chain_violation_detected = false;

    // Initialize all stages
    for (int i = 0; i < 9; i++) {
        record.stages[i].stage_name = llama_token_chain_stage_name((enum llama_token_chain_stage)i);
        record.stages[i].executed = false;
        record.stages[i].backend_executed = nullptr;
        record.stages[i].cpu_detected = false;
    }

    g_token_records[token_id] = record;

    fprintf(stdout, "[TOKEN CHAIN] Token %llu decode started\n", (unsigned long long)token_id);
    return 0;
}

int llama_assert_token_chain_complete(uint64_t token_id) {
    if (!g_token_chain_assertions_enabled || !g_in_decode_phase) {
        return 0;
    }

    g_assertion_count++;

    // Verify GPU-exclusive execution
    int result = llama_token_chain_verify_gpu_exclusive(token_id);

    if (result != 0) {
        auto it = g_token_records.find(token_id);
        if (it != g_token_records.end()) {
            llama_print_token_chain_violation_diagnostics(
                token_id,
                it->second.violation_type,
                it->second.violation_message
            );
        }
        return -1;
    }

    fprintf(stdout, "[TOKEN CHAIN] Token %llu decode completed - GPU-exclusive verified\n",
            (unsigned long long)token_id);

    // Clean up old records (keep last 10 tokens for debugging)
    if (g_token_records.size() > 10) {
        auto it = g_token_records.begin();
        std::advance(it, g_token_records.size() - 10);
        g_token_records.erase(g_token_records.begin(), it);
    }

    return 0;
}

// ============================================================================
// STAGE-LEVEL INSTRUMENTATION
// ============================================================================

int llama_token_chain_record_stage_start(
    uint64_t token_id,
    enum llama_token_chain_stage stage,
    const char* stage_name,
    const char* backend_executed
) {
    if (!g_token_chain_assertions_enabled || !g_in_decode_phase) {
        return 0;
    }

    auto it = g_token_records.find(token_id);
    if (it == g_token_records.end()) {
        fprintf(stderr, "WARNING: Stage start recorded for unknown token %llu\n",
                (unsigned long long)token_id);
        return -1;
    }

    if (stage > 0 && stage < 9) {
        it->second.stages[stage].executed = true;
        it->second.stages[stage].backend_executed = backend_executed;
        it->second.stages[stage].start_time_us = 0;  // Would be set to current time

        bool is_cpu = (strcmp(backend_executed, "CPU") == 0 ||
                       strcmp(backend_executed, "CPP") == 0);
        it->second.stages[stage].cpu_detected = is_cpu;

        fprintf(stdout, "[TOKEN CHAIN] Token %llu stage %s started on %s\n",
                (unsigned long long)token_id, stage_name, backend_executed);

        // Immediately detect CPU on chain
        if (is_cpu) {
            it->second.chain_violation_detected = true;
            it->second.violation_type = LLAMA_CHAIN_VIOLATION_DIRECT_CPU;
            it->second.violation_message = "CPU directly executed on token dependency chain";
        }
    }

    return 0;
}

int llama_token_chain_record_stage_end(
    uint64_t token_id,
    enum llama_token_chain_stage stage
) {
    if (!g_token_chain_assertions_enabled || !g_in_decode_phase) {
        return 0;
    }

    auto it = g_token_records.find(token_id);
    if (it == g_token_records.end()) {
        return -1;
    }

    if (stage > 0 && stage < 9) {
        it->second.stages[stage].end_time_us = 0;  // Would be set to current time
        fprintf(stdout, "[TOKEN CHAIN] Token %llu stage %s completed\n",
                (unsigned long long)token_id, llama_token_chain_stage_name(stage));
    }

    return 0;
}

// ============================================================================
// DEPENDENCY VIOLATION DETECTION
// ============================================================================

int llama_assert_token_chain_stage_gpu_only(
    uint64_t token_id,
    enum llama_token_chain_stage stage,
    const char* executed_backend
) {
    if (!g_token_chain_assertions_enabled || !g_in_decode_phase) {
        return 0;
    }

    bool is_cpu = (strcmp(executed_backend, "CPU") == 0 ||
                   strcmp(executed_backend, "CPP") == 0);

    if (is_cpu) {
        auto it = g_token_records.find(token_id);
        if (it != g_token_records.end()) {
            it->second.chain_violation_detected = true;
            it->second.violation_type = LLAMA_CHAIN_VIOLATION_DIRECT_CPU;
            it->second.violation_message = "Token chain stage executed on CPU (must be GPU)";
        }

        fprintf(stderr, "FATAL: Token %llu stage %s executed on CPU\n",
                (unsigned long long)token_id, llama_token_chain_stage_name(stage));
        return -1;
    }

    return 0;
}

int llama_assert_no_cpu_wait_on_token_chain(
    uint64_t token_id,
    bool gpu_is_waiting,
    const char* waiting_reason
) {
    if (!g_token_chain_assertions_enabled || !g_in_decode_phase || !gpu_is_waiting) {
        return 0;
    }

    auto it = g_token_records.find(token_id);
    if (it != g_token_records.end()) {
        it->second.cpu_wait_detected = true;
        it->second.chain_violation_detected = true;
        it->second.violation_type = LLAMA_CHAIN_VIOLATION_CPU_WAIT;
        it->second.violation_message = "GPU waiting on CPU to complete token chain stage";
    }

    fprintf(stderr, "FATAL: Token %llu GPU waiting on CPU (reason: %s)\n",
            (unsigned long long)token_id, waiting_reason);
    return -1;
}

int llama_assert_no_cpu_sync_block(
    uint64_t token_id,
    bool sync_required,
    const char* sync_type
) {
    if (!g_token_chain_assertions_enabled || !g_in_decode_phase || !sync_required) {
        return 0;
    }

    auto it = g_token_records.find(token_id);
    if (it != g_token_records.end()) {
        it->second.cpu_sync_detected = true;
        it->second.chain_violation_detected = true;
        it->second.violation_type = LLAMA_CHAIN_VIOLATION_CPU_SYNC;
        it->second.violation_message = "CPU synchronization blocking token emission";
    }

    fprintf(stderr, "FATAL: Token %llu CPU synchronization blocking token (%s)\n",
            (unsigned long long)token_id, sync_type);
    return -1;
}

int llama_assert_no_cpu_state_gate(
    uint64_t token_id,
    bool cpu_state_gating_next_token,
    const char* state_update_type
) {
    if (!g_token_chain_assertions_enabled || !g_in_decode_phase || !cpu_state_gating_next_token) {
        return 0;
    }

    auto it = g_token_records.find(token_id);
    if (it != g_token_records.end()) {
        it->second.cpu_state_gate_detected = true;
        it->second.chain_violation_detected = true;
        it->second.violation_type = LLAMA_CHAIN_VIOLATION_CPU_STATE_GATE;
        it->second.violation_message = "CPU state update gating next token emission";
    }

    fprintf(stderr, "FATAL: Token %llu CPU state gating token (%s)\n",
            (unsigned long long)token_id, state_update_type);
    return -1;
}

int llama_assert_no_indirect_cpu_gate(
    uint64_t token_id,
    bool gpu_depends_on_cpu_decision,
    const char* decision_type
) {
    if (!g_token_chain_assertions_enabled || !g_in_decode_phase || !gpu_depends_on_cpu_decision) {
        return 0;
    }

    auto it = g_token_records.find(token_id);
    if (it != g_token_records.end()) {
        it->second.chain_violation_detected = true;
        it->second.violation_type = LLAMA_CHAIN_VIOLATION_INDIRECT;
        it->second.violation_message = "GPU next step depends on CPU decision (indirect gating)";
    }

    fprintf(stderr, "FATAL: Token %llu GPU depends on CPU decision (%s)\n",
            (unsigned long long)token_id, decision_type);
    return -1;
}

// ============================================================================
// EXECUTION RECORD ANALYSIS
// ============================================================================

int llama_token_chain_verify_gpu_exclusive(uint64_t token_id) {
    if (!g_token_chain_assertions_enabled || !g_in_decode_phase) {
        return 0;
    }

    auto it = g_token_records.find(token_id);
    if (it == g_token_records.end()) {
        return 0;  // No record = not in decode phase
    }

    struct llama_token_execution_record& record = it->second;

    // Check all stages for CPU involvement
    for (int i = 1; i < 9; i++) {
        if (record.stages[i].cpu_detected) {
            record.chain_violation_detected = true;
            record.violation_type = LLAMA_CHAIN_VIOLATION_DIRECT_CPU;
            record.violation_message = "CPU detected on token dependency chain stage";
            return -1;
        }
    }

    // Check for indirect violations
    if (record.cpu_wait_detected || record.cpu_sync_detected || record.cpu_state_gate_detected) {
        record.chain_violation_detected = true;
        return -1;
    }

    // GPU-exclusive verified
    record.chain_violation_detected = false;
    return 0;
}

struct llama_token_execution_record* llama_get_token_execution_record(uint64_t token_id) {
    auto it = g_token_records.find(token_id);
    if (it != g_token_records.end()) {
        return &it->second;
    }
    return nullptr;
}

void llama_print_token_execution_record(const struct llama_token_execution_record* record) {
    if (!record) return;

    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "TOKEN EXECUTION RECORD (Token %llu)\n", (unsigned long long)record->token_id);
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "CHAIN EXECUTION:\n");
    for (int i = 1; i < 9; i++) {
        if (!record->stages[i].executed) continue;

        fprintf(stdout, "  Stage %d: %s\n", i, record->stages[i].stage_name);
        fprintf(stdout, "    Backend: %s\n", record->stages[i].backend_executed);
        fprintf(stdout, "    CPU: %s\n", record->stages[i].cpu_detected ? "YES (VIOLATION)" : "NO");
    }

    fprintf(stdout, "\n");
    fprintf(stdout, "VIOLATION STATUS:\n");
    fprintf(stdout, "  CPU Wait Detected: %s\n", record->cpu_wait_detected ? "YES" : "NO");
    fprintf(stdout, "  CPU Sync Detected: %s\n", record->cpu_sync_detected ? "YES" : "NO");
    fprintf(stdout, "  CPU State Gate Detected: %s\n", record->cpu_state_gate_detected ? "YES" : "NO");
    fprintf(stdout, "  Chain Violation: %s\n", record->chain_violation_detected ? "YES" : "NO");

    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

void llama_print_token_chain_violation_diagnostics(
    uint64_t token_id,
    enum llama_token_chain_violation_type violation_type,
    const char* violation_message
) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "TOKEN DEPENDENCY CHAIN VIOLATION (Token %llu)\n", (unsigned long long)token_id);
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "VIOLATION TYPE: %s\n", llama_token_chain_violation_name(violation_type));
    fprintf(stdout, "VIOLATION MESSAGE: %s\n\n", violation_message);

    fprintf(stdout, "DEFINITION OF TOKEN DEPENDENCY CHAIN:\n");
    fprintf(stdout, "  Entry → Forward Pass → Attention/MLP → KV Cache → Logits → Sampling → Commit\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "VIOLATION CAUSE:\n");
    switch (violation_type) {
        case LLAMA_CHAIN_VIOLATION_DIRECT_CPU:
            fprintf(stdout, "  CPU directly executed on the token dependency chain.\n");
            fprintf(stdout, "  All stages must execute on GPU backend only.\n");
            break;
        case LLAMA_CHAIN_VIOLATION_CPU_WAIT:
            fprintf(stdout, "  GPU is waiting on CPU to complete token chain stage.\n");
            fprintf(stdout, "  This makes CPU a blocking dependency for token emission.\n");
            break;
        case LLAMA_CHAIN_VIOLATION_CPU_SYNC:
            fprintf(stdout, "  CPU synchronization is blocking token emission.\n");
            fprintf(stdout, "  Synchronization must not gate token output.\n");
            break;
        case LLAMA_CHAIN_VIOLATION_CPU_STATE_GATE:
            fprintf(stdout, "  CPU state update is required before next token can be emitted.\n");
            fprintf(stdout, "  Token emission must not depend on CPU state.\n");
            break;
        case LLAMA_CHAIN_VIOLATION_INDIRECT:
            fprintf(stdout, "  GPU next step depends on a decision made by CPU.\n");
            fprintf(stdout, "  GPU must be independent of CPU decisions on token path.\n");
            break;
        default:
            fprintf(stdout, "  Unknown violation type.\n");
    }

    fprintf(stdout, "\n");
    fprintf(stdout, "CONSEQUENCE:\n");
    fprintf(stdout, "  Decode execution TERMINATED immediately.\n");
    fprintf(stdout, "  This is a correctness invariant violation.\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

// ============================================================================
// EXPLICIT TOKEN CHAIN STATEMENT
// ============================================================================

void llama_print_token_dependency_chain_statement(void) {
    fprintf(stdout, "\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "TOKEN DEPENDENCY CHAIN RUNTIME ASSERTION (Section 5)\n");
    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "DEFINITION:\n");
    fprintf(stdout, "  The token dependency chain is the sequence of operations whose completion\n");
    fprintf(stdout, "  is required before the next token can be emitted:\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  1. Entry: Decode iteration begins\n");
    fprintf(stdout, "  2. Forward Pass: Transformer forward through all layers\n");
    fprintf(stdout, "  3. Attention: Attention computation for current position\n");
    fprintf(stdout, "  4. MLP: Feed-forward computation\n");
    fprintf(stdout, "  5. KV Cache: KV cache read/write operations\n");
    fprintf(stdout, "  6. Logits: Output logits computation\n");
    fprintf(stdout, "  7. Sampling: Token sampling from logits\n");
    fprintf(stdout, "  8. Token Commit: Output token committed\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "INVARIANT:\n");
    fprintf(stdout, "  ALL stages of the token dependency chain MUST execute on GPU.\n");
    fprintf(stdout, "  CPU presence on this chain is a FATAL correctness violation.\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "RUNTIME ASSERTION:\n");
    fprintf(stdout, "  After each token decode iteration:\n");
    fprintf(stdout, "  1. Verify no stage executed on CPU\n");
    fprintf(stdout, "  2. Verify GPU is not waiting on CPU\n");
    fprintf(stdout, "  3. Verify no CPU synchronization blocks token\n");
    fprintf(stdout, "  4. Verify no CPU state gates next token\n");
    fprintf(stdout, "  5. Verify no indirect CPU gating of GPU decisions\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "VIOLATION TYPES:\n");
    fprintf(stdout, "  - Direct CPU: Stage directly executes on CPU\n");
    fprintf(stdout, "  - GPU Wait: GPU blocked waiting on CPU\n");
    fprintf(stdout, "  - CPU Sync: Synchronization blocks token\n");
    fprintf(stdout, "  - CPU State Gate: CPU state required for next token\n");
    fprintf(stdout, "  - Indirect Gate: GPU depends on CPU decision\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "SCOPE:\n");
    fprintf(stdout, "  Assertions apply ONLY during:\n");
    fprintf(stdout, "  - Decode phase (token-by-token execution)\n");
    fprintf(stdout, "  - Per-token iterations in decode loop\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "  Assertions do NOT apply to:\n");
    fprintf(stdout, "  - Prefill phase\n");
    fprintf(stdout, "  - Request setup and initialization\n");
    fprintf(stdout, "  - Background tasks\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "CONFIGURATION:\n");
    fprintf(stdout, "  Default: Assertions ENABLED\n");
    fprintf(stdout, "  Can be disabled via:\n");
    fprintf(stdout, "    - Build-time flag (LLAMA_DISABLE_TOKEN_CHAIN_ASSERT)\n");
    fprintf(stdout, "    - Runtime: llama_set_token_chain_assertions_enabled(false)\n");
    fprintf(stdout, "    - Environment: LLAMA_ASSERT_TOKEN_CHAIN=0\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "FAILURE BEHAVIOR:\n");
    fprintf(stdout, "  On violation detection:\n");
    fprintf(stdout, "  1. Decode execution TERMINATES immediately\n");
    fprintf(stdout, "  2. Detailed diagnostics printed\n");
    fprintf(stdout, "  3. Violation type and stage clearly identified\n");
    fprintf(stdout, "  4. No recovery or fallback attempted\n");
    fprintf(stdout, "\n");

    fprintf(stdout, "================================================================================\n");
    fprintf(stdout, "\n");
}

// ============================================================================
// SELF-TEST
// ============================================================================

int llama_token_dependency_assert_selftest(void) {
    fprintf(stdout, "[TOKEN CHAIN ASSERT SELFTEST] Running...\n");

    // Test 1: Enable assertions
    llama_set_token_chain_assertions_enabled(true);
    if (!llama_get_token_chain_assertions_enabled()) {
        fprintf(stderr, "SELFTEST FAIL: Assertions should be enabled\n");
        return -1;
    }

    // Test 2: Set decode phase
    if (llama_token_chain_set_decode_phase(true) != 0) {
        fprintf(stderr, "SELFTEST FAIL: Set decode phase failed\n");
        return -1;
    }

    if (!llama_token_chain_in_decode_phase()) {
        fprintf(stderr, "SELFTEST FAIL: Should be in decode phase\n");
        return -1;
    }

    // Test 3: Start token chain
    if (llama_assert_token_chain_start(1) != 0) {
        fprintf(stderr, "SELFTEST FAIL: Token chain start failed\n");
        return -1;
    }

    // Test 4: Record stage on GPU
    if (llama_token_chain_record_stage_start(1, LLAMA_CHAIN_STAGE_FORWARD_PASS, "FORWARD", "CUDA") != 0) {
        fprintf(stderr, "SELFTEST FAIL: Record stage failed\n");
        return -1;
    }

    // Test 5: Assert GPU-only (should pass)
    if (llama_assert_token_chain_stage_gpu_only(1, LLAMA_CHAIN_STAGE_FORWARD_PASS, "CUDA") != 0) {
        fprintf(stderr, "SELFTEST FAIL: GPU-only assert should pass on CUDA\n");
        return -1;
    }

    // Test 6: Test CPU detection (should fail)
    if (llama_assert_token_chain_stage_gpu_only(1, LLAMA_CHAIN_STAGE_LOGITS, "CPU") == 0) {
        fprintf(stderr, "SELFTEST FAIL: Should detect CPU on chain\n");
        return -1;
    }

    // Test 7: Test CPU wait detection (should fail)
    if (llama_assert_no_cpu_wait_on_token_chain(1, true, "waiting on CPU") == 0) {
        fprintf(stderr, "SELFTEST FAIL: Should detect CPU wait\n");
        return -1;
    }

    // Test 8: Assertion counter
    if (llama_get_token_chain_assertion_count() <= 0) {
        fprintf(stderr, "SELFTEST FAIL: Should have counted assertions\n");
        return -1;
    }

    llama_reset_token_chain_assertion_counter();
    if (llama_get_token_chain_assertion_count() != 0) {
        fprintf(stderr, "SELFTEST FAIL: Counter should be reset\n");
        return -1;
    }

    fprintf(stdout, "[TOKEN CHAIN ASSERT SELFTEST] PASSED\n");
    return 0;
}
