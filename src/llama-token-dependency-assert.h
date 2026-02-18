/**
 * SECTION 5: Add Runtime Assertion - CPU Not on Token Dependency Chain
 *
 * This file implements a runtime assertion mechanism that verifies—during execution—
 * that the CPU is never part of the dependency chain that gates token emission.
 *
 * Core Principle:
 * "The token dependency chain must be GPU-exclusive at runtime.
 *  CPU presence on this chain is a fatal invariant violation.
 *  Any detection causes immediate abort."
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>

// ============================================================================
// TOKEN DEPENDENCY CHAIN DEFINITION
// ============================================================================

/**
 * Enum defining stages of the token dependency chain
 *
 * The token dependency chain is:
 * Entry → Forward Pass → Attention/MLP → KV Cache → Logits → Sampling → Commit
 * All stages must be GPU-exclusive.
 */
enum llama_token_chain_stage {
    LLAMA_CHAIN_STAGE_UNKNOWN = 0,
    LLAMA_CHAIN_STAGE_ENTRY = 1,           // Decode iteration entry point
    LLAMA_CHAIN_STAGE_FORWARD_PASS = 2,    // Transformer forward pass
    LLAMA_CHAIN_STAGE_ATTENTION = 3,       // Attention computation
    LLAMA_CHAIN_STAGE_MLP = 4,             // MLP computation
    LLAMA_CHAIN_STAGE_KV_CACHE = 5,        // KV cache read/write
    LLAMA_CHAIN_STAGE_LOGITS = 6,          // Logits computation
    LLAMA_CHAIN_STAGE_SAMPLING = 7,        // Sampling operation
    LLAMA_CHAIN_STAGE_TOKEN_COMMIT = 8,    // Token commit (output)
};

/**
 * Enum defining violation detection type
 */
enum llama_token_chain_violation_type {
    LLAMA_CHAIN_VIOLATION_UNKNOWN = 0,
    LLAMA_CHAIN_VIOLATION_DIRECT_CPU = 1,        // CPU directly executed stage
    LLAMA_CHAIN_VIOLATION_CPU_WAIT = 2,          // GPU waiting on CPU
    LLAMA_CHAIN_VIOLATION_CPU_SYNC = 3,          // CPU synchronization blocking decode
    LLAMA_CHAIN_VIOLATION_CPU_STATE_GATE = 4,    // CPU state update blocking next token
    LLAMA_CHAIN_VIOLATION_INDIRECT = 5,          // Indirect CPU gating detected
};

// ============================================================================
// PER-TOKEN EXECUTION RECORD
// ============================================================================

/**
 * Structure recording which backend executed work for one token
 */
struct llama_token_execution_record {
    uint64_t token_id;                          // Token ID/sequence number

    // Backend execution tracking per stage
    struct {
        const char* stage_name;                 // Stage name
        bool executed;                          // Stage was executed
        const char* backend_executed;           // Which backend executed this stage
        uint64_t start_time_us;                 // Stage start time
        uint64_t end_time_us;                   // Stage end time
        bool cpu_detected;                      // CPU detected on this stage
    } stages[9];  // For each chain stage

    // Dependency tracking
    bool cpu_wait_detected;                     // GPU waited on CPU
    bool cpu_sync_detected;                     // CPU synchronization blocked decode
    bool cpu_state_gate_detected;               // CPU state gated next token

    // Overall violation
    bool chain_violation_detected;              // Any violation detected
    enum llama_token_chain_violation_type violation_type;  // Type of violation
    const char* violation_message;              // Detailed violation message
};

// ============================================================================
// ASSERTION CONTROL
// ============================================================================

/**
 * Enable/disable token dependency chain assertions
 * When enabled, CPU presence on token chain causes immediate abort.
 * Can be overridden by compile-time flag or environment variable.
 */
void llama_set_token_chain_assertions_enabled(bool enabled);

/**
 * Get current assertion state
 */
bool llama_get_token_chain_assertions_enabled(void);

/**
 * Get assertion violation count
 */
int llama_get_token_chain_assertion_count(void);

/**
 * Reset assertion counter
 */
void llama_reset_token_chain_assertion_counter(void);

// ============================================================================
// DECODE ITERATION INSTRUMENTATION
// ============================================================================

/**
 * Called at the START of each token decode iteration
 * Records the starting point of the token dependency chain.
 *
 * Returns: 0 = OK, -1 = Previous iteration detected violation
 */
int llama_assert_token_chain_start(uint64_t token_id);

/**
 * Called at the END of each token decode iteration
 * Analyzes the execution record and asserts CPU was not on the chain.
 *
 * Returns: 0 = No violations, -1 = CPU detected on chain (FATAL)
 */
int llama_assert_token_chain_complete(uint64_t token_id);

// ============================================================================
// STAGE-LEVEL INSTRUMENTATION
// ============================================================================

/**
 * Record execution of a stage on a specific backend
 * Called when a token-dependency-chain stage begins execution.
 */
int llama_token_chain_record_stage_start(
    uint64_t token_id,
    enum llama_token_chain_stage stage,
    const char* stage_name,
    const char* backend_executed
);

/**
 * Record completion of a stage
 * Called when a token-dependency-chain stage completes execution.
 */
int llama_token_chain_record_stage_end(
    uint64_t token_id,
    enum llama_token_chain_stage stage
);

// ============================================================================
// DEPENDENCY VIOLATION DETECTION
// ============================================================================

/**
 * Assert that a stage was executed on GPU, not CPU
 * Called after stage completion.
 *
 * Returns: 0 = Stage on GPU (OK), -1 = Stage on CPU (FATAL)
 */
int llama_assert_token_chain_stage_gpu_only(
    uint64_t token_id,
    enum llama_token_chain_stage stage,
    const char* executed_backend
);

/**
 * Detect if GPU is waiting on CPU to unblock
 * Called when GPU would block waiting for CPU work.
 *
 * Returns: 0 = No CPU blocking, -1 = CPU is blocking GPU (FATAL)
 */
int llama_assert_no_cpu_wait_on_token_chain(
    uint64_t token_id,
    bool gpu_is_waiting,
    const char* waiting_reason
);

/**
 * Detect if CPU synchronization is blocking token emission
 * Called when synchronization points exist.
 *
 * Returns: 0 = No blocking sync, -1 = Sync blocks token (FATAL)
 */
int llama_assert_no_cpu_sync_block(
    uint64_t token_id,
    bool sync_required,
    const char* sync_type
);

/**
 * Detect if CPU state update gates the next token
 * Called when state updates occur.
 *
 * Returns: 0 = State doesn't gate, -1 = State gates token (FATAL)
 */
int llama_assert_no_cpu_state_gate(
    uint64_t token_id,
    bool cpu_state_gating_next_token,
    const char* state_update_type
);

/**
 * Detect indirect CPU gating (CPU makes decision for GPU)
 * Called when GPU next step depends on CPU decision.
 *
 * Returns: 0 = No indirect gating, -1 = CPU gates GPU decision (FATAL)
 */
int llama_assert_no_indirect_cpu_gate(
    uint64_t token_id,
    bool gpu_depends_on_cpu_decision,
    const char* decision_type
);

// ============================================================================
// EXECUTION RECORD ANALYSIS
// ============================================================================

/**
 * Analyze token execution record for any CPU involvement
 * Called at end of token decode to verify invariant.
 *
 * Returns: 0 = CPU not on chain, -1 = CPU detected (FATAL)
 */
int llama_token_chain_verify_gpu_exclusive(uint64_t token_id);

/**
 * Get execution record for a token (for debugging)
 */
struct llama_token_execution_record* llama_get_token_execution_record(uint64_t token_id);

/**
 * Print detailed execution record
 */
void llama_print_token_execution_record(
    const struct llama_token_execution_record* record
);

// ============================================================================
// VIOLATION REPORTING
// ============================================================================

/**
 * Print token dependency chain violation diagnostics
 * Called when assertion fails.
 */
void llama_print_token_chain_violation_diagnostics(
    uint64_t token_id,
    enum llama_token_chain_violation_type violation_type,
    const char* violation_message
);

/**
 * Convert stage enum to string name
 */
static inline const char* llama_token_chain_stage_name(enum llama_token_chain_stage stage) {
    switch (stage) {
        case LLAMA_CHAIN_STAGE_UNKNOWN:        return "UNKNOWN";
        case LLAMA_CHAIN_STAGE_ENTRY:          return "ENTRY";
        case LLAMA_CHAIN_STAGE_FORWARD_PASS:   return "FORWARD_PASS";
        case LLAMA_CHAIN_STAGE_ATTENTION:      return "ATTENTION";
        case LLAMA_CHAIN_STAGE_MLP:            return "MLP";
        case LLAMA_CHAIN_STAGE_KV_CACHE:       return "KV_CACHE";
        case LLAMA_CHAIN_STAGE_LOGITS:         return "LOGITS";
        case LLAMA_CHAIN_STAGE_SAMPLING:       return "SAMPLING";
        case LLAMA_CHAIN_STAGE_TOKEN_COMMIT:   return "TOKEN_COMMIT";
        default:                                return "(invalid)";
    }
}

/**
 * Convert violation type enum to string name
 */
static inline const char* llama_token_chain_violation_name(
    enum llama_token_chain_violation_type violation_type
) {
    switch (violation_type) {
        case LLAMA_CHAIN_VIOLATION_UNKNOWN:       return "UNKNOWN";
        case LLAMA_CHAIN_VIOLATION_DIRECT_CPU:   return "DIRECT_CPU_EXECUTION";
        case LLAMA_CHAIN_VIOLATION_CPU_WAIT:     return "GPU_WAITING_ON_CPU";
        case LLAMA_CHAIN_VIOLATION_CPU_SYNC:     return "CPU_SYNC_BLOCKING_DECODE";
        case LLAMA_CHAIN_VIOLATION_CPU_STATE_GATE: return "CPU_STATE_GATES_TOKEN";
        case LLAMA_CHAIN_VIOLATION_INDIRECT:     return "INDIRECT_CPU_GATING";
        default:                                   return "(invalid)";
    }
}

// ============================================================================
// EXPLICIT TOKEN CHAIN DEFINITION STATEMENT
// ============================================================================

/**
 * Print the token dependency chain definition and assertion principle
 */
void llama_print_token_dependency_chain_statement(void);

// ============================================================================
// SCOPE DEFINITION
// ============================================================================

/**
 * Mark that we are entering the decode phase (token-by-token)
 * Assertions apply only during decode, not prefill or setup.
 */
int llama_token_chain_set_decode_phase(bool in_decode_phase);

/**
 * Get current phase
 */
bool llama_token_chain_in_decode_phase(void);

// ============================================================================
// VALIDATION AND SELF-TEST
// ============================================================================

/**
 * Self-test: verify assertion mechanism works correctly
 */
int llama_token_dependency_assert_selftest(void);

