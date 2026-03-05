#pragma once

/**
 * Decode-Exclusive Success Criteria for LLAMA GPU-Exclusive Architecture
 *
 * Formally defined non-negotiable acceptance gates that determine whether
 * the system satisfies the GPU-exclusive decode architecture.
 *
 * These criteria are BINARY. Partial compliance is FAILURE.
 *
 * The system is accepted only if ALL criteria hold simultaneously:
 * 1. GPU-Exclusive Decode Invariant
 * 2. CPU Dependency Chain Elimination
 * 3. Zero Hybrid Execution
 * 4. Zero Silent Fallback
 * 5. Zero Per-Token Host↔Device Transfers
 * 6. No Decode-Time Allocation
 * 7. Stable Backend Binding
 * 8. GPU Utilization Gate
 * 9. CPU Saturation Gate
 * 10. Determinism Gate
 * 11. Long-Run Stability Gate
 * 12. Throughput Improvement Gate
 *
 * Any violation invalidates the architecture.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CRITERIA_STATE_UNINITIALIZED = 0,
    CRITERIA_STATE_INITIALIZED = 1,
    CRITERIA_STATE_VALIDATING = 2,
    CRITERIA_STATE_COMPLETE = 3
} acceptance_criteria_state;

typedef enum {
    GATE_GPU_EXCLUSIVE_DECODE = 0,
    GATE_CPU_DEPENDENCY_ELIMINATION = 1,
    GATE_ZERO_HYBRID = 2,
    GATE_ZERO_SILENT_FALLBACK = 3,
    GATE_ZERO_PCIE_TRANSFERS = 4,
    GATE_NO_DECODE_ALLOCATION = 5,
    GATE_STABLE_BACKEND_BINDING = 6,
    GATE_GPU_UTILIZATION = 7,
    GATE_CPU_SATURATION = 8,
    GATE_DETERMINISM = 9,
    GATE_LONG_RUN_STABILITY = 10,
    GATE_THROUGHPUT_IMPROVEMENT = 11
} acceptance_gate;

typedef struct {
    acceptance_gate gate_id;
    const char * gate_name;
    const char * gate_description;
    bool gate_passed;
    const char * failure_reason;
    uint64_t validation_timestamp_ns;
    bool is_critical;  // true = blocks all acceptance
} gate_validation_result;

typedef struct {
    // Gate pass/fail status
    bool gpu_exclusive_decode;
    bool cpu_dependency_eliminated;
    bool zero_hybrid_execution;
    bool zero_silent_fallback;
    bool zero_pcie_transfers;
    bool no_decode_allocation;
    bool stable_backend_binding;
    bool gpu_utilization_stable;
    bool cpu_not_saturated;
    bool output_deterministic;
    bool long_run_stability;
    bool throughput_improved;

    // Overall acceptance
    bool all_gates_passed;
    bool system_accepted;

    // Metrics
    uint32_t gates_passed;
    uint32_t gates_failed;
    uint32_t gates_total;

    // Evidence
    const char * cpu_execution_violations;
    const char * cpu_dependency_chain;
    const char * hybrid_execution_evidence;
    const char * fallback_events;
    const char * pcie_violations;
    const char * allocation_events;
    const char * backend_mutations;
    const char * gpu_idle_evidence;
    const char * cpu_saturation_evidence;
    const char * determinism_failures;
    const char * stability_violations;
    double throughput_improvement_percent;

    uint64_t validation_timestamp_ns;
} acceptance_validation_result;

typedef struct {
    const char * violation_description;
    acceptance_gate related_gate;
    uint64_t violation_timestamp_ns;
    const char * evidence_source;
    bool is_blocking;
} acceptance_violation;

class decode_acceptance_validator {
private:
    acceptance_criteria_state current_state;
    std::vector<gate_validation_result> gate_results;
    acceptance_validation_result overall_result;
    std::vector<acceptance_violation> violations;

    std::atomic<bool> validator_enabled;
    std::atomic<bool> validation_running;

    // Thresholds
    double min_gpu_utilization;
    double max_cpu_utilization;
    double max_throughput_drift;
    double min_throughput_improvement;
    uint64_t max_pcie_h2d_bytes;
    uint64_t max_pcie_d2h_bytes;

    // Collected evidence
    uint32_t cpu_execution_count;
    bool cpu_on_dependency_chain;
    bool hybrid_mode_detected;
    uint32_t silent_fallback_count;
    uint64_t total_h2d_bytes;
    uint64_t total_d2h_bytes;
    uint32_t allocation_events;
    uint32_t backend_mutations;
    bool determinism_violated;
    double throughput_improvement;

public:
    decode_acceptance_validator();

    bool initialize();
    bool enable_validator(bool enable) { validator_enabled.store(enable); return true; }
    bool is_validator_enabled() const { return validator_enabled.load(); }

    bool begin_validation();
    bool end_validation();

    // Gate validation methods
    bool validate_gpu_exclusive_decode();
    bool validate_cpu_dependency_elimination();
    bool validate_zero_hybrid_execution();
    bool validate_zero_silent_fallback();
    bool validate_zero_pcie_transfers();
    bool validate_no_decode_allocation();
    bool validate_stable_backend_binding();
    bool validate_gpu_utilization();
    bool validate_cpu_saturation();
    bool validate_determinism();
    bool validate_long_run_stability();
    bool validate_throughput_improvement();

    bool run_all_validations();
    bool finalize_validation();

    // Query functions
    acceptance_criteria_state get_current_state() const { return current_state; }
    const acceptance_validation_result & get_overall_result() const { return overall_result; }
    std::vector<gate_validation_result> get_gate_results() const { return gate_results; }
    std::vector<acceptance_violation> get_violations() const { return violations; }

    bool is_system_accepted() const { return overall_result.system_accepted; }
    uint32_t get_gates_passed() const { return overall_result.gates_passed; }
    uint32_t get_gates_failed() const { return overall_result.gates_failed; }

    // Configuration
    void set_thresholds(double min_gpu_util, double max_cpu_util,
                       double max_drift, double min_improvement) {
        min_gpu_utilization = min_gpu_util;
        max_cpu_utilization = max_cpu_util;
        max_throughput_drift = max_drift;
        min_throughput_improvement = min_improvement;
    }

    // Evidence recording
    bool record_cpu_execution_event();
    bool record_cpu_dependency_chain();
    bool record_hybrid_mode_activation();
    bool record_silent_fallback_event();
    bool record_pcie_transfer(bool is_h2d, uint64_t size_bytes);
    bool record_allocation_event();
    bool record_backend_mutation();
    bool record_determinism_failure();
    bool record_throughput_improvement(double improvement_percent);

    // Reporting
    std::string generate_acceptance_report() const;
    std::string generate_detailed_report() const;
    std::string generate_json_report() const;
    std::string format_gate_status(bool passed) const;

private:
    bool record_gate_result(acceptance_gate gate_id,
                           const char * gate_name,
                           const char * description,
                           bool passed,
                           const char * failure_reason,
                           bool is_critical);

    bool record_violation(const char * description,
                         acceptance_gate gate,
                         const char * evidence_source,
                         bool is_blocking);
};

class acceptance_validator_guard {
private:
    bool guard_active;
    decode_acceptance_validator * validator;

public:
    acceptance_validator_guard(decode_acceptance_validator * validator_ptr);
    ~acceptance_validator_guard();

    bool is_guard_active() const { return guard_active; }
};

extern decode_acceptance_validator * g_decode_acceptance_validator;

bool llama_init_acceptance_validator();
bool llama_enable_acceptance_validator(bool enable);
bool llama_is_acceptance_validator_enabled();

bool llama_begin_acceptance_validation();
bool llama_end_acceptance_validation();

bool llama_run_all_acceptance_gates();
bool llama_finalize_acceptance_validation();

bool llama_record_cpu_execution();
bool llama_record_cpu_dependency();
bool llama_record_hybrid_mode();
bool llama_record_fallback_event();
// Note: llama_record_pcie_transfer and llama_record_allocation are declared elsewhere
// See: llama-pcie-traffic-watchdog.h and llama-gpu-allocation-alignment.h
bool llama_record_backend_mutation();
bool llama_record_determinism_failure();
bool llama_record_throughput_improvement(double improvement);

// Acceptance-specific PCIE and allocation recording (used by acceptance validator)
bool llama_acceptance_record_pcie_transfer(bool is_h2d, uint64_t size);
bool llama_acceptance_record_allocation();

const acceptance_validation_result * llama_get_acceptance_result();
bool llama_is_system_accepted();
uint32_t llama_get_gates_passed();
uint32_t llama_get_gates_failed();

void llama_print_acceptance_report();
void llama_print_detailed_acceptance_report();
void llama_export_acceptance_json(const char * filename);

// Acceptance gates (binary checks)
#ifdef LLAMA_DECODE_ACCEPTANCE_GATES

#define VALIDATE_ACCEPTANCE_CRITERIA() \
    do { \
        if (g_decode_acceptance_validator && \
            llama_is_acceptance_validator_enabled()) { \
            if (!llama_run_all_acceptance_gates()) { \
                FATAL("System failed GPU-exclusive decode acceptance criteria"); \
            } \
        } \
    } while(0)

#define CHECK_GPU_EXCLUSIVE_DECODE() \
    do { \
        if (g_decode_acceptance_validator) { \
            llama_record_cpu_execution(); \
        } \
    } while(0)

#define CHECK_CPU_DEPENDENCY() \
    do { \
        if (g_decode_acceptance_validator) { \
            llama_record_cpu_dependency(); \
        } \
    } while(0)

#define CHECK_HYBRID_MODE() \
    do { \
        if (g_decode_acceptance_validator) { \
            llama_record_hybrid_mode(); \
        } \
    } while(0)

#define CHECK_FALLBACK() \
    do { \
        if (g_decode_acceptance_validator) { \
            llama_record_fallback_event(); \
        } \
    } while(0)

#define CHECK_PCIE_TRANSFER(is_h2d, size) \
    do { \
        if (g_decode_acceptance_validator) { \
            llama_acceptance_record_pcie_transfer(is_h2d, size); \
        } \
    } while(0)

#else // LLAMA_DECODE_ACCEPTANCE_GATES

#define VALIDATE_ACCEPTANCE_CRITERIA() do { } while(0)
#define CHECK_GPU_EXCLUSIVE_DECODE() do { } while(0)
#define CHECK_CPU_DEPENDENCY() do { } while(0)
#define CHECK_HYBRID_MODE() do { } while(0)
#define CHECK_FALLBACK() do { } while(0)
#define CHECK_PCIE_TRANSFER(is_h2d, size) do { } while(0)

#endif // LLAMA_DECODE_ACCEPTANCE_GATES

#ifdef __cplusplus
}
#endif
