#pragma once

/**
 * Long-Run Decode Stability Test Harness for LLAMA
 *
 * Dedicated long-run decode stress harness that validates:
 * - GPU-exclusive decode invariant
 * - Zero PCIe transfers during decode
 * - Stable tokens/sec over time
 * - No memory leaks
 * - No GPU fragmentation drift
 * - No CPU regression
 * - No correctness deviation
 *
 * This is not a benchmark. This is a structural stability validator.
 *
 * Detects regressions that only appear under:
 * - 5k–50k token decode runs
 * - High context growth
 * - Long-lived server processes
 * - Repeated decode sessions
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HARNESS_STATE_UNINITIALIZED = 0,
    HARNESS_STATE_SETUP = 1,
    HARNESS_STATE_RUNNING = 2,
    HARNESS_STATE_COMPLETE = 3,
    HARNESS_STATE_FAILED = 4
} stability_harness_state;

typedef enum {
    STRESS_MODE_STANDARD = 0,
    STRESS_MODE_LONG_CONTEXT = 1,  // 8k–16k
    STRESS_MODE_QUANTIZED_MMQ = 2,
    STRESS_MODE_CUBLAS_DENSE = 3,
    STRESS_MODE_FLASH_ATTENTION = 4,
    STRESS_MODE_SERVER = 5
} stress_test_mode;

typedef struct {
    uint64_t token_number;

    // GPU timing
    double gpu_active_time_ms;
    double wall_time_ms;
    double idle_gap_ms;
    double gpu_utilization_ratio;

    // PCIe traffic
    uint64_t h2d_bytes;
    uint64_t d2h_bytes;
    uint64_t d2d_bytes;

    // CPU metrics
    double cpu_utilization_percent;

    // Memory metrics
    uint64_t gpu_memory_used_bytes;
    uint64_t gpu_memory_free_bytes;
    size_t context_length;

    // Throughput
    double tokens_per_sec;

    // Invariant violations
    bool cpu_execution_detected;
    bool pcie_violation_detected;
    bool allocation_during_decode;

    uint64_t measurement_timestamp_ns;
} stability_token_sample;

typedef struct {
    uint64_t total_tokens_generated;
    uint32_t total_samples_collected;

    // Throughput statistics
    double avg_tokens_per_sec;
    double initial_tokens_per_sec;
    double final_tokens_per_sec;
    double min_tokens_per_sec;
    double max_tokens_per_sec;
    double throughput_variance_percent;
    double throughput_drift_percent;

    // GPU metrics
    double avg_gpu_utilization;
    double min_gpu_utilization;
    double max_gpu_utilization;
    double avg_idle_gap_ms;

    // PCIe metrics
    uint64_t total_h2d_bytes;
    uint64_t total_d2h_bytes;
    uint64_t total_d2d_bytes;

    // Memory metrics
    uint64_t initial_gpu_memory;
    uint64_t final_gpu_memory;
    int64_t gpu_memory_delta;
    bool memory_leak_detected;
    bool gpu_fragmentation_stable;

    // Invariant violations
    uint32_t cpu_execution_violations;
    uint32_t pcie_violations;
    uint32_t allocation_events_during_decode;
    uint32_t kv_reallocation_events;

    // CPU metrics
    double avg_cpu_utilization;
    double max_cpu_utilization;
    bool cpu_regression_detected;

    // Overall status
    bool all_invariants_held;
    bool stability_test_passed;
    bool deterministic_output_verified;

    uint64_t test_start_timestamp_ns;
    uint64_t test_end_timestamp_ns;
    double test_duration_seconds;
} stability_test_summary;

typedef struct {
    const char * invariant_name;
    const char * violation_description;
    uint64_t token_at_violation;
    double violation_value;
    double threshold_value;
    uint64_t violation_timestamp_ns;
    bool is_critical;
} stability_invariant_violation;

class decode_stability_harness {
private:
    stability_harness_state current_state;
    stress_test_mode test_mode;
    std::vector<stability_token_sample> token_samples;
    stability_test_summary summary;
    std::vector<stability_invariant_violation> invariant_violations;

    std::atomic<bool> harness_enabled;
    std::atomic<bool> test_running;
    std::atomic<uint64_t> tokens_generated;
    std::atomic<uint64_t> samples_collected;

    // Configuration
    uint64_t target_tokens;             // e.g., 10,000
    uint32_t sample_interval;           // e.g., every token or every 10
    uint32_t drift_check_interval;      // e.g., every 100 tokens
    double throughput_drift_threshold;  // e.g., 0.10 (10%)
    double min_gpu_utilization_threshold;  // e.g., 0.85
    double cpu_utilization_threshold;   // e.g., 0.95
    bool deterministic_mode;            // temp=0, fixed seed
    bool strict_mode;                   // Fail on first violation?

    // Tracking
    double initial_avg_tps;
    double window_avg_tps;
    uint64_t last_drift_check_token;
    uint64_t initial_gpu_memory_free;

public:
    decode_stability_harness();

    bool initialize(uint64_t target_token_count, stress_test_mode mode);
    bool configure_thresholds(double drift_threshold,
                            double min_gpu_util,
                            double cpu_util_threshold);
    bool enable_harness(bool enable) { harness_enabled.store(enable); return true; }
    bool is_harness_enabled() const { return harness_enabled.load(); }

    bool begin_stability_test();
    bool end_stability_test();

    bool record_token_sample(const stability_token_sample & sample);
    bool check_invariants_at_token(uint64_t token_num);
    bool check_drift_every_n_tokens(uint64_t token_num);

    bool validate_gpu_exclusive_execution();
    bool validate_pcie_cleanliness();
    bool validate_throughput_stability();
    bool validate_gpu_utilization_stability();
    bool validate_no_allocations();
    bool validate_memory_stability();
    bool validate_cpu_not_saturated();
    bool validate_kv_cache_stability();

    bool finalize_test();
    bool generate_stability_report();

    // Query functions
    stability_harness_state get_current_state() const { return current_state; }
    bool is_test_running() const { return test_running.load(); }

    const stability_test_summary & get_summary() const { return summary; }
    std::vector<stability_token_sample> get_samples() const { return token_samples; }
    std::vector<stability_invariant_violation> get_violations() const { return invariant_violations; }

    // Configuration
    void set_target_tokens(uint64_t target) { target_tokens = target; }
    void set_sample_interval(uint32_t interval) { sample_interval = interval; }
    void set_drift_check_interval(uint32_t interval) { drift_check_interval = interval; }
    void set_strict_mode(bool strict) { strict_mode = strict; }
    void set_deterministic_mode(bool det) { deterministic_mode = det; }

    // Reporting
    std::string generate_report() const;
    std::string generate_json_report() const;
    std::string format_test_mode(stress_test_mode mode) const;
    std::string format_status(bool passed) const;

private:
    bool record_violation(const char * invariant_name,
                         const char * description,
                         uint64_t token_num,
                         double value,
                         double threshold,
                         bool is_critical);

    double compute_throughput_variance();
    double compute_throughput_drift();
};

class stability_harness_guard {
private:
    bool guard_active;
    decode_stability_harness * harness;

public:
    stability_harness_guard(decode_stability_harness * harness_ptr);
    ~stability_harness_guard();

    bool is_guard_active() const { return guard_active; }
};

extern decode_stability_harness * g_decode_stability_harness;

bool llama_init_stability_harness(uint64_t target_tokens, int stress_mode);
bool llama_enable_stability_harness(bool enable);
bool llama_is_stability_harness_enabled();

bool llama_begin_stability_test();
bool llama_end_stability_test();

bool llama_record_stability_sample(const stability_token_sample * sample);
bool llama_check_stability_invariants(uint64_t token_num);

bool llama_finalize_stability_test();
bool llama_generate_stability_report();
bool llama_validate_all_stability_invariants();

const stability_test_summary * llama_get_stability_summary();
const char * llama_get_stability_report();

void llama_print_stability_report();
void llama_print_stability_summary();
void llama_print_stability_violations();
void llama_export_stability_json(const char * filename);

// Macro guards (compile out when disabled)
#ifdef LLAMA_DECODE_STABILITY_HARNESS

#define INIT_STABILITY_HARNESS(target_tokens, mode) \
    llama_init_stability_harness(target_tokens, mode)

#define BEGIN_STABILITY_TEST() \
    do { \
        if (g_decode_stability_harness) { \
            llama_begin_stability_test(); \
        } \
    } while(0)

#define RECORD_STABILITY_SAMPLE(sample) \
    do { \
        if (g_decode_stability_harness && llama_is_stability_harness_enabled()) { \
            llama_record_stability_sample(sample); \
        } \
    } while(0)

#define CHECK_STABILITY_INVARIANTS(token_num) \
    do { \
        if (g_decode_stability_harness && llama_is_stability_harness_enabled()) { \
            llama_check_stability_invariants(token_num); \
        } \
    } while(0)

#define END_STABILITY_TEST() \
    do { \
        if (g_decode_stability_harness) { \
            llama_end_stability_test(); \
        } \
    } while(0)

#define FINALIZE_STABILITY_TEST() \
    do { \
        if (g_decode_stability_harness) { \
            llama_finalize_stability_test(); \
        } \
    } while(0)

#else // LLAMA_DECODE_STABILITY_HARNESS

// No-op macros when disabled
#define INIT_STABILITY_HARNESS(target_tokens, mode) true
#define BEGIN_STABILITY_TEST() do { } while(0)
#define RECORD_STABILITY_SAMPLE(sample) do { } while(0)
#define CHECK_STABILITY_INVARIANTS(token_num) do { } while(0)
#define END_STABILITY_TEST() do { } while(0)
#define FINALIZE_STABILITY_TEST() do { } while(0)

#endif // LLAMA_DECODE_STABILITY_HARNESS

#ifdef __cplusplus
}
#endif
