#pragma once

/**
 * Per-Token GPU Utilization Probe for LLAMA Decode
 *
 * Decode-phase GPU utilization probe that measures actual GPU activity per token
 * and detects idle gaps. This is instrumentation for validation — not runtime
 * scheduling logic.
 *
 * The probe must be:
 * - Decode-phase only
 * - Zero control-path interference
 * - No CPU-side polling loops
 * - Disabled by default in production builds
 *
 * Measurements per token:
 * - GPU active time (via CUDA events)
 * - Token wall-clock time
 * - Idle gap between token executions
 * - Effective GPU occupancy ratio
 *
 * Formula:
 * gpu_util_ratio = gpu_active_time / token_wall_time
 *
 * Target: gpu_util_ratio → ~1.0
 * If < 0.8 consistently → decode path not GPU-dominant
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
    PROBE_STATE_UNINITIALIZED = 0,
    PROBE_STATE_READY = 1,
    PROBE_STATE_MEASURING = 2,
    PROBE_STATE_COMPLETE = 3,
    PROBE_STATE_LOCKED = 4
} gpu_probe_state;

typedef struct {
    uint64_t token_number;
    uint64_t token_sequence_id;

    // GPU timing (from CUDA events)
    double gpu_active_time_ms;      // Time GPU was actively executing kernels
    double gpu_idle_time_ms;        // Time between kernel end and next start

    // Wall-clock timing
    double token_wall_time_ms;      // Total wall time for this token

    // Computed metrics
    double idle_gap_ms;             // wall_time - gpu_active_time
    double gpu_utilization_ratio;   // gpu_active_time / token_wall_time
    double effective_throughput_tokens_per_sec;

    // Thresholds
    bool idle_gap_flagged;          // true if idle_gap > threshold
    bool underutilized_flagged;     // true if ratio < 0.80

    // Timestamps
    uint64_t measurement_timestamp_ns;
    bool measurement_valid;
} gpu_token_measurement;

typedef struct {
    uint64_t total_tokens_measured;
    uint64_t tokens_with_valid_data;

    // Aggregated GPU timing
    double total_gpu_active_time_ms;
    double avg_gpu_active_time_ms;
    double min_gpu_active_time_ms;
    double max_gpu_active_time_ms;

    // Aggregated wall-clock timing
    double total_wall_time_ms;
    double avg_wall_time_ms;
    double min_wall_time_ms;
    double max_wall_time_ms;

    // Aggregated idle gap
    double total_idle_gap_ms;
    double avg_idle_gap_ms;
    double min_idle_gap_ms;
    double max_idle_gap_ms;

    // Utilization statistics
    double avg_utilization_ratio;
    double min_utilization_ratio;
    double max_utilization_ratio;
    uint64_t underutilized_count;   // count of tokens with ratio < 0.80
    uint64_t critically_underutilized_count;  // count < 0.60

    // Throughput
    double avg_tokens_per_sec;
    double min_tokens_per_sec;
    double max_tokens_per_sec;

    // Health indicators
    bool gpu_dominant;              // avg_ratio > 0.80?
    bool critically_underutilized;  // avg_ratio < 0.60?
    uint64_t measurement_timestamp_ns;
} gpu_utilization_summary;

typedef struct {
    const char * alert_description;
    uint64_t token_number;
    double detected_value;
    double threshold_value;
    uint64_t alert_timestamp_ns;
} gpu_utilization_alert;

class gpu_utilization_probe {
private:
    gpu_probe_state current_state;
    std::vector<gpu_token_measurement> measurements;
    gpu_utilization_summary summary;
    std::vector<gpu_utilization_alert> alerts;

    std::atomic<bool> probe_enabled;
    std::atomic<bool> probe_active;
    std::atomic<uint64_t> token_counter;
    std::atomic<uint64_t> measurements_count;

    // Thresholds
    double idle_gap_threshold_ms;           // e.g., 0.2ms at batch=1
    double underutilization_threshold;      // e.g., 0.80
    double critical_underutilization_threshold;  // e.g., 0.60
    uint32_t measurement_window_size;       // e.g., 50 tokens before reporting

    // For per-token measurement
    std::chrono::high_resolution_clock::time_point token_wall_start;
    uint64_t current_measurement_gpu_active_time_ns;

    // Stats tracking
    std::map<std::string, uint32_t> alert_types;

public:
    gpu_utilization_probe();

    bool initialize();
    bool enable_probe(bool enable);
    bool is_probe_enabled() const { return probe_enabled.load(); }

    bool begin_token_measurement(uint64_t token_number);
    bool record_gpu_active_time(double gpu_active_time_ms);
    bool end_token_measurement();

    bool finalize_measurements();
    bool generate_utilization_report();
    bool validate_utilization_metrics();

    // Query functions
    gpu_probe_state get_current_state() const { return current_state; }
    bool is_measurement_complete() const { return measurements_count.load() >= 50; }

    const gpu_utilization_summary & get_summary() const { return summary; }
    std::vector<gpu_token_measurement> get_measurements() const { return measurements; }
    std::vector<gpu_utilization_alert> get_alerts() const { return alerts; }

    // Alert generation
    bool check_idle_gap_alert(const gpu_token_measurement & measurement);
    bool check_underutilization_alert(const gpu_token_measurement & measurement);
    bool record_alert(const char * description, uint64_t token_num,
                      double detected, double threshold);

    // Reporting
    std::string format_measurement(const gpu_token_measurement & m) const;
    std::string generate_report() const;
    std::string generate_json_report() const;

    // Validation
    bool verify_gpu_dominance() const;
    bool verify_no_critical_underutilization() const;
    bool verify_measurement_consistency() const;

    // Statistics
    size_t get_measurement_count() const { return measurements.size(); }
    uint64_t get_alert_count() const { return alerts.size(); }
    uint64_t get_underutilized_count() const { return summary.underutilized_count; }

    // Thresholds
    void set_idle_gap_threshold(double threshold_ms) { idle_gap_threshold_ms = threshold_ms; }
    void set_underutilization_threshold(double ratio) { underutilization_threshold = ratio; }
    void set_measurement_window_size(uint32_t size) { measurement_window_size = size; }

    double get_idle_gap_threshold() const { return idle_gap_threshold_ms; }
    double get_underutilization_threshold() const { return underutilization_threshold; }
};

class gpu_probe_guard {
private:
    bool guard_active;
    gpu_utilization_probe * probe;

public:
    gpu_probe_guard(gpu_utilization_probe * probe_ptr);
    ~gpu_probe_guard();

    bool is_guard_active() const { return guard_active; }
};

extern gpu_utilization_probe * g_gpu_utilization_probe;

bool llama_init_gpu_utilization_probe();
bool llama_enable_gpu_utilization_probe(bool enable);
bool llama_is_gpu_utilization_probe_enabled();

bool llama_begin_gpu_token_measurement(uint64_t token_number);
bool llama_record_gpu_active_time(double gpu_active_time_ms);
bool llama_end_gpu_token_measurement();

bool llama_finalize_gpu_measurements();
bool llama_generate_gpu_utilization_report();
bool llama_validate_gpu_utilization();

const gpu_utilization_summary * llama_get_gpu_utilization_summary();
const char * llama_get_gpu_utilization_report();
const gpu_token_measurement * llama_get_token_measurement(uint64_t index);

void llama_print_gpu_utilization_report();
void llama_print_gpu_utilization_summary();
void llama_print_gpu_utilization_alerts();
void llama_print_gpu_token_measurements(uint32_t limit);
void llama_export_gpu_utilization_json(const char * filename);

// Macro guards for integration (compile out when disabled)
#ifdef LLAMA_DECODE_GPU_PROBE

#define INIT_GPU_UTILIZATION_PROBE() \
    do { \
        llama_init_gpu_utilization_probe(); \
    } while(0)

#define ENABLE_GPU_UTILIZATION_PROBE(enable) \
    do { \
        if (g_gpu_utilization_probe) { \
            llama_enable_gpu_utilization_probe(enable); \
        } \
    } while(0)

#define BEGIN_GPU_TOKEN_MEASUREMENT(token_num) \
    do { \
        if (g_gpu_utilization_probe && llama_is_gpu_utilization_probe_enabled()) { \
            llama_begin_gpu_token_measurement(token_num); \
        } \
    } while(0)

#define RECORD_GPU_ACTIVE_TIME(gpu_time_ms) \
    do { \
        if (g_gpu_utilization_probe && llama_is_gpu_utilization_probe_enabled()) { \
            llama_record_gpu_active_time(gpu_time_ms); \
        } \
    } while(0)

#define END_GPU_TOKEN_MEASUREMENT() \
    do { \
        if (g_gpu_utilization_probe && llama_is_gpu_utilization_probe_enabled()) { \
            llama_end_gpu_token_measurement(); \
        } \
    } while(0)

#define FINALIZE_GPU_MEASUREMENTS() \
    do { \
        if (g_gpu_utilization_probe && llama_is_gpu_utilization_probe_enabled()) { \
            llama_finalize_gpu_measurements(); \
        } \
    } while(0)

#else // LLAMA_DECODE_GPU_PROBE

// No-op macros when disabled
#define INIT_GPU_UTILIZATION_PROBE() do { } while(0)
#define ENABLE_GPU_UTILIZATION_PROBE(enable) do { } while(0)
#define BEGIN_GPU_TOKEN_MEASUREMENT(token_num) do { } while(0)
#define RECORD_GPU_ACTIVE_TIME(gpu_time_ms) do { } while(0)
#define END_GPU_TOKEN_MEASUREMENT() do { } while(0)
#define FINALIZE_GPU_MEASUREMENTS() do { } while(0)

#endif // LLAMA_DECODE_GPU_PROBE

#ifdef __cplusplus
}
#endif
