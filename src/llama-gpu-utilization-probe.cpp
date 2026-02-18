#include "llama-gpu-utilization-probe.h"
#include <cstring>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

// Global state
gpu_utilization_probe * g_gpu_utilization_probe = nullptr;

// ============================================================================
// gpu_utilization_probe Implementation
// ============================================================================

gpu_utilization_probe::gpu_utilization_probe()
    : current_state(PROBE_STATE_UNINITIALIZED),
      probe_enabled(false),
      probe_active(false),
      token_counter(0),
      measurements_count(0),
      idle_gap_threshold_ms(0.2),
      underutilization_threshold(0.80),
      critical_underutilization_threshold(0.60),
      measurement_window_size(50),
      current_measurement_gpu_active_time_ns(0) {
    std::memset(&summary, 0, sizeof(summary));
}

bool gpu_utilization_probe::initialize() {
    if (current_state != PROBE_STATE_UNINITIALIZED) {
        fprintf(stderr, "[PROBE] ERROR: Already initialized (state=%d)\n", current_state);
        return false;
    }

    current_state = PROBE_STATE_READY;
    probe_enabled.store(false);  // Disabled by default
    probe_active.store(false);
    token_counter.store(0);
    measurements_count.store(0);

    fprintf(stdout, "[PROBE] GPU utilization probe initialized\n");
    fprintf(stdout, "[PROBE] Idle gap threshold: %.3f ms\n", idle_gap_threshold_ms);
    fprintf(stdout, "[PROBE] Underutilization threshold: %.2f\n", underutilization_threshold);
    fprintf(stdout, "[PROBE] Measurement window: %u tokens\n", measurement_window_size);

    return true;
}

bool gpu_utilization_probe::enable_probe(bool enable) {
    if (current_state == PROBE_STATE_UNINITIALIZED) {
        fprintf(stderr, "[PROBE] ERROR: Not initialized\n");
        return false;
    }

    probe_enabled.store(enable);
    fprintf(stdout, "[PROBE] Probe %s\n", enable ? "ENABLED" : "DISABLED");

    return true;
}

bool gpu_utilization_probe::begin_token_measurement(uint64_t token_number) {
    if (!probe_enabled.load()) {
        return true;  // No-op when disabled
    }

    if (current_state != PROBE_STATE_READY && current_state != PROBE_STATE_MEASURING) {
        return false;
    }

    current_state = PROBE_STATE_MEASURING;
    token_counter.store(token_number);

    // Record wall-clock start time
    token_wall_start = std::chrono::high_resolution_clock::now();
    current_measurement_gpu_active_time_ns = 0;

    return true;
}

bool gpu_utilization_probe::record_gpu_active_time(double gpu_active_time_ms) {
    if (!probe_enabled.load()) {
        return true;  // No-op when disabled
    }

    if (current_state != PROBE_STATE_MEASURING) {
        fprintf(stderr, "[PROBE] ERROR: Not measuring\n");
        return false;
    }

    // Record GPU active time (convert to nanoseconds for precision)
    current_measurement_gpu_active_time_ns = (uint64_t)(gpu_active_time_ms * 1000000.0);

    return true;
}

bool gpu_utilization_probe::end_token_measurement() {
    if (!probe_enabled.load()) {
        return true;  // No-op when disabled
    }

    if (current_state != PROBE_STATE_MEASURING) {
        fprintf(stderr, "[PROBE] ERROR: Not measuring\n");
        return false;
    }

    // Record wall-clock end time
    auto token_wall_end = std::chrono::high_resolution_clock::now();
    auto wall_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        token_wall_end - token_wall_start);
    double token_wall_time_ms = wall_duration.count() / 1000000.0;

    double gpu_active_time_ms = current_measurement_gpu_active_time_ns / 1000000.0;
    double idle_gap_ms = token_wall_time_ms - gpu_active_time_ms;

    // Clamp to zero if negative (measurement jitter)
    if (idle_gap_ms < 0.0) {
        idle_gap_ms = 0.0;
    }

    // Compute utilization ratio
    double util_ratio = (token_wall_time_ms > 0.0) ?
                       (gpu_active_time_ms / token_wall_time_ms) : 0.0;
    util_ratio = std::min(1.0, util_ratio);  // Cap at 1.0

    // Compute tokens per second
    double tokens_per_sec = (token_wall_time_ms > 0.0) ?
                           (1000.0 / token_wall_time_ms) : 0.0;

    // Create measurement record
    gpu_token_measurement measurement = {
        token_counter.load(),
        measurements_count.load(),
        gpu_active_time_ms,
        0.0,  // gpu_idle_time_ms (could be populated separately)
        token_wall_time_ms,
        idle_gap_ms,
        util_ratio,
        tokens_per_sec,
        false,  // idle_gap_flagged (set below)
        false,  // underutilized_flagged (set below)
        static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
        true
    };

    // Check thresholds
    measurement.idle_gap_flagged = (idle_gap_ms > idle_gap_threshold_ms);
    measurement.underutilized_flagged = (util_ratio < underutilization_threshold);

    // Record measurement
    measurements.push_back(measurement);
    measurements_count.store(measurements_count.load() + 1);

    // Check for alerts
    if (measurement.idle_gap_flagged) {
        check_idle_gap_alert(measurement);
    }
    if (measurement.underutilized_flagged) {
        check_underutilization_alert(measurement);
    }

    current_state = PROBE_STATE_READY;

    // Debug output (optional)
    if (measurements_count.load() % 10 == 0) {
        fprintf(stdout, "[PROBE] Token %llu: GPU %.2f ms, Wall %.2f ms, Idle %.2f ms, Util %.2f%%\n",
                (unsigned long long)token_counter.load(),
                gpu_active_time_ms, token_wall_time_ms, idle_gap_ms,
                util_ratio * 100.0);
    }

    return true;
}

bool gpu_utilization_probe::check_idle_gap_alert(const gpu_token_measurement & measurement) {
    if (measurement.idle_gap_ms > idle_gap_threshold_ms) {
        std::ostringstream oss;
        oss << "Idle gap exceeded: " << std::fixed << std::setprecision(3)
            << measurement.idle_gap_ms << " ms > " << idle_gap_threshold_ms << " ms";
        return record_alert(oss.str().c_str(), measurement.token_number,
                          measurement.idle_gap_ms, idle_gap_threshold_ms);
    }
    return true;
}

bool gpu_utilization_probe::check_underutilization_alert(const gpu_token_measurement & measurement) {
    if (measurement.gpu_utilization_ratio < underutilization_threshold) {
        std::ostringstream oss;
        oss << "GPU underutilized: " << std::fixed << std::setprecision(2)
            << (measurement.gpu_utilization_ratio * 100.0) << "% < "
            << (underutilization_threshold * 100.0) << "%";
        return record_alert(oss.str().c_str(), measurement.token_number,
                          measurement.gpu_utilization_ratio, underutilization_threshold);
    }
    return true;
}

bool gpu_utilization_probe::record_alert(const char * description, uint64_t token_num,
                                        double detected, double threshold) {
    if (!description) {
        return false;
    }

    gpu_utilization_alert alert = {
        description,
        token_num,
        detected,
        threshold,
        static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())
    };

    alerts.push_back(alert);
    alert_types[description]++;

    return true;
}

bool gpu_utilization_probe::finalize_measurements() {
    if (measurements.size() == 0) {
        fprintf(stderr, "[PROBE] WARNING: No measurements recorded\n");
        return false;
    }

    current_state = PROBE_STATE_COMPLETE;

    // Compute summary statistics
    summary.total_tokens_measured = measurements.size();
    summary.tokens_with_valid_data = 0;

    // Initialize aggregates
    summary.total_gpu_active_time_ms = 0.0;
    summary.min_gpu_active_time_ms = 1e9;
    summary.max_gpu_active_time_ms = 0.0;

    summary.total_wall_time_ms = 0.0;
    summary.min_wall_time_ms = 1e9;
    summary.max_wall_time_ms = 0.0;

    summary.total_idle_gap_ms = 0.0;
    summary.min_idle_gap_ms = 1e9;
    summary.max_idle_gap_ms = 0.0;

    summary.min_utilization_ratio = 1e9;
    summary.max_utilization_ratio = 0.0;
    summary.underutilized_count = 0;
    summary.critically_underutilized_count = 0;

    summary.min_tokens_per_sec = 1e9;
    summary.max_tokens_per_sec = 0.0;

    // Aggregate measurements
    for (const auto & m : measurements) {
        if (!m.measurement_valid) continue;

        summary.tokens_with_valid_data++;

        // GPU active time
        summary.total_gpu_active_time_ms += m.gpu_active_time_ms;
        summary.min_gpu_active_time_ms = std::min(summary.min_gpu_active_time_ms,
                                                  m.gpu_active_time_ms);
        summary.max_gpu_active_time_ms = std::max(summary.max_gpu_active_time_ms,
                                                  m.gpu_active_time_ms);

        // Wall time
        summary.total_wall_time_ms += m.token_wall_time_ms;
        summary.min_wall_time_ms = std::min(summary.min_wall_time_ms, m.token_wall_time_ms);
        summary.max_wall_time_ms = std::max(summary.max_wall_time_ms, m.token_wall_time_ms);

        // Idle gap
        summary.total_idle_gap_ms += m.idle_gap_ms;
        summary.min_idle_gap_ms = std::min(summary.min_idle_gap_ms, m.idle_gap_ms);
        summary.max_idle_gap_ms = std::max(summary.max_idle_gap_ms, m.idle_gap_ms);

        // Utilization
        summary.min_utilization_ratio = std::min(summary.min_utilization_ratio,
                                                 m.gpu_utilization_ratio);
        summary.max_utilization_ratio = std::max(summary.max_utilization_ratio,
                                                 m.gpu_utilization_ratio);

        if (m.gpu_utilization_ratio < underutilization_threshold) {
            summary.underutilized_count++;
        }
        if (m.gpu_utilization_ratio < critical_underutilization_threshold) {
            summary.critically_underutilized_count++;
        }

        // Throughput
        summary.min_tokens_per_sec = std::min(summary.min_tokens_per_sec,
                                             m.effective_throughput_tokens_per_sec);
        summary.max_tokens_per_sec = std::max(summary.max_tokens_per_sec,
                                             m.effective_throughput_tokens_per_sec);
    }

    // Compute averages
    if (summary.tokens_with_valid_data > 0) {
        summary.avg_gpu_active_time_ms = summary.total_gpu_active_time_ms /
                                         summary.tokens_with_valid_data;
        summary.avg_wall_time_ms = summary.total_wall_time_ms /
                                   summary.tokens_with_valid_data;
        summary.avg_idle_gap_ms = summary.total_idle_gap_ms /
                                  summary.tokens_with_valid_data;

        // Recompute average utilization from averages
        summary.avg_utilization_ratio = (summary.avg_wall_time_ms > 0.0) ?
                                       (summary.avg_gpu_active_time_ms /
                                        summary.avg_wall_time_ms) : 0.0;

        summary.avg_tokens_per_sec = (summary.avg_wall_time_ms > 0.0) ?
                                    (1000.0 / summary.avg_wall_time_ms) : 0.0;
    }

    // Health indicators
    summary.gpu_dominant = (summary.avg_utilization_ratio > underutilization_threshold);
    summary.critically_underutilized =
        (summary.avg_utilization_ratio < critical_underutilization_threshold);

    summary.measurement_timestamp_ns =
        static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    fprintf(stdout, "[PROBE] Finalized %llu measurements\n",
            (unsigned long long)summary.total_tokens_measured);

    return true;
}

bool gpu_utilization_probe::generate_utilization_report() {
    if (current_state != PROBE_STATE_COMPLETE) {
        fprintf(stderr, "[PROBE] ERROR: Not complete\n");
        return false;
    }

    std::ostringstream oss;

    // Header
    oss << "\n";
    oss << "==== GPU UTILIZATION REPORT ====\n";
    oss << "\n";

    // Summary statistics
    oss << "MEASUREMENT STATISTICS:\n";
    oss << "  Total tokens measured:    " << summary.total_tokens_measured << "\n";
    oss << "  Valid measurements:       " << summary.tokens_with_valid_data << "\n";
    oss << "\n";

    oss << "GPU ACTIVE TIME (CUDA events):\n";
    oss << "  Average:                  " << std::fixed << std::setprecision(3)
        << summary.avg_gpu_active_time_ms << " ms\n";
    oss << "  Min:                      " << summary.min_gpu_active_time_ms << " ms\n";
    oss << "  Max:                      " << summary.max_gpu_active_time_ms << " ms\n";
    oss << "\n";

    oss << "WALL-CLOCK TIME:\n";
    oss << "  Average:                  " << std::fixed << std::setprecision(3)
        << summary.avg_wall_time_ms << " ms\n";
    oss << "  Min:                      " << summary.min_wall_time_ms << " ms\n";
    oss << "  Max:                      " << summary.max_wall_time_ms << " ms\n";
    oss << "\n";

    oss << "IDLE GAP (wall - GPU active):\n";
    oss << "  Average:                  " << std::fixed << std::setprecision(3)
        << summary.avg_idle_gap_ms << " ms\n";
    oss << "  Min:                      " << summary.min_idle_gap_ms << " ms\n";
    oss << "  Max:                      " << summary.max_idle_gap_ms << " ms\n";
    oss << "  Threshold:                " << idle_gap_threshold_ms << " ms\n";
    oss << "\n";

    oss << "GPU UTILIZATION RATIO (GPU active / wall):\n";
    oss << "  Average:                  " << std::fixed << std::setprecision(3)
        << (summary.avg_utilization_ratio * 100.0) << "%\n";
    oss << "  Min:                      " << (summary.min_utilization_ratio * 100.0) << "%\n";
    oss << "  Max:                      " << (summary.max_utilization_ratio * 100.0) << "%\n";
    oss << "  Threshold:                " << (underutilization_threshold * 100.0) << "%\n";
    oss << "\n";

    oss << "UNDERUTILIZATION FLAGS:\n";
    oss << "  Tokens below threshold:   " << summary.underutilized_count
        << " / " << summary.total_tokens_measured << "\n";
    oss << "  Critically underutilized: " << summary.critically_underutilized_count
        << " / " << summary.total_tokens_measured << "\n";
    oss << "\n";

    oss << "THROUGHPUT (tokens/sec):\n";
    oss << "  Average:                  " << std::fixed << std::setprecision(2)
        << summary.avg_tokens_per_sec << " tokens/sec\n";
    oss << "  Min:                      " << summary.min_tokens_per_sec << " tokens/sec\n";
    oss << "  Max:                      " << summary.max_tokens_per_sec << " tokens/sec\n";
    oss << "\n";

    // Health assessment
    oss << "GPU DOMINANCE ASSESSMENT:\n";
    if (summary.gpu_dominant) {
        oss << "  ✅ GPU is dominant: " << (summary.avg_utilization_ratio * 100.0)
            << "% >= " << (underutilization_threshold * 100.0) << "%\n";
    } else {
        oss << "  ⚠️  GPU underutilized: " << (summary.avg_utilization_ratio * 100.0)
            << "% < " << (underutilization_threshold * 100.0) << "%\n";
        oss << "  WARNING: Possible CPU gating or sync overhead detected\n";
    }

    if (summary.critically_underutilized) {
        oss << "  ❌ CRITICALLY underutilized: " << (summary.avg_utilization_ratio * 100.0)
            << "% < " << (critical_underutilization_threshold * 100.0) << "%\n";
    }

    oss << "\n";

    // Sample measurements (first 20)
    if (measurements.size() > 0) {
        oss << "SAMPLE MEASUREMENTS (first 20):\n";
        uint32_t shown = 0;
        for (const auto & m : measurements) {
            if (shown >= 20) {
                oss << "  ... and " << (measurements.size() - 20) << " more\n";
                break;
            }

            oss << "  Token " << std::setw(5) << std::setfill('0') << m.token_number
                << ": GPU " << std::fixed << std::setprecision(2) << std::setfill(' ')
                << std::setw(6) << m.gpu_active_time_ms << " ms | "
                << "Wall " << std::setw(6) << m.token_wall_time_ms << " ms | "
                << "Idle " << std::setw(6) << m.idle_gap_ms << " ms | "
                << "Util " << std::setw(5) << (m.gpu_utilization_ratio * 100.0) << "%\n";

            shown++;
        }
        oss << "\n";
    }

    // Alerts (if any)
    if (alerts.size() > 0) {
        oss << "DETECTED ALERTS (" << alerts.size() << " total):\n";
        uint32_t shown = 0;
        for (const auto & alert : alerts) {
            if (shown >= 10) {
                oss << "  ... and " << (alerts.size() - 10) << " more\n";
                break;
            }
            oss << "  [Token " << alert.token_number << "] " << alert.alert_description << "\n";
            shown++;
        }
        oss << "\n";
    }

    // Footer
    oss << "================================\n";
    oss << "\n";

    fprintf(stdout, "%s", oss.str().c_str());

    return true;
}

bool gpu_utilization_probe::validate_utilization_metrics() {
    if (current_state != PROBE_STATE_COMPLETE) {
        fprintf(stderr, "[PROBE] ERROR: Not complete\n");
        return false;
    }

    bool all_valid = true;

    all_valid &= verify_gpu_dominance();
    all_valid &= verify_no_critical_underutilization();
    all_valid &= verify_measurement_consistency();

    if (all_valid) {
        fprintf(stdout, "[PROBE] All validations passed ✅\n");
    } else {
        fprintf(stderr, "[PROBE] Some validations failed ❌\n");
    }

    return all_valid;
}

bool gpu_utilization_probe::verify_gpu_dominance() const {
    if (summary.avg_utilization_ratio >= underutilization_threshold) {
        fprintf(stdout, "[PROBE] GPU dominance verified: %.2f%% >= %.2f%%\n",
                summary.avg_utilization_ratio * 100.0,
                underutilization_threshold * 100.0);
        return true;
    } else {
        fprintf(stderr, "[PROBE] GPU NOT dominant: %.2f%% < %.2f%%\n",
                summary.avg_utilization_ratio * 100.0,
                underutilization_threshold * 100.0);
        return false;
    }
}

bool gpu_utilization_probe::verify_no_critical_underutilization() const {
    if (!summary.critically_underutilized) {
        fprintf(stdout, "[PROBE] No critical underutilization detected\n");
        return true;
    } else {
        fprintf(stderr, "[PROBE] CRITICAL underutilization detected: %.2f%% < %.2f%%\n",
                summary.avg_utilization_ratio * 100.0,
                critical_underutilization_threshold * 100.0);
        return false;
    }
}

bool gpu_utilization_probe::verify_measurement_consistency() const {
    if (summary.tokens_with_valid_data == 0) {
        fprintf(stderr, "[PROBE] No valid measurements\n");
        return false;
    }

    if (summary.min_wall_time_ms <= 0.0 || summary.max_wall_time_ms <= 0.0) {
        fprintf(stderr, "[PROBE] Invalid wall time measurements\n");
        return false;
    }

    if (summary.avg_utilization_ratio < 0.0 || summary.avg_utilization_ratio > 1.0) {
        fprintf(stderr, "[PROBE] Invalid utilization ratio\n");
        return false;
    }

    fprintf(stdout, "[PROBE] Measurement consistency verified\n");
    return true;
}

std::string gpu_utilization_probe::format_measurement(const gpu_token_measurement & m) const {
    std::ostringstream oss;
    oss << "Token " << m.token_number << ": "
        << "GPU " << std::fixed << std::setprecision(2) << m.gpu_active_time_ms << "ms, "
        << "Wall " << m.token_wall_time_ms << "ms, "
        << "Idle " << m.idle_gap_ms << "ms, "
        << "Util " << (m.gpu_utilization_ratio * 100.0) << "%";
    return oss.str();
}

std::string gpu_utilization_probe::generate_report() const {
    if (current_state != PROBE_STATE_COMPLETE) {
        return "ERROR: Probe not complete\n";
    }

    std::ostringstream oss;

    oss << "GPU Utilization Report\n";
    oss << "Tokens: " << summary.total_tokens_measured << "\n";
    oss << "Avg GPU active: " << std::fixed << std::setprecision(3)
        << summary.avg_gpu_active_time_ms << " ms\n";
    oss << "Avg wall time: " << summary.avg_wall_time_ms << " ms\n";
    oss << "Avg idle gap: " << summary.avg_idle_gap_ms << " ms\n";
    oss << "Avg utilization: " << (summary.avg_utilization_ratio * 100.0) << "%\n";

    return oss.str();
}

std::string gpu_utilization_probe::generate_json_report() const {
    std::ostringstream oss;

    oss << "{\n";
    oss << "  \"total_tokens\": " << summary.total_tokens_measured << ",\n";
    oss << "  \"valid_measurements\": " << summary.tokens_with_valid_data << ",\n";
    oss << "  \"avg_gpu_active_ms\": " << std::fixed << std::setprecision(3)
        << summary.avg_gpu_active_time_ms << ",\n";
    oss << "  \"avg_wall_time_ms\": " << summary.avg_wall_time_ms << ",\n";
    oss << "  \"avg_idle_gap_ms\": " << summary.avg_idle_gap_ms << ",\n";
    oss << "  \"avg_utilization\": " << summary.avg_utilization_ratio << ",\n";
    oss << "  \"gpu_dominant\": " << (summary.gpu_dominant ? "true" : "false") << ",\n";
    oss << "  \"underutilized_tokens\": " << summary.underutilized_count << ",\n";
    oss << "  \"critically_underutilized_tokens\": " << summary.critically_underutilized_count << ",\n";
    oss << "  \"alerts\": " << alerts.size() << "\n";
    oss << "}\n";

    return oss.str();
}

// ============================================================================
// gpu_probe_guard Implementation
// ============================================================================

gpu_probe_guard::gpu_probe_guard(gpu_utilization_probe * probe_ptr)
    : guard_active(false), probe(probe_ptr) {
    if (probe) {
        guard_active = true;
    }
}

gpu_probe_guard::~gpu_probe_guard() {
    guard_active = false;
}

// ============================================================================
// C-Style Wrapper Functions
// ============================================================================

bool llama_init_gpu_utilization_probe() {
    if (g_gpu_utilization_probe != nullptr) {
        fprintf(stderr, "[PROBE] Already initialized\n");
        return false;
    }

    g_gpu_utilization_probe = new gpu_utilization_probe();
    if (!g_gpu_utilization_probe->initialize()) {
        fprintf(stderr, "[PROBE] Failed to initialize\n");
        delete g_gpu_utilization_probe;
        g_gpu_utilization_probe = nullptr;
        return false;
    }

    return true;
}

bool llama_enable_gpu_utilization_probe(bool enable) {
    if (!g_gpu_utilization_probe) {
        return false;
    }
    return g_gpu_utilization_probe->enable_probe(enable);
}

bool llama_is_gpu_utilization_probe_enabled() {
    if (!g_gpu_utilization_probe) {
        return false;
    }
    return g_gpu_utilization_probe->is_probe_enabled();
}

bool llama_begin_gpu_token_measurement(uint64_t token_number) {
    if (!g_gpu_utilization_probe) {
        return true;  // No-op
    }
    return g_gpu_utilization_probe->begin_token_measurement(token_number);
}

bool llama_record_gpu_active_time(double gpu_active_time_ms) {
    if (!g_gpu_utilization_probe) {
        return true;  // No-op
    }
    return g_gpu_utilization_probe->record_gpu_active_time(gpu_active_time_ms);
}

bool llama_end_gpu_token_measurement() {
    if (!g_gpu_utilization_probe) {
        return true;  // No-op
    }
    return g_gpu_utilization_probe->end_token_measurement();
}

bool llama_finalize_gpu_measurements() {
    if (!g_gpu_utilization_probe) {
        return false;
    }
    if (!g_gpu_utilization_probe->finalize_measurements()) {
        return false;
    }
    if (!g_gpu_utilization_probe->generate_utilization_report()) {
        return false;
    }
    return g_gpu_utilization_probe->validate_utilization_metrics();
}

bool llama_generate_gpu_utilization_report() {
    if (!g_gpu_utilization_probe) {
        return false;
    }
    return g_gpu_utilization_probe->generate_utilization_report();
}

bool llama_validate_gpu_utilization() {
    if (!g_gpu_utilization_probe) {
        return false;
    }
    return g_gpu_utilization_probe->validate_utilization_metrics();
}

const gpu_utilization_summary * llama_get_gpu_utilization_summary() {
    if (!g_gpu_utilization_probe) {
        return nullptr;
    }
    return &g_gpu_utilization_probe->get_summary();
}

const char * llama_get_gpu_utilization_report() {
    if (!g_gpu_utilization_probe) {
        return "";
    }
    return g_gpu_utilization_probe->generate_report().c_str();
}

const gpu_token_measurement * llama_get_token_measurement(uint64_t index) {
    if (!g_gpu_utilization_probe) {
        return nullptr;
    }
    const auto & measurements = g_gpu_utilization_probe->get_measurements();
    if (index >= measurements.size()) {
        return nullptr;
    }
    return &measurements[index];
}

void llama_print_gpu_utilization_report() {
    if (!g_gpu_utilization_probe) {
        return;
    }
    g_gpu_utilization_probe->generate_utilization_report();
}

void llama_print_gpu_utilization_summary() {
    if (!g_gpu_utilization_probe) {
        return;
    }

    const auto & summary = g_gpu_utilization_probe->get_summary();
    printf("\n=== GPU UTILIZATION SUMMARY ===\n");
    printf("Tokens measured: %llu\n", (unsigned long long)summary.total_tokens_measured);
    printf("Avg GPU active: %.2f ms\n", summary.avg_gpu_active_time_ms);
    printf("Avg wall time: %.2f ms\n", summary.avg_wall_time_ms);
    printf("Avg idle gap: %.2f ms\n", summary.avg_idle_gap_ms);
    printf("Avg utilization: %.2f%%\n", summary.avg_utilization_ratio * 100.0);
    printf("GPU dominant: %s\n", summary.gpu_dominant ? "YES" : "NO");
    printf("================================\n\n");
}

void llama_print_gpu_utilization_alerts() {
    if (!g_gpu_utilization_probe) {
        return;
    }

    const auto & alerts = g_gpu_utilization_probe->get_alerts();
    if (alerts.empty()) {
        printf("No alerts detected\n");
        return;
    }

    printf("\n=== GPU UTILIZATION ALERTS ===\n");
    for (const auto & alert : alerts) {
        printf("Token %llu: %s\n",
               (unsigned long long)alert.token_number,
               alert.alert_description);
    }
    printf("===============================\n\n");
}

void llama_print_gpu_token_measurements(uint32_t limit) {
    if (!g_gpu_utilization_probe) {
        return;
    }

    const auto & measurements = g_gpu_utilization_probe->get_measurements();
    printf("\n=== GPU TOKEN MEASUREMENTS ===\n");
    printf("Total measurements: %zu\n", measurements.size());

    uint32_t shown = 0;
    for (const auto & m : measurements) {
        if (limit > 0 && shown >= limit) break;

        printf("Token %5llu: GPU %.2f ms | Wall %.2f ms | Idle %.2f ms | Util %.2f%%\n",
               (unsigned long long)m.token_number,
               m.gpu_active_time_ms,
               m.token_wall_time_ms,
               m.idle_gap_ms,
               m.gpu_utilization_ratio * 100.0);

        shown++;
    }
    printf("==============================\n\n");
}

void llama_export_gpu_utilization_json(const char * filename) {
    if (!g_gpu_utilization_probe || !filename) {
        fprintf(stderr, "[PROBE] Invalid probe or filename\n");
        return;
    }

    std::string json = g_gpu_utilization_probe->generate_json_report();
    FILE * f = fopen(filename, "w");
    if (f) {
        fprintf(f, "%s", json.c_str());
        fclose(f);
        printf("[PROBE] JSON report exported to %s\n", filename);
    } else {
        fprintf(stderr, "[PROBE] Failed to open %s for writing\n", filename);
    }
}

// ============================================================================
// Self-Test Suite (11 comprehensive tests)
// ============================================================================

static bool gpu_probe_initialization_test() {
    fprintf(stdout, "\n[TEST] GPU Probe Initialization Test\n");

    auto * probe = new gpu_utilization_probe();
    if (!probe->initialize()) {
        fprintf(stderr, "  FAILED: Initialization\n");
        delete probe;
        return false;
    }

    if (!probe->is_probe_enabled()) {
        fprintf(stdout, "  PASSED (disabled by default) ✅\n");
        delete probe;
        return true;
    }

    fprintf(stderr, "  FAILED: Should be disabled by default\n");
    delete probe;
    return false;
}

static bool gpu_probe_enable_test() {
    fprintf(stdout, "\n[TEST] GPU Probe Enable Test\n");

    auto * probe = new gpu_utilization_probe();
    probe->initialize();

    if (!probe->enable_probe(true)) {
        fprintf(stderr, "  FAILED: Enable\n");
        delete probe;
        return false;
    }

    if (!probe->is_probe_enabled()) {
        fprintf(stderr, "  FAILED: Not enabled\n");
        delete probe;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete probe;
    return true;
}

static bool gpu_probe_measurement_test() {
    fprintf(stdout, "\n[TEST] GPU Probe Measurement Test\n");

    auto * probe = new gpu_utilization_probe();
    probe->initialize();
    probe->enable_probe(true);

    // Record a measurement
    if (!probe->begin_token_measurement(1)) {
        fprintf(stderr, "  FAILED: begin_token_measurement\n");
        delete probe;
        return false;
    }

    // Simulate GPU active time (2.5ms)
    if (!probe->record_gpu_active_time(2.5)) {
        fprintf(stderr, "  FAILED: record_gpu_active_time\n");
        delete probe;
        return false;
    }

    // Small delay to get wall time
    std::this_thread::sleep_for(std::chrono::milliseconds(3));

    if (!probe->end_token_measurement()) {
        fprintf(stderr, "  FAILED: end_token_measurement\n");
        delete probe;
        return false;
    }

    if (probe->get_measurement_count() != 1) {
        fprintf(stderr, "  FAILED: Measurement not recorded\n");
        delete probe;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete probe;
    return true;
}

static bool gpu_probe_multiple_measurements_test() {
    fprintf(stdout, "\n[TEST] Multiple Measurements Test\n");

    auto * probe = new gpu_utilization_probe();
    probe->initialize();
    probe->enable_probe(true);

    // Record 10 measurements
    for (int i = 0; i < 10; i++) {
        probe->begin_token_measurement(i);
        probe->record_gpu_active_time(2.0 + (i % 3) * 0.5);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        probe->end_token_measurement();
    }

    if (probe->get_measurement_count() != 10) {
        fprintf(stderr, "  FAILED: Expected 10 measurements, got %zu\n",
                probe->get_measurement_count());
        delete probe;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete probe;
    return true;
}

static bool gpu_probe_finalize_test() {
    fprintf(stdout, "\n[TEST] Finalization Test\n");

    auto * probe = new gpu_utilization_probe();
    probe->initialize();
    probe->enable_probe(true);

    // Record measurements
    for (int i = 0; i < 50; i++) {
        probe->begin_token_measurement(i);
        probe->record_gpu_active_time(2.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        probe->end_token_measurement();
    }

    if (!probe->finalize_measurements()) {
        fprintf(stderr, "  FAILED: finalize_measurements\n");
        delete probe;
        return false;
    }

    const auto & summary = probe->get_summary();
    if (summary.total_tokens_measured != 50) {
        fprintf(stderr, "  FAILED: Summary not computed\n");
        delete probe;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete probe;
    return true;
}

static bool gpu_probe_utilization_ratio_test() {
    fprintf(stdout, "\n[TEST] Utilization Ratio Test\n");

    auto * probe = new gpu_utilization_probe();
    probe->initialize();
    probe->enable_probe(true);

    // Record measurements with controlled timing
    for (int i = 0; i < 50; i++) {
        probe->begin_token_measurement(i);
        probe->record_gpu_active_time(2.4);  // 2.4ms GPU active
        std::this_thread::sleep_for(std::chrono::milliseconds(3));  // ~3ms wall
        probe->end_token_measurement();
    }

    probe->finalize_measurements();

    const auto & summary = probe->get_summary();
    // Expected ratio ~0.80 (2.4/3.0)
    if (summary.avg_utilization_ratio < 0.70 || summary.avg_utilization_ratio > 0.95) {
        fprintf(stderr, "  FAILED: Utilization ratio out of range: %.2f\n",
                summary.avg_utilization_ratio);
        delete probe;
        return false;
    }

    fprintf(stdout, "  PASSED (utilization: %.2f%%) ✅\n",
            summary.avg_utilization_ratio * 100.0);
    delete probe;
    return true;
}

static bool gpu_probe_idle_gap_detection_test() {
    fprintf(stdout, "\n[TEST] Idle Gap Detection Test\n");

    auto * probe = new gpu_utilization_probe();
    probe->initialize();
    probe->enable_probe(true);
    probe->set_idle_gap_threshold(0.3);  // 0.3ms threshold

    // Record measurements with large idle gaps
    for (int i = 0; i < 50; i++) {
        probe->begin_token_measurement(i);
        probe->record_gpu_active_time(1.0);  // 1ms GPU active
        std::this_thread::sleep_for(std::chrono::milliseconds(3));  // ~3ms wall, ~2ms idle
        probe->end_token_measurement();
    }

    probe->finalize_measurements();

    if (probe->get_alert_count() == 0) {
        fprintf(stderr, "  FAILED: No idle gap alerts detected\n");
        delete probe;
        return false;
    }

    fprintf(stdout, "  PASSED (alerts: %llu) ✅\n", (unsigned long long)probe->get_alert_count());
    delete probe;
    return true;
}

static bool gpu_probe_underutilization_detection_test() {
    fprintf(stdout, "\n[TEST] Underutilization Detection Test\n");

    auto * probe = new gpu_utilization_probe();
    probe->initialize();
    probe->enable_probe(true);
    probe->set_underutilization_threshold(0.90);  // 90% threshold

    // Record measurements with low utilization
    for (int i = 0; i < 50; i++) {
        probe->begin_token_measurement(i);
        probe->record_gpu_active_time(1.0);  // 1ms GPU active
        std::this_thread::sleep_for(std::chrono::milliseconds(3));  // ~3ms wall, 0.33 util
        probe->end_token_measurement();
    }

    probe->finalize_measurements();

    const auto & summary = probe->get_summary();
    if (summary.underutilized_count == 0) {
        fprintf(stderr, "  FAILED: No underutilization detected\n");
        delete probe;
        return false;
    }

    fprintf(stdout, "  PASSED (underutilized: %llu) ✅\n",
            (unsigned long long)summary.underutilized_count);
    delete probe;
    return true;
}

static bool gpu_probe_json_export_test() {
    fprintf(stdout, "\n[TEST] JSON Export Test\n");

    auto * probe = new gpu_utilization_probe();
    probe->initialize();
    probe->enable_probe(true);

    for (int i = 0; i < 50; i++) {
        probe->begin_token_measurement(i);
        probe->record_gpu_active_time(2.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        probe->end_token_measurement();
    }

    probe->finalize_measurements();

    std::string json = probe->generate_json_report();
    if (json.empty() || json.find("\"avg_utilization\"") == std::string::npos) {
        fprintf(stderr, "  FAILED: Invalid JSON\n");
        delete probe;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete probe;
    return true;
}

static bool gpu_probe_disabled_noop_test() {
    fprintf(stdout, "\n[TEST] Disabled No-Op Test\n");

    auto * probe = new gpu_utilization_probe();
    probe->initialize();
    // Don't enable probe

    // These should be no-ops
    if (!probe->begin_token_measurement(1)) {
        fprintf(stderr, "  FAILED: begin should succeed even when disabled\n");
        delete probe;
        return false;
    }

    if (probe->get_measurement_count() != 0) {
        fprintf(stderr, "  FAILED: Measurements should not be recorded when disabled\n");
        delete probe;
        return false;
    }

    fprintf(stdout, "  PASSED (no-ops when disabled) ✅\n");
    delete probe;
    return true;
}

static bool gpu_probe_full_workflow_test() {
    fprintf(stdout, "\n[TEST] Full Workflow Test\n");

    auto * probe = new gpu_utilization_probe();

    if (!probe->initialize()) {
        fprintf(stderr, "  FAILED: initialize\n");
        delete probe;
        return false;
    }

    if (!probe->enable_probe(true)) {
        fprintf(stderr, "  FAILED: enable\n");
        delete probe;
        return false;
    }

    // Record 60 measurements
    for (int i = 0; i < 60; i++) {
        probe->begin_token_measurement(i);
        probe->record_gpu_active_time(2.3);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        probe->end_token_measurement();
    }

    if (!probe->finalize_measurements()) {
        fprintf(stderr, "  FAILED: finalize\n");
        delete probe;
        return false;
    }

    if (!probe->generate_utilization_report()) {
        fprintf(stderr, "  FAILED: generate report\n");
        delete probe;
        return false;
    }

    if (!probe->validate_utilization_metrics()) {
        fprintf(stderr, "  FAILED: validate\n");
        delete probe;
        return false;
    }

    const auto & summary = probe->get_summary();
    if (summary.total_tokens_measured != 60) {
        fprintf(stderr, "  FAILED: Final summary incorrect\n");
        delete probe;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete probe;
    return true;
}

// Self-test runner
static bool run_gpu_probe_self_tests() {
    fprintf(stdout, "\n========================================\n");
    fprintf(stdout, "Running GPU Utilization Probe Self-Tests\n");
    fprintf(stdout, "========================================\n");

    bool all_passed = true;
    all_passed &= gpu_probe_initialization_test();
    all_passed &= gpu_probe_enable_test();
    all_passed &= gpu_probe_measurement_test();
    all_passed &= gpu_probe_multiple_measurements_test();
    all_passed &= gpu_probe_finalize_test();
    all_passed &= gpu_probe_utilization_ratio_test();
    all_passed &= gpu_probe_idle_gap_detection_test();
    all_passed &= gpu_probe_underutilization_detection_test();
    all_passed &= gpu_probe_json_export_test();
    all_passed &= gpu_probe_disabled_noop_test();
    all_passed &= gpu_probe_full_workflow_test();

    fprintf(stdout, "\n========================================\n");
    if (all_passed) {
        fprintf(stdout, "All tests PASSED ✅\n");
    } else {
        fprintf(stdout, "Some tests FAILED ❌\n");
    }
    fprintf(stdout, "========================================\n\n");

    return all_passed;
}

// Auto-run self-tests on module load
__attribute__((constructor))
static void gpu_probe_self_tests_ctor() {
    // Uncomment to auto-run tests on module load:
    // run_gpu_probe_self_tests();
}
