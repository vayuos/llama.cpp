#include "llama-decode-stability-harness.h"
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <chrono>

// Global state
decode_stability_harness * g_decode_stability_harness = nullptr;

// ============================================================================
// decode_stability_harness Implementation
// ============================================================================

decode_stability_harness::decode_stability_harness()
    : current_state(HARNESS_STATE_UNINITIALIZED),
      test_mode(STRESS_MODE_STANDARD),
      harness_enabled(false),
      test_running(false),
      tokens_generated(0),
      samples_collected(0),
      target_tokens(10000),
      sample_interval(10),
      drift_check_interval(100),
      throughput_drift_threshold(0.10),  // 10%
      min_gpu_utilization_threshold(0.85),
      cpu_utilization_threshold(0.95),
      deterministic_mode(true),
      strict_mode(false),
      initial_avg_tps(0.0),
      window_avg_tps(0.0),
      last_drift_check_token(0),
      initial_gpu_memory_free(0) {
    std::memset(&summary, 0, sizeof(summary));
}

bool decode_stability_harness::initialize(uint64_t target_token_count,
                                         stress_test_mode mode) {
    if (current_state != HARNESS_STATE_UNINITIALIZED) {
        fprintf(stderr, "[HARNESS] ERROR: Already initialized\n");
        return false;
    }

    current_state = HARNESS_STATE_SETUP;
    test_mode = mode;
    target_tokens = target_token_count;

    fprintf(stdout, "[HARNESS] Decode Stability Harness Initialized\n");
    fprintf(stdout, "[HARNESS] Target tokens: %llu\n", (unsigned long long)target_tokens);
    fprintf(stdout, "[HARNESS] Test mode: %s\n", format_test_mode(mode).c_str());
    fprintf(stdout, "[HARNESS] Drift threshold: %.1f%%\n", throughput_drift_threshold * 100.0);
    fprintf(stdout, "[HARNESS] Min GPU utilization: %.1f%%\n",
            min_gpu_utilization_threshold * 100.0);
    fprintf(stdout, "[HARNESS] CPU saturation threshold: %.1f%%\n",
            cpu_utilization_threshold * 100.0);
    fprintf(stdout, "[HARNESS] Deterministic mode: %s\n", deterministic_mode ? "ON" : "OFF");

    return true;
}

bool decode_stability_harness::configure_thresholds(double drift_threshold,
                                                    double min_gpu_util,
                                                    double cpu_util_threshold) {
    if (drift_threshold < 0.0 || drift_threshold > 1.0) {
        fprintf(stderr, "[HARNESS] ERROR: Invalid drift threshold\n");
        return false;
    }

    throughput_drift_threshold = drift_threshold;
    min_gpu_utilization_threshold = min_gpu_util;
    cpu_utilization_threshold = cpu_util_threshold;

    fprintf(stdout, "[HARNESS] Thresholds configured\n");
    return true;
}

bool decode_stability_harness::begin_stability_test() {
    if (current_state != HARNESS_STATE_SETUP) {
        fprintf(stderr, "[HARNESS] ERROR: Invalid state for test start\n");
        return false;
    }

    current_state = HARNESS_STATE_RUNNING;
    test_running.store(true);
    tokens_generated.store(0);
    samples_collected.store(0);

    summary.test_start_timestamp_ns =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    fprintf(stdout, "[HARNESS] Stability test started\n");
    return true;
}

bool decode_stability_harness::end_stability_test() {
    if (current_state != HARNESS_STATE_RUNNING) {
        fprintf(stderr, "[HARNESS] ERROR: Test not running\n");
        return false;
    }

    test_running.store(false);
    summary.test_end_timestamp_ns =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    summary.test_duration_seconds =
        (summary.test_end_timestamp_ns - summary.test_start_timestamp_ns) / 1e9;

    fprintf(stdout, "[HARNESS] Stability test ended\n");
    fprintf(stdout, "[HARNESS] Test duration: %.2f seconds\n", summary.test_duration_seconds);

    return true;
}

bool decode_stability_harness::record_token_sample(const stability_token_sample & sample) {
    if (!harness_enabled.load() || !test_running.load()) {
        return true;  // No-op when disabled
    }

    if (samples_collected.load() % sample_interval == 0) {
        token_samples.push_back(sample);
        samples_collected.store(samples_collected.load() + 1);
    }

    tokens_generated.store(sample.token_number);

    // Check for invariant violations
    if (sample.cpu_execution_detected) {
        record_violation("GPU-Exclusive Execution", "CPU execution detected",
                        sample.token_number, 1.0, 0.0, true);
    }

    if (sample.pcie_violation_detected) {
        record_violation("PCIe Cleanliness",
                        "H2D or D2H transfers detected",
                        sample.token_number,
                        sample.h2d_bytes + sample.d2h_bytes, 0.0, true);
    }

    if (sample.allocation_during_decode) {
        record_violation("No Allocations During Decode",
                        "Allocation event detected",
                        sample.token_number, 1.0, 0.0, false);
    }

    if (sample.gpu_utilization_ratio < min_gpu_utilization_threshold) {
        record_violation("GPU Utilization Stability",
                        "GPU utilization below threshold",
                        sample.token_number,
                        sample.gpu_utilization_ratio,
                        min_gpu_utilization_threshold, false);
    }

    return true;
}

bool decode_stability_harness::check_invariants_at_token(uint64_t token_num) {
    if (!harness_enabled.load() || !test_running.load()) {
        return true;
    }

    // Check drift every N tokens
    if (token_num - last_drift_check_token >= drift_check_interval) {
        if (!check_drift_every_n_tokens(token_num)) {
            if (strict_mode) {
                fprintf(stderr, "[HARNESS] STRICT MODE: Aborting on drift violation\n");
                return false;
            }
        }
        last_drift_check_token = token_num;
    }

    return true;
}

bool decode_stability_harness::check_drift_every_n_tokens(uint64_t token_num) {
    if (token_samples.size() < 2) {
        return true;
    }

    // Compute current window average
    std::vector<double> recent_tps;
    size_t window_size = std::min((size_t)drift_check_interval, token_samples.size());

    for (size_t i = token_samples.size() - window_size; i < token_samples.size(); i++) {
        recent_tps.push_back(token_samples[i].tokens_per_sec);
    }

    if (recent_tps.empty()) {
        return true;
    }

    window_avg_tps = std::accumulate(recent_tps.begin(), recent_tps.end(), 0.0) /
                    recent_tps.size();

    // Compare with initial average
    if (initial_avg_tps == 0.0 && token_samples.size() > 0) {
        initial_avg_tps = token_samples[0].tokens_per_sec;
    }

    if (initial_avg_tps > 0.0) {
        double drift = (initial_avg_tps - window_avg_tps) / initial_avg_tps;

        if (drift > throughput_drift_threshold) {
            fprintf(stderr, "[HARNESS] Throughput drift detected at token %llu: %.1f%% drop\n",
                   (unsigned long long)token_num, drift * 100.0);
            record_violation("Throughput Stability",
                           "Performance drift detected",
                           token_num, window_avg_tps, initial_avg_tps, true);
            return false;
        }
    }

    return true;
}

bool decode_stability_harness::finalize_test() {
    if (current_state != HARNESS_STATE_RUNNING) {
        fprintf(stderr, "[HARNESS] ERROR: Test not running\n");
        return false;
    }

    end_stability_test();

    // Compute summary statistics
    summary.total_tokens_generated = tokens_generated.load();
    summary.total_samples_collected = samples_collected.load();

    if (token_samples.size() == 0) {
        fprintf(stderr, "[HARNESS] ERROR: No samples collected\n");
        return false;
    }

    // Throughput statistics
    std::vector<double> tps_values;
    for (const auto & sample : token_samples) {
        tps_values.push_back(sample.tokens_per_sec);
    }

    summary.avg_tokens_per_sec = std::accumulate(tps_values.begin(), tps_values.end(), 0.0) /
                                 tps_values.size();
    summary.initial_tokens_per_sec = token_samples[0].tokens_per_sec;
    summary.final_tokens_per_sec = token_samples[token_samples.size() - 1].tokens_per_sec;
    summary.min_tokens_per_sec = *std::min_element(tps_values.begin(), tps_values.end());
    summary.max_tokens_per_sec = *std::max_element(tps_values.begin(), tps_values.end());

    summary.throughput_variance_percent = compute_throughput_variance();
    summary.throughput_drift_percent = compute_throughput_drift();

    // GPU statistics
    std::vector<double> gpu_util_values;
    for (const auto & sample : token_samples) {
        gpu_util_values.push_back(sample.gpu_utilization_ratio);
        summary.avg_idle_gap_ms += sample.idle_gap_ms;
        summary.total_h2d_bytes += sample.h2d_bytes;
        summary.total_d2h_bytes += sample.d2h_bytes;
        summary.total_d2d_bytes += sample.d2d_bytes;

        summary.cpu_execution_violations += sample.cpu_execution_detected ? 1 : 0;
        summary.pcie_violations += sample.pcie_violation_detected ? 1 : 0;
        summary.allocation_events_during_decode += sample.allocation_during_decode ? 1 : 0;
    }

    if (gpu_util_values.size() > 0) {
        summary.avg_gpu_utilization = std::accumulate(gpu_util_values.begin(),
                                                     gpu_util_values.end(), 0.0) /
                                      gpu_util_values.size();
        summary.min_gpu_utilization = *std::min_element(gpu_util_values.begin(),
                                                       gpu_util_values.end());
        summary.max_gpu_utilization = *std::max_element(gpu_util_values.begin(),
                                                       gpu_util_values.end());
    }

    summary.avg_idle_gap_ms /= token_samples.size();

    // Memory leak detection
    if (summary.initial_gpu_memory > 0 && summary.final_gpu_memory > 0) {
        summary.gpu_memory_delta = (int64_t)summary.final_gpu_memory -
                                  (int64_t)summary.initial_gpu_memory;
        // Flag leak if memory decreased by more than 100MB
        summary.memory_leak_detected = (summary.gpu_memory_delta < -100 * 1024 * 1024);
    }

    // Overall invariant status
    summary.all_invariants_held = (summary.cpu_execution_violations == 0 &&
                                  summary.pcie_violations == 0 &&
                                  summary.throughput_drift_percent <= throughput_drift_threshold * 100.0 &&
                                  summary.avg_gpu_utilization >= min_gpu_utilization_threshold &&
                                  !summary.memory_leak_detected);

    summary.stability_test_passed = (summary.all_invariants_held &&
                                    invariant_violations.size() == 0);

    current_state = summary.stability_test_passed ? HARNESS_STATE_COMPLETE : HARNESS_STATE_FAILED;

    fprintf(stdout, "[HARNESS] Finalization complete. Status: %s\n",
            summary.stability_test_passed ? "PASS ✅" : "FAIL ❌");

    return summary.stability_test_passed;
}

bool decode_stability_harness::generate_stability_report() {
    if (current_state != HARNESS_STATE_COMPLETE && current_state != HARNESS_STATE_FAILED) {
        fprintf(stderr, "[HARNESS] ERROR: Test not complete\n");
        return false;
    }

    std::ostringstream oss;

    // Header
    oss << "\n";
    oss << "=== LONG RUN DECODE STABILITY REPORT ===\n";
    oss << "\n";

    // Test configuration
    oss << "TEST CONFIGURATION:\n";
    oss << "  Mode:                     " << format_test_mode(test_mode) << "\n";
    oss << "  Duration:                 " << std::fixed << std::setprecision(2)
        << summary.test_duration_seconds << " seconds\n";
    oss << "  Deterministic:            " << (deterministic_mode ? "YES" : "NO") << "\n";
    oss << "\n";

    // Token statistics
    oss << "TOKEN GENERATION:\n";
    oss << "  Total tokens:             " << summary.total_tokens_generated << "\n";
    oss << "  Samples collected:        " << summary.total_samples_collected << "\n";
    oss << "\n";

    // Throughput statistics
    oss << "THROUGHPUT (tokens/sec):\n";
    oss << "  Average:                  " << std::fixed << std::setprecision(2)
        << summary.avg_tokens_per_sec << "\n";
    oss << "  Initial:                  " << summary.initial_tokens_per_sec << "\n";
    oss << "  Final:                    " << summary.final_tokens_per_sec << "\n";
    oss << "  Min:                      " << summary.min_tokens_per_sec << "\n";
    oss << "  Max:                      " << summary.max_tokens_per_sec << "\n";
    oss << "  Variance:                 " << summary.throughput_variance_percent << "%\n";
    oss << "  Drift:                    " << summary.throughput_drift_percent << "%\n";
    oss << "\n";

    // GPU metrics
    oss << "GPU METRICS:\n";
    oss << "  Avg utilization:          " << std::fixed << std::setprecision(2)
        << (summary.avg_gpu_utilization * 100.0) << "%\n";
    oss << "  Min utilization:          " << (summary.min_gpu_utilization * 100.0) << "%\n";
    oss << "  Max utilization:          " << (summary.max_gpu_utilization * 100.0) << "%\n";
    oss << "  Avg idle gap:             " << summary.avg_idle_gap_ms << " ms\n";
    oss << "\n";

    // PCIe metrics
    oss << "PCIe TRANSFERS:\n";
    oss << "  Total H2D bytes:          " << summary.total_h2d_bytes << "\n";
    oss << "  Total D2H bytes:          " << summary.total_d2h_bytes << "\n";
    oss << "  Total D2D bytes:          " << summary.total_d2d_bytes << "\n";
    oss << "\n";

    // Invariant violations
    oss << "INVARIANT VIOLATIONS:\n";
    oss << "  CPU execution violations: " << summary.cpu_execution_violations << "\n";
    oss << "  PCIe violations:          " << summary.pcie_violations << "\n";
    oss << "  Allocations during decode:" << summary.allocation_events_during_decode << "\n";
    oss << "  KV reallocations:         " << summary.kv_reallocation_events << "\n";
    oss << "\n";

    // Memory metrics
    oss << "MEMORY:\n";
    oss << "  Initial GPU free:         "
        << std::fixed << std::setprecision(0) << (summary.initial_gpu_memory / (1024.0*1024.0))
        << " MB\n";
    oss << "  Final GPU free:           "
        << (summary.final_gpu_memory / (1024.0*1024.0)) << " MB\n";
    oss << "  Delta:                    "
        << (summary.gpu_memory_delta / (1024.0*1024.0)) << " MB\n";
    oss << "  Memory leak detected:     " << (summary.memory_leak_detected ? "YES" : "NO") << "\n";
    oss << "  GPU fragmentation stable: " << (summary.gpu_fragmentation_stable ? "YES" : "NO") << "\n";
    oss << "\n";

    // CPU metrics
    oss << "CPU:\n";
    oss << "  Avg utilization:          " << std::fixed << std::setprecision(1)
        << summary.avg_cpu_utilization << "%\n";
    oss << "  Max utilization:          " << summary.max_cpu_utilization << "%\n";
    oss << "  Regression detected:      " << (summary.cpu_regression_detected ? "YES" : "NO") << "\n";
    oss << "\n";

    // Overall status
    oss << "STATUS: " << format_status(summary.stability_test_passed) << "\n";
    if (!summary.stability_test_passed && invariant_violations.size() > 0) {
        oss << "REASON: " << invariant_violations[0].violation_description << "\n";
    }

    oss << "\n";
    oss << "=========================================\n";
    oss << "\n";

    printf("%s", oss.str().c_str());

    return true;
}

bool decode_stability_harness::record_violation(const char * invariant_name,
                                               const char * description,
                                               uint64_t token_num,
                                               double value,
                                               double threshold,
                                               bool is_critical) {
    stability_invariant_violation violation = {
        invariant_name,
        description,
        token_num,
        value,
        threshold,
        static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
        is_critical
    };

    invariant_violations.push_back(violation);

    if (is_critical) {
        fprintf(stderr, "[HARNESS] CRITICAL: %s - %s (token %llu)\n",
                invariant_name, description, (unsigned long long)token_num);
    } else {
        fprintf(stdout, "[HARNESS] Warning: %s - %s (token %llu)\n",
                invariant_name, description, (unsigned long long)token_num);
    }

    return true;
}

double decode_stability_harness::compute_throughput_variance() {
    if (token_samples.size() < 2) {
        return 0.0;
    }

    std::vector<double> tps_values;
    for (const auto & sample : token_samples) {
        tps_values.push_back(sample.tokens_per_sec);
    }

    double mean = std::accumulate(tps_values.begin(), tps_values.end(), 0.0) /
                  tps_values.size();

    double variance = 0.0;
    for (double val : tps_values) {
        variance += (val - mean) * (val - mean);
    }
    variance /= tps_values.size();

    double stddev = std::sqrt(variance);
    return (stddev / mean) * 100.0;
}

double decode_stability_harness::compute_throughput_drift() {
    if (token_samples.size() < 2) {
        return 0.0;
    }

    double initial = token_samples[0].tokens_per_sec;
    double final = token_samples[token_samples.size() - 1].tokens_per_sec;

    if (initial <= 0.0) {
        return 0.0;
    }

    return ((final - initial) / initial) * 100.0;
}

std::string decode_stability_harness::generate_report() const {
    std::ostringstream oss;
    oss << "Stability Test Report\n";
    oss << "Tokens: " << summary.total_tokens_generated << "\n";
    oss << "Duration: " << std::fixed << std::setprecision(2)
        << summary.test_duration_seconds << "s\n";
    oss << "Avg throughput: " << summary.avg_tokens_per_sec << " t/s\n";
    oss << "Status: " << (summary.stability_test_passed ? "PASS" : "FAIL") << "\n";
    return oss.str();
}

std::string decode_stability_harness::generate_json_report() const {
    std::ostringstream oss;

    oss << "{\n";
    oss << "  \"tokens_generated\": " << summary.total_tokens_generated << ",\n";
    oss << "  \"avg_tokens_per_sec\": " << std::fixed << std::setprecision(2)
        << summary.avg_tokens_per_sec << ",\n";
    oss << "  \"throughput_drift\": " << summary.throughput_drift_percent << ",\n";
    oss << "  \"avg_gpu_utilization\": " << summary.avg_gpu_utilization << ",\n";
    oss << "  \"h2d_bytes\": " << summary.total_h2d_bytes << ",\n";
    oss << "  \"d2h_bytes\": " << summary.total_d2h_bytes << ",\n";
    oss << "  \"violations\": {\n";
    oss << "    \"cpu_execution\": " << summary.cpu_execution_violations << ",\n";
    oss << "    \"pcie\": " << summary.pcie_violations << ",\n";
    oss << "    \"allocations\": " << summary.allocation_events_during_decode << "\n";
    oss << "  },\n";
    oss << "  \"memory_leak\": " << (summary.memory_leak_detected ? "true" : "false") << ",\n";
    oss << "  \"passed\": " << (summary.stability_test_passed ? "true" : "false") << "\n";
    oss << "}\n";

    return oss.str();
}

std::string decode_stability_harness::format_test_mode(stress_test_mode mode) const {
    switch (mode) {
        case STRESS_MODE_STANDARD:
            return "Standard";
        case STRESS_MODE_LONG_CONTEXT:
            return "Long-Context (8k-16k)";
        case STRESS_MODE_QUANTIZED_MMQ:
            return "Quantized MMQ";
        case STRESS_MODE_CUBLAS_DENSE:
            return "cuBLAS Dense";
        case STRESS_MODE_FLASH_ATTENTION:
            return "Flash-Attention";
        case STRESS_MODE_SERVER:
            return "Server";
        default:
            return "Unknown";
    }
}

std::string decode_stability_harness::format_status(bool passed) const {
    return passed ? "PASS ✅" : "FAIL ❌";
}

// ============================================================================
// stability_harness_guard Implementation
// ============================================================================

stability_harness_guard::stability_harness_guard(decode_stability_harness * harness_ptr)
    : guard_active(false), harness(harness_ptr) {
    if (harness) {
        guard_active = true;
    }
}

stability_harness_guard::~stability_harness_guard() {
    guard_active = false;
}

// ============================================================================
// C-Style Wrapper Functions
// ============================================================================

bool llama_init_stability_harness(uint64_t target_tokens, int stress_mode) {
    if (g_decode_stability_harness != nullptr) {
        fprintf(stderr, "[HARNESS] Already initialized\n");
        return false;
    }

    g_decode_stability_harness = new decode_stability_harness();
    if (!g_decode_stability_harness->initialize(target_tokens, (stress_test_mode)stress_mode)) {
        fprintf(stderr, "[HARNESS] Failed to initialize\n");
        delete g_decode_stability_harness;
        g_decode_stability_harness = nullptr;
        return false;
    }

    return true;
}

bool llama_enable_stability_harness(bool enable) {
    if (!g_decode_stability_harness) {
        return false;
    }
    return g_decode_stability_harness->enable_harness(enable);
}

bool llama_is_stability_harness_enabled() {
    if (!g_decode_stability_harness) {
        return false;
    }
    return g_decode_stability_harness->is_harness_enabled();
}

bool llama_begin_stability_test() {
    if (!g_decode_stability_harness) {
        return false;
    }
    return g_decode_stability_harness->begin_stability_test();
}

bool llama_end_stability_test() {
    if (!g_decode_stability_harness) {
        return false;
    }
    return g_decode_stability_harness->end_stability_test();
}

bool llama_record_stability_sample(const stability_token_sample * sample) {
    if (!g_decode_stability_harness || !sample) {
        return false;
    }
    return g_decode_stability_harness->record_token_sample(*sample);
}

bool llama_check_stability_invariants(uint64_t token_num) {
    if (!g_decode_stability_harness) {
        return true;
    }
    return g_decode_stability_harness->check_invariants_at_token(token_num);
}

bool llama_finalize_stability_test() {
    if (!g_decode_stability_harness) {
        return false;
    }
    return g_decode_stability_harness->finalize_test();
}

bool llama_generate_stability_report() {
    if (!g_decode_stability_harness) {
        return false;
    }
    return g_decode_stability_harness->generate_stability_report();
}

bool llama_validate_all_stability_invariants() {
    if (!g_decode_stability_harness) {
        return false;
    }

    const auto & summary = g_decode_stability_harness->get_summary();
    return summary.stability_test_passed;
}

const stability_test_summary * llama_get_stability_summary() {
    if (!g_decode_stability_harness) {
        return nullptr;
    }
    return &g_decode_stability_harness->get_summary();
}

const char * llama_get_stability_report() {
    if (!g_decode_stability_harness) {
        return "";
    }
    return g_decode_stability_harness->generate_report().c_str();
}

void llama_print_stability_report() {
    if (g_decode_stability_harness) {
        g_decode_stability_harness->generate_stability_report();
    }
}

void llama_print_stability_summary() {
    if (!g_decode_stability_harness) {
        return;
    }

    const auto & summary = g_decode_stability_harness->get_summary();
    printf("\n=== STABILITY TEST SUMMARY ===\n");
    printf("Tokens: %llu\n", (unsigned long long)summary.total_tokens_generated);
    printf("Avg throughput: %.2f t/s\n", summary.avg_tokens_per_sec);
    printf("Drift: %.2f%%\n", summary.throughput_drift_percent);
    printf("GPU utilization: %.2f%%\n", summary.avg_gpu_utilization * 100.0);
    printf("Status: %s\n", summary.stability_test_passed ? "PASS ✅" : "FAIL ❌");
    printf("==============================\n\n");
}

void llama_print_stability_violations() {
    if (!g_decode_stability_harness) {
        return;
    }

    const auto & violations = g_decode_stability_harness->get_violations();
    if (violations.empty()) {
        printf("No violations detected\n");
        return;
    }

    printf("\n=== STABILITY TEST VIOLATIONS ===\n");
    for (const auto & v : violations) {
        printf("[Token %llu] %s: %s\n",
               (unsigned long long)v.token_at_violation,
               v.invariant_name,
               v.violation_description);
    }
    printf("=================================\n\n");
}

void llama_export_stability_json(const char * filename) {
    if (!g_decode_stability_harness || !filename) {
        fprintf(stderr, "[HARNESS] Invalid harness or filename\n");
        return;
    }

    std::string json = g_decode_stability_harness->generate_json_report();
    FILE * f = fopen(filename, "w");
    if (f) {
        fprintf(f, "%s", json.c_str());
        fclose(f);
        printf("[HARNESS] JSON report exported to %s\n", filename);
    } else {
        fprintf(stderr, "[HARNESS] Failed to open %s for writing\n", filename);
    }
}

// ============================================================================
// Self-Test Suite (8 comprehensive tests)
// ============================================================================

static bool stability_harness_initialization_test() {
    fprintf(stdout, "\n[TEST] Stability Harness Initialization Test\n");

    auto * harness = new decode_stability_harness();
    if (!harness->initialize(10000, STRESS_MODE_STANDARD)) {
        fprintf(stderr, "  FAILED: Initialization\n");
        delete harness;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete harness;
    return true;
}

static bool stability_harness_test_lifecycle_test() {
    fprintf(stdout, "\n[TEST] Test Lifecycle Test\n");

    auto * harness = new decode_stability_harness();
    harness->initialize(100, STRESS_MODE_STANDARD);
    harness->enable_harness(true);

    if (!harness->begin_stability_test()) {
        fprintf(stderr, "  FAILED: begin_stability_test\n");
        delete harness;
        return false;
    }

    stability_token_sample sample = {
        1, 2.5, 2.7, 0.2, 0.93, 0, 0, 0, 10.0, 2048ULL*1024*1024, 4096ULL*1024*1024, 1024,
        40.0, false, false, false, 0
    };

    for (int i = 0; i < 10; i++) {
        sample.token_number = i;
        harness->record_token_sample(sample);
    }

    if (!harness->end_stability_test()) {
        fprintf(stderr, "  FAILED: end_stability_test\n");
        delete harness;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete harness;
    return true;
}

static bool stability_harness_sample_recording_test() {
    fprintf(stdout, "\n[TEST] Sample Recording Test\n");

    auto * harness = new decode_stability_harness();
    harness->initialize(100, STRESS_MODE_STANDARD);
    harness->enable_harness(true);
    harness->begin_stability_test();

    stability_token_sample sample = {
        1, 2.5, 2.7, 0.2, 0.93, 0, 0, 0, 10.0, 2048ULL*1024*1024, 4096ULL*1024*1024, 1024,
        40.0, false, false, false, 0
    };

    for (int i = 0; i < 50; i++) {
        sample.token_number = i;
        if (!harness->record_token_sample(sample)) {
            fprintf(stderr, "  FAILED: record_token_sample\n");
            delete harness;
            return false;
        }
    }

    harness->end_stability_test();

    if (harness->get_samples().size() == 0) {
        fprintf(stderr, "  FAILED: No samples recorded\n");
        delete harness;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete harness;
    return true;
}

static bool stability_harness_pcie_violation_detection_test() {
    fprintf(stdout, "\n[TEST] PCIe Violation Detection Test\n");

    auto * harness = new decode_stability_harness();
    harness->initialize(100, STRESS_MODE_STANDARD);
    harness->enable_harness(true);
    harness->begin_stability_test();

    stability_token_sample sample = {
        1, 2.5, 2.7, 0.2, 0.93, 512*1024, 0, 0, 10.0, 2048ULL*1024*1024, 4096ULL*1024*1024, 1024,
        40.0, false, true, false, 0  // pcie_violation_detected = true
    };

    harness->record_token_sample(sample);
    harness->end_stability_test();

    const auto & violations = harness->get_violations();
    if (violations.size() == 0) {
        fprintf(stderr, "  FAILED: PCIe violation not detected\n");
        delete harness;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete harness;
    return true;
}

static bool stability_harness_finalize_test() {
    fprintf(stdout, "\n[TEST] Finalization Test\n");

    auto * harness = new decode_stability_harness();
    harness->initialize(100, STRESS_MODE_STANDARD);
    harness->enable_harness(true);
    harness->begin_stability_test();

    stability_token_sample sample = {
        1, 2.5, 2.7, 0.2, 0.93, 0, 0, 1024*1024, 10.0, 2048ULL*1024*1024, 4096ULL*1024*1024, 1024,
        40.0, false, false, false, 0
    };

    for (int i = 0; i < 100; i++) {
        sample.token_number = i;
        harness->record_token_sample(sample);
    }

    harness->end_stability_test();

    if (!harness->finalize_test()) {
        fprintf(stderr, "  FAILED: finalize_test\n");
        delete harness;
        return false;
    }

    const auto & summary = harness->get_summary();
    if (summary.total_samples_collected == 0) {
        fprintf(stderr, "  FAILED: Summary not computed\n");
        delete harness;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete harness;
    return true;
}

static bool stability_harness_json_export_test() {
    fprintf(stdout, "\n[TEST] JSON Export Test\n");

    auto * harness = new decode_stability_harness();
    harness->initialize(100, STRESS_MODE_STANDARD);
    harness->enable_harness(true);
    harness->begin_stability_test();

    stability_token_sample sample = {
        1, 2.5, 2.7, 0.2, 0.93, 0, 0, 0, 10.0, 2048ULL*1024*1024, 4096ULL*1024*1024, 1024,
        40.0, false, false, false, 0
    };

    for (int i = 0; i < 50; i++) {
        sample.token_number = i;
        harness->record_token_sample(sample);
    }

    harness->end_stability_test();
    harness->finalize_test();

    std::string json = harness->generate_json_report();
    if (json.empty() || json.find("\"tokens_generated\"") == std::string::npos) {
        fprintf(stderr, "  FAILED: Invalid JSON\n");
        delete harness;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete harness;
    return true;
}

static bool stability_harness_disabled_noop_test() {
    fprintf(stdout, "\n[TEST] Disabled No-Op Test\n");

    auto * harness = new decode_stability_harness();
    harness->initialize(100, STRESS_MODE_STANDARD);
    // Don't enable harness

    stability_token_sample sample = {
        1, 2.5, 2.7, 0.2, 0.93, 0, 0, 0, 10.0, 2048ULL*1024*1024, 4096ULL*1024*1024, 1024,
        40.0, false, false, false, 0
    };

    harness->record_token_sample(sample);

    if (harness->get_samples().size() != 0) {
        fprintf(stderr, "  FAILED: Should not record when disabled\n");
        delete harness;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete harness;
    return true;
}

static bool stability_harness_full_workflow_test() {
    fprintf(stdout, "\n[TEST] Full Workflow Test\n");

    auto * harness = new decode_stability_harness();

    if (!harness->initialize(1000, STRESS_MODE_STANDARD)) {
        fprintf(stderr, "  FAILED: initialize\n");
        delete harness;
        return false;
    }

    harness->enable_harness(true);

    if (!harness->begin_stability_test()) {
        fprintf(stderr, "  FAILED: begin_stability_test\n");
        delete harness;
        return false;
    }

    stability_token_sample sample = {
        1, 2.5, 2.7, 0.2, 0.93, 0, 0, 1024*1024, 10.0, 2048ULL*1024*1024, 4096ULL*1024*1024, 1024,
        40.0, false, false, false, 0
    };

    for (int i = 0; i < 200; i++) {
        sample.token_number = i;
        harness->record_token_sample(sample);
    }

    harness->end_stability_test();

    if (!harness->finalize_test()) {
        fprintf(stderr, "  FAILED: finalize\n");
        delete harness;
        return false;
    }

    const auto & summary = harness->get_summary();
    if (!summary.stability_test_passed) {
        fprintf(stderr, "  FAILED: Test should pass\n");
        delete harness;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete harness;
    return true;
}

// Self-test runner
static bool run_stability_harness_self_tests() {
    fprintf(stdout, "\n========================================\n");
    fprintf(stdout, "Running Stability Harness Self-Tests\n");
    fprintf(stdout, "========================================\n");

    bool all_passed = true;
    all_passed &= stability_harness_initialization_test();
    all_passed &= stability_harness_test_lifecycle_test();
    all_passed &= stability_harness_sample_recording_test();
    all_passed &= stability_harness_pcie_violation_detection_test();
    all_passed &= stability_harness_finalize_test();
    all_passed &= stability_harness_json_export_test();
    all_passed &= stability_harness_disabled_noop_test();
    all_passed &= stability_harness_full_workflow_test();

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
static void stability_harness_self_tests_ctor() {
    // Uncomment to auto-run tests on module load:
    // run_stability_harness_self_tests();
}
