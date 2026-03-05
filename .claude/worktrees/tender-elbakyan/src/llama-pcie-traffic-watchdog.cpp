#include "llama-pcie-traffic-watchdog.h"
#include <cstring>
#include <cstdio>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>

// Global state
pcie_traffic_watchdog * g_pcie_traffic_watchdog = nullptr;

// ============================================================================
// pcie_traffic_watchdog Implementation
// ============================================================================

pcie_traffic_watchdog::pcie_traffic_watchdog()
    : current_state(WATCHDOG_STATE_UNINITIALIZED),
      watchdog_enabled(false),
      monitoring_active(false),
      decode_active_count(0),
      token_number(0),
      transfer_counter(0),
      current_h2d_bytes(0),
      current_d2h_bytes(0),
      current_d2d_bytes(0),
      strict_mode(false),
      report_all_transfers(false),
      violation_threshold(0) {
    std::memset(&current_token_stats, 0, sizeof(current_token_stats));
    std::memset(&summary, 0, sizeof(summary));
}

bool pcie_traffic_watchdog::initialize() {
    if (current_state != WATCHDOG_STATE_UNINITIALIZED) {
        fprintf(stderr, "[WATCHDOG] ERROR: Already initialized (state=%d)\n", current_state);
        return false;
    }

    current_state = WATCHDOG_STATE_READY;
    watchdog_enabled.store(false);  // Disabled by default
    monitoring_active.store(false);
    decode_active_count.store(0);
    token_number.store(0);
    transfer_counter.store(0);

    fprintf(stdout, "[WATCHDOG] PCIe traffic watchdog initialized\n");
    fprintf(stdout, "[WATCHDOG] Strict mode: %s\n", strict_mode ? "ON" : "OFF");
    fprintf(stdout, "[WATCHDOG] Report all transfers: %s\n",
            report_all_transfers ? "ON" : "OFF");

    return true;
}

bool pcie_traffic_watchdog::enable_watchdog(bool enable) {
    if (current_state == WATCHDOG_STATE_UNINITIALIZED) {
        fprintf(stderr, "[WATCHDOG] ERROR: Not initialized\n");
        return false;
    }

    watchdog_enabled.store(enable);
    fprintf(stdout, "[WATCHDOG] Watchdog %s\n", enable ? "ENABLED" : "DISABLED");

    return true;
}

bool pcie_traffic_watchdog::begin_decode_phase() {
    if (current_state != WATCHDOG_STATE_READY && current_state != WATCHDOG_STATE_MONITORING) {
        return false;
    }

    if (decode_active_count.load() == 0) {
        fprintf(stdout, "[WATCHDOG] Decode phase started\n");
        current_state = WATCHDOG_STATE_MONITORING;
    }

    decode_active_count.store(decode_active_count.load() + 1);
    monitoring_active.store(true);

    return true;
}

bool pcie_traffic_watchdog::end_decode_phase() {
    uint64_t active = decode_active_count.load();
    if (active == 0) {
        fprintf(stderr, "[WATCHDOG] ERROR: Decode phase not active\n");
        return false;
    }

    decode_active_count.store(active - 1);

    if (active == 1) {
        fprintf(stdout, "[WATCHDOG] Decode phase ended\n");
        monitoring_active.store(false);
        current_state = WATCHDOG_STATE_PAUSED;
    }

    return true;
}

bool pcie_traffic_watchdog::begin_token() {
    if (!watchdog_enabled.load() || !monitoring_active.load()) {
        return true;  // No-op when disabled
    }

    if (current_state != WATCHDOG_STATE_MONITORING) {
        return false;
    }

    // Reset per-token counters
    std::memset(&current_token_stats, 0, sizeof(current_token_stats));
    current_h2d_bytes.store(0);
    current_d2h_bytes.store(0);
    current_d2d_bytes.store(0);

    current_token_stats.token_number = token_number.load();
    current_token_stats.token_sequence_id = token_stats.size();
    current_token_stats.measurement_timestamp_ns =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    return true;
}

bool pcie_traffic_watchdog::record_transfer(pcie_transfer_type type,
                                           pcie_transfer_direction direction,
                                           uint64_t size_bytes,
                                           const char * source_location) {
    if (!watchdog_enabled.load() || !monitoring_active.load()) {
        return true;  // No-op when disabled
    }

    if (!source_location) {
        source_location = "unknown";
    }

    // Create transfer record
    pcie_transfer_record record = {
        transfer_counter.load(),
        token_number.load(),
        type,
        direction,
        size_bytes,
        source_location,
        static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
        monitoring_active.load(),
        false  // violation (set below)
    };

    // Track by direction
    switch (direction) {
        case TRANSFER_DIRECTION_H2D:
            current_token_stats.h2d_bytes += size_bytes;
            current_token_stats.h2d_count++;
            current_h2d_bytes.store(current_h2d_bytes.load() + size_bytes);
            record.violation = true;  // H2D forbidden during decode
            break;

        case TRANSFER_DIRECTION_D2H:
            current_token_stats.d2h_bytes += size_bytes;
            current_token_stats.d2h_count++;
            current_d2h_bytes.store(current_d2h_bytes.load() + size_bytes);
            record.violation = true;  // D2H forbidden during decode

            // Special case: logits copied to host
            if (strstr(source_location, "logits") || strstr(source_location, "sampling")) {
                current_token_stats.logits_copied_to_host = true;
            }
            break;

        case TRANSFER_DIRECTION_D2D:
            current_token_stats.d2d_bytes += size_bytes;
            current_token_stats.d2d_count++;
            current_d2d_bytes.store(current_d2d_bytes.load() + size_bytes);
            record.violation = false;  // D2D allowed
            break;

        default:
            break;
    }

    // Log transfer
    transfer_log.push_back(record);
    transfer_counter.store(transfer_counter.load() + 1);

    // Report if violation and report_all is enabled
    if (report_all_transfers || record.violation) {
        fprintf(stdout, "[WATCHDOG] Transfer: %s %s %s (token %llu)\n",
                format_transfer_type(type).c_str(),
                format_transfer_direction(direction).c_str(),
                format_size_bytes(size_bytes).c_str(),
                (unsigned long long)token_number.load());
    }

    return true;
}

bool pcie_traffic_watchdog::end_token() {
    if (!watchdog_enabled.load() || !monitoring_active.load()) {
        return true;  // No-op when disabled
    }

    // Mark violation if H2D or D2H transfers occurred
    if (current_token_stats.h2d_bytes > 0 || current_token_stats.d2h_bytes > 0) {
        current_token_stats.has_violation = true;

        // Record violation
        std::ostringstream oss;
        oss << "PCIe violation detected: ";
        if (current_token_stats.h2d_bytes > 0) {
            oss << "H2D " << format_size_bytes(current_token_stats.h2d_bytes);
        }
        if (current_token_stats.d2h_bytes > 0) {
            if (current_token_stats.h2d_bytes > 0) oss << " + ";
            oss << "D2H " << format_size_bytes(current_token_stats.d2h_bytes);
        }

        bool is_critical = (current_token_stats.h2d_bytes > violation_threshold) ||
                          (current_token_stats.d2h_bytes > violation_threshold);

        record_violation(oss.str().c_str(),
                        current_token_stats.h2d_bytes > 0 ?
                            TRANSFER_DIRECTION_H2D : TRANSFER_DIRECTION_D2H,
                        current_token_stats.h2d_bytes + current_token_stats.d2h_bytes,
                        "per-token-accumulation",
                        is_critical);

        fprintf(stderr, "[WATCHDOG] ❌ Token %llu: H2D=%s D2H=%s\n",
                (unsigned long long)current_token_stats.token_number,
                format_size_bytes(current_token_stats.h2d_bytes).c_str(),
                format_size_bytes(current_token_stats.d2h_bytes).c_str());

        if (strict_mode) {
            fprintf(stderr, "[WATCHDOG] STRICT MODE: Aborting on violation\n");
            return false;
        }
    }

    // Store token stats
    token_stats.push_back(current_token_stats);
    token_number.store(token_number.load() + 1);

    return true;
}

bool pcie_traffic_watchdog::record_violation(const char * description,
                                            pcie_transfer_direction direction,
                                            uint64_t size,
                                            const char * location,
                                            bool is_critical) {
    if (!description || !location) {
        return false;
    }

    pcie_watchdog_violation violation = {
        description,
        token_number.load(),
        direction,
        size,
        location,
        static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
        is_critical
    };

    violations.push_back(violation);
    violation_by_location[location]++;

    fprintf(stderr, "[WATCHDOG] %s at %s (token %llu)\n",
            description, location, (unsigned long long)token_number.load());

    return true;
}

bool pcie_traffic_watchdog::finalize_monitoring() {
    if (monitoring_active.load()) {
        fprintf(stderr, "[WATCHDOG] WARNING: Monitoring still active\n");
        end_decode_phase();
    }

    current_state = WATCHDOG_STATE_COMPLETE;

    // Compute summary
    summary.total_tokens_observed = token_stats.size();
    summary.total_h2d_bytes = 0;
    summary.total_d2h_bytes = 0;
    summary.total_d2d_bytes = 0;

    summary.total_h2d_transfers = 0;
    summary.total_d2h_transfers = 0;
    summary.total_d2d_transfers = 0;

    summary.logits_host_copies = 0;
    summary.sampling_transfers = 0;
    summary.unified_memory_migrations = 0;

    summary.tokens_with_violations = 0;
    summary.any_violation_detected = false;

    // Aggregate token statistics
    for (const auto & ts : token_stats) {
        summary.total_h2d_bytes += ts.h2d_bytes;
        summary.total_d2h_bytes += ts.d2h_bytes;
        summary.total_d2d_bytes += ts.d2d_bytes;

        summary.total_h2d_transfers += ts.h2d_count;
        summary.total_d2h_transfers += ts.d2h_count;
        summary.total_d2d_transfers += ts.d2d_count;

        if (ts.logits_copied_to_host) {
            summary.logits_host_copies++;
        }

        if (ts.has_violation) {
            summary.tokens_with_violations++;
            summary.any_violation_detected = true;
        }
    }

    summary.is_decode_pcie_clean = (summary.total_h2d_bytes == 0 &&
                                   summary.total_d2h_bytes == 0);

    summary.measurement_timestamp_ns =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();

    fprintf(stdout, "[WATCHDOG] Monitoring finalized: %llu tokens, %s\n",
            (unsigned long long)summary.total_tokens_observed,
            summary.is_decode_pcie_clean ? "CLEAN ✅" : "VIOLATIONS DETECTED ❌");

    return true;
}

bool pcie_traffic_watchdog::generate_watchdog_report() {
    if (current_state != WATCHDOG_STATE_COMPLETE) {
        fprintf(stderr, "[WATCHDOG] ERROR: Not complete\n");
        return false;
    }

    std::ostringstream oss;

    // Header
    oss << "\n";
    oss << "==== PCIe WATCHDOG REPORT ====\n";
    oss << "\n";

    // Summary statistics
    oss << "OBSERVATION STATISTICS:\n";
    oss << "  Tokens observed:              " << summary.total_tokens_observed << "\n";
    oss << "  Tokens with violations:       " << summary.tokens_with_violations << "\n";
    oss << "\n";

    oss << "PCIE TRANSFER STATISTICS:\n";
    oss << "  H2D (forbidden):              " << summary.total_h2d_transfers << " transfers, "
        << format_size_bytes(summary.total_h2d_bytes) << "\n";
    oss << "  D2H (forbidden):              " << summary.total_d2h_transfers << " transfers, "
        << format_size_bytes(summary.total_d2h_bytes) << "\n";
    oss << "  D2D (allowed):                " << summary.total_d2d_transfers << " transfers, "
        << format_size_bytes(summary.total_d2d_bytes) << "\n";
    oss << "\n";

    oss << "SPECIAL CASES:\n";
    oss << "  Logits copied to host:        " << summary.logits_host_copies << "\n";
    oss << "  Sampling transfers:           " << summary.sampling_transfers << "\n";
    oss << "  Unified memory migrations:    " << summary.unified_memory_migrations << "\n";
    oss << "\n";

    // Status
    if (summary.is_decode_pcie_clean) {
        oss << "STATUS: ✅ CLEAN\n";
        oss << "No host↔device transfers during decode\n";
    } else {
        oss << "STATUS: ❌ VIOLATION\n";
        oss << "Host↔device transfers detected during decode\n";
    }

    oss << "\n";

    // Violation details (if any)
    if (violations.size() > 0) {
        oss << "VIOLATIONS DETECTED (" << violations.size() << " total):\n";
        uint32_t shown = 0;
        for (const auto & v : violations) {
            if (shown >= 10) {
                oss << "  ... and " << (violations.size() - 10) << " more\n";
                break;
            }

            oss << "  [Token " << v.token_number << "] "
                << v.violation_description << " at " << v.transfer_location << "\n";

            shown++;
        }
        oss << "\n";
    }

    // Transfer log (first 20)
    if (transfer_log.size() > 0) {
        oss << "TRANSFER LOG (first 20):\n";
        uint32_t shown = 0;
        for (const auto & t : transfer_log) {
            if (shown >= 20) {
                oss << "  ... and " << (transfer_log.size() - 20) << " more\n";
                break;
            }

            if (report_all_transfers || t.violation) {
                oss << "  [" << t.transfer_id << "] Token " << t.token_number << ": "
                    << format_transfer_type(t.transfer_type) << " "
                    << format_transfer_direction(t.direction) << " "
                    << format_size_bytes(t.size_bytes)
                    << (t.violation ? " ❌" : "") << "\n";
                shown++;
            }
        }
        oss << "\n";
    }

    // Footer
    oss << "==============================\n";
    oss << "\n";

    printf("%s", oss.str().c_str());

    return true;
}

bool pcie_traffic_watchdog::validate_pcie_cleanliness() {
    if (current_state != WATCHDOG_STATE_COMPLETE) {
        fprintf(stderr, "[WATCHDOG] ERROR: Not complete\n");
        return false;
    }

    bool all_valid = true;

    all_valid &= verify_no_h2d_transfers();
    all_valid &= verify_no_d2h_transfers();
    all_valid &= verify_decode_pcie_clean();

    if (all_valid) {
        fprintf(stdout, "[WATCHDOG] All validations passed ✅\n");
    } else {
        fprintf(stderr, "[WATCHDOG] Validation failed ❌\n");
    }

    return all_valid;
}

bool pcie_traffic_watchdog::verify_no_h2d_transfers() const {
    if (summary.total_h2d_bytes == 0) {
        fprintf(stdout, "[WATCHDOG] No H2D transfers detected\n");
        return true;
    } else {
        fprintf(stderr, "[WATCHDOG] H2D VIOLATION: %s transferred\n",
                format_size_bytes(summary.total_h2d_bytes).c_str());
        return false;
    }
}

bool pcie_traffic_watchdog::verify_no_d2h_transfers() const {
    if (summary.total_d2h_bytes == 0) {
        fprintf(stdout, "[WATCHDOG] No D2H transfers detected\n");
        return true;
    } else {
        fprintf(stderr, "[WATCHDOG] D2H VIOLATION: %s transferred\n",
                format_size_bytes(summary.total_d2h_bytes).c_str());
        return false;
    }
}

bool pcie_traffic_watchdog::verify_decode_pcie_clean() const {
    if (summary.is_decode_pcie_clean) {
        fprintf(stdout, "[WATCHDOG] PCIe is clean during decode\n");
        return true;
    } else {
        fprintf(stderr, "[WATCHDOG] PCIe violations detected\n");
        return false;
    }
}

std::string pcie_traffic_watchdog::format_transfer_direction(pcie_transfer_direction dir) const {
    switch (dir) {
        case TRANSFER_DIRECTION_H2D:
            return "H2D";
        case TRANSFER_DIRECTION_D2H:
            return "D2H";
        case TRANSFER_DIRECTION_D2D:
            return "D2D";
        default:
            return "???";
    }
}

std::string pcie_traffic_watchdog::format_transfer_type(pcie_transfer_type type) const {
    switch (type) {
        case TRANSFER_TYPE_MEMCPY:
            return "memcpy";
        case TRANSFER_TYPE_MEMCPY_ASYNC:
            return "memcpy_async";
        case TRANSFER_TYPE_MEMCPY_2D:
            return "memcpy2d";
        case TRANSFER_TYPE_MEMCPY_FROM_SYMBOL:
            return "memcpy_sym";
        case TRANSFER_TYPE_MEMADVISE:
            return "memadvise";
        case TRANSFER_TYPE_PAGE_MIGRATION:
            return "page_migration";
        default:
            return "unknown";
    }
}

std::string pcie_traffic_watchdog::format_size_bytes(uint64_t bytes) const {
    std::ostringstream oss;
    if (bytes < 1024) {
        oss << bytes << "B";
    } else if (bytes < 1024 * 1024) {
        oss << std::fixed << std::setprecision(2) << (bytes / 1024.0) << "KB";
    } else if (bytes < 1024 * 1024 * 1024) {
        oss << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0)) << "MB";
    } else {
        oss << std::fixed << std::setprecision(2)
            << (bytes / (1024.0 * 1024.0 * 1024.0)) << "GB";
    }
    return oss.str();
}

std::string pcie_traffic_watchdog::generate_report() const {
    if (current_state != WATCHDOG_STATE_COMPLETE) {
        return "ERROR: Watchdog not complete\n";
    }

    std::ostringstream oss;
    oss << "PCIe Watchdog Report\n";
    oss << "Tokens: " << summary.total_tokens_observed << "\n";
    oss << "H2D: " << format_size_bytes(summary.total_h2d_bytes) << "\n";
    oss << "D2H: " << format_size_bytes(summary.total_d2h_bytes) << "\n";
    oss << "D2D: " << format_size_bytes(summary.total_d2d_bytes) << "\n";
    oss << "Status: " << (summary.is_decode_pcie_clean ? "CLEAN" : "VIOLATIONS") << "\n";

    return oss.str();
}

std::string pcie_traffic_watchdog::generate_json_report() const {
    std::ostringstream oss;

    oss << "{\n";
    oss << "  \"total_tokens\": " << summary.total_tokens_observed << ",\n";
    oss << "  \"h2d_transfers\": " << summary.total_h2d_transfers << ",\n";
    oss << "  \"d2h_transfers\": " << summary.total_d2h_transfers << ",\n";
    oss << "  \"d2d_transfers\": " << summary.total_d2d_transfers << ",\n";
    oss << "  \"h2d_bytes\": " << summary.total_h2d_bytes << ",\n";
    oss << "  \"d2h_bytes\": " << summary.total_d2h_bytes << ",\n";
    oss << "  \"d2d_bytes\": " << summary.total_d2d_bytes << ",\n";
    oss << "  \"violations\": " << summary.tokens_with_violations << ",\n";
    oss << "  \"is_clean\": " << (summary.is_decode_pcie_clean ? "true" : "false") << ",\n";
    oss << "  \"logits_copies\": " << summary.logits_host_copies << "\n";
    oss << "}\n";

    return oss.str();
}

// ============================================================================
// pcie_watchdog_guard Implementation
// ============================================================================

pcie_watchdog_guard::pcie_watchdog_guard(pcie_traffic_watchdog * watchdog_ptr)
    : guard_active(false), watchdog(watchdog_ptr) {
    if (watchdog) {
        guard_active = true;
    }
}

pcie_watchdog_guard::~pcie_watchdog_guard() {
    guard_active = false;
}

// ============================================================================
// C-Style Wrapper Functions
// ============================================================================

bool llama_init_pcie_watchdog() {
    if (g_pcie_traffic_watchdog != nullptr) {
        fprintf(stderr, "[WATCHDOG] Already initialized\n");
        return false;
    }

    g_pcie_traffic_watchdog = new pcie_traffic_watchdog();
    if (!g_pcie_traffic_watchdog->initialize()) {
        fprintf(stderr, "[WATCHDOG] Failed to initialize\n");
        delete g_pcie_traffic_watchdog;
        g_pcie_traffic_watchdog = nullptr;
        return false;
    }

    return true;
}

bool llama_enable_pcie_watchdog(bool enable) {
    if (!g_pcie_traffic_watchdog) {
        return false;
    }
    return g_pcie_traffic_watchdog->enable_watchdog(enable);
}

bool llama_is_pcie_watchdog_enabled() {
    if (!g_pcie_traffic_watchdog) {
        return false;
    }
    return g_pcie_traffic_watchdog->is_watchdog_enabled();
}

bool llama_begin_decode_pcie_monitoring() {
    if (!g_pcie_traffic_watchdog) {
        return false;
    }
    return g_pcie_traffic_watchdog->begin_decode_phase();
}

bool llama_end_decode_pcie_monitoring() {
    if (!g_pcie_traffic_watchdog) {
        return false;
    }
    return g_pcie_traffic_watchdog->end_decode_phase();
}

bool llama_begin_pcie_token() {
    if (!g_pcie_traffic_watchdog) {
        return true;
    }
    return g_pcie_traffic_watchdog->begin_token();
}

bool llama_record_pcie_transfer(int transfer_type,
                                int direction,
                                uint64_t size_bytes,
                                const char * source_location) {
    if (!g_pcie_traffic_watchdog) {
        return true;
    }
    return g_pcie_traffic_watchdog->record_transfer(
        (pcie_transfer_type)transfer_type,
        (pcie_transfer_direction)direction,
        size_bytes,
        source_location);
}

bool llama_end_pcie_token() {
    if (!g_pcie_traffic_watchdog) {
        return true;
    }
    return g_pcie_traffic_watchdog->end_token();
}

bool llama_finalize_pcie_monitoring() {
    if (!g_pcie_traffic_watchdog) {
        return false;
    }
    if (!g_pcie_traffic_watchdog->finalize_monitoring()) {
        return false;
    }
    return g_pcie_traffic_watchdog->generate_watchdog_report();
}

bool llama_generate_pcie_watchdog_report() {
    if (!g_pcie_traffic_watchdog) {
        return false;
    }
    return g_pcie_traffic_watchdog->generate_watchdog_report();
}

bool llama_validate_pcie_cleanliness() {
    if (!g_pcie_traffic_watchdog) {
        return false;
    }
    return g_pcie_traffic_watchdog->validate_pcie_cleanliness();
}

const pcie_watchdog_summary * llama_get_pcie_watchdog_summary() {
    if (!g_pcie_traffic_watchdog) {
        return nullptr;
    }
    return &g_pcie_traffic_watchdog->get_summary();
}

const char * llama_get_pcie_watchdog_report() {
    if (!g_pcie_traffic_watchdog) {
        return "";
    }
    return g_pcie_traffic_watchdog->generate_report().c_str();
}

void llama_print_pcie_watchdog_report() {
    if (!g_pcie_traffic_watchdog) {
        return;
    }
    g_pcie_traffic_watchdog->generate_watchdog_report();
}

void llama_print_pcie_watchdog_summary() {
    if (!g_pcie_traffic_watchdog) {
        return;
    }

    const auto & summary = g_pcie_traffic_watchdog->get_summary();
    printf("\n=== PCIe WATCHDOG SUMMARY ===\n");
    printf("Tokens observed: %llu\n", (unsigned long long)summary.total_tokens_observed);
    printf("Tokens with violations: %llu\n",
           (unsigned long long)summary.tokens_with_violations);
    printf("H2D bytes: %s\n", g_pcie_traffic_watchdog->format_size_bytes(
           summary.total_h2d_bytes).c_str());
    printf("D2H bytes: %s\n", g_pcie_traffic_watchdog->format_size_bytes(
           summary.total_d2h_bytes).c_str());
    printf("D2D bytes: %s\n", g_pcie_traffic_watchdog->format_size_bytes(
           summary.total_d2d_bytes).c_str());
    printf("Status: %s\n", summary.is_decode_pcie_clean ? "CLEAN ✅" : "VIOLATIONS ❌");
    printf("=============================\n\n");
}

void llama_print_pcie_watchdog_violations() {
    if (!g_pcie_traffic_watchdog) {
        return;
    }

    const auto & violations = g_pcie_traffic_watchdog->get_violations();
    if (violations.empty()) {
        printf("No violations detected\n");
        return;
    }

    printf("\n=== PCIe WATCHDOG VIOLATIONS ===\n");
    for (const auto & v : violations) {
        printf("Token %llu: %s at %s\n",
               (unsigned long long)v.token_number,
               v.violation_description,
               v.transfer_location);
    }
    printf("================================\n\n");
}

void llama_print_pcie_transfer_log(uint32_t limit) {
    if (!g_pcie_traffic_watchdog) {
        return;
    }

    const auto & transfers = g_pcie_traffic_watchdog->get_transfer_log();
    printf("\n=== PCIe TRANSFER LOG ===\n");
    printf("Total transfers: %zu\n", transfers.size());

    uint32_t shown = 0;
    for (const auto & t : transfers) {
        if (limit > 0 && shown >= limit) break;

        printf("Transfer %llu: Token %llu %s %s %s%s\n",
               (unsigned long long)t.transfer_id,
               (unsigned long long)t.token_number,
               g_pcie_traffic_watchdog->format_transfer_type(t.transfer_type).c_str(),
               g_pcie_traffic_watchdog->format_transfer_direction(t.direction).c_str(),
               g_pcie_traffic_watchdog->format_size_bytes(t.size_bytes).c_str(),
               t.violation ? " ❌" : "");

        shown++;
    }
    printf("========================\n\n");
}

void llama_export_pcie_watchdog_json(const char * filename) {
    if (!g_pcie_traffic_watchdog || !filename) {
        fprintf(stderr, "[WATCHDOG] Invalid watchdog or filename\n");
        return;
    }

    std::string json = g_pcie_traffic_watchdog->generate_json_report();
    FILE * f = fopen(filename, "w");
    if (f) {
        fprintf(f, "%s", json.c_str());
        fclose(f);
        printf("[WATCHDOG] JSON report exported to %s\n", filename);
    } else {
        fprintf(stderr, "[WATCHDOG] Failed to open %s for writing\n", filename);
    }
}

// ============================================================================
// Self-Test Suite (12 comprehensive tests)
// ============================================================================

static bool pcie_watchdog_initialization_test() {
    fprintf(stdout, "\n[TEST] PCIe Watchdog Initialization Test\n");

    auto * watchdog = new pcie_traffic_watchdog();
    if (!watchdog->initialize()) {
        fprintf(stderr, "  FAILED: Initialization\n");
        delete watchdog;
        return false;
    }

    if (watchdog->is_watchdog_enabled()) {
        fprintf(stderr, "  FAILED: Should be disabled by default\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

static bool pcie_watchdog_enable_test() {
    fprintf(stdout, "\n[TEST] PCIe Watchdog Enable Test\n");

    auto * watchdog = new pcie_traffic_watchdog();
    watchdog->initialize();

    if (!watchdog->enable_watchdog(true)) {
        fprintf(stderr, "  FAILED: Enable\n");
        delete watchdog;
        return false;
    }

    if (!watchdog->is_watchdog_enabled()) {
        fprintf(stderr, "  FAILED: Not enabled\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

static bool pcie_watchdog_phase_test() {
    fprintf(stdout, "\n[TEST] PCIe Watchdog Phase Test\n");

    auto * watchdog = new pcie_traffic_watchdog();
    watchdog->initialize();
    watchdog->enable_watchdog(true);

    if (!watchdog->begin_decode_phase()) {
        fprintf(stderr, "  FAILED: begin_decode_phase\n");
        delete watchdog;
        return false;
    }

    if (!watchdog->end_decode_phase()) {
        fprintf(stderr, "  FAILED: end_decode_phase\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

static bool pcie_watchdog_d2d_transfer_test() {
    fprintf(stdout, "\n[TEST] D2D Transfer Test (Allowed)\n");

    auto * watchdog = new pcie_traffic_watchdog();
    watchdog->initialize();
    watchdog->enable_watchdog(true);
    watchdog->begin_decode_phase();

    watchdog->begin_token();
    watchdog->record_transfer(TRANSFER_TYPE_MEMCPY, TRANSFER_DIRECTION_D2D,
                             1024 * 1024, "test_d2d");
    watchdog->end_token();

    watchdog->end_decode_phase();

    const auto & stats = watchdog->get_token_stats();
    if (stats.size() != 1 || stats[0].d2d_bytes != 1024 * 1024) {
        fprintf(stderr, "  FAILED: D2D bytes not recorded\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

static bool pcie_watchdog_h2d_violation_test() {
    fprintf(stdout, "\n[TEST] H2D Violation Detection Test\n");

    auto * watchdog = new pcie_traffic_watchdog();
    watchdog->initialize();
    watchdog->enable_watchdog(true);
    watchdog->begin_decode_phase();

    watchdog->begin_token();
    watchdog->record_transfer(TRANSFER_TYPE_MEMCPY, TRANSFER_DIRECTION_H2D,
                             512 * 1024, "test_h2d");
    watchdog->end_token();

    watchdog->end_decode_phase();

    const auto & stats = watchdog->get_token_stats();
    if (stats.size() != 1 || !stats[0].has_violation) {
        fprintf(stderr, "  FAILED: Violation not detected\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

static bool pcie_watchdog_d2h_violation_test() {
    fprintf(stdout, "\n[TEST] D2H Violation Detection Test\n");

    auto * watchdog = new pcie_traffic_watchdog();
    watchdog->initialize();
    watchdog->enable_watchdog(true);
    watchdog->begin_decode_phase();

    watchdog->begin_token();
    watchdog->record_transfer(TRANSFER_TYPE_MEMCPY, TRANSFER_DIRECTION_D2H,
                             256 * 1024, "test_d2h");
    watchdog->end_token();

    watchdog->end_decode_phase();

    const auto & stats = watchdog->get_token_stats();
    if (stats.size() != 1 || !stats[0].has_violation) {
        fprintf(stderr, "  FAILED: Violation not detected\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

static bool pcie_watchdog_multiple_tokens_test() {
    fprintf(stdout, "\n[TEST] Multiple Tokens Test\n");

    auto * watchdog = new pcie_traffic_watchdog();
    watchdog->initialize();
    watchdog->enable_watchdog(true);
    watchdog->begin_decode_phase();

    for (int i = 0; i < 10; i++) {
        watchdog->begin_token();
        if (i % 2 == 0) {
            watchdog->record_transfer(TRANSFER_TYPE_MEMCPY, TRANSFER_DIRECTION_D2D,
                                     1024 * 1024, "d2d_transfer");
        } else {
            watchdog->record_transfer(TRANSFER_TYPE_MEMCPY, TRANSFER_DIRECTION_H2D,
                                     512 * 1024, "h2d_violation");
        }
        watchdog->end_token();
    }

    watchdog->end_decode_phase();

    const auto & stats = watchdog->get_token_stats();
    if (stats.size() != 10) {
        fprintf(stderr, "  FAILED: Not all tokens recorded\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

static bool pcie_watchdog_finalize_test() {
    fprintf(stdout, "\n[TEST] Finalization Test\n");

    auto * watchdog = new pcie_traffic_watchdog();
    watchdog->initialize();
    watchdog->enable_watchdog(true);
    watchdog->begin_decode_phase();

    for (int i = 0; i < 50; i++) {
        watchdog->begin_token();
        watchdog->record_transfer(TRANSFER_TYPE_MEMCPY, TRANSFER_DIRECTION_D2D,
                                 1024 * 1024, "test");
        watchdog->end_token();
    }

    watchdog->end_decode_phase();

    if (!watchdog->finalize_monitoring()) {
        fprintf(stderr, "  FAILED: finalize_monitoring\n");
        delete watchdog;
        return false;
    }

    const auto & summary = watchdog->get_summary();
    if (summary.total_tokens_observed != 50) {
        fprintf(stderr, "  FAILED: Summary not computed\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

static bool pcie_watchdog_clean_report_test() {
    fprintf(stdout, "\n[TEST] Clean Report Test\n");

    auto * watchdog = new pcie_traffic_watchdog();
    watchdog->initialize();
    watchdog->enable_watchdog(true);
    watchdog->begin_decode_phase();

    for (int i = 0; i < 50; i++) {
        watchdog->begin_token();
        watchdog->record_transfer(TRANSFER_TYPE_MEMCPY, TRANSFER_DIRECTION_D2D,
                                 1024 * 1024, "test");
        watchdog->end_token();
    }

    watchdog->end_decode_phase();
    watchdog->finalize_monitoring();

    const auto & summary = watchdog->get_summary();
    if (!summary.is_decode_pcie_clean) {
        fprintf(stderr, "  FAILED: Should be clean\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

static bool pcie_watchdog_violation_report_test() {
    fprintf(stdout, "\n[TEST] Violation Report Test\n");

    auto * watchdog = new pcie_traffic_watchdog();
    watchdog->initialize();
    watchdog->enable_watchdog(true);
    watchdog->begin_decode_phase();

    watchdog->begin_token();
    watchdog->record_transfer(TRANSFER_TYPE_MEMCPY, TRANSFER_DIRECTION_H2D,
                             512 * 1024, "violation");
    watchdog->end_token();

    watchdog->end_decode_phase();
    watchdog->finalize_monitoring();

    const auto & summary = watchdog->get_summary();
    if (summary.is_decode_pcie_clean || summary.total_h2d_bytes == 0) {
        fprintf(stderr, "  FAILED: Should have violations\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

static bool pcie_watchdog_json_export_test() {
    fprintf(stdout, "\n[TEST] JSON Export Test\n");

    auto * watchdog = new pcie_traffic_watchdog();
    watchdog->initialize();
    watchdog->enable_watchdog(true);
    watchdog->begin_decode_phase();

    for (int i = 0; i < 10; i++) {
        watchdog->begin_token();
        watchdog->record_transfer(TRANSFER_TYPE_MEMCPY, TRANSFER_DIRECTION_D2D,
                                 1024 * 1024, "test");
        watchdog->end_token();
    }

    watchdog->end_decode_phase();
    watchdog->finalize_monitoring();

    std::string json = watchdog->generate_json_report();
    if (json.empty() || json.find("\"d2d_bytes\"") == std::string::npos) {
        fprintf(stderr, "  FAILED: Invalid JSON\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

static bool pcie_watchdog_disabled_noop_test() {
    fprintf(stdout, "\n[TEST] Disabled No-Op Test\n");

    auto * watchdog = new pcie_traffic_watchdog();
    watchdog->initialize();
    // Don't enable

    if (!watchdog->begin_decode_phase()) {
        fprintf(stderr, "  FAILED: Should succeed even disabled\n");
        delete watchdog;
        return false;
    }

    watchdog->begin_token();
    watchdog->record_transfer(TRANSFER_TYPE_MEMCPY, TRANSFER_DIRECTION_H2D,
                             512 * 1024, "test");
    watchdog->end_token();
    watchdog->end_decode_phase();

    if (watchdog->get_transfer_log().size() != 0) {
        fprintf(stderr, "  FAILED: Should not record when disabled\n");
        delete watchdog;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete watchdog;
    return true;
}

// Self-test runner
static bool run_pcie_watchdog_self_tests() {
    fprintf(stdout, "\n========================================\n");
    fprintf(stdout, "Running PCIe Traffic Watchdog Self-Tests\n");
    fprintf(stdout, "========================================\n");

    bool all_passed = true;
    all_passed &= pcie_watchdog_initialization_test();
    all_passed &= pcie_watchdog_enable_test();
    all_passed &= pcie_watchdog_phase_test();
    all_passed &= pcie_watchdog_d2d_transfer_test();
    all_passed &= pcie_watchdog_h2d_violation_test();
    all_passed &= pcie_watchdog_d2h_violation_test();
    all_passed &= pcie_watchdog_multiple_tokens_test();
    all_passed &= pcie_watchdog_finalize_test();
    all_passed &= pcie_watchdog_clean_report_test();
    all_passed &= pcie_watchdog_violation_report_test();
    all_passed &= pcie_watchdog_json_export_test();
    all_passed &= pcie_watchdog_disabled_noop_test();

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
static void pcie_watchdog_self_tests_ctor() {
    // Uncomment to auto-run tests on module load:
    // run_pcie_watchdog_self_tests();
}
