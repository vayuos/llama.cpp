#pragma once

/**
 * PCIe Traffic Watchdog for LLAMA Decode
 *
 * Decode-phase PCIe transfer watchdog that detects and reports any host↔device
 * memory traffic occurring during token generation.
 *
 * This enforces the invariant:
 * No per-token host↔device transfers are allowed in the decode-critical path.
 *
 * The watchdog is passive instrumentation, not a synchronization mechanism.
 *
 * Monitored Operations:
 * - cudaMemcpy / cudaMemcpyAsync
 * - cudaMemcpy2D
 * - cudaMemcpyFromSymbol
 * - D2H (device → host) transfers
 * - H2D (host → device) transfers
 * - Unified memory page migration (if enabled)
 *
 * Decode transfers are forbidden. Prefill transfers are allowed.
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
    TRANSFER_DIRECTION_H2D = 0,  // Host → Device (forbidden in decode)
    TRANSFER_DIRECTION_D2H = 1,  // Device → Host (forbidden in decode)
    TRANSFER_DIRECTION_D2D = 2,  // Device → Device (allowed)
    TRANSFER_DIRECTION_UNKNOWN = 3
} pcie_transfer_direction;

typedef enum {
    TRANSFER_TYPE_MEMCPY = 0,
    TRANSFER_TYPE_MEMCPY_ASYNC = 1,
    TRANSFER_TYPE_MEMCPY_2D = 2,
    TRANSFER_TYPE_MEMCPY_FROM_SYMBOL = 3,
    TRANSFER_TYPE_MEMADVISE = 4,
    TRANSFER_TYPE_PAGE_MIGRATION = 5,
    TRANSFER_TYPE_UNKNOWN = 6
} pcie_transfer_type;

typedef enum {
    WATCHDOG_STATE_UNINITIALIZED = 0,
    WATCHDOG_STATE_READY = 1,
    WATCHDOG_STATE_MONITORING = 2,
    WATCHDOG_STATE_PAUSED = 3,
    WATCHDOG_STATE_COMPLETE = 4,
    WATCHDOG_STATE_LOCKED = 5
} pcie_watchdog_state;

typedef struct {
    uint64_t transfer_id;
    uint64_t token_number;
    pcie_transfer_type transfer_type;
    pcie_transfer_direction direction;
    uint64_t size_bytes;
    const char * source_location;  // Where in code (function name)
    uint64_t transfer_timestamp_ns;
    bool during_decode;
    bool violation;
} pcie_transfer_record;

typedef struct {
    uint64_t token_number;
    uint64_t token_sequence_id;

    uint64_t h2d_bytes;        // Host → Device (FORBIDDEN)
    uint64_t d2h_bytes;        // Device → Host (FORBIDDEN)
    uint64_t d2d_bytes;        // Device → Device (ALLOWED)

    uint32_t h2d_count;
    uint32_t d2h_count;
    uint32_t d2d_count;

    bool has_violation;        // true if H2D or D2H > 0
    bool logits_copied_to_host;
    bool sampling_transferred;

    uint64_t measurement_timestamp_ns;
} pcie_token_transfer_stats;

typedef struct {
    uint64_t total_tokens_observed;
    uint64_t tokens_with_violations;

    uint64_t total_h2d_bytes;
    uint64_t total_d2h_bytes;
    uint64_t total_d2d_bytes;

    uint64_t total_h2d_transfers;
    uint64_t total_d2h_transfers;
    uint64_t total_d2d_transfers;

    uint32_t logits_host_copies;
    uint32_t sampling_transfers;
    uint32_t unified_memory_migrations;

    bool any_violation_detected;
    bool is_decode_pcie_clean;

    uint64_t measurement_timestamp_ns;
} pcie_watchdog_summary;

typedef struct {
    const char * violation_description;
    uint64_t token_number;
    pcie_transfer_direction direction;
    uint64_t transfer_size;
    const char * transfer_location;
    uint64_t violation_timestamp_ns;
    bool is_critical;
} pcie_watchdog_violation;

class pcie_traffic_watchdog {
private:
    pcie_watchdog_state current_state;
    std::vector<pcie_transfer_record> transfer_log;
    std::vector<pcie_token_transfer_stats> token_stats;
    pcie_watchdog_summary summary;
    std::vector<pcie_watchdog_violation> violations;

    std::atomic<bool> watchdog_enabled;
    std::atomic<bool> monitoring_active;
    std::atomic<uint64_t> decode_active_count;
    std::atomic<uint64_t> token_number;
    std::atomic<uint64_t> transfer_counter;

    // Current token tracking
    pcie_token_transfer_stats current_token_stats;
    std::atomic<uint64_t> current_h2d_bytes;
    std::atomic<uint64_t> current_d2h_bytes;
    std::atomic<uint64_t> current_d2d_bytes;

    // Configuration
    bool strict_mode;           // Abort on violation?
    bool report_all_transfers;  // Log all or only violations?
    uint32_t violation_threshold;  // bytes that trigger warning

    // Statistics
    std::map<std::string, uint64_t> violation_by_location;

public:
    pcie_traffic_watchdog();

    bool initialize();
    bool enable_watchdog(bool enable);
    bool is_watchdog_enabled() const { return watchdog_enabled.load(); }

    bool begin_decode_phase();
    bool end_decode_phase();

    bool begin_token();
    bool record_transfer(pcie_transfer_type type,
                        pcie_transfer_direction direction,
                        uint64_t size_bytes,
                        const char * source_location);
    bool end_token();

    bool finalize_monitoring();
    bool generate_watchdog_report();
    bool validate_pcie_cleanliness();

    // Query functions
    pcie_watchdog_state get_current_state() const { return current_state; }
    bool is_monitoring_active() const { return monitoring_active.load(); }

    const pcie_watchdog_summary & get_summary() const { return summary; }
    std::vector<pcie_transfer_record> get_transfer_log() const { return transfer_log; }
    std::vector<pcie_token_transfer_stats> get_token_stats() const { return token_stats; }
    std::vector<pcie_watchdog_violation> get_violations() const { return violations; }

    // Configuration
    void set_strict_mode(bool strict) { strict_mode = strict; }
    void set_report_all_transfers(bool report_all) { report_all_transfers = report_all; }
    void set_violation_threshold(uint32_t threshold) { violation_threshold = threshold; }

    bool get_strict_mode() const { return strict_mode; }
    bool get_report_all_transfers() const { return report_all_transfers; }
    uint32_t get_violation_threshold() const { return violation_threshold; }

    // Reporting
    std::string format_transfer_direction(pcie_transfer_direction dir) const;
    std::string format_transfer_type(pcie_transfer_type type) const;
    std::string format_size_bytes(uint64_t bytes) const;
    std::string generate_report() const;
    std::string generate_json_report() const;

    // Validation
    bool verify_no_h2d_transfers() const;
    bool verify_no_d2h_transfers() const;
    bool verify_decode_pcie_clean() const;

    // Statistics
    size_t get_transfer_log_size() const { return transfer_log.size(); }
    size_t get_violation_count() const { return violations.size(); }
    uint64_t get_h2d_bytes() const { return summary.total_h2d_bytes; }
    uint64_t get_d2h_bytes() const { return summary.total_d2h_bytes; }
    uint64_t get_d2d_bytes() const { return summary.total_d2d_bytes; }

private:
    bool record_violation(const char * description,
                         pcie_transfer_direction direction,
                         uint64_t size,
                         const char * location,
                         bool is_critical);
};

class pcie_watchdog_guard {
private:
    bool guard_active;
    pcie_traffic_watchdog * watchdog;

public:
    pcie_watchdog_guard(pcie_traffic_watchdog * watchdog_ptr);
    ~pcie_watchdog_guard();

    bool is_guard_active() const { return guard_active; }
};

extern pcie_traffic_watchdog * g_pcie_traffic_watchdog;

bool llama_init_pcie_watchdog();
bool llama_enable_pcie_watchdog(bool enable);
bool llama_is_pcie_watchdog_enabled();

bool llama_begin_decode_pcie_monitoring();
bool llama_end_decode_pcie_monitoring();

bool llama_begin_pcie_token();
bool llama_record_pcie_transfer(int transfer_type,
                                int direction,
                                uint64_t size_bytes,
                                const char * source_location);
bool llama_end_pcie_token();

bool llama_finalize_pcie_monitoring();
bool llama_generate_pcie_watchdog_report();
bool llama_validate_pcie_cleanliness();

const pcie_watchdog_summary * llama_get_pcie_watchdog_summary();
const char * llama_get_pcie_watchdog_report();

void llama_print_pcie_watchdog_report();
void llama_print_pcie_watchdog_summary();
void llama_print_pcie_watchdog_violations();
void llama_print_pcie_transfer_log(uint32_t limit);
void llama_export_pcie_watchdog_json(const char * filename);

// Macro-based wrappers (compile out when disabled)
#ifdef LLAMA_DECODE_PCIE_WATCHDOG

#define INIT_PCIE_WATCHDOG() \
    llama_init_pcie_watchdog()

#define BEGIN_DECODE_PCIE_MONITORING() \
    do { \
        if (g_pcie_traffic_watchdog) { \
            llama_begin_decode_pcie_monitoring(); \
        } \
    } while(0)

#define END_DECODE_PCIE_MONITORING() \
    do { \
        if (g_pcie_traffic_watchdog) { \
            llama_end_decode_pcie_monitoring(); \
        } \
    } while(0)

#define BEGIN_PCIE_TOKEN() \
    do { \
        if (g_pcie_traffic_watchdog && llama_is_pcie_watchdog_enabled()) { \
            llama_begin_pcie_token(); \
        } \
    } while(0)

#define RECORD_PCIE_TRANSFER(type, direction, size, location) \
    do { \
        if (g_pcie_traffic_watchdog && llama_is_pcie_watchdog_enabled()) { \
            llama_record_pcie_transfer(type, direction, size, location); \
        } \
    } while(0)

#define END_PCIE_TOKEN() \
    do { \
        if (g_pcie_traffic_watchdog && llama_is_pcie_watchdog_enabled()) { \
            llama_end_pcie_token(); \
        } \
    } while(0)

#define VERIFY_PCIE_CLEAN() \
    do { \
        if (g_pcie_traffic_watchdog && llama_is_pcie_watchdog_enabled()) { \
            if (!llama_validate_pcie_cleanliness()) { \
                llama_print_pcie_watchdog_report(); \
                FATAL("PCIe watchdog validation failed"); \
            } \
        } \
    } while(0)

#else // LLAMA_DECODE_PCIE_WATCHDOG

// No-op macros when disabled
#define INIT_PCIE_WATCHDOG() do { } while(0)
#define BEGIN_DECODE_PCIE_MONITORING() do { } while(0)
#define END_DECODE_PCIE_MONITORING() do { } while(0)
#define BEGIN_PCIE_TOKEN() do { } while(0)
#define RECORD_PCIE_TRANSFER(type, direction, size, location) do { } while(0)
#define END_PCIE_TOKEN() do { } while(0)
#define VERIFY_PCIE_CLEAN() do { } while(0)

#endif // LLAMA_DECODE_PCIE_WATCHDOG

#ifdef __cplusplus
}
#endif
