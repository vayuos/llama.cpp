#pragma once

/**
 * Configuration Freeze Enforcement for LLAMA
 *
 * All configuration decisions must be finalized before decode begins.
 * No runtime flag evaluation is allowed inside the decode-critical path.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CONFIG_FREEZE_UNINITIALIZED = 0,
    CONFIG_FREEZE_STARTUP = 1,
    CONFIG_FREEZE_INITIALIZATION = 2,
    CONFIG_FREEZE_LOCKED = 3
} config_freeze_phase;

typedef struct {
    int backend_type;
    bool backend_forced;
    int sampling_mode;
    float temperature;
    int top_k_value;
    float top_p_value;
    int decode_thread_count;
    int worker_thread_count;
    bool thread_affinity_pinned;
    bool kv_cache_allocated;
    bool use_flash_attention;
    bool use_quantization;
    uint64_t freeze_timestamp_ns;
} config_snapshot;

typedef struct {
    const char * file_path;
    int line_number;
    const char * function_name;
    const char * config_accessed;
    const char * access_type;
    bool is_during_decode;
    bool is_violation;
} config_access_audit_entry;

typedef struct {
    bool is_frozen;
    bool decode_immutable;
    uint32_t runtime_flag_reads_count;
    uint32_t configuration_changes_rejected;
    uint64_t total_decode_cycles;
    bool all_branches_static;
} config_freeze_validation_result;

class config_freeze_engine {
private:
    config_snapshot precomputed_config;
    std::vector<config_access_audit_entry> config_access_log;
    std::vector<config_access_audit_entry> violation_log;

    std::atomic<config_freeze_phase> current_phase;
    std::atomic<bool> configuration_locked;
    std::atomic<bool> strict_enforcement;

    std::function<void()> backend_dispatch_fn;
    std::function<void()> sampling_dispatch_fn;
    std::function<void()> memory_dispatch_fn;

    std::atomic<uint64_t> total_static_dispatches;
    std::atomic<uint64_t> branch_avoidance_count;
    std::atomic<uint64_t> configuration_lock_attempts;
    std::atomic<uint64_t> lock_rejections;

public:
    config_freeze_engine();

    bool initialize();
    bool enable_enforcement(bool enable);

    bool resolve_cli_flags(int argc, const char * const * argv);
    bool resolve_server_config();
    bool resolve_environment_variables();
    bool resolve_backend_selection();
    bool resolve_sampling_configuration();
    bool resolve_threading_topology();
    bool resolve_memory_strategy();

    const config_snapshot & get_frozen_config() const { return precomputed_config; }
    bool is_config_frozen() const { return configuration_locked.load(); }
    config_freeze_phase get_current_phase() const { return current_phase.load(); }

    void enter_initialization_phase();
    void enter_decode_phase();
    void lock_configuration();
    bool attempt_configuration_change(const char * config_name);

    void audit_config_access(const char * file, int line, const char * func,
                             const char * config_name, const char * access_type,
                             bool during_decode);
    void record_violation(const config_access_audit_entry & entry);

    size_t get_access_log_count() const { return config_access_log.size(); }
    size_t get_violation_count() const { return violation_log.size(); }
    std::vector<config_access_audit_entry> get_access_log() const { return config_access_log; }
    std::vector<config_access_audit_entry> get_violations() const { return violation_log; }

    void bind_backend_dispatch(std::function<void()> fn) { backend_dispatch_fn = fn; }
    void bind_sampling_dispatch(std::function<void()> fn) { sampling_dispatch_fn = fn; }
    void bind_memory_dispatch(std::function<void()> fn) { memory_dispatch_fn = fn; }

    void execute_backend_dispatch() { if (backend_dispatch_fn) backend_dispatch_fn(); }
    void execute_sampling_dispatch() { if (sampling_dispatch_fn) sampling_dispatch_fn(); }
    void execute_memory_dispatch() { if (memory_dispatch_fn) memory_dispatch_fn(); }

    uint64_t get_static_dispatch_count() const { return total_static_dispatches.load(); }
    uint64_t get_branch_avoidance_count() const { return branch_avoidance_count.load(); }
    uint64_t get_lock_rejection_count() const { return lock_rejections.load(); }

    void record_static_dispatch() { total_static_dispatches.fetch_add(1); }
    void record_branch_avoidance() { branch_avoidance_count.fetch_add(1); }

    config_freeze_validation_result validate_configuration_frozen() const;
    bool verify_zero_runtime_flag_reads() const;
    bool verify_all_branches_static() const;
    bool verify_decode_immutability() const;
    bool verify_backend_frozen() const;
    bool verify_sampling_frozen() const;
    bool verify_threading_frozen() const;
};

class config_access_guard {
private:
    const char * config_name;
    bool is_during_decode;

public:
    config_access_guard(const char * name);
    ~config_access_guard();

    bool is_read_allowed() const;
    void record_attempted_read();
};

extern config_freeze_engine * g_config_freeze_engine;

bool llama_init_config_freeze();
bool llama_enable_config_freeze(bool enable);
void llama_set_strict_config_enforcement(bool strict);

void llama_enter_initialization_phase();
void llama_enter_decode_phase();
void llama_lock_configuration();

bool llama_resolve_all_startup_flags();
bool llama_resolve_cli_configuration(int argc, const char * const * argv);
bool llama_resolve_environment_configuration();

bool llama_attempt_config_modification(const char * config_name);
const config_snapshot * llama_get_frozen_config();
bool llama_is_configuration_frozen();
bool llama_is_decode_phase_active();

void llama_audit_config_access(const char * file, int line, const char * func,
                               const char * config_name, const char * access_type);

bool llama_validate_config_frozen();
bool llama_validate_zero_runtime_reads();
bool llama_validate_all_branches_static();

void llama_print_config_freeze_report();
void llama_print_config_access_audit();
void llama_print_config_violation_report();
void llama_dump_frozen_config();
void llama_dump_config_statistics();

#define CONFIG_FREEZE_GUARD(config_name) \
    do { \
        if (g_config_freeze_engine && g_config_freeze_engine->is_config_frozen()) { \
            g_config_freeze_engine->record_static_dispatch(); \
        } \
    } while(0)

#define STATIC_DISPATCH_BACKEND() \
    do { \
        if (g_config_freeze_engine) { \
            g_config_freeze_engine->execute_backend_dispatch(); \
            g_config_freeze_engine->record_branch_avoidance(); \
        } \
    } while(0)

#define STATIC_DISPATCH_SAMPLING() \
    do { \
        if (g_config_freeze_engine) { \
            g_config_freeze_engine->execute_sampling_dispatch(); \
            g_config_freeze_engine->record_branch_avoidance(); \
        } \
    } while(0)

#define CONFIG_MODIFICATION_GUARD(config_name) \
    do { \
        if (g_config_freeze_engine && !g_config_freeze_engine->attempt_configuration_change(config_name)) { \
            return -1; \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
