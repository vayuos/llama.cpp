#pragma once

/**
 * Decode-Time Feature Probing Removal for LLAMA
 *
 * All capability detection, backend probing, and conditional feature
 * selection must be resolved before decode begins, never during.
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
    PROBING_REMOVAL_UNINITIALIZED = 0,
    PROBING_REMOVAL_STARTUP = 1,
    PROBING_REMOVAL_VALIDATION = 2,
    PROBING_REMOVAL_LOCKED = 3
} probing_removal_phase;

typedef struct {
    bool cuda_available;
    bool tensor_cores_available;
    bool mmq_compatible;
    bool flash_attention_compatible;
    int compute_capability;
    int device_architecture;
    bool all_ops_gpu_compatible;
    bool backend_validated;
    uint64_t validation_timestamp_ns;
} capability_snapshot;

typedef struct {
    const char * file_path;
    int line_number;
    const char * function_name;
    const char * probe_type;
    const char * location_description;
    bool is_in_decode_path;
    bool is_violation;
} probing_audit_entry;

typedef struct {
    bool all_probing_removed;
    uint32_t probing_checks_found;
    uint32_t in_decode_path_count;
    uint32_t removed_checks;
    bool capabilities_locked;
    bool decode_path_clean;
} probing_removal_validation_result;

class probing_removal_engine {
private:
    capability_snapshot immutable_capabilities;
    std::vector<probing_audit_entry> probing_audit_log;
    std::vector<probing_audit_entry> violation_log;

    std::atomic<probing_removal_phase> current_phase;
    std::atomic<bool> capabilities_frozen;
    std::atomic<bool> strict_validation;

    std::atomic<uint64_t> total_probes_found;
    std::atomic<uint64_t> decode_path_probes;
    std::atomic<uint64_t> probes_removed;
    std::atomic<uint64_t> fallback_paths_removed;

public:
    probing_removal_engine();

    bool initialize();
    bool validate_capabilities_at_startup();
    bool detect_gpu_architecture();
    bool detect_compute_capability();
    bool detect_tensor_core_support();
    bool detect_mmq_compatibility();
    bool detect_flash_attention_compatibility();
    bool validate_all_ops_gpu_compatible();

    const capability_snapshot & get_immutable_capabilities() const { return immutable_capabilities; }
    bool are_capabilities_frozen() const { return capabilities_frozen.load(); }
    probing_removal_phase get_current_phase() const { return current_phase.load(); }

    void lock_capabilities();
    void audit_probing_check(const char * file, int line, const char * func,
                             const char * probe_type, const char * location,
                             bool in_decode_path);
    void record_violation(const probing_audit_entry & entry);
    void record_probe_removal();
    void record_fallback_removal();

    size_t get_audit_count() const { return probing_audit_log.size(); }
    size_t get_violation_count() const { return violation_log.size(); }
    std::vector<probing_audit_entry> get_audit_log() const { return probing_audit_log; }
    std::vector<probing_audit_entry> get_violations() const { return violation_log; }

    uint64_t get_probes_found() const { return total_probes_found.load(); }
    uint64_t get_decode_path_probes() const { return decode_path_probes.load(); }
    uint64_t get_probes_removed() const { return probes_removed.load(); }
    uint64_t get_fallbacks_removed() const { return fallback_paths_removed.load(); }

    void record_probe_found() { total_probes_found.fetch_add(1); }
    void record_decode_path_probe() { decode_path_probes.fetch_add(1); }

    probing_removal_validation_result validate_probing_removal() const;
    bool verify_no_runtime_probing() const;
    bool verify_capabilities_immutable() const;
    bool verify_decode_path_clean() const;
    bool verify_backend_precondition() const;
};

class capability_guard {
private:
    const char * capability_name;
    bool is_available;
    bool is_in_decode;

public:
    capability_guard(const char * name);
    ~capability_guard();

    bool is_capability_available() const;
    void record_capability_check();
};

extern probing_removal_engine * g_probing_removal_engine;

bool llama_init_probing_removal();
bool llama_validate_capabilities_at_startup();
bool llama_detect_gpu_architecture();
void llama_lock_capabilities();

bool llama_is_cuda_available();
bool llama_has_tensor_cores();
bool llama_is_mmq_compatible();
bool llama_is_flash_attention_compatible();

void llama_audit_probing_check(const char * file, int line, const char * func,
                                const char * probe_type, const char * location);
void llama_record_probe_removal();

bool llama_validate_probing_removal();
bool llama_validate_no_runtime_probing();
bool llama_validate_decode_path_clean();

void llama_print_probing_audit_report();
void llama_print_probing_removal_validation();
void llama_print_capabilities_snapshot();
void llama_dump_probing_statistics();

#define PROBING_GUARD(probe_type, location) \
    do { \
        if (g_probing_removal_engine) { \
            g_probing_removal_engine->record_probe_found(); \
        } \
    } while(0)

#define CAPABILITY_CHECK_GUARD(capability) \
    do { \
        if (g_probing_removal_engine && g_probing_removal_engine->are_capabilities_frozen()) { \
            g_probing_removal_engine->record_decode_path_probe(); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
