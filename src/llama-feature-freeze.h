#pragma once

/**
 * Feature Flag Build-Time Freezing for LLAMA
 *
 * All optional features affecting decode behavior decided at compile time.
 * No decode-path branching based on feature availability permitted.
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

#ifndef LLAMA_FEATURE_CUDA_ENABLED
#define LLAMA_FEATURE_CUDA_ENABLED 1
#endif
#ifndef LLAMA_FEATURE_CPU_ENABLED
#define LLAMA_FEATURE_CPU_ENABLED 0
#endif
#ifndef LLAMA_FEATURE_CUBLAS_ENABLED
#define LLAMA_FEATURE_CUBLAS_ENABLED 1
#endif
#ifndef LLAMA_FEATURE_MMQ_ENABLED
#define LLAMA_FEATURE_MMQ_ENABLED 1
#endif
#ifndef LLAMA_FEATURE_FLASH_ATTENTION_ENABLED
#define LLAMA_FEATURE_FLASH_ATTENTION_ENABLED 0
#endif
#ifndef LLAMA_FEATURE_CUDA_GRAPHS_ENABLED
#define LLAMA_FEATURE_CUDA_GRAPHS_ENABLED 0
#endif
#ifndef LLAMA_FEATURE_HYBRID_MEMORY_ENABLED
#define LLAMA_FEATURE_HYBRID_MEMORY_ENABLED 0
#endif
#ifndef LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED
#define LLAMA_FEATURE_SPECULATIVE_DECODE_ENABLED 0
#endif
#ifndef LLAMA_FEATURE_DETERMINISM_STRICT
#define LLAMA_FEATURE_DETERMINISM_STRICT 0
#endif
#ifndef LLAMA_FEATURE_EXPERIMENTAL_KERNELS
#define LLAMA_FEATURE_EXPERIMENTAL_KERNELS 0
#endif
#ifndef LLAMA_FEATURE_PROFILE_NAME
#define LLAMA_FEATURE_PROFILE_NAME "default"
#endif
#ifndef LLAMA_FEATURE_FREEZE_PROFILE
#define LLAMA_FEATURE_FREEZE_PROFILE 1
#endif

typedef enum {
    LLAMA_FEATURE_FREEZE_STATE_UNINITIALIZED = 0,
    LLAMA_FEATURE_FREEZE_STATE_VALIDATED = 1,
    LLAMA_FEATURE_FREEZE_STATE_IMMUTABLE = 2,
    LLAMA_FEATURE_FREEZE_STATE_HARDWARE_MISMATCH = 3,
    LLAMA_FEATURE_FREEZE_STATE_ERROR = 4
} llama_feature_freeze_state;

typedef struct {
    llama_feature_freeze_state state;
    int hardware_compatible;
    int validation_error_code;
    const char * validation_error_message;
} llama_feature_freeze_validation_state;

typedef struct {
    bool cuda_enabled;
    bool cpu_enabled;
    bool cublas_enabled;
    bool mmq_enabled;
    bool flash_attention_enabled;
    bool cuda_graphs_enabled;
    bool hybrid_memory_enabled;
    bool speculative_decode_enabled;
    bool determinism_strict;
    bool experimental_kernels;
    uint32_t reserved;
} llama_feature_freeze_capabilities;

#ifdef __cplusplus
}  // extern "C"
#endif

typedef struct {
    std::atomic<uint64_t> build_time_features_resolved;
    std::atomic<uint64_t> runtime_feature_checks_blocked;
    std::atomic<uint64_t> compile_out_paths_eliminated;
    std::atomic<uint64_t> decode_branches_removed;
    std::atomic<uint64_t> feature_symbol_lookups_eliminated;
} llama_feature_freeze_metrics;

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*llama_feature_freeze_validate_hardware_fn)(void);
typedef int (*llama_feature_freeze_validate_features_fn)(void);

typedef struct {
    const char * profile_name;
    uint32_t profile_id;
    llama_feature_freeze_capabilities capabilities;
    void (*compute_dispatch)(void);
    void (*memory_dispatch)(void);
    void (*sync_dispatch)(void);
    llama_feature_freeze_validate_hardware_fn validate_hardware;
    llama_feature_freeze_validate_features_fn validate_features;
} llama_feature_freeze_dispatch_table;

typedef enum {
    FEATURE_FREEZE_UNINITIALIZED = 0,
    FEATURE_FREEZE_STARTUP = 1,
    FEATURE_FREEZE_VALIDATION = 2,
    FEATURE_FREEZE_LOCKED = 3
} feature_freeze_phase;

typedef struct {
    bool cuda_enabled;
    bool cublas_enabled;
    bool mmq_enabled;
    bool flash_attention_enabled;
    bool cuda_graphs_enabled;
    bool hybrid_memory_enabled;
    bool speculative_decoding_enabled;
    bool determinism_enforced;
    bool debug_code_compiled;
    bool verbose_logging_compiled;
    bool experimental_kernels_enabled;
    bool cpu_backend_compiled;
    bool openmp_enabled;
    uint32_t feature_flags_mask;
    uint64_t build_timestamp_ns;
} feature_snapshot;

typedef struct {
    const char * feature_name;
    bool is_compiled_in;
    bool is_used_in_decode;
    const char * build_config;
    bool is_required;
} feature_audit_entry;

typedef struct {
    bool all_features_frozen;
    uint32_t total_features_evaluated;
    uint32_t runtime_feature_checks;
    uint32_t compiled_out_features;
    bool decode_path_clean;
    bool no_runtime_branching;
} feature_freeze_validation_result;

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
class feature_freeze_engine {
private:
    feature_snapshot compiled_features;
    std::vector<feature_audit_entry> feature_audit_log;
    std::vector<feature_audit_entry> violation_log;

    std::atomic<feature_freeze_phase> current_phase;
    std::atomic<bool> features_locked;
    std::atomic<bool> strict_validation;

    std::atomic<uint64_t> total_runtime_checks;
    std::atomic<uint64_t> compiled_out_count;
    std::atomic<uint64_t> incompatibility_detected;

public:
    feature_freeze_engine();

    bool initialize();
    bool validate_build_configuration();
    bool validate_hardware_compatibility();

    const feature_snapshot & get_compiled_features() const { return compiled_features; }
    bool are_features_frozen() const { return features_locked.load(); }
    feature_freeze_phase get_current_phase() const { return current_phase.load(); }

    void lock_features();
    void audit_feature_usage(const char * feature_name, bool is_compiled,
                             bool is_used_in_decode, const char * build_config);
    void record_runtime_feature_check();
    void record_incompatibility();

    size_t get_audit_count() const { return feature_audit_log.size(); }
    size_t get_violation_count() const { return violation_log.size(); }
    std::vector<feature_audit_entry> get_audit_log() const { return feature_audit_log; }
    std::vector<feature_audit_entry> get_violations() const { return violation_log; }

    uint64_t get_runtime_checks_count() const { return total_runtime_checks.load(); }
    uint64_t get_compiled_out_count() const { return compiled_out_count.load(); }

    void record_runtime_check() { total_runtime_checks.fetch_add(1); }
    void record_compiled_out() { compiled_out_count.fetch_add(1); }

    feature_freeze_validation_result validate_feature_freeze() const;
    bool verify_no_runtime_branching() const;
    bool verify_all_features_compiled_decision() const;
    bool verify_decode_path_clean() const;
    bool verify_hardware_compatibility() const;
};

class feature_availability_guard {
private:
    const char * feature_name;
    bool is_available;

public:
    feature_availability_guard(const char * name);
    ~feature_availability_guard();

    bool is_feature_available() const;
    void record_feature_check();
};
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern feature_freeze_engine * g_feature_freeze_engine;

bool llama_init_feature_freeze();
bool llama_validate_build_features();
bool llama_validate_hardware_features();
void llama_lock_features();

bool llama_is_cuda_enabled();
bool llama_is_flash_attention_enabled();
bool llama_is_mmq_enabled();
bool llama_is_determinism_enforced();
bool llama_is_experimental_kernels_enabled();

void llama_audit_feature_usage(const char * feature_name, bool compiled,
                               bool in_decode_path, const char * config);
void llama_record_feature_check();

bool llama_validate_feature_freeze();
bool llama_validate_no_runtime_branching();

void llama_print_feature_audit_report();
void llama_print_feature_freeze_validation();
void llama_print_compiled_features();
void llama_dump_feature_statistics();

// Additional function declarations
int llama_feature_freeze_init(void);
const llama_feature_freeze_validation_state* llama_feature_freeze_get_validation_state(void);
uint32_t llama_feature_freeze_get_profile(void);
const char* llama_feature_freeze_get_profile_name(void);
const llama_feature_freeze_capabilities* llama_feature_freeze_get_features(void);
int llama_feature_freeze_validate_integrity(void);
const llama_feature_freeze_metrics* llama_feature_freeze_get_metrics(void);
void llama_feature_freeze_log_config(void);
int llama_feature_freeze_is_feature_enabled(uint32_t feature_id);
void llama_feature_freeze_register_validator(llama_feature_freeze_validate_hardware_fn fn);

#ifdef __cplusplus
}
#endif

#define FEATURE_COMPILE_CHECK(feature_name) \
    do { \
        if (g_feature_freeze_engine) { \
            g_feature_freeze_engine->record_runtime_check(); \
        } \
    } while(0)

#define FEATURE_HARDWARE_CHECK(feature_name) \
    do { \
        if (!llama_is_cuda_enabled()) { \
            return false; \
        } \
    } while(0)
