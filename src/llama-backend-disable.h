#pragma once

/**
 * Backend Disable at Build Time for LLAMA
 *
 * All optional backends must be disabled at compile-time, not runtime.
 * No runtime backend selection branching is permitted.
 * Zero dynamic dispatch for backend operations in decode path.
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
    BACKEND_DISABLE_UNINITIALIZED = 0,
    BACKEND_DISABLE_STARTUP = 1,
    BACKEND_DISABLE_VALIDATION = 2,
    BACKEND_DISABLE_LOCKED = 3
} backend_disable_phase;

typedef struct {
    bool cuda_enabled;
    bool rocm_enabled;
    bool metal_enabled;
    bool vulkan_enabled;
    bool cpu_backend_enabled;
    bool mmq_enabled;
    bool cutlass_enabled;
    bool tensorrt_enabled;
    uint64_t lock_timestamp_ns;
} backend_configuration;

typedef struct {
    const char * file_path;
    int line_number;
    const char * function_name;
    const char * backend_name;
    const char * dispatch_type;
    bool is_during_decode;
    bool is_violation;
} backend_dispatch_audit_entry;

typedef struct {
    bool backends_locked;
    bool single_backend_enforced;
    uint32_t runtime_backend_checks;
    uint32_t dispatch_violations;
    uint32_t backends_disabled_at_build;
    bool decode_path_single_target;
} backend_disable_validation_result;

class backend_disable_engine {
private:
    backend_configuration compile_time_config;
    backend_configuration immutable_config;
    std::vector<backend_dispatch_audit_entry> dispatch_audit_log;
    std::vector<backend_dispatch_audit_entry> violation_log;

    std::atomic<backend_disable_phase> current_phase;
    std::atomic<bool> backends_locked;
    std::atomic<bool> strict_enforcement;

    std::atomic<uint32_t> runtime_backend_checks;
    std::atomic<uint32_t> dispatch_violations;
    std::atomic<uint32_t> backends_disabled;
    std::atomic<uint32_t> static_dispatches;

public:
    backend_disable_engine();

    bool initialize();
    bool enable_enforcement(bool enable);

    bool validate_backends_at_startup();
    bool verify_single_active_backend();
    bool verify_backend_availability();
    bool disable_unused_backends();

    const backend_configuration & get_active_backend_config() const { return immutable_config; }
    bool are_backends_locked() const { return backends_locked.load(); }
    backend_disable_phase get_current_phase() const { return current_phase.load(); }

    void enter_startup_phase();
    void enter_validation_phase();
    void lock_backends();
    bool attempt_backend_change(const char * backend_name);

    void audit_backend_dispatch(const char * file, int line, const char * func,
                               const char * backend_name, const char * dispatch_type,
                               bool during_decode);
    void record_violation(const backend_dispatch_audit_entry & entry);

    size_t get_audit_log_count() const { return dispatch_audit_log.size(); }
    size_t get_violation_count() const { return violation_log.size(); }
    std::vector<backend_dispatch_audit_entry> get_audit_log() const { return dispatch_audit_log; }
    std::vector<backend_dispatch_audit_entry> get_violations() const { return violation_log; }

    void record_runtime_check() { runtime_backend_checks.fetch_add(1); }
    void record_dispatch_violation() { dispatch_violations.fetch_add(1); }
    void record_backend_disabled() { backends_disabled.fetch_add(1); }
    void record_static_dispatch() { static_dispatches.fetch_add(1); }

    backend_disable_validation_result validate_backend_disable() const;
    bool verify_no_runtime_backend_checks() const;
    bool verify_single_backend_enforced() const;
    bool verify_decode_path_exclusive() const;
    bool verify_cuda_or_rocm_only() const;
    bool verify_no_cpu_fallback() const;
};

class backend_dispatch_guard {
private:
    const char * backend_name;
    bool is_during_decode;
    bool is_allowed;

public:
    backend_dispatch_guard(const char * name, bool decode_context);
    ~backend_dispatch_guard();

    bool is_dispatch_allowed() const;
    void record_dispatch_attempt();
};

extern backend_disable_engine * g_backend_disable_engine;

bool llama_init_backend_disable();
bool llama_enable_backend_disable_enforcement(bool enable);
void llama_set_strict_backend_enforcement(bool strict);

bool llama_validate_backends_at_startup();
bool llama_verify_single_active_backend();
bool llama_disable_unused_backends();

void llama_lock_backend_configuration();
bool llama_attempt_backend_change(const char * backend_name);

bool llama_is_cuda_backend_enabled();
bool llama_is_rocm_backend_enabled();
bool llama_is_metal_backend_enabled();
bool llama_is_cpu_backend_enabled();
bool llama_is_mmq_backend_enabled();

void llama_audit_backend_dispatch(const char * file, int line, const char * func,
                                 const char * backend_name, const char * dispatch_type);

bool llama_validate_backend_disable();
bool llama_validate_no_runtime_checks();
bool llama_validate_single_backend();

void llama_print_backend_audit_report();
void llama_print_backend_disable_validation();
void llama_print_active_backend_snapshot();
void llama_dump_backend_statistics();

// Module initialization
bool llama_init_backend_disable_module(void);
void llama_cleanup_backend_disable_module(void);

#define BACKEND_DISPATCH_GUARD(backend_name, decode_context) \
    do { \
        if (g_backend_disable_engine && g_backend_disable_engine->are_backends_locked()) { \
            g_backend_disable_engine->record_static_dispatch(); \
        } \
    } while(0)

#define STATIC_BACKEND_DISPATCH(backend_target) \
    do { \
        if (g_backend_disable_engine) { \
            g_backend_disable_engine->record_static_dispatch(); \
        } \
    } while(0)

#define BACKEND_MODIFICATION_GUARD(backend_name) \
    do { \
        if (g_backend_disable_engine && !g_backend_disable_engine->attempt_backend_change(backend_name)) { \
            return -1; \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
