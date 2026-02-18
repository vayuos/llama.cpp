#pragma once

/**
 * MMQ Backend Enforcement for Quantized Decode in LLAMA
 *
 * For quantized model decode, MMQ (MatMul Quantized) backend must be the ONLY
 * valid path. cuBLAS fallback is forbidden. Decode must guarantee MMQ usage
 * or hard-fail with explicit error message.
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
    MMQ_ENFORCE_UNINITIALIZED = 0,
    MMQ_ENFORCE_STARTUP = 1,
    MMQ_ENFORCE_VALIDATION = 2,
    MMQ_ENFORCE_LOCKED = 3
} mmq_enforce_phase;

typedef enum {
    QUANTIZATION_NONE = 0,
    QUANTIZATION_Q4_0 = 1,
    QUANTIZATION_Q4_1 = 2,
    QUANTIZATION_Q5_0 = 3,
    QUANTIZATION_Q5_1 = 4,
    QUANTIZATION_Q8_0 = 5,
    QUANTIZATION_Q6_K = 6,
    QUANTIZATION_Q2_K = 7,
    QUANTIZATION_Q3_K = 8,
    QUANTIZATION_Q4_K = 9,
    QUANTIZATION_Q5_K = 10,
    QUANTIZATION_IQ2_XXS = 11,
    QUANTIZATION_IQ3_XXS = 12
} quantization_type;

typedef struct {
    bool model_quantized;
    quantization_type quant_type;
    bool mmq_available;
    bool mmq_forced;
    bool cublas_disabled;
    bool dense_cuda_disabled;
    bool cpu_fallback_disabled;
    uint64_t lock_timestamp_ns;
} mmq_configuration;

typedef struct {
    const char * file_path;
    int line_number;
    const char * function_name;
    const char * kernel_type;
    const char * backend_attempted;
    bool is_in_decode;
    bool is_violation;
} mmq_dispatch_audit_entry;

typedef struct {
    bool mmq_enforced;
    bool quantized_decode_safe;
    uint32_t mmq_kernel_invocations;
    uint32_t fallback_attempts;
    uint32_t fallback_rejections;
    uint32_t quantization_mismatches;
} mmq_enforcement_validation_result;

class mmq_enforcement_engine {
private:
    mmq_configuration immutable_mmq_config;
    std::vector<mmq_dispatch_audit_entry> dispatch_audit_log;
    std::vector<mmq_dispatch_audit_entry> violation_log;

    std::atomic<mmq_enforce_phase> current_phase;
    std::atomic<bool> mmq_enforcement_locked;
    std::atomic<bool> strict_enforcement;

    std::atomic<uint32_t> mmq_kernels_used;
    std::atomic<uint32_t> fallback_attempts_blocked;
    std::atomic<uint32_t> backend_mismatches;
    std::atomic<uint32_t> quantization_validation_passes;

public:
    mmq_enforcement_engine();

    bool initialize();
    bool enable_enforcement(bool enable);

    bool validate_mmq_availability_at_startup();
    bool verify_quantization_type();
    bool verify_mmq_kernel_compatibility();
    bool disable_cublas_fallback();
    bool disable_dense_fallback();
    bool disable_cpu_fallback();

    const mmq_configuration & get_mmq_config() const { return immutable_mmq_config; }
    bool is_mmq_enforcement_locked() const { return mmq_enforcement_locked.load(); }
    mmq_enforce_phase get_current_phase() const { return current_phase.load(); }

    void enter_startup_phase();
    void enter_validation_phase();
    void lock_mmq_enforcement();
    bool attempt_fallback_dispatch(const char * backend_name);

    void audit_mmq_dispatch(const char * file, int line, const char * func,
                           const char * kernel_type, const char * backend_attempted,
                           bool in_decode);
    void record_violation(const mmq_dispatch_audit_entry & entry);

    size_t get_audit_log_count() const { return dispatch_audit_log.size(); }
    size_t get_violation_count() const { return violation_log.size(); }
    std::vector<mmq_dispatch_audit_entry> get_audit_log() const { return dispatch_audit_log; }
    std::vector<mmq_dispatch_audit_entry> get_violations() const { return violation_log; }

    void record_mmq_kernel_invocation() { mmq_kernels_used.fetch_add(1); }
    void record_fallback_attempt() { fallback_attempts_blocked.fetch_add(1); }
    void record_backend_mismatch() { backend_mismatches.fetch_add(1); }
    void record_quantization_validation() { quantization_validation_passes.fetch_add(1); }

    mmq_enforcement_validation_result validate_mmq_enforcement() const;
    bool verify_only_mmq_used() const;
    bool verify_no_cublas_fallback() const;
    bool verify_no_dense_fallback() const;
    bool verify_quantization_type_safe() const;
    bool verify_decode_path_mmq_exclusive() const;
};

class mmq_dispatch_guard {
private:
    const char * backend_name;
    const char * kernel_type;
    bool is_in_decode;
    bool dispatch_allowed;

public:
    mmq_dispatch_guard(const char * backend, const char * kernel, bool decode_context);
    ~mmq_dispatch_guard();

    bool is_mmq_dispatch_allowed() const;
    void record_dispatch_attempt();
};

extern mmq_enforcement_engine * g_mmq_enforcement_engine;

bool llama_init_mmq_enforcement();
bool llama_enable_mmq_enforcement(bool enable);
void llama_set_strict_mmq_enforcement(bool strict);

bool llama_validate_mmq_at_startup();
bool llama_verify_quantization_type();
bool llama_verify_mmq_kernel_compatibility();

void llama_lock_mmq_enforcement();
bool llama_attempt_fallback_dispatch(const char * backend_name);

bool llama_is_quantized_model();
quantization_type llama_get_quantization_type();
bool llama_is_mmq_available();
bool llama_is_cublas_disabled();

void llama_audit_mmq_dispatch(const char * file, int line, const char * func,
                             const char * kernel_type, const char * backend_attempted);

bool llama_validate_mmq_enforcement();
bool llama_validate_only_mmq_used();
bool llama_validate_no_fallback();
bool llama_validate_quantization_safe();

void llama_print_mmq_dispatch_audit();
void llama_print_mmq_enforcement_validation();
void llama_print_mmq_configuration_snapshot();
void llama_dump_mmq_statistics();

#define MMQ_DISPATCH_GUARD(backend, kernel, decode_ctx) \
    do { \
        if (g_mmq_enforcement_engine && g_mmq_enforcement_engine->is_mmq_enforcement_locked()) { \
            g_mmq_enforcement_engine->record_mmq_kernel_invocation(); \
        } \
    } while(0)

#define ENFORCE_MMQ_ONLY() \
    do { \
        if (g_mmq_enforcement_engine && !g_mmq_enforcement_engine->is_mmq_enforcement_locked()) { \
            return -1; \
        } \
    } while(0)

#define REJECT_FALLBACK_DISPATCH(backend_name) \
    do { \
        if (g_mmq_enforcement_engine && g_mmq_enforcement_engine->attempt_fallback_dispatch(backend_name)) { \
            return -1; \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
