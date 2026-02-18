#pragma once

/**
 * Host Access Prevention for LLAMA Decode
 *
 * Guarantee that no CPU-side code reads, writes, maps, or touches any
 * decode-critical buffer during the decode phase.
 *
 * During token generation, all decode-path data must remain GPU-resident
 * and GPU-owned. Host access creates implicit synchronization, PCIe transfers,
 * pipeline stalls, and decode pacing dependencies.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BUFFER_OWNERSHIP_UNINITIALIZED = 0,
    BUFFER_OWNERSHIP_CLASSIFICATION = 1,
    BUFFER_OWNERSHIP_VALIDATION = 2,
    BUFFER_OWNERSHIP_LOCKED = 3
} buffer_ownership_phase;

typedef enum {
    BUFFER_CLASS_GPU_EXCLUSIVE = 0,  // Decode-critical, GPU-only
    BUFFER_CLASS_CPU_PERMITTED = 1,  // Non-critical, CPU accessible
    BUFFER_CLASS_SHARED = 2          // Both CPU and GPU (outside decode)
} buffer_classification;

typedef struct {
    const char * buffer_name;
    buffer_classification classification;
    bool is_gpu_resident;
    bool host_accessible;
    bool decode_critical;
    bool accessed_during_decode;
} buffer_ownership_record;

typedef struct {
    bool kv_cache_gpu_exclusive;
    bool activations_gpu_exclusive;
    bool logits_gpu_only;
    bool sampling_gpu_only;
    bool quantized_weights_gpu_locked;
    bool cuda_workspace_gpu_only;
    bool host_sync_blocked;
    bool pcie_transfer_blocked;
    uint64_t ownership_check_timestamp_ns;
} host_access_prevention_config;

typedef struct {
    const char * function_name;
    const char * buffer_name;
    bool was_gpu_resident;
    bool host_access_attempted;
    bool was_during_decode;
    bool was_blocked;
} host_access_violation_record;

typedef struct {
    size_t total_buffers_classified;
    size_t gpu_exclusive_count;
    size_t cpu_permitted_count;
    size_t host_access_attempts_blocked;
    size_t implicit_sync_prevented;
    size_t pcie_transfers_prevented;
} host_access_prevention_validation_result;

class host_access_prevention_engine {
private:
    host_access_prevention_config immutable_config;
    std::vector<buffer_ownership_record> buffer_classifications;
    std::vector<host_access_violation_record> violation_log;
    std::map<const char *, buffer_ownership_record> buffer_registry;
    std::set<const char *> host_access_blocked_functions;

    std::atomic<buffer_ownership_phase> current_phase;
    std::atomic<bool> ownership_enforced;
    std::atomic<bool> validation_complete;
    std::atomic<bool> decode_in_progress;

    std::atomic<uint32_t> gpu_exclusive_buffers;
    std::atomic<uint32_t> host_access_attempts;
    std::atomic<uint32_t> host_access_blocks;
    std::atomic<uint32_t> sync_prevents;

public:
    host_access_prevention_engine();

    bool initialize();
    bool enable_strict_mode(bool enable);

    bool classify_buffers();
    bool mark_kv_cache_gpu_exclusive();
    bool mark_activations_gpu_exclusive();
    bool mark_logits_gpu_only();
    bool mark_sampling_gpu_only();
    bool mark_quantized_weights_gpu_locked();
    bool mark_cuda_workspace_gpu_only();

    bool begin_decode_phase();
    bool end_decode_phase();

    bool attempt_host_access(const char * func_name, const char * buffer_name, bool is_gpu_resident);
    bool attempt_host_sync();
    bool attempt_pcie_transfer(const char * buffer_name, size_t size);

    bool register_buffer(const char * name, buffer_classification classification,
                        bool gpu_resident, bool host_accessible, bool decode_critical);
    bool validate_buffer_classification(const char * buffer_name);

    bool verify_kv_cache_gpu_exclusive() const;
    bool verify_logits_gpu_only() const;
    bool verify_sampling_gpu_only() const;
    bool verify_no_host_access() const;
    bool verify_no_implicit_sync() const;
    bool verify_pcie_flat() const;

    const host_access_prevention_config & get_config() const { return immutable_config; }
    bool is_decode_in_progress() const { return decode_in_progress.load(); }
    bool is_ownership_enforced() const { return ownership_enforced.load(); }
    buffer_ownership_phase get_current_phase() const { return current_phase.load(); }

    void record_host_access_violation(const char * func, const char * buffer,
                                     bool gpu_resident, bool during_decode);
    void record_sync_prevention(const char * reason);
    void record_pcie_prevention(const char * buffer_name, size_t size);

    size_t get_gpu_exclusive_count() const { return gpu_exclusive_buffers.load(); }
    size_t get_host_access_attempts() const { return host_access_attempts.load(); }
    size_t get_host_access_blocks() const { return host_access_blocks.load(); }

    std::vector<buffer_ownership_record> get_buffer_classifications() const { return buffer_classifications; }
    std::vector<host_access_violation_record> get_violations() const { return violation_log; }

    host_access_prevention_validation_result validate_host_access_prevention() const;
    bool verify_decode_gpu_ownership() const;
    bool verify_host_isolation() const;
};

class host_access_guard {
private:
    bool guard_active;
    bool decode_phase_started;

public:
    host_access_guard();
    ~host_access_guard();

    bool is_guard_active() const;
};

extern host_access_prevention_engine * g_host_access_prevention_engine;

bool llama_init_host_access_prevention();
bool llama_enable_host_access_strict_mode(bool enable);

bool llama_classify_buffers();
bool llama_mark_kv_cache_gpu_exclusive();
bool llama_mark_activations_gpu_exclusive();
bool llama_mark_logits_gpu_only();
bool llama_mark_sampling_gpu_only();
bool llama_mark_quantized_weights_gpu_locked();
bool llama_mark_cuda_workspace_gpu_only();

bool llama_begin_decode_phase_isolation();
bool llama_end_decode_phase_isolation();

bool llama_attempt_host_access(const char * func_name, const char * buffer_name, bool is_gpu_resident);
bool llama_attempt_host_sync();
bool llama_attempt_pcie_transfer(const char * buffer_name, size_t size);

bool llama_register_buffer(const char * name, int classification,
                          bool gpu_resident, bool host_accessible, bool decode_critical);
bool llama_validate_buffer_classification(const char * buffer_name);

bool llama_verify_kv_cache_gpu_exclusive();
bool llama_verify_logits_gpu_only();
bool llama_verify_sampling_gpu_only();
bool llama_verify_no_host_access();
bool llama_verify_no_implicit_sync();
bool llama_verify_pcie_flat();

bool llama_is_decode_isolated();
bool llama_is_ownership_enforced();

void llama_record_host_access_violation(const char * func, const char * buffer,
                                       bool gpu_resident, bool during_decode);
void llama_record_sync_prevention(const char * reason);
void llama_record_pcie_prevention(const char * buffer_name, size_t size);

bool llama_validate_host_access_prevention();
bool llama_verify_decode_gpu_ownership();
bool llama_verify_host_isolation();

void llama_print_host_access_prevention_status();
void llama_print_buffer_ownership_classification();
void llama_print_host_access_violations();
void llama_print_decode_isolation_statistics();

#define ASSERT_DECODE_GPU_OWNED(buffer_name, is_gpu_resident) \
    do { \
        if (g_host_access_prevention_engine && !llama_attempt_host_access(__FUNCTION__, buffer_name, is_gpu_resident)) { \
            return -1; \
        } \
    } while(0)

#define ASSERT_NO_HOST_SYNC() \
    do { \
        if (g_host_access_prevention_engine && !llama_attempt_host_sync()) { \
            return -1; \
        } \
    } while(0)

#define ASSERT_PCIE_FLAT(buffer_name, size) \
    do { \
        if (g_host_access_prevention_engine && !llama_attempt_pcie_transfer(buffer_name, size)) { \
            return -1; \
        } \
    } while(0)

#define GUARD_HOST_ACCESS(func_name, buffer_name, is_gpu_resident) \
    do { \
        if (g_host_access_prevention_engine && !llama_attempt_host_access(func_name, buffer_name, is_gpu_resident)) { \
            return nullptr; \
        } \
    } while(0)

#define MARK_DECODE_GPU_CRITICAL(buffer_name) \
    do { \
        if (g_host_access_prevention_engine) { \
            llama_register_buffer(buffer_name, BUFFER_CLASS_GPU_EXCLUSIVE, true, false, true); \
        } \
    } while(0)

#ifdef __cplusplus
}
bool llama_init_host_access_prevention_module(void);
void llama_cleanup_host_access_prevention_module(void);
#endif
