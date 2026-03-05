#pragma once

/**
 * CPU Dequantization Elimination for LLAMA
 *
 * All weight dequantization must occur inside GPU kernels only.
 * No host-side dequantization during token generation.
 * Quantized tensors must remain GPU-resident and GPU-executed.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CPU_DEQUANT_UNINITIALIZED = 0,
    CPU_DEQUANT_STARTUP = 1,
    CPU_DEQUANT_VALIDATION = 2,
    CPU_DEQUANT_LOCKED = 3
} cpu_dequant_elimination_phase;

typedef struct {
    bool decode_in_progress;
    bool cpu_dequant_forbidden;
    bool quant_tensors_gpu_resident;
    bool mmq_kernels_exclusive;
    bool no_host_buffers_allowed;
    uint64_t lock_timestamp_ns;
} cpu_dequant_config;

typedef struct {
    const char * file_path;
    int line_number;
    const char * function_name;
    const char * tensor_name;
    const char * dequant_type;
    bool was_cpu_dequant;
    bool was_blocked;
} cpu_dequant_attempt_record;

typedef struct {
    bool cpu_dequant_eliminated;
    bool all_quant_gpu_resident;
    uint32_t cpu_dequant_blocks;
    uint32_t host_buffer_prevents;
    uint32_t gpu_residency_enforces;
} cpu_dequant_elimination_validation_result;

class cpu_dequantization_elimination_engine {
private:
    cpu_dequant_config immutable_config;
    std::vector<cpu_dequant_attempt_record> elimination_audit_log;
    std::vector<cpu_dequant_attempt_record> blocked_attempt_log;

    std::atomic<cpu_dequant_elimination_phase> current_phase;
    std::atomic<bool> decode_active;
    std::atomic<bool> strict_enforcement;

    std::atomic<uint32_t> cpu_dequant_blocks;
    std::atomic<uint32_t> host_buffer_prevents;
    std::atomic<uint32_t> gpu_residency_enforces;
    std::atomic<uint32_t> backend_checks;

public:
    cpu_dequantization_elimination_engine();

    bool initialize();
    bool enable_strict_mode(bool enable);

    bool begin_decode_phase();
    bool end_decode_phase();
    bool lock_gpu_residency();

    bool attempt_cpu_dequantization(const char * tensor_name, const char * dequant_type);
    bool attempt_host_buffer_allocation(const char * tensor_name);
    bool attempt_quant_tensor_relocation(const char * tensor_name);

    const cpu_dequant_config & get_config() const { return immutable_config; }
    bool is_decode_in_progress() const { return decode_active.load(); }
    cpu_dequant_elimination_phase get_current_phase() const { return current_phase.load(); }

    void record_cpu_dequant_block(const char * file, int line, const char * func,
                                 const char * tensor, const char * dequant_type);
    void record_host_buffer_prevent(const char * tensor_name);
    void record_gpu_residency_enforce(const char * tensor_name);

    size_t get_audit_count() const { return elimination_audit_log.size(); }
    size_t get_blocked_count() const { return blocked_attempt_log.size(); }
    std::vector<cpu_dequant_attempt_record> get_audit_log() const { return elimination_audit_log; }
    std::vector<cpu_dequant_attempt_record> get_blocked() const { return blocked_attempt_log; }

    cpu_dequant_elimination_validation_result validate_cpu_dequant_elimination() const;
    bool verify_no_cpu_dequant() const;
    bool verify_gpu_residency_locked() const;
    bool verify_no_host_buffers() const;
    bool verify_mmq_exclusive() const;
    bool verify_decode_phase_clean() const;
};

class decode_phase_guard {
private:
    bool phase_started;

public:
    decode_phase_guard();
    ~decode_phase_guard();

    bool is_decode_active() const;
};

extern cpu_dequantization_elimination_engine * g_cpu_dequant_elimination_engine;

bool llama_init_cpu_dequant_elimination();
bool llama_enable_cpu_dequant_strict_mode(bool enable);

bool llama_begin_decode_phase();
bool llama_end_decode_phase();
bool llama_lock_gpu_residency();

bool llama_attempt_cpu_dequantization(const char * tensor_name, const char * dequant_type);
bool llama_attempt_host_buffer_allocation(const char * tensor_name);
bool llama_attempt_quant_tensor_relocation(const char * tensor_name);

bool llama_is_decode_phase_active();
bool llama_is_gpu_residency_locked();
bool llama_is_cpu_dequant_forbidden();

void llama_record_cpu_dequant_block(const char * file, int line, const char * func,
                                   const char * tensor, const char * dequant_type);
void llama_record_host_buffer_prevent(const char * tensor_name);
void llama_record_gpu_residency_enforce(const char * tensor_name);

bool llama_validate_cpu_dequant_elimination();
bool llama_validate_no_cpu_dequant();
bool llama_validate_gpu_residency();
bool llama_validate_no_host_buffers();
bool llama_validate_decode_phase_clean();

void llama_print_cpu_dequant_elimination_audit();
void llama_print_cpu_dequant_elimination_validation();
void llama_print_cpu_dequant_config_snapshot();
void llama_dump_cpu_dequant_statistics();

// Self-test module initialization (internal use)
bool llama_init_cpu_dequant_elimination_module(void);
void llama_cleanup_cpu_dequant_elimination_module(void);

#define DECODE_FORBIDDEN_CPU_DEQUANT(tensor, type) \
    do { \
        if (g_cpu_dequant_elimination_engine && llama_is_decode_phase_active()) { \
            if (!llama_attempt_cpu_dequantization(tensor, type)) { \
                return -1; \
            } \
        } \
    } while(0)

#define BLOCK_HOST_BUFFER_IN_DECODE(tensor) \
    do { \
        if (g_cpu_dequant_elimination_engine && llama_is_decode_phase_active()) { \
            if (!llama_attempt_host_buffer_allocation(tensor)) { \
                return -1; \
            } \
        } \
    } while(0)

#define ENFORCE_GPU_RESIDENCY(tensor) \
    do { \
        if (g_cpu_dequant_elimination_engine) { \
            llama_record_gpu_residency_enforce(tensor); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
