#pragma once

/**
 * Fused Quantized Kernel Coverage Validation for LLAMA
 *
 * Formally prove that every quantized decode-path operation
 * is handled by a fused CUDA kernel with no fallback paths.
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
    KERNEL_COVERAGE_UNINITIALIZED = 0,
    KERNEL_COVERAGE_ENUMERATION = 1,
    KERNEL_COVERAGE_MAPPING = 2,
    KERNEL_COVERAGE_VALIDATION = 3,
    KERNEL_COVERAGE_LOCKED = 4
} kernel_coverage_phase;

#include "llama-quantization-format-freeze.h"

typedef quantization_format_type active_quant_format;

typedef struct {
    active_quant_format format;
    bool has_cuda_kernel;
    bool is_fused;
    const char * kernel_symbol;
    uint64_t kernel_invocations;
} quant_format_kernel_mapping;

typedef struct {
    uint32_t active_format_count;
    uint32_t fused_kernel_count;
    uint32_t cpu_fallback_count;
    uint32_t dense_gemv_count;
    uint32_t split_kernel_count;
    bool all_formats_mapped;
} kernel_coverage_validation_result;

typedef struct {
    active_quant_format format;
    bool kernel_found;
    bool is_fused;
    const char * kernel_name;
    const char * backend_type;
} kernel_symbol_record;

class fused_kernel_coverage_engine {
private:
    std::set<active_quant_format> active_quant_formats;
    std::map<active_quant_format, quant_format_kernel_mapping> format_kernel_map;
    std::vector<kernel_symbol_record> symbol_resolution_log;

    std::atomic<kernel_coverage_phase> current_phase;
    std::atomic<bool> coverage_locked;
    std::atomic<bool> validation_complete;

    std::atomic<uint32_t> fused_kernel_launches;
    std::atomic<uint32_t> cpu_kernel_launches;
    std::atomic<uint32_t> dense_gemv_launches;
    std::atomic<uint32_t> split_kernel_launches;

public:
    fused_kernel_coverage_engine();

    bool initialize();

    bool enumerate_active_quantization_formats(const std::vector<active_quant_format> & formats);
    bool map_format_to_cuda_kernel(active_quant_format format, const char * kernel_symbol);
    bool prohibit_non_fused_paths();
    bool validate_kernel_symbol_resolution();
    bool validate_kernel_dispatch_integrity();

    const std::map<active_quant_format, quant_format_kernel_mapping> & get_format_map() const {
        return format_kernel_map;
    }
    bool is_coverage_locked() const { return coverage_locked.load(); }
    kernel_coverage_phase get_current_phase() const { return current_phase.load(); }

    void record_fused_kernel_launch(active_quant_format format);
    void record_cpu_kernel_launch(active_quant_format format);
    void record_dense_gemv_launch(active_quant_format format);
    void record_split_kernel_launch(active_quant_format format);

    void record_kernel_symbol_resolution(active_quant_format format, bool found, bool fused,
                                        const char * kernel_name, const char * backend);

    size_t get_active_format_count() const { return active_quant_formats.size(); }
    size_t get_symbol_resolution_count() const { return symbol_resolution_log.size(); }

    kernel_coverage_validation_result validate_kernel_coverage() const;
    bool verify_all_formats_have_kernels() const;
    bool verify_no_cpu_fallback() const;
    bool verify_no_dense_gemv() const;
    bool verify_no_split_kernels() const;
    bool verify_fused_only() const;
};

class kernel_coverage_validator {
private:
    bool validation_passed;

public:
    kernel_coverage_validator();
    ~kernel_coverage_validator();

    bool run_coverage_validation();
    bool get_validation_status() const;
};

extern fused_kernel_coverage_engine * g_fused_kernel_coverage_engine;

bool llama_init_fused_kernel_coverage();

bool llama_enumerate_active_quantization_formats(const std::vector<active_quant_format> & formats);
bool llama_map_format_to_cuda_kernel(active_quant_format format, const char * kernel_symbol);
bool llama_prohibit_non_fused_paths();
bool llama_validate_kernel_symbol_resolution();
bool llama_validate_kernel_dispatch_integrity();

void llama_record_fused_kernel_launch(active_quant_format format);
void llama_record_cpu_kernel_launch(active_quant_format format);
void llama_record_dense_gemv_launch(active_quant_format format);
void llama_record_split_kernel_launch(active_quant_format format);

void llama_record_kernel_symbol_resolution(active_quant_format format, bool found, bool fused,
                                          const char * kernel_name, const char * backend);

bool llama_validate_fused_kernel_coverage();
bool llama_verify_all_formats_have_kernels();
bool llama_verify_no_cpu_fallback();
bool llama_verify_no_dense_gemv();
bool llama_verify_no_split_kernels();
bool llama_verify_fused_only();

void llama_print_kernel_coverage_status();
void llama_print_kernel_symbol_resolution();
void llama_print_kernel_dispatch_statistics();
void llama_print_coverage_validation_result();

#define ASSERT_FUSED_KERNEL_COVERAGE() \
    do { \
        if (g_fused_kernel_coverage_engine && !g_fused_kernel_coverage_engine->is_coverage_locked()) { \
            return -1; \
        } \
    } while(0)

#define RECORD_QUANT_KERNEL_LAUNCH(format, is_fused, is_cpu) \
    do { \
        if (g_fused_kernel_coverage_engine) { \
            if (is_cpu) { \
                llama_record_cpu_kernel_launch(format); \
            } else if (is_fused) { \
                llama_record_fused_kernel_launch(format); \
            } \
        } \
    } while(0)

#ifdef __cplusplus
}
bool llama_init_fused_kernel_coverage_module(void);
void llama_cleanup_fused_kernel_coverage_module(void);
#endif
