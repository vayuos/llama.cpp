#pragma once

/**
 * GPU Allocation Alignment for LLAMA
 *
 * All GPU-resident buffers used in the decode path must be allocated with
 * explicit alignment guarantees suitable for:
 * - Tensor Core MMA instructions
 * - Vectorized global memory loads
 * - Fused quantized kernels (MMQ)
 * - Flash-attention kernels
 *
 * Misaligned allocations reduce memory throughput, break coalescing, and
 * degrade occupancy. Alignment must be structurally enforced, not assumed.
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

// Alignment policy constants
#define GPU_ALIGNMENT 256                    // Minimum global alignment
#define TENSOR_CORE_ALIGNMENT 128            // Tensor Core MMA alignment
#define KV_CACHE_ALIGNMENT 128               // KV cache stride alignment
#define QUANTIZED_BLOCK_ALIGNMENT 64         // Quantized block alignment
#define LOGITS_ALIGNMENT 128                 // Logits buffer alignment
#define SAMPLING_ALIGNMENT 128               // Sampling buffer alignment

typedef enum {
    ALIGNMENT_ENFORCEMENT_UNINITIALIZED = 0,
    ALIGNMENT_ENFORCEMENT_PLANNING = 1,
    ALIGNMENT_ENFORCEMENT_VALIDATION = 2,
    ALIGNMENT_ENFORCEMENT_LOCKED = 3
} alignment_enforcement_phase;

typedef struct {
    void * original_ptr;                     // Original allocation pointer
    void * aligned_ptr;                      // Aligned allocation pointer
    size_t requested_size;                   // Originally requested size
    size_t allocated_size;                   // Actually allocated size
    size_t alignment;                        // Alignment requirement
    const char * buffer_name;                // Buffer identifier
    bool is_aligned;                         // Alignment verified
} aligned_allocation_record;

typedef struct {
    bool global_alignment_enforced;
    bool tensor_core_alignment_enforced;
    bool kv_cache_alignment_enforced;
    bool quantized_block_alignment_enforced;
    bool logits_alignment_enforced;
    bool sampling_alignment_enforced;
    bool no_misaligned_views;
    bool pinned_memory_used;
    uint64_t alignment_check_timestamp_ns;
} gpu_alignment_enforcement_config;

typedef struct {
    const char * buffer_name;
    size_t size;
    size_t required_alignment;
    size_t actual_alignment;
    bool alignment_satisfied;
    bool misaligned_access_attempted;
} allocation_alignment_status;

typedef struct {
    size_t total_allocations;
    size_t aligned_allocations;
    size_t misaligned_allocations;
    size_t alignment_violations;
    size_t memory_coalescing_failures;
    bool all_allocations_aligned;
} gpu_alignment_validation_result;

class gpu_allocation_alignment_engine {
private:
    gpu_alignment_enforcement_config immutable_config;
    std::vector<aligned_allocation_record> allocation_records;
    std::vector<allocation_alignment_status> alignment_status;
    std::map<void *, aligned_allocation_record> active_allocations;

    std::atomic<alignment_enforcement_phase> current_phase;
    std::atomic<bool> alignment_enforced;
    std::atomic<bool> validation_complete;

    std::atomic<uint32_t> allocation_count;
    std::atomic<uint32_t> aligned_count;
    std::atomic<uint32_t> misaligned_blocks;
    std::atomic<uint32_t> alignment_violations;

public:
    gpu_allocation_alignment_engine();

    bool initialize();
    bool enable_strict_mode(bool enable);

    bool validate_alignment_policy();
    bool enforce_global_alignment();
    bool enforce_tensor_core_alignment();
    bool enforce_kv_cache_alignment();
    bool enforce_quantized_alignment();
    bool enforce_logits_alignment();
    bool enforce_sampling_alignment();

    void * allocate_aligned(const char * buffer_name, size_t size, size_t alignment);
    bool deallocate_aligned(void * ptr);
    bool validate_buffer_alignment(const char * buffer_name, void * ptr, size_t size, size_t alignment);

    bool attempt_misaligned_view(const char * buffer_name, size_t offset);
    bool verify_tensor_alignment(const char * tensor_name, void * data, size_t stride);
    bool verify_kv_cache_alignment(size_t n_layer, size_t stride);
    bool verify_quantized_alignment(const char * quant_format, void * data, size_t block_size);

    const gpu_alignment_enforcement_config & get_config() const { return immutable_config; }
    bool is_alignment_enforced() const { return alignment_enforced.load(); }
    bool is_validation_complete() const { return validation_complete.load(); }
    alignment_enforcement_phase get_current_phase() const { return current_phase.load(); }

    void record_allocation(const char * name, void * orig_ptr, void * aligned_ptr,
                          size_t requested_size, size_t allocated_size, size_t alignment);
    void record_alignment_status(const char * name, size_t size, size_t required_align,
                                size_t actual_align, bool satisfied);
    void record_alignment_violation(const char * buffer_name);

    size_t get_allocation_count() const { return allocation_records.size(); }
    size_t get_aligned_count() const { return aligned_count.load(); }
    size_t get_misaligned_count() const { return misaligned_blocks.load(); }

    std::vector<aligned_allocation_record> get_allocation_records() const { return allocation_records; }
    std::vector<allocation_alignment_status> get_alignment_status() const { return alignment_status; }

    gpu_alignment_validation_result validate_gpu_alignment() const;
    bool verify_all_allocations_aligned() const;
    bool verify_no_misaligned_views() const;
    bool verify_coalescing_safe() const;
    bool verify_tensor_core_compatible() const;
};

class gpu_alignment_guard {
private:
    bool guard_active;

public:
    gpu_alignment_guard();
    ~gpu_alignment_guard();

    bool is_guard_active() const;
};

extern gpu_allocation_alignment_engine * g_gpu_allocation_alignment_engine;

bool llama_init_gpu_allocation_alignment();
bool llama_enable_alignment_strict_mode(bool enable);

bool llama_validate_alignment_policy();
bool llama_enforce_global_alignment();
bool llama_enforce_tensor_core_alignment();
bool llama_enforce_kv_cache_alignment();
bool llama_enforce_quantized_alignment();
bool llama_enforce_logits_alignment();
bool llama_enforce_sampling_alignment();

void * llama_allocate_aligned(const char * buffer_name, size_t size, size_t alignment);
bool llama_deallocate_aligned(void * ptr);
bool llama_validate_buffer_alignment(const char * buffer_name, void * ptr, size_t size, size_t alignment);

bool llama_attempt_misaligned_view(const char * buffer_name, size_t offset);
bool llama_verify_tensor_alignment(const char * tensor_name, void * data, size_t stride);
bool llama_verify_kv_cache_alignment(size_t n_layer, size_t stride);
bool llama_verify_quantized_alignment(const char * quant_format, void * data, size_t block_size);

bool llama_is_alignment_enforced();
bool llama_is_alignment_validation_complete();

void llama_record_allocation(const char * name, void * orig_ptr, void * aligned_ptr,
                            size_t requested_size, size_t allocated_size, size_t alignment);
void llama_record_alignment_status(const char * name, size_t size, size_t required_align,
                                  size_t actual_align, bool satisfied);
void llama_record_alignment_violation(const char * buffer_name);

bool llama_validate_gpu_alignment();
bool llama_verify_all_allocations_aligned();
bool llama_verify_no_misaligned_views();
bool llama_verify_coalescing_safe();
bool llama_verify_tensor_core_compatible();

void llama_print_alignment_enforcement_status();
void llama_print_allocation_alignment_summary();
void llama_print_allocation_records();
void llama_print_alignment_violations();

#define ASSERT_GLOBAL_ALIGNMENT(ptr, size) \
    do { \
        if (g_gpu_allocation_alignment_engine && !llama_validate_buffer_alignment("check", ptr, size, GPU_ALIGNMENT)) { \
            return -1; \
        } \
    } while(0)

#define ASSERT_TENSOR_CORE_ALIGNMENT(ptr) \
    do { \
        if (g_gpu_allocation_alignment_engine && ((uintptr_t)(ptr) % TENSOR_CORE_ALIGNMENT != 0)) { \
            return -1; \
        } \
    } while(0)

#define ASSERT_KV_CACHE_ALIGNMENT(ptr) \
    do { \
        if (g_gpu_allocation_alignment_engine && ((uintptr_t)(ptr) % KV_CACHE_ALIGNMENT != 0)) { \
            return -1; \
        } \
    } while(0)

#define GUARD_MISALIGNED_VIEW(buffer_name, offset) \
    do { \
        if (g_gpu_allocation_alignment_engine && !llama_attempt_misaligned_view(buffer_name, offset)) { \
            return -1; \
        } \
    } while(0)

#define ALLOCATE_ALIGNED(name, size, alignment) \
    llama_allocate_aligned(name, size, alignment)

#define DEALLOCATE_ALIGNED(ptr) \
    llama_deallocate_aligned(ptr)

#ifdef __cplusplus
}
bool llama_init_gpu_allocation_alignment_module(void);
void llama_cleanup_gpu_allocation_alignment_module(void);
#endif
