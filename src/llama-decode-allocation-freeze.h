#pragma once

/**
 * Decode-Time Allocation Freeze for LLAMA
 *
 * Guarantee no dynamic memory allocation occurs on decode-critical path.
 * All memory must be preallocated and fixed-layout.
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
    ALLOC_FREEZE_UNINITIALIZED = 0,
    ALLOC_FREEZE_INIT_PHASE = 1,
    ALLOC_FREEZE_PREALLOCATE = 2,
    ALLOC_FREEZE_DECODE_PHASE = 3,
    ALLOC_FREEZE_LOCKED = 4
} allocation_freeze_phase;

typedef struct {
    size_t transformer_activations_bytes;
    size_t attention_buffer_bytes;
    size_t ffn_intermediate_bytes;
    size_t logits_buffer_bytes;
    size_t sampling_buffer_bytes;
    size_t kv_cache_bytes;
    size_t cuda_workspace_bytes;
    size_t graph_scratch_bytes;
    uint64_t total_preallocated_bytes;
} decode_buffer_allocation_plan;

typedef struct {
    bool decode_memory_frozen;
    bool all_buffers_preallocated;
    bool kv_cache_locked;
    bool graph_resources_frozen;
    bool allocator_guarded;
    uint64_t freeze_timestamp_ns;
} allocation_freeze_config;

typedef struct {
    const char * file_path;
    int line_number;
    const char * function_name;
    const char * allocation_type;
    size_t allocation_size;
    bool is_during_decode;
    bool was_blocked;
} decode_allocation_attempt_record;

typedef struct {
    bool zero_cpu_allocations;
    bool zero_gpu_allocations;
    uint32_t allocation_blocks;
    uint32_t pre_decode_allocations;
    uint32_t decode_phase_allocations;
    bool memory_footprint_stable;
} allocation_freeze_validation_result;

class decode_allocation_freeze_engine {
private:
    allocation_freeze_config immutable_config;
    decode_buffer_allocation_plan preallocated_plan;
    std::vector<decode_allocation_attempt_record> allocation_audit_log;
    std::vector<decode_allocation_attempt_record> blocked_allocation_log;

    std::atomic<allocation_freeze_phase> current_phase;
    std::atomic<bool> memory_frozen;
    std::atomic<bool> allocator_guarded;
    std::atomic<bool> strict_enforcement;

    std::atomic<uint32_t> cpu_allocation_blocks;
    std::atomic<uint32_t> gpu_allocation_blocks;
    std::atomic<uint32_t> pre_decode_allocations;
    std::atomic<uint32_t> decode_allocations;

public:
    decode_allocation_freeze_engine();

    bool initialize();
    bool enable_strict_mode(bool enable);

    bool compute_buffer_allocation_plan(size_t n_ctx, size_t n_layer,
                                       size_t n_embd, size_t quant_format);
    bool preallocate_all_decode_buffers();
    bool guard_allocator();
    bool enter_decode_phase();
    bool exit_decode_phase();

    bool attempt_cpu_allocation(const char * file, int line, const char * func,
                               const char * alloc_type, size_t size);
    bool attempt_gpu_allocation(const char * file, int line, const char * func,
                               const char * alloc_type, size_t size);
    bool attempt_vector_growth(const char * vector_name);
    bool attempt_kv_cache_reallocation();

    const allocation_freeze_config & get_config() const { return immutable_config; }
    const decode_buffer_allocation_plan & get_allocation_plan() const { return preallocated_plan; }
    bool is_memory_frozen() const { return memory_frozen.load(); }
    allocation_freeze_phase get_current_phase() const { return current_phase.load(); }

    void record_allocation_attempt(const char * file, int line, const char * func,
                                  const char * alloc_type, size_t size, bool is_decode);
    void record_blocked_allocation(const decode_allocation_attempt_record & record);

    size_t get_audit_count() const { return allocation_audit_log.size(); }
    size_t get_blocked_count() const { return blocked_allocation_log.size(); }
    std::vector<decode_allocation_attempt_record> get_audit_log() const { return allocation_audit_log; }
    std::vector<decode_allocation_attempt_record> get_blocked() const { return blocked_allocation_log; }

    allocation_freeze_validation_result validate_allocation_freeze() const;
    bool verify_zero_decode_allocations() const;
    bool verify_memory_footprint_stable() const;
    bool verify_all_buffers_preallocated() const;
    bool verify_kv_cache_immutable() const;
};

class allocation_freeze_guard {
private:
    bool guard_active;

public:
    allocation_freeze_guard();
    ~allocation_freeze_guard();

    bool is_guard_active() const;
};

extern decode_allocation_freeze_engine * g_decode_allocation_freeze_engine;

bool llama_init_decode_allocation_freeze();
bool llama_enable_allocation_freeze_strict_mode(bool enable);

bool llama_compute_buffer_allocation_plan(size_t n_ctx, size_t n_layer,
                                         size_t n_embd, size_t quant_format);
bool llama_preallocate_all_decode_buffers();
bool llama_guard_allocator();
bool llama_enter_decode_phase();
bool llama_exit_decode_phase();

bool llama_attempt_cpu_allocation(const char * file, int line, const char * func,
                                 const char * alloc_type, size_t size);
bool llama_attempt_gpu_allocation(const char * file, int line, const char * func,
                                 const char * alloc_type, size_t size);
bool llama_attempt_vector_growth(const char * vector_name);
bool llama_attempt_kv_cache_reallocation();

bool llama_is_memory_frozen();
bool llama_is_allocator_guarded();

void llama_record_allocation_attempt(const char * file, int line, const char * func,
                                    const char * alloc_type, size_t size);

bool llama_validate_allocation_freeze();
bool llama_verify_zero_decode_allocations();
bool llama_verify_memory_stable();
bool llama_verify_buffers_preallocated();
bool llama_verify_kv_immutable();

void llama_print_allocation_freeze_status();
void llama_print_buffer_allocation_plan();
void llama_print_allocation_audit_log();
void llama_print_allocation_freeze_validation();

// Self-test module initialization (internal use)
bool llama_init_decode_allocation_freeze_module(void);
void llama_cleanup_decode_allocation_freeze_module(void);

#define GUARD_CPU_ALLOCATION(alloc_type, size) \
    do { \
        if (g_decode_allocation_freeze_engine && !llama_attempt_cpu_allocation(__FILE__, __LINE__, __FUNCTION__, alloc_type, size)) { \
            return -1; \
        } \
    } while(0)

#define GUARD_GPU_ALLOCATION(alloc_type, size) \
    do { \
        if (g_decode_allocation_freeze_engine && !llama_attempt_gpu_allocation(__FILE__, __LINE__, __FUNCTION__, alloc_type, size)) { \
            return -1; \
        } \
    } while(0)

#define FREEZE_MEMORY() \
    do { \
        if (g_decode_allocation_freeze_engine) { \
            g_decode_allocation_freeze_engine->enter_decode_phase(); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
