#pragma once

/**
 * Decode Buffer Freeze for LLAMA
 *
 * Ensure every buffer used during decode is fully allocated, sized, bound,
 * and immutable before the first token is generated.
 * After context initialization completes, no buffer resizing, relocation,
 * rebinding, or structural mutation is allowed.
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
    BUFFER_FREEZE_UNINITIALIZED = 0,
    BUFFER_FREEZE_PLANNING = 1,
    BUFFER_FREEZE_ALLOCATION = 2,
    BUFFER_FREEZE_BINDING = 3,
    BUFFER_FREEZE_LOCKED = 4
} buffer_freeze_phase;

typedef struct {
    size_t transformer_activation_bytes;
    size_t attention_scratch_bytes;
    size_t mlp_scratch_bytes;
    size_t logits_buffer_bytes;
    size_t sampling_buffer_bytes;
    size_t kv_cache_bytes;
    size_t cuda_workspace_bytes;
    size_t graph_scratch_bytes;
    size_t streaming_buffer_bytes;
    uint64_t total_allocated_bytes;
} decode_buffer_allocation;

typedef struct {
    bool transformer_activations_frozen;
    bool attention_scratch_frozen;
    bool mlp_scratch_frozen;
    bool logits_buffer_frozen;
    bool sampling_buffers_frozen;
    bool kv_cache_structure_frozen;
    bool cuda_workspace_frozen;
    bool graph_tensors_bound;
    bool decode_graph_frozen;
    bool decode_memory_locked;
    uint64_t freeze_timestamp_ns;
} decode_buffer_freeze_config;

typedef struct {
    const char * buffer_name;
    size_t buffer_size;
    void * buffer_ptr;
    bool is_gpu_resident;
    bool is_frozen;
    bool relocation_attempted;
} buffer_binding_record;

typedef struct {
    size_t total_buffers;
    size_t frozen_buffers;
    size_t relocation_violations;
    size_t resize_violations;
    size_t rebind_violations;
    bool all_buffers_frozen;
} buffer_freeze_validation_result;

class decode_buffer_freeze_engine {
private:
    decode_buffer_freeze_config immutable_config;
    decode_buffer_allocation allocated_buffers;
    std::vector<buffer_binding_record> buffer_bindings;
    std::vector<buffer_binding_record> relocation_attempts;
    std::vector<buffer_binding_record> resize_attempts;

    std::atomic<buffer_freeze_phase> current_phase;
    std::atomic<bool> buffers_frozen;
    std::atomic<bool> graph_frozen;
    std::atomic<bool> structure_locked;

    std::atomic<uint32_t> buffer_count;
    std::atomic<uint32_t> relocation_blocks;
    std::atomic<uint32_t> resize_blocks;
    std::atomic<uint32_t> rebind_blocks;

public:
    decode_buffer_freeze_engine();

    bool initialize();
    bool enable_strict_mode(bool enable);

    bool plan_buffer_allocation(size_t n_ctx, size_t n_layer, size_t n_embd,
                               size_t max_batch, size_t max_seq_len);
    bool allocate_all_decode_buffers();
    bool bind_graph_tensors();
    bool freeze_decode_graph();
    bool lock_buffer_structure();

    bool attempt_buffer_relocation(const char * buffer_name);
    bool attempt_buffer_resize(const char * buffer_name, size_t new_size);
    bool attempt_tensor_rebinding(const char * tensor_name);

    const decode_buffer_freeze_config & get_config() const { return immutable_config; }
    const decode_buffer_allocation & get_allocation() const { return allocated_buffers; }
    bool are_buffers_frozen() const { return buffers_frozen.load(); }
    bool is_graph_frozen() const { return graph_frozen.load(); }
    bool is_structure_locked() const { return structure_locked.load(); }
    buffer_freeze_phase get_current_phase() const { return current_phase.load(); }

    void record_buffer_binding(const char * name, size_t size, void * ptr, bool gpu_resident);
    void record_relocation_attempt(const char * buffer_name);
    void record_resize_attempt(const char * buffer_name, size_t new_size);
    void record_rebind_attempt(const char * tensor_name);

    size_t get_buffer_count() const { return buffer_bindings.size(); }
    size_t get_relocation_count() const { return relocation_attempts.size(); }
    size_t get_resize_count() const { return resize_attempts.size(); }

    std::vector<buffer_binding_record> get_buffer_bindings() const { return buffer_bindings; }
    std::vector<buffer_binding_record> get_relocation_attempts() const { return relocation_attempts; }
    std::vector<buffer_binding_record> get_resize_attempts() const { return resize_attempts; }

    buffer_freeze_validation_result validate_buffer_freeze() const;
    bool verify_all_buffers_frozen() const;
    bool verify_no_relocation() const;
    bool verify_no_resizing() const;
    bool verify_graph_frozen() const;
    bool verify_structure_immutable() const;
};

class buffer_freeze_guard {
private:
    bool guard_active;

public:
    buffer_freeze_guard();
    ~buffer_freeze_guard();

    bool is_guard_active() const;
};

extern decode_buffer_freeze_engine * g_decode_buffer_freeze_engine;

bool llama_init_decode_buffer_freeze();
bool llama_enable_buffer_freeze_strict_mode(bool enable);

bool llama_plan_buffer_allocation(size_t n_ctx, size_t n_layer, size_t n_embd,
                                 size_t max_batch, size_t max_seq_len);
bool llama_allocate_all_decode_buffers();
bool llama_bind_graph_tensors();
bool llama_freeze_decode_graph();
bool llama_lock_buffer_structure();

bool llama_attempt_buffer_relocation(const char * buffer_name);
bool llama_attempt_buffer_resize(const char * buffer_name, size_t new_size);
bool llama_attempt_tensor_rebinding(const char * tensor_name);

bool llama_are_buffers_frozen();
bool llama_is_graph_frozen();
bool llama_is_structure_locked();

void llama_record_buffer_binding(const char * name, size_t size, void * ptr, bool gpu_resident);
void llama_record_relocation_attempt(const char * buffer_name);
void llama_record_resize_attempt(const char * buffer_name, size_t new_size);
void llama_record_rebind_attempt(const char * tensor_name);

bool llama_validate_buffer_freeze();
bool llama_verify_all_buffers_frozen();
bool llama_verify_no_relocation();
bool llama_verify_no_resizing();
bool llama_verify_graph_frozen();
bool llama_verify_structure_immutable();

void llama_print_buffer_freeze_status();
void llama_print_buffer_allocation_summary();
void llama_print_buffer_bindings();
void llama_print_buffer_freeze_violations();

#define ASSERT_BUFFERS_FROZEN() \
    do { \
        if (g_decode_buffer_freeze_engine && !llama_are_buffers_frozen()) { \
            return -1; \
        } \
    } while(0)

#define ASSERT_GRAPH_FROZEN() \
    do { \
        if (g_decode_buffer_freeze_engine && !llama_is_graph_frozen()) { \
            return -1; \
        } \
    } while(0)

#define ASSERT_STRUCTURE_LOCKED() \
    do { \
        if (g_decode_buffer_freeze_engine && !llama_is_structure_locked()) { \
            return -1; \
        } \
    } while(0)

#define GUARD_BUFFER_RELOCATION(buffer_name) \
    do { \
        if (g_decode_buffer_freeze_engine && !llama_attempt_buffer_relocation(buffer_name)) { \
            return -1; \
        } \
    } while(0)

#define GUARD_BUFFER_RESIZE(buffer_name, new_size) \
    do { \
        if (g_decode_buffer_freeze_engine && !llama_attempt_buffer_resize(buffer_name, new_size)) { \
            return -1; \
        } \
    } while(0)

#define GUARD_TENSOR_REBINDING(tensor_name) \
    do { \
        if (g_decode_buffer_freeze_engine && !llama_attempt_tensor_rebinding(tensor_name)) { \
            return -1; \
        } \
    } while(0)

#ifdef __cplusplus
}
bool llama_init_decode_buffer_freeze_module(void);
void llama_cleanup_decode_buffer_freeze_module(void);
#endif
