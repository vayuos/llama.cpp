#include "llama-expert-cache.h"
#include "llama-impl.h"
#include <cstring>
#include <algorithm>

#include "ggml-backend.h"

llama_expert_cache::llama_expert_cache(
    ggml_backend_t backend,
    size_t n_expert_total,
    size_t n_slots,
    size_t expert_size_bytes,
    int64_t ne0_gate, int64_t ne1_gate,
    int64_t ne0_up,   int64_t ne1_up,
    int64_t ne0_down, int64_t ne1_down)
    : backend(backend), n_expert_total(n_expert_total), n_slots(n_slots), expert_size_bytes(expert_size_bytes),
      ne0_gate(ne0_gate), ne1_gate(ne1_gate), ne0_up(ne0_up), ne1_up(ne1_up), ne0_down(ne0_down), ne1_down(ne1_down) {

    slots.resize(n_slots);

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * n_slots * 4 + 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctx = ggml_init(params);

    host_data = malloc(n_expert_total * expert_size_bytes);

    t_mapping = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_expert_total);

    ggml_type wtype = GGML_TYPE_F32; 

    t_gate_slots = ggml_new_tensor_3d(ctx, wtype, ne0_gate, ne1_gate, n_slots);
    t_up_slots   = ggml_new_tensor_3d(ctx, wtype, ne0_up, ne1_up, n_slots);
    t_down_slots = ggml_new_tensor_3d(ctx, wtype, ne0_down, ne1_down, n_slots);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    device_data = (void*) buf;

    std::vector<int32_t> initial_mapping(n_expert_total, -1);
    ggml_backend_tensor_set(t_mapping, initial_mapping.data(), 0, n_expert_total * 4);

    for (size_t i = 0; i < n_slots; ++i) {
        slots[i].expert_id = -1;
        slots[i].last_used_at = 0;
    }
}

llama_expert_cache::~llama_expert_cache() {
    if (ctx) ggml_free(ctx);
    if (host_data) free(host_data);
    if (device_data) ggml_backend_buffer_free((ggml_backend_buffer_t)device_data);
}

int32_t llama_expert_cache::acquire_expert(int32_t expert_id, void * stream) {
    (void)stream;
    auto it = expert_to_slot.find(expert_id);
    if (it != expert_to_slot.end()) {
        return it->second;
    }

    int32_t slot_idx = find_lru_slot();
    
    if (slots[slot_idx].expert_id != -1) {
        expert_to_slot.erase(slots[slot_idx].expert_id);
    }

    slots[slot_idx].expert_id = expert_id;
    expert_to_slot[expert_id] = slot_idx;

    int32_t val = slot_idx;
    ggml_backend_tensor_set(t_mapping, &val, expert_id * 4, 4);

    void * src = (uint8_t*)host_data + (size_t)expert_id * expert_size_bytes;
    
    size_t gate_size = ne0_gate * ne1_gate * ggml_type_size(GGML_TYPE_F32);
    size_t up_size   = ne0_up * ne1_up * ggml_type_size(GGML_TYPE_F32);
    size_t down_size = ne0_down * ne1_down * ggml_type_size(GGML_TYPE_F32);

    void * src_gate = src;
    void * src_up   = (uint8_t*)src + gate_size;
    void * src_down = (uint8_t*)src + gate_size + up_size;

    ggml_backend_tensor_set(t_gate_slots, src_gate, slot_idx * gate_size, gate_size);
    ggml_backend_tensor_set(t_up_slots,   src_up,   slot_idx * up_size,   up_size);
    ggml_backend_tensor_set(t_down_slots, src_down, slot_idx * down_size, down_size);


    return slot_idx;
}

llama_expert_cache::expert_slot & llama_expert_cache::get_slot(int32_t slot_idx) {
    return slots[slot_idx];
}

void llama_expert_cache::touch_expert(int32_t expert_id, uint64_t timestamp) {
    auto it = expert_to_slot.find(expert_id);
    if (it != expert_to_slot.end()) {
        slots[it->second].last_used_at = timestamp;
    }
}

int32_t llama_expert_cache::find_lru_slot() const {
    int32_t best_idx = 0;
    uint64_t min_time = UINT64_MAX;

    for (size_t i = 0; i < n_slots; ++i) {
        if (slots[i].expert_id == -1) {
            return (int32_t)i; // Free slot found
        }
        if (slots[i].last_used_at < min_time) {
            min_time = slots[i].last_used_at;
            best_idx = (int32_t)i;
        }
    }

    return best_idx;
}
