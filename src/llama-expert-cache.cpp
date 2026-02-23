#include "llama-expert-cache.h"
#include "llama-impl.h"
#include <cstring>
#include <algorithm>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

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

    // Initialize GGML context for metadata tensors
    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * n_slots * 4 + 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ctx = ggml_init(params);

#ifdef GGML_USE_CUDA
    // Allocate pinned host memory for backing store
    size_t total_host_size = n_expert_total * expert_size_bytes;
    cudaHostAlloc(&host_data, total_host_size, cudaHostAllocDefault);
    
    // Allocate device memory for VRAM slots
    size_t total_device_size = n_slots * expert_size_bytes;
    cudaMalloc(&device_data, total_device_size);
#else
    host_data = malloc(n_expert_total * expert_size_bytes);
    device_data = malloc(n_slots * expert_size_bytes);
#endif

    // Initialize slots with pointers into device_data
    t_mapping = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_expert_total);
#ifdef GGML_USE_CUDA
    cudaMalloc(&t_mapping->data, n_expert_total * 4);
    // Initialize mapping with -1
    std::vector<int32_t> initial_mapping(n_expert_total, -1);
    cudaMemcpy(t_mapping->data, initial_mapping.data(), n_expert_total * 4, cudaMemcpyHostToDevice);
#else
    t_mapping->data = malloc(n_expert_total * 4);
    memset(t_mapping->data, -1, n_expert_total * 4);
#endif

    // Assume weights are F16? Let's use F32 for now to match ggml_new_tensor call
    // In a real scenario, this would match model->type
    ggml_type wtype = GGML_TYPE_F32; 

    t_gate_slots = ggml_new_tensor_3d(ctx, wtype, ne0_gate, ne1_gate, n_slots);
    t_gate_slots->data = device_data; // Simple offset for now
    
    t_up_slots   = ggml_new_tensor_3d(ctx, wtype, ne0_up, ne1_up, n_slots);
    t_up_slots->data = (uint8_t*)device_data + (ne0_gate * ne1_gate * 4 * n_slots);
    
    t_down_slots = ggml_new_tensor_3d(ctx, wtype, ne0_down, ne1_down, n_slots);
    t_down_slots->data = (uint8_t*)device_data + ((ne0_gate * ne1_gate + ne0_up * ne1_up) * 4 * n_slots);

    for (size_t i = 0; i < n_slots; ++i) {
        slots[i].expert_id = -1;
        slots[i].last_used_at = 0;
    }
}

llama_expert_cache::~llama_expert_cache() {
    if (ctx) ggml_free(ctx);
#ifdef GGML_USE_CUDA
    if (host_data) cudaFreeHost(host_data);
    if (device_data) cudaFree(device_data);
#else
    if (host_data) free(host_data);
    if (device_data) free(device_data);
#endif
}

int32_t llama_expert_cache::acquire_expert(int32_t expert_id, void * stream) {
    auto it = expert_to_slot.find(expert_id);
    if (it != expert_to_slot.end()) {
        return it->second;
    }

    // Expert not in VRAM, find a slot to evict
    int32_t slot_idx = find_lru_slot();
    
    // Evict old expert
    if (slots[slot_idx].expert_id != -1) {
        expert_to_slot.erase(slots[slot_idx].expert_id);
    }

    // Load new expert weights asynchronously
    slots[slot_idx].expert_id = expert_id;
    expert_to_slot[expert_id] = slot_idx;

    // Update GPU mapping tensor
#ifdef GGML_USE_CUDA
    cudaMemcpyAsync((int32_t*)t_mapping->data + expert_id, &slot_idx, 4, cudaMemcpyHostToDevice, (cudaStream_t)stream);
#else
    ((int32_t*)t_mapping->data)[expert_id] = slot_idx;
#endif

#ifdef GGML_USE_CUDA
    void * src = (uint8_t*)host_data + (size_t)expert_id * expert_size_bytes;
    void * dst = (uint8_t*)device_data + (size_t)slot_idx * expert_size_bytes;
    cudaMemcpyAsync(dst, src, expert_size_bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
#else
    void * src = (uint8_t*)host_data + (size_t)expert_id * expert_size_bytes;
    void * dst = (uint8_t*)device_data + (size_t)slot_idx * expert_size_bytes;
    memcpy(dst, src, expert_size_bytes);
#endif

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
