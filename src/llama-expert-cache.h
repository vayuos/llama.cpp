#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

/**
 * llama_expert_cache
 * 
 * Manages a set of VRAM "slots" for MoE experts.
 * Provides asynchronous streaming of expert weights from host memory to GPU.
 */
class llama_expert_cache {
public:
    struct expert_slot {
        int32_t expert_id = -1; // -1 if empty
        struct ggml_tensor * t_gate = nullptr;
        struct ggml_tensor * t_down = nullptr;
        struct ggml_tensor * t_up   = nullptr;
        uint64_t last_used_at = 0;
    };

    llama_expert_cache(
        ggml_backend_t backend,
        size_t n_expert_total,
        size_t n_slots,
        size_t expert_size_bytes,
        int64_t ne0_gate, int64_t ne1_gate,
        int64_t ne0_up,   int64_t ne1_up,
        int64_t ne0_down, int64_t ne1_down);

    ~llama_expert_cache();

    // Acquire a slot for an expert. Trigger async load if not resident.
    // Returns slot index.
    int32_t acquire_expert(int32_t expert_id, void * stream);

    // Get tensors for a resident expert
    expert_slot & get_slot(int32_t slot_idx);

    // Update LRU state
    void touch_expert(int32_t expert_id, uint64_t timestamp);

    // Total experts managed
    size_t get_n_experts() const { return n_expert_total; }
    size_t get_n_slots() const { return n_slots; }

    struct ggml_tensor * get_mapping_tensor() const { return t_mapping; }
    struct ggml_tensor * get_up_slots()      const { return t_up_slots; }
    struct ggml_tensor * get_gate_slots()    const { return t_gate_slots; }
    struct ggml_tensor * get_down_slots()    const { return t_down_slots; }

private:
    ggml_backend_t backend;
    size_t n_expert_total;
    size_t n_slots;
    size_t expert_size_bytes;
    int64_t ne0_gate, ne1_gate;
    int64_t ne0_up,   ne1_up;
    int64_t ne0_down, ne1_down;

    std::vector<expert_slot> slots;
    std::map<int32_t, int32_t> expert_to_slot; // expert_id -> slot_idx

    // Pinned host memory for all experts (the "backing store")
    void * host_data = nullptr; 
    
    // Device memory for slots
    void * device_data = nullptr;

    struct ggml_context * ctx = nullptr;
    struct ggml_tensor * t_mapping = nullptr; // [n_expert_total] -> slot_idx
    struct ggml_tensor * t_up_slots   = nullptr; // [ne0, ne1, n_slots]
    struct ggml_tensor * t_gate_slots = nullptr;
    struct ggml_tensor * t_down_slots = nullptr;

    int32_t find_lru_slot() const;
};
