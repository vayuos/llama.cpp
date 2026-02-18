#pragma once

/**
 * GPU Memory Fragmentation Monitoring for LLAMA
 *
 * Guarantee that GPU memory remains structurally stable and fragmentation-free
 * across long-running decode sessions.
 *
 * Even without allocations per token, fragmentation can arise from:
 * - Improper workspace reuse
 * - Context recreation
 * - CUDA graph capture pools
 * - Asynchronous allocator churn
 * - Server request lifecycle
 *
 * Fragmentation increases allocation latency, reduces contiguous block availability,
 * and can silently degrade performance.
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
    FRAGMENTATION_MONITOR_UNINITIALIZED = 0,
    FRAGMENTATION_MONITOR_PLANNING = 1,
    FRAGMENTATION_MONITOR_BASELINE = 2,
    FRAGMENTATION_MONITOR_MONITORING = 3,
    FRAGMENTATION_MONITOR_LOCKED = 4
} fragmentation_monitor_phase;

typedef struct {
    uint64_t timestamp_ns;
    size_t total_memory;
    size_t free_memory;
    size_t used_memory;
    size_t largest_contiguous_block;
    double fragmentation_ratio;
    bool stable;
} memory_snapshot;

typedef struct {
    size_t initial_total;
    size_t initial_free;
    size_t initial_used;
    size_t initial_largest_block;
    uint64_t baseline_timestamp_ns;
    bool topology_locked;
    bool async_allocator_disabled;
} gpu_memory_baseline;

typedef struct {
    const char * buffer_name;
    void * base_pointer;
    size_t size;
    bool is_persistent;
    bool is_locked;
    uint64_t allocation_timestamp_ns;
} gpu_buffer_pointer_record;

typedef struct {
    uint32_t total_snapshots;
    uint32_t stable_snapshots;
    uint32_t drift_detected;
    uint32_t fragmentation_warnings;
    uint32_t allocation_events;
    uint32_t pool_growth_events;
    bool memory_stable;
} gpu_fragmentation_validation_result;

class gpu_memory_fragmentation_monitor {
private:
    gpu_memory_baseline baseline_state;
    std::vector<memory_snapshot> memory_snapshots;
    std::vector<gpu_buffer_pointer_record> buffer_pointers;
    std::map<const char *, gpu_buffer_pointer_record> pointer_registry;
    std::vector<gpu_fragmentation_validation_result> validation_results;

    std::atomic<fragmentation_monitor_phase> current_phase;
    std::atomic<bool> monitoring_active;
    std::atomic<bool> topology_locked;
    std::atomic<bool> memory_stable;

    std::atomic<uint32_t> snapshot_count;
    std::atomic<uint32_t> drift_events;
    std::atomic<uint32_t> fragmentation_events;
    std::atomic<uint32_t> allocation_events;

public:
    gpu_memory_fragmentation_monitor();

    bool initialize();
    bool enable_strict_mode(bool enable);

    bool freeze_allocation_topology();
    bool disable_async_allocator_growth();
    bool establish_baseline();
    bool begin_monitoring();
    bool end_monitoring();

    bool record_memory_snapshot();
    bool detect_fragmentation_risk();
    bool validate_memory_stability();
    bool verify_pointer_integrity();

    bool register_buffer_pointer(const char * name, void * ptr, size_t size, bool persistent);
    bool verify_pointer_unchanged(const char * buffer_name, void * current_ptr);

    bool attempt_new_allocation(const char * buffer_name, size_t size);
    bool detect_pool_growth();
    bool detect_memory_drift();

    const gpu_memory_baseline & get_baseline() const { return baseline_state; }
    bool is_topology_locked() const { return topology_locked.load(); }
    bool is_memory_stable() const { return memory_stable.load(); }
    fragmentation_monitor_phase get_current_phase() const { return current_phase.load(); }

    void record_allocation_event(const char * buffer_name);
    void record_pool_growth_event();
    void record_drift_event();
    void record_fragmentation_event();

    size_t get_snapshot_count() const { return memory_snapshots.size(); }
    size_t get_drift_count() const { return drift_events.load(); }
    size_t get_fragmentation_count() const { return fragmentation_events.load(); }

    std::vector<memory_snapshot> get_memory_snapshots() const { return memory_snapshots; }
    std::vector<gpu_buffer_pointer_record> get_buffer_pointers() const { return buffer_pointers; }

    gpu_fragmentation_validation_result validate_fragmentation_stability() const;
    bool verify_long_run_stability() const;
    bool verify_no_hidden_allocations() const;
    bool verify_pointer_stability() const;
    bool verify_kernel_performance_stable() const;
};

class memory_fragmentation_guard {
private:
    bool guard_active;

public:
    memory_fragmentation_guard();
    ~memory_fragmentation_guard();

    bool is_guard_active() const;
};

extern gpu_memory_fragmentation_monitor * g_gpu_memory_fragmentation_monitor;

bool llama_init_gpu_memory_fragmentation_monitor();
bool llama_enable_fragmentation_strict_mode(bool enable);

bool llama_freeze_allocation_topology();
bool llama_disable_async_allocator_growth();
bool llama_establish_memory_baseline();
bool llama_begin_memory_monitoring();
bool llama_end_memory_monitoring();

bool llama_record_memory_snapshot();
bool llama_detect_fragmentation_risk();
bool llama_validate_memory_stability();
bool llama_verify_pointer_integrity();

bool llama_register_buffer_pointer(const char * name, void * ptr, size_t size, bool persistent);
bool llama_verify_pointer_unchanged(const char * buffer_name, void * current_ptr);

bool llama_attempt_new_allocation(const char * buffer_name, size_t size);
bool llama_detect_pool_growth();
bool llama_detect_memory_drift();

bool llama_is_topology_locked();
bool llama_is_memory_stable();

void llama_record_allocation_event(const char * buffer_name);
void llama_record_pool_growth_event();
void llama_record_drift_event();
void llama_record_fragmentation_event();

bool llama_validate_fragmentation_stability();
bool llama_verify_long_run_stability();
bool llama_verify_no_hidden_allocations();
bool llama_verify_pointer_stability();
bool llama_verify_kernel_performance_stable();

void llama_print_fragmentation_monitor_status();
void llama_print_memory_baseline();
void llama_print_memory_snapshots();
void llama_print_pointer_integrity_status();
void llama_print_fragmentation_analysis();

#define ASSERT_TOPOLOGY_LOCKED() \
    do { \
        if (g_gpu_memory_fragmentation_monitor && !llama_is_topology_locked()) { \
            return -1; \
        } \
    } while(0)

#define ASSERT_MEMORY_STABLE() \
    do { \
        if (g_gpu_memory_fragmentation_monitor && !llama_is_memory_stable()) { \
            return -1; \
        } \
    } while(0)

#define GUARD_ALLOCATION_POST_INIT(buffer_name, size) \
    do { \
        if (g_gpu_memory_fragmentation_monitor && !llama_attempt_new_allocation(buffer_name, size)) { \
            return -1; \
        } \
    } while(0)

#define REGISTER_BUFFER_POINTER(name, ptr, size, persistent) \
    do { \
        if (g_gpu_memory_fragmentation_monitor) { \
            llama_register_buffer_pointer(name, ptr, size, persistent); \
        } \
    } while(0)

#define VERIFY_POINTER_UNCHANGED(name, current_ptr) \
    do { \
        if (g_gpu_memory_fragmentation_monitor && !llama_verify_pointer_unchanged(name, current_ptr)) { \
            return -1; \
        } \
    } while(0)

#ifdef __cplusplus
}
bool llama_init_gpu_memory_fragmentation_monitor_module(void);
void llama_cleanup_gpu_memory_fragmentation_monitor_module(void);
#endif
