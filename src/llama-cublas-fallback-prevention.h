#pragma once

/**
 * cuBLAS Fallback Prevention for LLAMA
 *
 * cuBLAS must never be selected or re-selected during the decode phase.
 * Decode backend binding must be immutable and enforced with hard guards.
 * No fallback to dense CUDA or cuBLAS allowed once decode begins.
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
    FALLBACK_PREVENT_UNINITIALIZED = 0,
    FALLBACK_PREVENT_STARTUP = 1,
    FALLBACK_PREVENT_BINDING = 2,
    FALLBACK_PREVENT_LOCKED = 3
} fallback_prevent_phase;

typedef enum {
    CUBLAS_BACKEND_NONE = 0,
    CUBLAS_BACKEND_MMQ = 1,
    CUBLAS_BACKEND_CUTLASS = 2,
    CUBLAS_BACKEND_CUBLAS = 3,
    CUBLAS_BACKEND_DENSE_CUDA = 4,
    CUBLAS_BACKEND_CPU = 5
} cublas_backend_selection;

typedef struct {
    cublas_backend_selection decode_backend;
    bool backend_locked;
    bool cublas_blocked;
    bool dense_cuda_blocked;
    bool cpu_blocked;
    bool graph_immutable;
    uint64_t lock_timestamp_ns;
} fallback_prevention_config;

typedef struct {
    const char * file_path;
    int line_number;
    const char * function_name;
    const char * attempted_backend;
    const char * reason;
    bool is_during_decode;
    bool was_blocked;
} fallback_attempt_record;

typedef struct {
    bool backend_locked;
    bool decode_backend_immutable;
    uint32_t fallback_attempts;
    uint32_t fallback_blocks;
    uint32_t backend_mismatches;
    bool no_mid_decode_switching;
} fallback_prevention_validation_result;

class fallback_prevention_engine {
private:
    fallback_prevention_config immutable_config;
    std::vector<fallback_attempt_record> attempt_audit_log;
    std::vector<fallback_attempt_record> blocked_attempt_log;

    std::atomic<fallback_prevent_phase> current_phase;
    std::atomic<bool> backend_locked;
    std::atomic<bool> strict_mode;

    std::atomic<uint32_t> fallback_attempts;
    std::atomic<uint32_t> fallback_blocks;
    std::atomic<uint32_t> backend_switches_blocked;
    std::atomic<uint32_t> graph_rebuilds_prevented;

public:
    fallback_prevention_engine();

    bool initialize();
    bool enable_strict_mode(bool enable);

    bool bind_decode_backend(cublas_backend_selection backend);
    bool lock_decode_backend();
    bool validate_no_mid_decode_switch();

    bool attempt_backend_switch(const char * new_backend);
    bool attempt_graph_rebuild();
    bool attempt_cublas_fallback(const char * reason);
    bool attempt_dense_cuda_fallback(const char * reason);
    bool attempt_cpu_fallback(const char * reason);

    const fallback_prevention_config & get_config() const { return immutable_config; }
    bool is_backend_locked() const { return backend_locked.load(); }
    fallback_prevent_phase get_current_phase() const { return current_phase.load(); }

    void record_fallback_attempt(const char * file, int line, const char * func,
                                const char * backend, const char * reason, bool in_decode);
    void record_blocked_fallback(const fallback_attempt_record & record);

    size_t get_attempt_count() const { return attempt_audit_log.size(); }
    size_t get_blocked_count() const { return blocked_attempt_log.size(); }
    std::vector<fallback_attempt_record> get_attempts() const { return attempt_audit_log; }
    std::vector<fallback_attempt_record> get_blocked() const { return blocked_attempt_log; }

    void record_fallback_block() { fallback_blocks.fetch_add(1); }
    void record_backend_switch_block() { backend_switches_blocked.fetch_add(1); }
    void record_graph_rebuild_prevent() { graph_rebuilds_prevented.fetch_add(1); }

    fallback_prevention_validation_result validate_fallback_prevention() const;
    bool verify_backend_locked() const;
    bool verify_no_cublas_used() const;
    bool verify_no_backend_switching() const;
    bool verify_no_graph_rebuild() const;
    bool verify_decode_path_immutable() const;
};

class decode_backend_guard {
private:
    cublas_backend_selection bound_backend;
    bool is_locked;

public:
    decode_backend_guard(cublas_backend_selection backend);
    ~decode_backend_guard();

    bool is_backend_bound() const;
    bool attempt_switch(cublas_backend_selection new_backend);
};

extern fallback_prevention_engine * g_fallback_prevention_engine;

bool llama_init_fallback_prevention();
bool llama_enable_fallback_prevention_strict_mode(bool enable);

bool llama_bind_decode_backend(cublas_backend_selection backend);
bool llama_lock_decode_backend();
bool llama_validate_no_mid_decode_switch();

bool llama_attempt_backend_switch(const char * new_backend);
bool llama_attempt_graph_rebuild();
bool llama_attempt_cublas_fallback(const char * reason);
bool llama_attempt_dense_cuda_fallback(const char * reason);
bool llama_attempt_cpu_fallback(const char * reason);

cublas_backend_selection llama_get_decode_backend();
bool llama_is_decode_backend_locked();
bool llama_is_cublas_fallback_blocked();
bool llama_is_dense_cuda_fallback_blocked();
bool llama_is_cpu_fallback_blocked();

void llama_record_fallback_attempt(const char * file, int line, const char * func,
                                  const char * backend, const char * reason);

bool llama_validate_fallback_prevention();
bool llama_validate_backend_locked();
bool llama_validate_no_cublas();
bool llama_validate_no_backend_switching();

void llama_print_fallback_attempt_audit();
void llama_print_fallback_prevention_validation();
void llama_print_backend_lock_status();
void llama_dump_fallback_statistics();

// Self-test module initialization (internal use)
bool llama_init_fallback_prevention_module(void);
void llama_cleanup_fallback_prevention_module(void);

#define DECODE_BACKEND_LOCK() \
    do { \
        if (g_fallback_prevention_engine && !g_fallback_prevention_engine->is_backend_locked()) { \
            return -1; \
        } \
    } while(0)

#define REJECT_BACKEND_SWITCH(new_backend) \
    do { \
        if (g_fallback_prevention_engine && !g_fallback_prevention_engine->attempt_backend_switch(new_backend)) { \
            return -1; \
        } \
    } while(0)

#define REJECT_CUBLAS_FALLBACK(reason) \
    do { \
        if (g_fallback_prevention_engine && !g_fallback_prevention_engine->attempt_cublas_fallback(reason)) { \
            return -1; \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
