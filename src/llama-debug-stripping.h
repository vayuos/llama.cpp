#pragma once

/**
 * Debug and Tracing Code Stripping for LLAMA
 *
 * All debug, tracing, and diagnostic instrumentation removed from decode path.
 * Decode executes with zero diagnostic branching.
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
    DEBUG_STRIPPING_UNINITIALIZED = 0,
    DEBUG_STRIPPING_STARTUP = 1,
    DEBUG_STRIPPING_VALIDATION = 2,
    DEBUG_STRIPPING_LOCKED = 3
} debug_stripping_phase;

typedef struct {
    bool debug_mode_enabled;
    bool logging_enabled;
    bool tracing_enabled;
    bool assertions_enabled;
    bool profiling_enabled;
    bool timing_instrumentation_enabled;
    bool verbose_mode;
    int log_level;
} debug_config;

typedef struct {
    const char * file_path;
    int line_number;
    const char * function_name;
    const char * debug_construct_type;
    const char * location_description;
    bool is_in_decode_path;
    bool is_violation;
} debug_audit_entry;

typedef struct {
    bool all_debug_stripped;
    uint32_t debug_constructs_found;
    uint32_t in_decode_path_count;
    uint32_t constructs_stripped;
    bool decode_path_clean;
    bool release_build_verified;
} debug_stripping_validation_result;

class debug_stripping_engine {
private:
    debug_config immutable_debug_config;
    std::vector<debug_audit_entry> debug_audit_log;
    std::vector<debug_audit_entry> violation_log;

    std::atomic<debug_stripping_phase> current_phase;
    std::atomic<bool> debug_config_frozen;
    std::atomic<bool> strict_validation;

    std::atomic<uint64_t> total_debug_constructs;
    std::atomic<uint64_t> decode_path_constructs;
    std::atomic<uint64_t> constructs_removed;

public:
    debug_stripping_engine();

    bool initialize();
    bool validate_debug_configuration();
    bool verify_no_debug_in_decode();
    bool verify_release_build();
    bool verify_no_assertions_in_loops();

    const debug_config & get_immutable_debug_config() const { return immutable_debug_config; }
    bool is_debug_config_frozen() const { return debug_config_frozen.load(); }
    debug_stripping_phase get_current_phase() const { return current_phase.load(); }

    void lock_debug_configuration();
    void audit_debug_construct(const char * file, int line, const char * func,
                               const char * construct_type, const char * location,
                               bool in_decode_path);
    void record_violation(const debug_audit_entry & entry);
    void record_construct_removal();

    size_t get_audit_count() const { return debug_audit_log.size(); }
    size_t get_violation_count() const { return violation_log.size(); }
    std::vector<debug_audit_entry> get_audit_log() const { return debug_audit_log; }
    std::vector<debug_audit_entry> get_violations() const { return violation_log; }

    uint64_t get_debug_constructs_found() const { return total_debug_constructs.load(); }
    uint64_t get_decode_path_constructs() const { return decode_path_constructs.load(); }
    uint64_t get_constructs_removed() const { return constructs_removed.load(); }

    void record_debug_construct() { total_debug_constructs.fetch_add(1); }
    void record_decode_path_construct() { decode_path_constructs.fetch_add(1); }

    debug_stripping_validation_result validate_debug_stripping() const;
    bool verify_no_logging_in_decode() const;
    bool verify_no_tracing_in_decode() const;
    bool verify_no_timing_in_decode() const;
};

extern debug_stripping_engine * g_debug_stripping_engine;

bool llama_init_debug_stripping();
bool llama_validate_debug_configuration();
bool llama_verify_release_build();
void llama_lock_debug_configuration();

bool llama_is_debug_mode_enabled();
bool llama_is_logging_enabled();
bool llama_is_tracing_enabled();

void llama_audit_debug_construct(const char * file, int line, const char * func,
                                  const char * construct_type, const char * location);

bool llama_validate_debug_stripping();
bool llama_validate_no_debug_in_decode();

void llama_print_debug_audit_report();
void llama_print_debug_stripping_validation();
void llama_dump_debug_configuration();
void llama_dump_debug_statistics();

#define DEBUG_CONSTRUCT_GUARD(construct_type, location) \
    do { \
        if (g_debug_stripping_engine) { \
            g_debug_stripping_engine->record_debug_construct(); \
        } \
    } while(0)

#define DECODE_PATH_DEBUG_CHECK() \
    do { \
        if (g_debug_stripping_engine && g_debug_stripping_engine->is_debug_config_frozen()) { \
            g_debug_stripping_engine->record_decode_path_construct(); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
