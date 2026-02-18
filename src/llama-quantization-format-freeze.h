#pragma once

/**
 * Quantization Format Freeze for LLAMA
 *
 * Quantization format must be treated as an immutable decode invariant.
 * No quantization reinterpretation, promotion, fallback, or format switching
 * may occur once decode begins.
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
    QUANT_FORMAT_NONE = 0,
    QUANT_FORMAT_Q4_0 = 1,
    QUANT_FORMAT_Q4_1 = 2,
    QUANT_FORMAT_Q5_0 = 3,
    QUANT_FORMAT_Q5_1 = 4,
    QUANT_FORMAT_Q8_0 = 5,
    QUANT_FORMAT_Q6_K = 6,
    QUANT_FORMAT_Q2_K = 7,
    QUANT_FORMAT_Q3_K = 8,
    QUANT_FORMAT_Q4_K = 9,
    QUANT_FORMAT_Q5_K = 10,
    QUANT_FORMAT_IQ2_XXS = 11,
    QUANT_FORMAT_IQ3_XXS = 12,
    QUANT_FORMAT_F16 = 13,
    QUANT_FORMAT_F32 = 14
} quantization_format_type;

typedef enum {
    QUANT_FREEZE_UNINITIALIZED = 0,
    QUANT_FREEZE_LOAD = 1,
    QUANT_FREEZE_VALIDATION = 2,
    QUANT_FREEZE_LOCKED = 3
} quantization_freeze_phase;

typedef struct {
    quantization_format_type primary_format;
    quantization_format_type kv_cache_format;
    quantization_format_type attention_format;
    bool format_locked;
    bool promotion_disabled;
    bool dequant_disabled;
    bool format_drift_prevented;
    uint64_t lock_timestamp_ns;
} quantization_format_config;

typedef struct {
    const char * file_path;
    int line_number;
    const char * function_name;
    const char * layer_name;
    quantization_format_type format_attempted;
    quantization_format_type format_expected;
    bool is_violation;
} quantization_format_violation_record;

typedef struct {
    bool format_locked;
    bool backend_compatible;
    bool kv_format_stable;
    uint32_t promotion_attempts_blocked;
    uint32_t format_mismatches;
    uint32_t dequant_attempts_blocked;
} quantization_format_validation_result;

class quantization_format_freeze_engine {
private:
    quantization_format_config immutable_quant_config;
    std::vector<quantization_format_violation_record> violation_log;
    std::map<std::string, quantization_format_type> layer_format_map;

    std::atomic<quantization_freeze_phase> current_phase;
    std::atomic<bool> format_locked;
    std::atomic<bool> strict_enforcement;

    std::atomic<uint32_t> promotion_blocks;
    std::atomic<uint32_t> format_mismatches;
    std::atomic<uint32_t> dequant_blocks;
    std::atomic<uint32_t> format_validations;

public:
    quantization_format_freeze_engine();

    bool initialize();
    bool enable_strict_mode(bool enable);

    bool resolve_quantization_format_at_load(quantization_format_type primary,
                                            quantization_format_type kv,
                                            quantization_format_type attention);
    bool validate_format_backend_compatibility(quantization_format_type format);
    bool lock_quantization_format();

    bool attempt_quantization_promotion(quantization_format_type from_format,
                                       quantization_format_type to_format);
    bool attempt_dequantization();
    bool attempt_format_drift(const char * layer_name);

    const quantization_format_config & get_config() const { return immutable_quant_config; }
    bool is_format_locked() const { return format_locked.load(); }
    quantization_freeze_phase get_current_phase() const { return current_phase.load(); }

    void record_layer_format(const char * layer_name, quantization_format_type format);
    void record_format_violation(const char * file, int line, const char * func,
                                const char * layer, quantization_format_type attempted,
                                quantization_format_type expected);

    size_t get_violation_count() const { return violation_log.size(); }
    std::vector<quantization_format_violation_record> get_violations() const { return violation_log; }

    void record_promotion_block() { promotion_blocks.fetch_add(1); }
    void record_format_mismatch() { format_mismatches.fetch_add(1); }
    void record_dequant_block() { dequant_blocks.fetch_add(1); }
    void record_validation_pass() { format_validations.fetch_add(1); }

    quantization_format_validation_result validate_quantization_format_freeze() const;
    bool verify_format_locked() const;
    bool verify_no_promotion() const;
    bool verify_kv_format_stable() const;
    bool verify_backend_compatibility() const;
    bool verify_decode_start_invariants() const;
};

class quantization_format_guard {
private:
    quantization_format_type bound_format;
    bool is_locked;

public:
    quantization_format_guard(quantization_format_type format);
    ~quantization_format_guard();

    bool is_format_valid() const;
    bool attempt_promotion(quantization_format_type new_format);
};

extern quantization_format_freeze_engine * g_quantization_format_freeze_engine;

bool llama_init_quantization_format_freeze();
bool llama_enable_quantization_format_strict_mode(bool enable);

bool llama_resolve_quantization_format_at_load(quantization_format_type primary,
                                              quantization_format_type kv,
                                              quantization_format_type attention);
bool llama_validate_format_backend_compatibility(quantization_format_type format);
bool llama_lock_quantization_format();

bool llama_attempt_quantization_promotion(quantization_format_type from_format,
                                         quantization_format_type to_format);
bool llama_attempt_dequantization();
bool llama_attempt_format_drift(const char * layer_name);

quantization_format_type llama_get_primary_quantization_format();
quantization_format_type llama_get_kv_cache_format();
quantization_format_type llama_get_attention_format();
bool llama_is_quantization_format_locked();

void llama_record_layer_quantization_format(const char * layer_name, quantization_format_type format);

bool llama_validate_quantization_format_freeze();
bool llama_validate_format_locked();
bool llama_validate_no_promotion();
bool llama_validate_kv_stable();
bool llama_validate_decode_start_invariants();

void llama_print_quantization_format_violations();
void llama_print_quantization_format_validation();
void llama_print_quantization_format_snapshot();
void llama_dump_quantization_format_statistics();

#define QUANTIZATION_FORMAT_CHECK(expected_format) \
    do { \
        if (g_quantization_format_freeze_engine && !g_quantization_format_freeze_engine->is_format_locked()) { \
            return -1; \
        } \
    } while(0)

#define REJECT_PROMOTION(from_fmt, to_fmt) \
    do { \
        if (g_quantization_format_freeze_engine && !g_quantization_format_freeze_engine->attempt_quantization_promotion(from_fmt, to_fmt)) { \
            return -1; \
        } \
    } while(0)

#define REJECT_DEQUANT() \
    do { \
        if (g_quantization_format_freeze_engine && !g_quantization_format_freeze_engine->attempt_dequantization()) { \
            return -1; \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
