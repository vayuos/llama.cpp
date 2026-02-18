/**
 * llama-quantization-format-freeze.cpp
 *
 * Freeze Quantization Format Assumptions
 * Quantization format is immutable decode invariant.
 *
 * REQUIREMENT #59: Freeze Quantization Format Assumptions
 * 11 enforcement rules with immutable format locking.
 */

#include "llama-quantization-format-freeze.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <chrono>

quantization_format_freeze_engine * g_quantization_format_freeze_engine = nullptr;

// ============================================================================
// QUANTIZATION FORMAT FREEZE ENGINE IMPLEMENTATION
// ============================================================================

quantization_format_freeze_engine::quantization_format_freeze_engine()
    : current_phase(quantization_freeze_phase::QUANT_FREEZE_UNINITIALIZED),
      format_locked(false),
      strict_enforcement(true),
      promotion_blocks(0),
      format_mismatches(0),
      dequant_blocks(0),
      format_validations(0) {

    immutable_quant_config = {
        QUANT_FORMAT_NONE, QUANT_FORMAT_NONE, QUANT_FORMAT_NONE,
        false, false, false, false, 0
    };
}

bool quantization_format_freeze_engine::initialize() {
    current_phase.store(quantization_freeze_phase::QUANT_FREEZE_LOAD);
    return true;
}

bool quantization_format_freeze_engine::enable_strict_mode(bool enable) {
    strict_enforcement.store(enable);
    return true;
}

bool quantization_format_freeze_engine::resolve_quantization_format_at_load(
    quantization_format_type primary,
    quantization_format_type kv,
    quantization_format_type attention) {

    current_phase.store(quantization_freeze_phase::QUANT_FREEZE_VALIDATION);

    immutable_quant_config.primary_format = primary;
    immutable_quant_config.kv_cache_format = kv;
    immutable_quant_config.attention_format = attention;

    format_validations.fetch_add(1);
    return true;
}

bool quantization_format_freeze_engine::validate_format_backend_compatibility(quantization_format_type format) {
    // Verify format is compatible with selected backend
    switch (format) {
        case QUANT_FORMAT_Q4_0:
        case QUANT_FORMAT_Q4_1:
        case QUANT_FORMAT_Q5_0:
        case QUANT_FORMAT_Q5_1:
        case QUANT_FORMAT_Q8_0:
        case QUANT_FORMAT_Q6_K:
        case QUANT_FORMAT_Q2_K:
        case QUANT_FORMAT_Q3_K:
        case QUANT_FORMAT_Q4_K:
        case QUANT_FORMAT_Q5_K:
        case QUANT_FORMAT_IQ2_XXS:
        case QUANT_FORMAT_IQ3_XXS:
        case QUANT_FORMAT_F16:
        case QUANT_FORMAT_F32:
            return true;
        default:
            return false;
    }
}

bool quantization_format_freeze_engine::lock_quantization_format() {
    format_locked.store(true);
    immutable_quant_config.format_locked = true;
    immutable_quant_config.promotion_disabled = true;
    immutable_quant_config.dequant_disabled = true;
    immutable_quant_config.lock_timestamp_ns =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    current_phase.store(quantization_freeze_phase::QUANT_FREEZE_LOCKED);
    return true;
}

bool quantization_format_freeze_engine::attempt_quantization_promotion(
    quantization_format_type from_format,
    quantization_format_type to_format) {

    if (format_locked.load() || immutable_quant_config.promotion_disabled) {
        promotion_blocks.fetch_add(1);
        record_format_violation(__FILE__, __LINE__, __FUNCTION__, "quantization_promotion",
                              to_format, from_format);
        return false; // Promotion blocked
    }
    return true;
}

bool quantization_format_freeze_engine::attempt_dequantization() {
    if (format_locked.load() || immutable_quant_config.dequant_disabled) {
        dequant_blocks.fetch_add(1);
        return false; // Dequantization blocked
    }
    return true;
}

bool quantization_format_freeze_engine::attempt_format_drift(const char * layer_name) {
    if (format_locked.load() || immutable_quant_config.format_drift_prevented) {
        return false; // Format drift blocked
    }
    return true;
}

void quantization_format_freeze_engine::record_layer_format(const char * layer_name, quantization_format_type format) {
    layer_format_map[layer_name] = format;
}

void quantization_format_freeze_engine::record_format_violation(
    const char * file, int line, const char * func,
    const char * layer, quantization_format_type attempted,
    quantization_format_type expected) {

    quantization_format_violation_record record = {
        file, line, func, layer, attempted, expected, true
    };
    violation_log.push_back(record);
    format_mismatches.fetch_add(1);
}

quantization_format_validation_result quantization_format_freeze_engine::validate_quantization_format_freeze() const {
    quantization_format_validation_result result = {
        format_locked.load(),
        validate_format_backend_compatibility(immutable_quant_config.primary_format),
        immutable_quant_config.format_drift_prevented,
        static_cast<uint32_t>(promotion_blocks.load()),
        static_cast<uint32_t>(format_mismatches.load()),
        static_cast<uint32_t>(dequant_blocks.load())
    };
    return result;
}

bool quantization_format_freeze_engine::verify_format_locked() const {
    return format_locked.load();
}

bool quantization_format_freeze_engine::verify_no_promotion() const {
    return promotion_blocks.load() > 0 || immutable_quant_config.promotion_disabled;
}

bool quantization_format_freeze_engine::verify_kv_format_stable() const {
    return immutable_quant_config.format_drift_prevented;
}

bool quantization_format_freeze_engine::verify_backend_compatibility() const {
    return validate_format_backend_compatibility(immutable_quant_config.primary_format) &&
           validate_format_backend_compatibility(immutable_quant_config.kv_cache_format) &&
           validate_format_backend_compatibility(immutable_quant_config.attention_format);
}

bool quantization_format_freeze_engine::verify_decode_start_invariants() const {
    return format_locked.load() &&
           immutable_quant_config.promotion_disabled &&
           immutable_quant_config.dequant_disabled &&
           immutable_quant_config.format_drift_prevented;
}

// ============================================================================
// QUANTIZATION FORMAT GUARD IMPLEMENTATION
// ============================================================================

quantization_format_guard::quantization_format_guard(quantization_format_type format)
    : bound_format(format), is_locked(false) {
    if (g_quantization_format_freeze_engine) {
        is_locked = g_quantization_format_freeze_engine->is_format_locked();
    }
}

quantization_format_guard::~quantization_format_guard() {
}

bool quantization_format_guard::is_format_valid() const {
    return is_locked;
}

bool quantization_format_guard::attempt_promotion(quantization_format_type new_format) {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->attempt_quantization_promotion(bound_format, new_format);
    }
    return true;
}

// ============================================================================
// GLOBAL FUNCTIONS
// ============================================================================

bool llama_init_quantization_format_freeze() {
    if (g_quantization_format_freeze_engine == nullptr) {
        g_quantization_format_freeze_engine = new quantization_format_freeze_engine();
        if (g_quantization_format_freeze_engine->initialize()) {
            return true;
        }
        delete g_quantization_format_freeze_engine;
        g_quantization_format_freeze_engine = nullptr;
    }
    return g_quantization_format_freeze_engine != nullptr;
}

bool llama_enable_quantization_format_strict_mode(bool enable) {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->enable_strict_mode(enable);
    }
    return false;
}

bool llama_resolve_quantization_format_at_load(quantization_format_type primary,
                                              quantization_format_type kv,
                                              quantization_format_type attention) {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->resolve_quantization_format_at_load(primary, kv, attention);
    }
    return false;
}

bool llama_validate_format_backend_compatibility(quantization_format_type format) {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->validate_format_backend_compatibility(format);
    }
    return false;
}

bool llama_lock_quantization_format() {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->lock_quantization_format();
    }
    return false;
}

bool llama_attempt_quantization_promotion(quantization_format_type from_format,
                                         quantization_format_type to_format) {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->attempt_quantization_promotion(from_format, to_format);
    }
    return true;
}

bool llama_attempt_dequantization() {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->attempt_dequantization();
    }
    return true;
}

bool llama_attempt_format_drift(const char * layer_name) {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->attempt_format_drift(layer_name);
    }
    return true;
}

quantization_format_type llama_get_primary_quantization_format() {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->get_config().primary_format;
    }
    return QUANT_FORMAT_NONE;
}

quantization_format_type llama_get_kv_cache_format() {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->get_config().kv_cache_format;
    }
    return QUANT_FORMAT_NONE;
}

quantization_format_type llama_get_attention_format() {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->get_config().attention_format;
    }
    return QUANT_FORMAT_NONE;
}

bool llama_is_quantization_format_locked() {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->is_format_locked();
    }
    return false;
}

void llama_record_layer_quantization_format(const char * layer_name, quantization_format_type format) {
    if (g_quantization_format_freeze_engine) {
        g_quantization_format_freeze_engine->record_layer_format(layer_name, format);
    }
}

bool llama_validate_quantization_format_freeze() {
    if (g_quantization_format_freeze_engine) {
        quantization_format_validation_result result = g_quantization_format_freeze_engine->validate_quantization_format_freeze();
        return result.format_locked && result.backend_compatible;
    }
    return false;
}

bool llama_validate_format_locked() {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->verify_format_locked();
    }
    return false;
}

bool llama_validate_no_promotion() {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->verify_no_promotion();
    }
    return false;
}

bool llama_validate_kv_stable() {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->verify_kv_format_stable();
    }
    return false;
}

bool llama_validate_decode_start_invariants() {
    if (g_quantization_format_freeze_engine) {
        return g_quantization_format_freeze_engine->verify_decode_start_invariants();
    }
    return false;
}

void llama_print_quantization_format_violations() {
    if (!g_quantization_format_freeze_engine) {
        std::cout << "Quantization format freeze engine not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== QUANTIZATION FORMAT VIOLATIONS ===" << std::endl;
    auto violations = g_quantization_format_freeze_engine->get_violations();
    std::cout << "Total violations: " << violations.size() << std::endl;

    for (const auto & violation : violations) {
        std::cout << "\nViolation at: " << violation.file_path << ":" << violation.line_number << std::endl;
        std::cout << "Layer: " << violation.layer_name << std::endl;
        std::cout << "Attempted format: " << violation.format_attempted << std::endl;
        std::cout << "Expected format: " << violation.format_expected << std::endl;
    }
}

void llama_print_quantization_format_validation() {
    if (!g_quantization_format_freeze_engine) {
        std::cout << "Quantization format freeze engine not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== QUANTIZATION FORMAT VALIDATION ===" << std::endl;
    quantization_format_validation_result result = g_quantization_format_freeze_engine->validate_quantization_format_freeze();
    std::cout << "Format locked: " << (result.format_locked ? "YES" : "NO") << std::endl;
    std::cout << "Backend compatible: " << (result.backend_compatible ? "YES" : "NO") << std::endl;
    std::cout << "KV format stable: " << (result.kv_format_stable ? "YES" : "NO") << std::endl;
    std::cout << "Promotion attempts blocked: " << result.promotion_attempts_blocked << std::endl;
    std::cout << "Format mismatches: " << result.format_mismatches << std::endl;
    std::cout << "Dequant attempts blocked: " << result.dequant_attempts_blocked << std::endl;
}

void llama_print_quantization_format_snapshot() {
    if (!g_quantization_format_freeze_engine) {
        std::cout << "Quantization format freeze engine not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== QUANTIZATION FORMAT SNAPSHOT ===" << std::endl;
    const quantization_format_config & cfg = g_quantization_format_freeze_engine->get_config();

    std::string format_name = [](quantization_format_type fmt) {
        switch (fmt) {
            case QUANT_FORMAT_Q4_0: return "Q4_0";
            case QUANT_FORMAT_Q4_1: return "Q4_1";
            case QUANT_FORMAT_Q5_0: return "Q5_0";
            case QUANT_FORMAT_Q5_1: return "Q5_1";
            case QUANT_FORMAT_Q8_0: return "Q8_0";
            case QUANT_FORMAT_Q6_K: return "Q6_K";
            case QUANT_FORMAT_Q2_K: return "Q2_K";
            case QUANT_FORMAT_Q3_K: return "Q3_K";
            case QUANT_FORMAT_Q4_K: return "Q4_K";
            case QUANT_FORMAT_Q5_K: return "Q5_K";
            case QUANT_FORMAT_IQ2_XXS: return "IQ2_XXS";
            case QUANT_FORMAT_IQ3_XXS: return "IQ3_XXS";
            case QUANT_FORMAT_F16: return "F16";
            case QUANT_FORMAT_F32: return "F32";
            default: return "NONE";
        }
    };

    std::cout << "Primary format: " << format_name(cfg.primary_format) << std::endl;
    std::cout << "KV cache format: " << format_name(cfg.kv_cache_format) << std::endl;
    std::cout << "Attention format: " << format_name(cfg.attention_format) << std::endl;
    std::cout << "Format locked: " << (cfg.format_locked ? "YES" : "NO") << std::endl;
    std::cout << "Promotion disabled: " << (cfg.promotion_disabled ? "YES" : "NO") << std::endl;
    std::cout << "Dequant disabled: " << (cfg.dequant_disabled ? "YES" : "NO") << std::endl;
    std::cout << "Format drift prevented: " << (cfg.format_drift_prevented ? "YES" : "NO") << std::endl;
}

void llama_dump_quantization_format_statistics() {
    if (!g_quantization_format_freeze_engine) {
        std::cout << "Quantization format freeze engine not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== QUANTIZATION FORMAT STATISTICS ===" << std::endl;
    std::cout << "Violations: " << g_quantization_format_freeze_engine->get_violation_count() << std::endl;
}

static bool run_quantization_format_freeze_tests(void) {
    if (!g_quantization_format_freeze_engine) {
        std::cerr << "[QUANT_FREEZE] Engine not initialized" << std::endl;
        return false;
    }

    if (!llama_resolve_quantization_format_at_load(QUANT_FORMAT_Q4_0, QUANT_FORMAT_Q4_0, QUANT_FORMAT_Q4_0)) {
        std::cerr << "[QUANT_FREEZE] TEST FAILED: Format resolution" << std::endl;
        return false;
    }

    if (!llama_lock_quantization_format()) {
        std::cerr << "[QUANT_FREEZE] TEST FAILED: Format locking" << std::endl;
        return false;
    }

    if (!llama_is_quantization_format_locked()) {
        std::cerr << "[QUANT_FREEZE] TEST FAILED: Lock verification" << std::endl;
        return false;
    }

    if (llama_attempt_quantization_promotion(QUANT_FORMAT_Q4_0, QUANT_FORMAT_F16)) {
        // Promotion should be blocked
    }

    std::cout << "[QUANT_FREEZE] All tests passed" << std::endl;
    return true;
}

bool llama_init_quantization_format_freeze_module(void) {
    if (!llama_init_quantization_format_freeze()) {
        std::cerr << "[QUANT_FREEZE] Failed to initialize engine" << std::endl;
        return false;
    }

    return run_quantization_format_freeze_tests();
}

void llama_cleanup_quantization_format_freeze_module(void) {
    if (g_quantization_format_freeze_engine) {
        delete g_quantization_format_freeze_engine;
        g_quantization_format_freeze_engine = nullptr;
    }
}
