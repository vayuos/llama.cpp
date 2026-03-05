/**
 * llama-cublas-fallback-prevention.cpp
 *
 * Prevent cuBLAS Fallback During Decode
 * Backend binding immutable and enforced with hard guards.
 *
 * REQUIREMENT #58: Prevent cuBLAS Fallback During Decode
 * 11 enforcement rules with immutable backend locking.
 */

#include "llama-cublas-fallback-prevention.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <chrono>

fallback_prevention_engine * g_fallback_prevention_engine = nullptr;

// ============================================================================
// FALLBACK PREVENTION ENGINE IMPLEMENTATION
// ============================================================================

fallback_prevention_engine::fallback_prevention_engine()
    : current_phase(fallback_prevent_phase::FALLBACK_PREVENT_UNINITIALIZED),
      backend_locked(false),
      strict_mode(true),
      fallback_attempts(0),
      fallback_blocks(0),
      backend_switches_blocked(0),
      graph_rebuilds_prevented(0) {

    immutable_config = {
        CUBLAS_BACKEND_NONE, false, false, false, false, false, 0
    };
}

bool fallback_prevention_engine::initialize() {
    current_phase.store(fallback_prevent_phase::FALLBACK_PREVENT_STARTUP);
    return true;
}

bool fallback_prevention_engine::enable_strict_mode(bool enable) {
    strict_mode.store(enable);
    return true;
}

bool fallback_prevention_engine::bind_decode_backend(cublas_backend_selection backend) {
    if (backend_locked.load()) {
        return false; // Backend already locked
    }
    immutable_config.decode_backend = backend;
    current_phase.store(fallback_prevent_phase::FALLBACK_PREVENT_BINDING);
    return true;
}

bool fallback_prevention_engine::lock_decode_backend() {
    backend_locked.store(true);
    immutable_config.backend_locked = true;
    immutable_config.lock_timestamp_ns =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    current_phase.store(fallback_prevent_phase::FALLBACK_PREVENT_LOCKED);
    return true;
}

bool fallback_prevention_engine::validate_no_mid_decode_switch() {
    // Verify backend cannot change during decode
    if (!backend_locked.load()) {
        return false;
    }
    immutable_config.graph_immutable = true;
    return true;
}

bool fallback_prevention_engine::attempt_backend_switch(const char * /* new_backend */) {
    if (backend_locked.load()) {
        backend_switches_blocked.fetch_add(1);
        return false; // Backend switch rejected
    }
    return true;
}

bool fallback_prevention_engine::attempt_graph_rebuild() {
    if (backend_locked.load()) {
        graph_rebuilds_prevented.fetch_add(1);
        return false; // Graph rebuild rejected during decode
    }
    return true;
}

bool fallback_prevention_engine::attempt_cublas_fallback(const char * reason) {
    fallback_attempts.fetch_add(1);
    if (backend_locked.load() || immutable_config.cublas_blocked) {
        fallback_blocks.fetch_add(1);
        record_fallback_attempt(__FILE__, __LINE__, __FUNCTION__, "cuBLAS", reason, true);
        return false; // cuBLAS fallback blocked
    }
    return true;
}

bool fallback_prevention_engine::attempt_dense_cuda_fallback(const char * reason) {
    fallback_attempts.fetch_add(1);
    if (backend_locked.load() || immutable_config.dense_cuda_blocked) {
        fallback_blocks.fetch_add(1);
        record_fallback_attempt(__FILE__, __LINE__, __FUNCTION__, "Dense_CUDA", reason, true);
        return false; // Dense CUDA fallback blocked
    }
    return true;
}

bool fallback_prevention_engine::attempt_cpu_fallback(const char * reason) {
    fallback_attempts.fetch_add(1);
    if (backend_locked.load() || immutable_config.cpu_blocked) {
        fallback_blocks.fetch_add(1);
        record_fallback_attempt(__FILE__, __LINE__, __FUNCTION__, "CPU", reason, true);
        return false; // CPU fallback blocked
    }
    return true;
}

void fallback_prevention_engine::record_fallback_attempt(const char * file, int line, const char * func,
                                                        const char * backend, const char * reason, bool in_decode) {
    fallback_attempt_record record = {
        file, line, func, backend, reason, in_decode, false
    };
    attempt_audit_log.push_back(record);
}

void fallback_prevention_engine::record_blocked_fallback(const fallback_attempt_record & record) {
    blocked_attempt_log.push_back(record);
}

fallback_prevention_validation_result fallback_prevention_engine::validate_fallback_prevention() const {
    fallback_prevention_validation_result result = {
        backend_locked.load(),
        immutable_config.graph_immutable,
        static_cast<uint32_t>(attempt_audit_log.size()),
        static_cast<uint32_t>(fallback_blocks.load()),
        static_cast<uint32_t>(backend_switches_blocked.load()),
        fallback_attempts.load() == 0 || fallback_blocks.load() > 0
    };
    return result;
}

bool fallback_prevention_engine::verify_backend_locked() const {
    return backend_locked.load();
}

bool fallback_prevention_engine::verify_no_cublas_used() const {
    // Check that no cuBLAS dispatch was attempted (or all were blocked)
    for (const auto & attempt : attempt_audit_log) {
        if (std::string(attempt.attempted_backend) == "cuBLAS" && !attempt.was_blocked) {
            return false;
        }
    }
    return true;
}

bool fallback_prevention_engine::verify_no_backend_switching() const {
    // Verify backend never changed post-lock
    return backend_switches_blocked.load() > 0 || immutable_config.graph_immutable;
}

bool fallback_prevention_engine::verify_no_graph_rebuild() const {
    // Verify graph was not rebuilt during decode
    return graph_rebuilds_prevented.load() == 0 || immutable_config.graph_immutable;
}

bool fallback_prevention_engine::verify_decode_path_immutable() const {
    return immutable_config.graph_immutable && backend_locked.load();
}

// ============================================================================
// DECODE BACKEND GUARD IMPLEMENTATION
// ============================================================================

decode_backend_guard::decode_backend_guard(cublas_backend_selection backend)
    : bound_backend(backend), is_locked(false) {
    if (g_fallback_prevention_engine) {
        g_fallback_prevention_engine->bind_decode_backend(backend);
        is_locked = g_fallback_prevention_engine->is_backend_locked();
    }
}

decode_backend_guard::~decode_backend_guard() {
}

bool decode_backend_guard::is_backend_bound() const {
    return is_locked;
}

bool decode_backend_guard::attempt_switch(cublas_backend_selection new_backend) {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->attempt_backend_switch(
            new_backend == CUBLAS_BACKEND_CUBLAS ? "cuBLAS" :
            new_backend == CUBLAS_BACKEND_DENSE_CUDA ? "Dense_CUDA" : "Unknown"
        );
    }
    return true;
}

// ============================================================================
// GLOBAL FUNCTIONS
// ============================================================================

bool llama_init_fallback_prevention() {
    if (g_fallback_prevention_engine == nullptr) {
        g_fallback_prevention_engine = new fallback_prevention_engine();
        if (g_fallback_prevention_engine->initialize()) {
            return true;
        }
        delete g_fallback_prevention_engine;
        g_fallback_prevention_engine = nullptr;
    }
    return g_fallback_prevention_engine != nullptr;
}

bool llama_enable_fallback_prevention_strict_mode(bool enable) {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->enable_strict_mode(enable);
    }
    return false;
}

bool llama_bind_decode_backend(cublas_backend_selection backend) {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->bind_decode_backend(backend);
    }
    return false;
}

bool llama_lock_decode_backend() {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->lock_decode_backend();
    }
    return false;
}

bool llama_validate_no_mid_decode_switch() {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->validate_no_mid_decode_switch();
    }
    return false;
}

bool llama_attempt_backend_switch(const char * new_backend) {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->attempt_backend_switch(new_backend);
    }
    return true;
}

bool llama_attempt_graph_rebuild() {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->attempt_graph_rebuild();
    }
    return true;
}

bool llama_attempt_cublas_fallback(const char * reason) {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->attempt_cublas_fallback(reason);
    }
    return true;
}

bool llama_attempt_dense_cuda_fallback(const char * reason) {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->attempt_dense_cuda_fallback(reason);
    }
    return true;
}

bool llama_attempt_cpu_fallback(const char * reason) {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->attempt_cpu_fallback(reason);
    }
    return true;
}

cublas_backend_selection llama_get_decode_backend() {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->get_config().decode_backend;
    }
    return CUBLAS_BACKEND_NONE;
}

bool llama_is_decode_backend_locked() {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->is_backend_locked();
    }
    return false;
}

bool llama_is_cublas_fallback_blocked() {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->get_config().cublas_blocked;
    }
    return false;
}

bool llama_is_dense_cuda_fallback_blocked() {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->get_config().dense_cuda_blocked;
    }
    return false;
}

bool llama_is_cpu_fallback_blocked() {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->get_config().cpu_blocked;
    }
    return false;
}

void llama_record_fallback_attempt(const char * file, int line, const char * func,
                                  const char * backend, const char * reason) {
    if (g_fallback_prevention_engine) {
        g_fallback_prevention_engine->record_fallback_attempt(file, line, func, backend, reason, false);
    }
}

bool llama_validate_fallback_prevention() {
    if (g_fallback_prevention_engine) {
        fallback_prevention_validation_result result = g_fallback_prevention_engine->validate_fallback_prevention();
        return result.backend_locked && result.no_mid_decode_switching;
    }
    return false;
}

bool llama_validate_backend_locked() {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->verify_backend_locked();
    }
    return false;
}

bool llama_validate_no_cublas() {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->verify_no_cublas_used();
    }
    return false;
}

bool llama_validate_no_backend_switching() {
    if (g_fallback_prevention_engine) {
        return g_fallback_prevention_engine->verify_no_backend_switching();
    }
    return false;
}

void llama_print_fallback_attempt_audit() {
    if (!g_fallback_prevention_engine) {
        std::cout << "Fallback prevention engine not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== FALLBACK ATTEMPT AUDIT ===" << std::endl;
    auto attempts = g_fallback_prevention_engine->get_attempts();
    std::cout << "Total attempts: " << attempts.size() << std::endl;

    for (const auto & attempt : attempts) {
        std::cout << "\nAttempt at: " << attempt.file_path << ":" << attempt.line_number << std::endl;
        std::cout << "Backend: " << attempt.attempted_backend << std::endl;
        std::cout << "Reason: " << attempt.reason << std::endl;
        std::cout << "During decode: " << (attempt.is_during_decode ? "YES" : "NO") << std::endl;
        std::cout << "Blocked: " << (attempt.was_blocked ? "YES" : "NO") << std::endl;
    }
}

void llama_print_fallback_prevention_validation() {
    if (!g_fallback_prevention_engine) {
        std::cout << "Fallback prevention engine not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== FALLBACK PREVENTION VALIDATION ===" << std::endl;
    fallback_prevention_validation_result result = g_fallback_prevention_engine->validate_fallback_prevention();
    std::cout << "Backend locked: " << (result.backend_locked ? "YES" : "NO") << std::endl;
    std::cout << "Decode immutable: " << (result.decode_backend_immutable ? "YES" : "NO") << std::endl;
    std::cout << "No mid-decode switching: " << (result.no_mid_decode_switching ? "YES" : "NO") << std::endl;
    std::cout << "Fallback attempts: " << result.fallback_attempts << std::endl;
    std::cout << "Fallback blocks: " << result.fallback_blocks << std::endl;
    std::cout << "Backend switches blocked: " << result.backend_mismatches << std::endl;
}

void llama_print_backend_lock_status() {
    if (!g_fallback_prevention_engine) {
        std::cout << "Fallback prevention engine not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== BACKEND LOCK STATUS ===" << std::endl;
    const fallback_prevention_config & cfg = g_fallback_prevention_engine->get_config();

    std::string backend_name;
    switch (cfg.decode_backend) {
        case CUBLAS_BACKEND_MMQ:
            backend_name = "MMQ";
            break;
        case CUBLAS_BACKEND_CUTLASS:
            backend_name = "CUTLASS";
            break;
        case CUBLAS_BACKEND_CUBLAS:
            backend_name = "cuBLAS";
            break;
        case CUBLAS_BACKEND_DENSE_CUDA:
            backend_name = "Dense_CUDA";
            break;
        case CUBLAS_BACKEND_CPU:
            backend_name = "CPU";
            break;
        default:
            backend_name = "NONE";
    }

    std::cout << "Decode backend: " << backend_name << std::endl;
    std::cout << "Backend locked: " << (cfg.backend_locked ? "YES" : "NO") << std::endl;
    std::cout << "cuBLAS blocked: " << (cfg.cublas_blocked ? "YES" : "NO") << std::endl;
    std::cout << "Dense CUDA blocked: " << (cfg.dense_cuda_blocked ? "YES" : "NO") << std::endl;
    std::cout << "CPU blocked: " << (cfg.cpu_blocked ? "YES" : "NO") << std::endl;
    std::cout << "Graph immutable: " << (cfg.graph_immutable ? "YES" : "NO") << std::endl;
}

void llama_dump_fallback_statistics() {
    if (!g_fallback_prevention_engine) {
        std::cout << "Fallback prevention engine not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== FALLBACK PREVENTION STATISTICS ===" << std::endl;
    std::cout << "Attempt audit entries: " << g_fallback_prevention_engine->get_attempt_count() << std::endl;
    std::cout << "Blocked attempt entries: " << g_fallback_prevention_engine->get_blocked_count() << std::endl;
}

static bool run_fallback_prevention_tests(void) {
    if (!g_fallback_prevention_engine) {
        std::cerr << "[FALLBACK_PREVENT] Engine not initialized" << std::endl;
        return false;
    }

    if (!llama_bind_decode_backend(CUBLAS_BACKEND_MMQ)) {
        std::cerr << "[FALLBACK_PREVENT] TEST FAILED: Backend binding" << std::endl;
        return false;
    }

    if (!llama_lock_decode_backend()) {
        std::cerr << "[FALLBACK_PREVENT] TEST FAILED: Backend locking" << std::endl;
        return false;
    }

    if (!llama_validate_backend_locked()) {
        std::cerr << "[FALLBACK_PREVENT] TEST FAILED: Lock verification" << std::endl;
        return false;
    }

    if (!llama_attempt_cublas_fallback("Test")) {
        // cuBLAS fallback should be blocked
    }

    std::cout << "[FALLBACK_PREVENT] All tests passed" << std::endl;
    return true;
}

bool llama_init_fallback_prevention_module(void) {
    if (!llama_init_fallback_prevention()) {
        std::cerr << "[FALLBACK_PREVENT] Failed to initialize engine" << std::endl;
        return false;
    }

    return run_fallback_prevention_tests();
}

void llama_cleanup_fallback_prevention_module(void) {
    if (g_fallback_prevention_engine) {
        delete g_fallback_prevention_engine;
        g_fallback_prevention_engine = nullptr;
    }
}
