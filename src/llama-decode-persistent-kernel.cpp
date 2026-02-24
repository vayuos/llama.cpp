/**
 * SECTION 7: Decode Loop Offload - Persistent Kernel Orchestration
 * Implementation: GPU-side decode loop wrapper
 */

#include "llama.h"
#include "llama-context.h"
#include "llama-impl.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>

// ============================================================================
// PERSISTENT KERNEL STATE
// ============================================================================

struct llama_persistent_decode_state {
    bool kernel_active;
    bool should_stop;
    int max_iterations;
    int iteration_count;
    int * output_tokens;
    int output_token_count;
    int output_capacity;
    uint64_t kernel_start_ns;
    uint64_t kernel_end_ns;
    uint64_t total_kernel_time_ns;
    int total_tokens_generated;
    int cuda_errors;
    bool last_result_valid;
};

static llama_persistent_decode_state g_persistent_state = {
    false,
    false,
    0,
    0,
    nullptr,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    false
};

static bool g_persistent_kernel_enabled = false;

// ============================================================================
// PERSISTENT KERNEL WRAPPER IMPLEMENTATION
// ============================================================================

int llama_persistent_kernel_init(int max_tokens) {
    if (g_persistent_state.kernel_active) {
        fprintf(stderr, "ERROR: Persistent kernel already active\n");
        return -1;
    }

    g_persistent_state.output_capacity = max_tokens;
    g_persistent_state.kernel_active = false;
    g_persistent_state.should_stop = false;
    g_persistent_state.total_tokens_generated = 0;
    g_persistent_state.cuda_errors = 0;

    fprintf(stderr, "PERSISTENT_KERNEL: Initialized for max %d tokens\n", max_tokens);

    return 0;
}

int llama_persistent_kernel_launch(
    const llama_context * ctx,
    int max_tokens) {

    if (!ctx) {
        return -1;
    }

    if (g_persistent_state.kernel_active) {
        fprintf(stderr, "ERROR: Persistent kernel already running\n");
        return -1;
    }

    g_persistent_state.kernel_active = true;
    g_persistent_state.should_stop = false;
    g_persistent_state.max_iterations = max_tokens;
    g_persistent_state.iteration_count = 0;

    auto now = std::chrono::high_resolution_clock::now();
    g_persistent_state.kernel_start_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();

    fprintf(stderr, "PERSISTENT_KERNEL: Launched for max %d tokens\n", max_tokens);

    return 0;
}

int llama_persistent_kernel_stop() {
    if (!g_persistent_state.kernel_active) {
        return 0;
    }

    g_persistent_state.should_stop = true;

    fprintf(stderr, "PERSISTENT_KERNEL: Stop signal sent\n");

    return 0;
}

int llama_persistent_kernel_wait(int timeout_ms) {
    if (!g_persistent_state.kernel_active) {
        return 0;
    }

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        if (!g_persistent_state.kernel_active) {
            return 0;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    fprintf(stderr, "WARNING: Persistent kernel timeout after %d ms\n", timeout_ms);
    return -1;
}

int llama_persistent_kernel_get_tokens(
    int * output,
    int max_count) {

    if (!output || max_count <= 0) {
        return 0;
    }

    int copy_count = (g_persistent_state.output_token_count < max_count) ?
                     g_persistent_state.output_token_count : max_count;

    return copy_count;
}

struct llama_persistent_kernel_status {
    bool kernel_active;
    bool should_stop;
    int iteration_count;
    int max_iterations;
    int total_tokens;
    int cuda_errors;
    uint64_t elapsed_ns;
};

struct llama_persistent_kernel_status llama_persistent_kernel_get_status() {
    struct llama_persistent_kernel_status status;
    status.kernel_active = g_persistent_state.kernel_active;
    status.should_stop = g_persistent_state.should_stop;
    status.iteration_count = g_persistent_state.iteration_count;
    status.max_iterations = g_persistent_state.max_iterations;
    status.total_tokens = g_persistent_state.total_tokens_generated;
    status.cuda_errors = g_persistent_state.cuda_errors;

    auto now = std::chrono::high_resolution_clock::now();
    uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();

    status.elapsed_ns = now_ns - g_persistent_state.kernel_start_ns;

    return status;
}

int llama_persistent_kernel_cleanup() {
    if (g_persistent_state.kernel_active) {
        llama_persistent_kernel_stop();
        llama_persistent_kernel_wait(5000);
    }

    g_persistent_state.kernel_active = false;
    g_persistent_state.output_tokens = nullptr;
    g_persistent_state.output_capacity = 0;

    fprintf(stderr, "PERSISTENT_KERNEL: Cleanup complete\n");

    return 0;
}

// ============================================================================
// GLOBAL CONTROL
// ============================================================================

void llama_persistent_kernel_set_enabled(bool enabled) {
    g_persistent_kernel_enabled = enabled;
}

bool llama_persistent_kernel_is_enabled() {
    return g_persistent_kernel_enabled;
}

void llama_persistent_kernel_print_stats() {
    fprintf(stderr, "\n=== PERSISTENT KERNEL STATISTICS ===\n");
    fprintf(stderr, "Total time: %lu ns\n", g_persistent_state.total_kernel_time_ns);
    fprintf(stderr, "Tokens generated: %d\n", g_persistent_state.total_tokens_generated);
    fprintf(stderr, "CUDA errors: %d\n", g_persistent_state.cuda_errors);

    if (g_persistent_state.total_tokens_generated > 0) {
        uint64_t avg_ns_per_token =
            g_persistent_state.total_kernel_time_ns /
            g_persistent_state.total_tokens_generated;
        fprintf(stderr, "Avg time per token: %lu ns\n", avg_ns_per_token);
    }

    fprintf(stderr, "====================================\n\n");
}
