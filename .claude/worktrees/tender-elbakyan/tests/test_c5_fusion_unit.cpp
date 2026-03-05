/**
 * Phase C5: Unit Tests for Kernel Fusion Enforcement
 * Tests individual validators and support functions
 */

#include <cstdio>
#include <cassert>
#include <cstring>
#include <vector>

// Forward declarations (would include headers in real build)
extern "C" {
    // From llama-kernel-fusion-enforce.h
    typedef struct {
        uint64_t total_launches;
        uint64_t launches_per_token;
        uint64_t launches_per_layer;
        uint32_t qkv_fusion_state;
        uint32_t norm_matmul_fusion;
        uint32_t bias_activation_fusion;
        uint32_t attention_kernel_type;
        uint32_t sampling_kernel_type;
    } llama_kernel_metrics;

    typedef struct {
        bool enforce_active;
        uint64_t baseline_launches;
        uint32_t target_max_launches;
        uint32_t layer_count;
        uint32_t max_launches_per_layer;
        bool enforce_qkv_fusion;
        bool enforce_norm_matmul_fusion;
        bool enforce_bias_activation;
        bool enforce_flash_attention;
        bool enforce_single_stream;
        bool enforce_persistent_kernels;
        llama_kernel_metrics metrics;
    } llama_kernel_fusion_state;

    void llama_kernel_fusion_init(llama_kernel_fusion_state * state);
    void llama_kernel_fusion_activate(llama_kernel_fusion_state * state, uint32_t n_layers, uint32_t target_launches);
    void llama_kernel_fusion_deactivate(llama_kernel_fusion_state * state);
    llama_kernel_metrics llama_kernel_fusion_get_metrics(const llama_kernel_fusion_state * state);
}

// ============================================================================
// TEST SUITE 1: STATE INITIALIZATION
// ============================================================================

void test_fusion_state_init() {
    printf("TEST 1: Fusion state initialization\n");
    
    llama_kernel_fusion_state state = {};
    llama_kernel_fusion_init(&state);

    assert(state.enforce_active == false);
    assert(state.baseline_launches == 0);
    assert(state.target_max_launches == 0);
    assert(state.layer_count == 0);
    assert(state.max_launches_per_layer == 0);
    assert(state.enforce_qkv_fusion == false);
    assert(state.enforce_norm_matmul_fusion == false);
    assert(state.enforce_bias_activation == false);
    assert(state.enforce_flash_attention == false);
    assert(state.enforce_single_stream == false);
    assert(state.enforce_persistent_kernels == false);
    assert(state.metrics.qkv_fusion_state == 0);
    assert(state.metrics.attention_kernel_type == 2);
    assert(state.metrics.sampling_kernel_type == 2);

    printf("  ✓ State initialized correctly\n\n");
}

// ============================================================================
// TEST SUITE 2: ACTIVATION
// ============================================================================

void test_fusion_activation() {
    printf("TEST 2: Fusion enforcement activation\n");
    
    llama_kernel_fusion_state state = {};
    llama_kernel_fusion_init(&state);
    assert(state.enforce_active == false);

    llama_kernel_fusion_activate(&state, 49, 4);

    assert(state.enforce_active == true);
    assert(state.layer_count == 49);
    assert(state.target_max_launches == 4);
    assert(state.max_launches_per_layer == 1);  // (4 + 49 - 1) / 49 = 1
    assert(state.enforce_qkv_fusion == true);
    assert(state.enforce_norm_matmul_fusion == true);
    assert(state.enforce_bias_activation == true);
    assert(state.enforce_flash_attention == true);
    assert(state.enforce_single_stream == true);
    assert(state.enforce_persistent_kernels == true);

    printf("  ✓ State activated correctly\n");
    printf("  ✓ All enforcement flags enabled\n");
    printf("  ✓ Max launches per layer calculated: %u\n\n", state.max_launches_per_layer);
}

// ============================================================================
// TEST SUITE 3: DEACTIVATION
// ============================================================================

void test_fusion_deactivation() {
    printf("TEST 3: Fusion enforcement deactivation\n");
    
    llama_kernel_fusion_state state = {};
    llama_kernel_fusion_init(&state);
    llama_kernel_fusion_activate(&state, 49, 4);
    assert(state.enforce_active == true);

    llama_kernel_fusion_deactivate(&state);
    assert(state.enforce_active == false);

    printf("  ✓ State deactivated correctly\n\n");
}

// ============================================================================
// TEST SUITE 4: METRICS TRACKING
// ============================================================================

void test_fusion_metrics() {
    printf("TEST 4: Kernel fusion metrics tracking\n");
    
    llama_kernel_fusion_state state = {};
    llama_kernel_fusion_init(&state);
    llama_kernel_fusion_activate(&state, 49, 4);

    llama_kernel_metrics metrics = llama_kernel_fusion_get_metrics(&state);
    
    assert(metrics.total_launches == 0);
    assert(metrics.launches_per_token == 0);
    assert(metrics.qkv_fusion_state == 0);
    assert(metrics.norm_matmul_fusion == 0);
    assert(metrics.bias_activation_fusion == 0);
    assert(metrics.attention_kernel_type == 2);
    assert(metrics.sampling_kernel_type == 2);

    printf("  ✓ Metrics initialized correctly\n");
    printf("  ✓ All metrics default to 0/2\n\n");
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main() {
    printf("\n========== PHASE C5: FUSION UNIT TESTS ==========\n\n");
    
    try {
        test_fusion_state_init();
        test_fusion_activation();
        test_fusion_deactivation();
        test_fusion_metrics();

        printf("========== ALL TESTS PASSED ✓ ==========\n\n");
        return 0;
    } catch (const std::exception& e) {
        printf("ASSERTION FAILED: %s\n", e.what());
        return 1;
    }
}
