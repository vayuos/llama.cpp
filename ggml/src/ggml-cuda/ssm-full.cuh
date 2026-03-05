/**
 * SECTION 11: SSM Acceleration (Qwen3Next)
 * Header: Complete SSM CUDA kernel API
 */

#pragma once

#include "common.cuh"
#include <cstdint>

#if defined(GGML_USE_HIP)
#    include "vendors/hip.h"
#elif defined(GGML_USE_MUSA)
#    include "vendors/musa.h"
#else
#    include "vendors/cuda.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// VALIDATION RESULTS
// ============================================================================

struct ggml_cuda_ssm_validation_result {
    bool convolution_ok;
    bool state_update_ok;
    bool gating_ok;
    bool fused_ok;
    const char * error_msg;
};

// ============================================================================
// INDIVIDUAL KERNEL API
// ============================================================================

/**
 * SSM Convolution Kernel
 * Applies 1D convolution for SSM input transformation.
 *
 * @param d_input Input sequence [T, D] on GPU
 * @param d_weights Convolution weights [K]
 * @param d_output Output [T, D]
 * @param T Sequence length
 * @param D State dimension
 * @param K Kernel length
 * @param stream CUDA stream
 */
void ggml_cuda_ssm_convolve(
    ggml_backend_cuda_context & ctx,
    const float * d_input,
    const float * d_weights,
    float * d_output,
    int32_t T,
    int32_t D,
    int32_t K,
    cudaStream_t stream);

/**
 * SSM State Update Kernel
 * Computes state transition: h_t = A @ h_{t-1} + B @ u_t
 *
 * @param d_A Transition matrix [D, D]
 * @param d_B Input matrix [D, 1]
 * @param d_u Input sequence [T]
 * @param d_h State trajectory [T, D]
 * @param T Sequence length
 * @param D State dimension
 * @param stream CUDA stream
 */
void ggml_cuda_ssm_state_update(
    ggml_backend_cuda_context & ctx,
    const float * d_A,
    const float * d_B,
    const float * d_u,
    float * d_h,
    int32_t T,
    int32_t D,
    cudaStream_t stream);

/**
 * SSM Gated Recurrence Kernel
 * Computes gated output: y_t = C @ h_t, then gate_t * y_t + (1-gate_t) * x_t
 *
 * @param d_h State [T, D]
 * @param d_C Output projection [1, D]
 * @param d_gate Gate values [T, 1] (should be sigmoid-activated)
 * @param d_x Residual input [T, 1]
 * @param d_y Output [T, 1]
 * @param T Sequence length
 * @param D State dimension
 * @param stream CUDA stream
 */
void ggml_cuda_ssm_gated_recurrence(
    ggml_backend_cuda_context & ctx,
    const float * d_h,
    const float * d_C,
    const float * d_gate,
    const float * d_x,
    float * d_y,
    int32_t T,
    int32_t D,
    cudaStream_t stream);

// ============================================================================
// FUSED API
// ============================================================================

/**
 * Fused complete SSM forward pass.
 * Combines convolution, state update, and gating.
 * Minimizes kernel launch overhead and data movement.
 *
 * Equivalent to:
 * 1. convolved = convolve(input, weights)
 * 2. state = state_update(A, B, convolved)
 * 3. output = gated_recurrence(state, C, gate, input)
 *
 * @param d_input Input [T, D]
 * @param d_weights Convolution weights [K]
 * @param d_A Transition matrix [D, D]
 * @param d_B Input matrix [D]
 * @param d_C Output projection [D]
 * @param d_gate Gate values [T] (sigmoid-activated)
 * @param d_output Output [T]
 * @param T Sequence length
 * @param D State dimension
 * @param K Kernel length
 * @param stream CUDA stream
 */
void ggml_cuda_ssm_forward_fused(
    ggml_backend_cuda_context & ctx,
    const float * d_input,
    const float * d_weights,
    const float * d_A,
    const float * d_B,
    const float * d_C,
    const float * d_gate,
    float * d_output,
    int32_t T,
    int32_t D,
    int32_t K,
    cudaStream_t stream);

// ============================================================================
// VALIDATION AND DEBUGGING
// ============================================================================

/**
 * Validate SSM kernels.
 * Runs small test cases to verify correctness.
 * Returns results of all kernel validations.
 */
struct ggml_cuda_ssm_validation_result ggml_cuda_ssm_validate();

#ifdef __cplusplus
}
#endif
