#pragma once

/**
 * llama-mmq-enforcement.h
 *
 * Exclusive MMQ backend binding for all quantized decode operations.
 *
 * Implements 11 enforcement rules with:
 * - Quantization type detection at model load time
 * - MMQ backend binding at graph build time
 * - Immutable backend configuration for decode lifetime
 * - Runtime assertions and validation
 * - Complete fallback prevention
 * - Hybrid placement prohibition
 *
 * Requirements Enforced (11 Rules):
 *
 * 1. Detect Quantized Model at Load Time
 *    - Inspect GGUF tensor types during model load
 *    - Detect: Q4_*, Q5_*, Q6_*, Q8_*, IQ*, *_K
 *    - Mark context: ctx->decode_quantized = true
 *    - Flag becomes immutable for context lifetime
 *
 * 2. Bind Quantized Decode to MMQ at Graph Build
 *    - If ctx->decode_quantized == true
 *    - Force backend selection to: GGML_BACKEND_CUDA_MMQ
 *    - No fallback to: cuBLAS, CUDA dense, CPU
 *    - Backend decision cached and immutable
 *
 * 3. Disable cuBLAS Path for Quantized Decode
 *    - In ggml-backend.cpp, ggml-backend-reg.cpp
 *    - If quantized and decode mode: skip cuBLAS registration
 *    - Skip dense CUDA backend
 *    - Only expose MMQ kernel
 *
 * 4. Prohibit CPU Fallback for Quantized Ops
 *    - If MMQ kernel not available: hard error
 *    - FATAL: "Quantized decode requires MMQ backend"
 *    - No silent: dequantize to FP16, CPU GEMV, backend switch
 *
 * 5. Enforce Fused Dequant + MatMul
 *    - MMQ kernels: dequant + multiply-accumulate + accumulation
 *    - No separate dequantization
 *    - No host-side dequant
 *    - No CPU dequant
 *    - All quantized math in fused CUDA kernels
 *
 * 6. Lock Backend at Decode Start
 *    - Assert: ctx->decode_backend == CUDA_MMQ
 *    - Prevent runtime backend switching
 *    - Decode-time backend immutable
 *
 * 7. Disable Hybrid Layer Placement for Quantized Decode
 *    - If quantized decode active
 *    - Disallow CPU layer placement
 *    - Disallow split-layer execution
 *    - Enforce full GPU residency for: weights, KV cache, activations
 *
 * 8. Force MMQ Build Configuration
 *    - For quantized decode builds
 *    - -DGGML_CUDA_MMQ=ON
 *    - -DGGML_CUDA_FORCE_MMQ=ON
 *    - -DGGML_CUDA_FORCE_CUBLAS=OFF
 *    - Verify MMQ selection in logs
 *
 * 9. Add Runtime Assertion
 *    - At first decode step
 *    - assert(ctx->decode_quantized == false || ctx->decode_backend == CUDA_MMQ)
 *    - If violated: abort immediately
 *
 * 10. Prevent Mixed Backend Graph Nodes
 *     - During graph build
 *     - Verify no quantized node assigned: CPU, cuBLAS
 *     - If detected: abort graph construction
 *
 * 11. Expected Outcome
 *     - No CPU dequantization
 *     - No backend switching
 *     - Fewer kernel launches (fusion)
 *     - Higher arithmetic efficiency
 *     - Stable GPU utilization
 *     - Zero hybrid ambiguity
 *     - Strictly MMQ-driven GPU pipeline
 *
 * Key Metrics Tracked:
 * - Quantized decode detection rate (target: 100%)
 * - MMQ backend binding rate (target: 100%)
 * - CPU fallback prevention (target: 100%)
 * - cuBLAS usage in quantized decode (target: 0%)
 * - Hybrid placement attempts (target: 0%)
 * - Mixed backend graph nodes (target: 0%)
 */

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <array>
#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <stdexcept>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

// Maximum number of quantization types to track
#define LLAMA_MMQ_ENFORCEMENT_MAX_QUANT_TYPES 20

// MMQ enforcement state machine states
#define LLAMA_MMQ_ENFORCEMENT_STATE_UNINITIALIZED 0
#define LLAMA_MMQ_ENFORCEMENT_STATE_QUANTIZED_DETECTED 1
#define LLAMA_MMQ_ENFORCEMENT_STATE_BACKEND_BOUND 2
#define LLAMA_MMQ_ENFORCEMENT_STATE_DECODE_LOCKED 3
#define LLAMA_MMQ_ENFORCEMENT_STATE_ERROR 4

// Maximum graph violations to accumulate before abort
#define LLAMA_MMQ_ENFORCEMENT_MAX_VIOLATIONS 100

// Compile-time configuration
#ifndef LLAMA_MMQ_ENFORCEMENT_ENABLED
#define LLAMA_MMQ_ENFORCEMENT_ENABLED 1
#endif

#ifndef LLAMA_MMQ_ENFORCEMENT_STRICT_MODE
#define LLAMA_MMQ_ENFORCEMENT_STRICT_MODE 1
#endif

#ifndef LLAMA_MMQ_ENFORCEMENT_COLLECT_METRICS
#define LLAMA_MMQ_ENFORCEMENT_COLLECT_METRICS 1
#endif

// ============================================================================
// QUANTIZATION TYPE DETECTION ENUMS
// ============================================================================

/**
 * Quantization categories detected at model load
 */
typedef enum {
    LLAMA_QUANT_CATEGORY_NONE = 0,           // No quantization (F16, F32)
    LLAMA_QUANT_CATEGORY_Q4 = 1,             // Q4_0, Q4_1, Q4_K
    LLAMA_QUANT_CATEGORY_Q5 = 2,             // Q5_0, Q5_1, Q5_K
    LLAMA_QUANT_CATEGORY_Q6 = 3,             // Q6_K
    LLAMA_QUANT_CATEGORY_Q8 = 4,             // Q8_0, Q8_1, Q8_K
    LLAMA_QUANT_CATEGORY_IQ = 5,             // IQ2_*, IQ3_*, IQ4_*, IQ1_*
    LLAMA_QUANT_CATEGORY_K_VARIANTS = 6,     // Q*_K (K-quant variants)
    LLAMA_QUANT_CATEGORY_MIXED = 7           // Multiple quantization types detected
} llama_quant_category_t;

/**
 * MMQ backend binding states
 */
typedef enum {
    LLAMA_MMQ_BACKEND_UNBOUND = 0,            // Backend not yet bound
    LLAMA_MMQ_BACKEND_CUDA_MMQ = 1,           // Bound to CUDA MMQ
    LLAMA_MMQ_BACKEND_FORCED = 2              // Forced binding (immutable)
} llama_mmq_backend_binding_t;

/**
 * Decode isolation levels
 */
typedef enum {
    LLAMA_DECODE_ISOLATION_NONE = 0,          // No isolation
    LLAMA_DECODE_ISOLATION_BACKEND_LOCKED = 1, // Backend locked for decode
    LLAMA_DECODE_ISOLATION_FULL = 2            // Full isolation: backend + memory + placement
} llama_decode_isolation_level_t;

// ============================================================================
// QUANTIZATION DETECTION STRUCTURES
// ============================================================================

/**
 * Quantized tensor detected during model load
 */
typedef struct {
    const char * tensor_name;                 // Tensor name from GGUF
    int ggml_type;                            // GGML type code (GGML_TYPE_Q4_0, etc.)
    uint64_t element_count;                   // Number of elements
    uint64_t size_bytes;                      // Storage size in bytes
    bool is_weight;                           // true if weight tensor, false if activation
    bool is_kv_cache;                         // true if KV cache tensor
} llama_quantized_tensor_t;

/**
 * Quantization detection summary for context
 */
typedef struct {
    bool model_is_quantized;                  // true if any quantized tensors detected
    llama_quant_category_t dominant_category; // Most common quantization type
    std::atomic<size_t> total_quantized_tensors;  // Count of quantized tensors
    std::atomic<uint64_t> total_quantized_bytes;  // Total bytes in quantized tensors
    std::atomic<size_t> unique_quant_types;      // Number of distinct quantization types
    llama_quantized_tensor_t * first_quantized_tensor; // Pointer to first quantized tensor
} llama_quantization_detection_t;

/**
 * Quantization detection summary for external query (non-atomic)
 */
typedef struct {
    bool model_is_quantized;
    llama_quant_category_t dominant_category;
    size_t total_quantized_tensors;
    uint64_t total_quantized_bytes;
    size_t unique_quant_types;
    llama_quantized_tensor_t * first_quantized_tensor;
} llama_quantization_detection_summary_t;

// ============================================================================
// MMQ BINDING AND ENFORCEMENT STRUCTURES
// ============================================================================

/**
 * MMQ backend binding configuration (immutable after binding)
 */
typedef struct {
    llama_mmq_backend_binding_t binding_state; // Current binding state
    uint32_t backend_type;                    // GGML_BACKEND_CUDA_MMQ
    bool supports_mmq_fusion;                 // true if MMQ fusion available
    bool prohibit_cublas;                     // true if cuBLAS explicitly disabled
    bool prohibit_cpu_fallback;               // true if CPU fallback disabled
    std::atomic<bool> binding_locked;         // Immutable after decode start
} llama_mmq_backend_binding_t_struct;

/**
 * Graph node backend validation entry
 */
typedef struct {
    void * node_ptr;                          // Opaque pointer to graph node
    const char * node_name;                   // Operation name
    int node_ggml_type;                       // Input/output type
    bool is_quantized_op;                     // true if operates on quantized tensors
    uint32_t assigned_backend;                // Backend assignment for this node
    bool violates_mmq_policy;                 // true if backend assignment violates policy
} llama_graph_node_validation_entry_t;

// ============================================================================
// ENFORCEMENT METRICS STRUCTURES
// ============================================================================

/**
 * Atomic enforcement violation tracking (lock-free)
 */
typedef struct {
    std::atomic<uint64_t> quantized_models_detected;        // Models loaded with quantization
    std::atomic<uint64_t> mmq_bindings_succeeded;           // Successful MMQ backend bindings
    std::atomic<uint64_t> cpu_fallback_attempts;            // Attempted CPU fallbacks (blocked)
    std::atomic<uint64_t> cublas_path_violations;           // cuBLAS path access attempts (blocked)
    std::atomic<uint64_t> backend_switch_attempts;          // Runtime backend switch attempts (blocked)
    std::atomic<uint64_t> unused_backend_symbols_found;     // Detected unused backend symbols
    std::atomic<uint64_t> hybrid_placement_attempts;        // Hybrid layer placement attempts (blocked)
    std::atomic<uint64_t> mixed_backend_graph_nodes;        // Mixed backend nodes in graphs (blocked)
    std::atomic<uint64_t> fused_kernel_launches;            // Successful fused kernel launches
    std::atomic<uint64_t> decode_locks_enforced;            // Decode-time backend locks
    std::atomic<uint64_t> runtime_assertions_passed;        // Runtime assertions that passed
    std::atomic<uint64_t> total_enforcement_violations;     // All enforcement violations
} llama_mmq_enforcement_violations_t;

/**
 * Snapshot of violations for reporting (non-atomic)
 */
typedef struct {
    uint64_t quantized_models_detected;
    uint64_t mmq_bindings_succeeded;
    uint64_t cpu_fallback_attempts;
    uint64_t cublas_path_violations;
    uint64_t backend_switch_attempts;
    uint64_t hybrid_placement_attempts;
    uint64_t mixed_backend_graph_nodes;
    uint64_t fused_kernel_launches;
    uint64_t decode_locks_enforced;
    uint64_t runtime_assertions_passed;
    uint64_t total_enforcement_violations;
} llama_mmq_enforcement_violations_log_t;

/**
 * MMQ enforcement metrics (lock-free)
 */
typedef struct {
    std::atomic<size_t> total_models_processed;             // Models loaded
    std::atomic<size_t> quantized_models;                   // Quantized models
    std::atomic<size_t> quantized_graphs_built;             // Graphs with quantized tensors
    std::atomic<size_t> mmq_backend_bound_graphs;           // Graphs bound to MMQ
    std::atomic<uint64_t> cumulative_quantized_bytes;       // Total quantized tensor bytes processed
    std::atomic<uint64_t> kernel_fusion_bytes;              // Bytes processed by fused kernels
    std::atomic<uint64_t> verification_time_ns;             // Last verification duration (nanoseconds)
    std::atomic<uint32_t> verification_count;               // Total verification runs
    std::atomic<bool> last_verification_passed;             // Result of last verification
    std::atomic<double> cpu_fallback_prevention_rate;       // Percentage of fallbacks prevented (0-100)
} llama_mmq_enforcement_metrics_t;

/**
 * MMQ enforcement metrics snapshot (non-atomic)
 */
typedef struct {
    size_t total_models_processed;
    size_t quantized_models;
    size_t quantized_graphs_built;
    size_t mmq_backend_bound_graphs;
    uint64_t cumulative_quantized_bytes;
    uint64_t kernel_fusion_bytes;
    uint64_t verification_time_ns;
    uint32_t verification_count;
    bool last_verification_passed;
    double cpu_fallback_prevention_rate;
} llama_mmq_enforcement_metrics_log_t;

/**
 * MMQ enforcement state machine
 */
typedef struct {
    uint32_t state;                           // Current state (see LLAMA_MMQ_ENFORCEMENT_STATE_*)
    bool model_quantized;                     // true if model contains quantized tensors
    bool mmq_backend_bound;                   // true if MMQ backend is bound
    bool decode_backend_locked;               // true if decode backend is locked
    llama_quant_category_t dominant_category; // Most common quantization type
    llama_decode_isolation_level_t isolation_level; // Decode isolation level
    llama_mmq_enforcement_violations_t violations;
    llama_mmq_enforcement_metrics_t metrics;
    std::vector<std::string> * deferred_violations; // Buffer for violation details
    std::atomic<bool> abort_on_violation;    // Immediately abort if violation detected
} llama_mmq_enforcement_state_t;

// ============================================================================
// COMPILE-TIME CHECK MACROS (15+ total)
// ============================================================================

/**
 * LLAMA_MMQ_ENFORCEMENT_ASSERT_QUANTIZED_DETECTED
 * Runtime check that quantization was detected at load time
 */
#define LLAMA_MMQ_ENFORCEMENT_ASSERT_QUANTIZED_DETECTED(ctx) \
    llama_mmq_enforcement_assert_quantized_detected(ctx)

/**
 * LLAMA_MMQ_ENFORCEMENT_ASSERT_MMQ_BOUND
 * Runtime check that MMQ backend is bound
 */
#define LLAMA_MMQ_ENFORCEMENT_ASSERT_MMQ_BOUND(ctx) \
    llama_mmq_enforcement_assert_mmq_bound(ctx)

/**
 * LLAMA_MMQ_ENFORCEMENT_LOCK_DECODE_BACKEND
 * Lock backend at decode start (immutable)
 */
#define LLAMA_MMQ_ENFORCEMENT_LOCK_DECODE_BACKEND(ctx) \
    llama_mmq_enforcement_lock_decode_backend(ctx)

/**
 * LLAMA_MMQ_ENFORCEMENT_PROHIBIT_CUBLAS
 * Disable cuBLAS path for quantized decode
 */
#define LLAMA_MMQ_ENFORCEMENT_PROHIBIT_CUBLAS() \
    llama_mmq_enforcement_prohibit_cublas()

/**
 * LLAMA_MMQ_ENFORCEMENT_PROHIBIT_CPU_FALLBACK
 * Disable CPU fallback for quantized operations
 */
#define LLAMA_MMQ_ENFORCEMENT_PROHIBIT_CPU_FALLBACK() \
    llama_mmq_enforcement_prohibit_cpu_fallback()

/**
 * LLAMA_MMQ_ENFORCEMENT_VERIFY_GRAPH_BACKEND
 * Verify all quantized nodes use MMQ backend
 */
#define LLAMA_MMQ_ENFORCEMENT_VERIFY_GRAPH_BACKEND(graph_ptr) \
    llama_mmq_enforcement_verify_graph_backend(graph_ptr)

/**
 * LLAMA_MMQ_ENFORCEMENT_DISABLE_HYBRID_PLACEMENT
 * Prevent hybrid CPU/GPU layer placement
 */
#define LLAMA_MMQ_ENFORCEMENT_DISABLE_HYBRID_PLACEMENT(ctx) \
    llama_mmq_enforcement_disable_hybrid_placement(ctx)

/**
 * LLAMA_MMQ_ENFORCEMENT_FORCE_MMQ_BUILD
 * Verify MMQ build configuration
 */
#define LLAMA_MMQ_ENFORCEMENT_FORCE_MMQ_BUILD \
    do { \
        static_assert(defined(GGML_CUDA_MMQ), "GGML_CUDA_MMQ must be enabled"); \
        static_assert(defined(GGML_CUDA_FORCE_MMQ), "GGML_CUDA_FORCE_MMQ must be enabled"); \
        static_assert(!defined(GGML_CUDA_FORCE_CUBLAS), "GGML_CUDA_FORCE_CUBLAS must be disabled"); \
    } while(0)

/**
 * LLAMA_MMQ_ENFORCEMENT_FIRST_DECODE_STEP
 * Runtime assertion at first decode step
 */
#define LLAMA_MMQ_ENFORCEMENT_FIRST_DECODE_STEP(ctx) \
    llama_mmq_enforcement_first_decode_step(ctx)

/**
 * LLAMA_MMQ_ENFORCEMENT_PREVENT_BACKEND_SWITCH
 * Assert that backend cannot be switched during decode
 */
#define LLAMA_MMQ_ENFORCEMENT_PREVENT_BACKEND_SWITCH(ctx) \
    llama_mmq_enforcement_prevent_backend_switch(ctx)

/**
 * LLAMA_MMQ_ENFORCEMENT_FENCE
 * Memory fence to ensure enforcement state propagation
 */
#define LLAMA_MMQ_ENFORCEMENT_FENCE() \
    std::atomic_thread_fence(std::memory_order_seq_cst)

/**
 * LLAMA_MMQ_ENFORCEMENT_ABORT_ON_VIOLATION
 * Configure behavior when violations detected
 */
#define LLAMA_MMQ_ENFORCEMENT_ABORT_ON_VIOLATION(should_abort) \
    llama_mmq_enforcement_set_abort_on_violation(should_abort)

/**
 * LLAMA_MMQ_ENFORCEMENT_CHECK_QUANTIZED_DECODE
 * Comprehensive check for quantized decode state
 */
#define LLAMA_MMQ_ENFORCEMENT_CHECK_QUANTIZED_DECODE(ctx) \
    llama_mmq_enforcement_check_quantized_decode(ctx)

/**
 * LLAMA_MMQ_ENFORCEMENT_GUARD_GRAPH_NODE
 * Guard a graph node backend assignment
 */
#define LLAMA_MMQ_ENFORCEMENT_GUARD_GRAPH_NODE(node_ptr, ggml_type) \
    llama_mmq_enforcement_guard_graph_node(node_ptr, ggml_type)

// ============================================================================
// QUANTIZATION DETECTION FUNCTIONS (5+)
// ============================================================================

/**
 * Initialize MMQ enforcement for a context
 * @param ctx_ptr Opaque context pointer
 * @return Enforcement state handle
 */
extern llama_mmq_enforcement_state_t * llama_mmq_enforcement_init(void * ctx_ptr);

/**
 * Detect quantization types at model load time
 * @param state State handle
 * @param tensor_name Name of tensor being loaded
 * @param ggml_type GGML type from GGUF (see GGML_TYPE_Q4_0, etc.)
 * @return true if tensor is quantized and detection succeeded
 */
extern bool llama_mmq_enforcement_detect_quantized_tensor(
    llama_mmq_enforcement_state_t * state,
    const char * tensor_name,
    int ggml_type
);

/**
 * Finalize quantization detection after model load
 * @param state State handle
 * @return Dominant quantization category detected
 */
extern llama_quant_category_t llama_mmq_enforcement_finalize_detection(
    llama_mmq_enforcement_state_t * state
);

/**
 * Check if model is quantized
 * @param state State handle
 * @return true if model contains any quantized tensors
 */
extern bool llama_mmq_enforcement_is_quantized_model(
    const llama_mmq_enforcement_state_t * state
);

/**
 * Get quantization detection summary
 * @param state State handle
 * @return Detection summary struct
 */
extern llama_quantization_detection_summary_t llama_mmq_enforcement_get_detection_summary(
    const llama_mmq_enforcement_state_t * state
);

// ============================================================================
// MMQ BACKEND BINDING FUNCTIONS (6+)
// ============================================================================

/**
 * Bind MMQ backend to quantized decode at graph build time
 * @param state State handle
 * @param graph_ptr Opaque pointer to computation graph
 * @return true if binding succeeded
 */
extern bool llama_mmq_enforcement_bind_mmq_backend(
    llama_mmq_enforcement_state_t * state,
    void * graph_ptr
);

/**
 * Assert that MMQ backend is bound
 * @param state State handle
 * @throws std::runtime_error if not bound
 */
extern void llama_mmq_enforcement_assert_mmq_bound(
    llama_mmq_enforcement_state_t * state
);

/**
 * Lock backend at decode start (immutable)
 * @param state State handle
 */
extern void llama_mmq_enforcement_lock_decode_backend(
    llama_mmq_enforcement_state_t * state
);

/**
 * Check if decode backend is locked
 * @param state State handle
 * @return true if backend is locked
 */
extern bool llama_mmq_enforcement_is_decode_locked(
    const llama_mmq_enforcement_state_t * state
);

/**
 * Disable cuBLAS path for quantized decode
 * @return true if cuBLAS was successfully disabled
 */
extern bool llama_mmq_enforcement_prohibit_cublas(void);

/**
 * Disable CPU fallback for quantized operations
 * @return true if CPU fallback was successfully disabled
 */
extern bool llama_mmq_enforcement_prohibit_cpu_fallback(void);

// ============================================================================
// RUNTIME ASSERTION FUNCTIONS (5+)
// ============================================================================

/**
 * Assert quantized model detected at load time
 * @param state State handle
 * @throws std::runtime_error if not detected
 */
extern void llama_mmq_enforcement_assert_quantized_detected(
    llama_mmq_enforcement_state_t * state
);

/**
 * Runtime assertion at first decode step
 * @param state State handle
 * @throws std::runtime_error if constraints violated
 */
extern void llama_mmq_enforcement_first_decode_step(
    llama_mmq_enforcement_state_t * state
);

/**
 * Prevent runtime backend switching
 * @param state State handle
 * @throws std::runtime_error if switch attempted
 */
extern void llama_mmq_enforcement_prevent_backend_switch(
    llama_mmq_enforcement_state_t * state
);

/**
 * Comprehensive check for quantized decode state
 * @param state State handle
 * @return Number of violations found
 */
extern size_t llama_mmq_enforcement_check_quantized_decode(
    llama_mmq_enforcement_state_t * state
);

/**
 * Guard a graph node backend assignment
 * @param node_ptr Opaque pointer to graph node
 * @param ggml_type GGML type of node input/output
 * @throws std::runtime_error if node violates policy
 */
extern void llama_mmq_enforcement_guard_graph_node(
    void * node_ptr,
    int ggml_type
);

// ============================================================================
// HYBRID PLACEMENT PREVENTION FUNCTIONS (3+)
// ============================================================================

/**
 * Disable hybrid CPU/GPU layer placement
 * @param state State handle
 * @return true if hybrid placement successfully disabled
 */
extern bool llama_mmq_enforcement_disable_hybrid_placement(
    llama_mmq_enforcement_state_t * state
);

/**
 * Assert no CPU layers for quantized decode
 * @param state State handle
 * @throws std::runtime_error if CPU layers detected
 */
extern void llama_mmq_enforcement_assert_no_cpu_layers(
    llama_mmq_enforcement_state_t * state
);

/**
 * Verify full GPU residency for quantized tensors
 * @param state State handle
 * @return true if all quantized tensors on GPU
 */
extern bool llama_mmq_enforcement_verify_gpu_residency(
    llama_mmq_enforcement_state_t * state
);

// ============================================================================
// GRAPH VALIDATION FUNCTIONS (4+)
// ============================================================================

/**
 * Validate that all quantized nodes use MMQ backend
 * @param graph_ptr Opaque pointer to computation graph
 * @return Number of mixed backend nodes found (should be 0)
 */
extern size_t llama_mmq_enforcement_verify_graph_backend(
    void * graph_ptr
);

/**
 * Check for mixed backend graph nodes
 * @param graph_ptr Opaque pointer to computation graph
 * @param state State handle for tracking
 * @return true if no mixed backend nodes detected
 */
extern bool llama_mmq_enforcement_check_mixed_backend_nodes(
    void * graph_ptr,
    llama_mmq_enforcement_state_t * state
);

/**
 * Get graph validation details
 * @param graph_ptr Opaque pointer to computation graph
 * @return Array of validation entries (terminated by null)
 */
extern const llama_graph_node_validation_entry_t *
llama_mmq_enforcement_get_graph_validation_details(
    void * graph_ptr
);

/**
 * Report all graph validation violations
 * @param graph_ptr Opaque pointer to computation graph
 * @return Human-readable violation report
 */
extern const char * llama_mmq_enforcement_get_graph_violation_report(
    void * graph_ptr
);

// ============================================================================
// METRIC COLLECTION AND REPORTING FUNCTIONS (6+)
// ============================================================================

/**
 * Collect and report enforcement metrics
 * @param state State handle
 * @return Metrics struct
 */
extern llama_mmq_enforcement_metrics_log_t llama_mmq_enforcement_get_metrics(
    llama_mmq_enforcement_state_t * state
);

/**
 * Collect violation tracking metrics
 * @param state State handle
 * @return Violation tracking struct
 */
extern llama_mmq_enforcement_violations_log_t llama_mmq_enforcement_get_violations(
    llama_mmq_enforcement_state_t * state
);

/**
 * Report all detected violations
 * @param state State handle
 * @return Human-readable violation report
 */
extern const char * llama_mmq_enforcement_get_violation_report(
    llama_mmq_enforcement_state_t * state
);

/**
 * Get CPU fallback prevention rate
 * @param state State handle
 * @return Prevention rate as percentage (0-100)
 */
extern double llama_mmq_enforcement_get_cpu_fallback_prevention_rate(
    llama_mmq_enforcement_state_t * state
);

/**
 * Get MMQ backend binding success rate
 * @param state State handle
 * @return Success rate as percentage (0-100)
 */
extern double llama_mmq_enforcement_get_mmq_binding_rate(
    llama_mmq_enforcement_state_t * state
);

/**
 * Reset all violation counters
 * @param state State handle
 */
extern void llama_mmq_enforcement_reset_violations(
    llama_mmq_enforcement_state_t * state
);

// ============================================================================
// CONFIGURATION AND STATE FUNCTIONS (5+)
// ============================================================================

/**
 * Set behavior when violations are detected
 * @param should_abort If true, immediately abort; if false, defer
 */
extern void llama_mmq_enforcement_set_abort_on_violation(bool should_abort);

/**
 * Get current MMQ enforcement state
 * @param state State handle
 * @return Current state value (see LLAMA_MMQ_ENFORCEMENT_STATE_*)
 */
extern uint32_t llama_mmq_enforcement_get_state(
    const llama_mmq_enforcement_state_t * state
);

/**
 * Validate enforcement state consistency
 * @param state State handle
 * @return true if state is consistent
 */
extern bool llama_mmq_enforcement_validate_state_consistency(
    llama_mmq_enforcement_state_t * state
);

/**
 * Get detailed enforcement status report
 * @param state State handle
 * @return Human-readable status report
 */
extern const char * llama_mmq_enforcement_get_status_report(
    llama_mmq_enforcement_state_t * state
);

/**
 * Free MMQ enforcement state
 * @param state State handle
 */
extern void llama_mmq_enforcement_free(llama_mmq_enforcement_state_t * state);

// ============================================================================
// QUANTIZATION TYPE HELPER FUNCTIONS (5+)
// ============================================================================

/**
 * Check if GGML type is quantized
 * @param ggml_type GGML type code
 * @return true if type is quantized (Q4_*, Q5_*, Q6_*, Q8_*, IQ*, *_K)
 */
extern bool llama_mmq_enforcement_is_quantized_type(int ggml_type);

/**
 * Get quantization category for GGML type
 * @param ggml_type GGML type code
 * @return Quantization category
 */
extern llama_quant_category_t llama_mmq_enforcement_get_quant_category(int ggml_type);

/**
 * Get human-readable quantization type name
 * @param ggml_type GGML type code
 * @return Type name string (e.g., "Q4_K", "IQ2_XS")
 */
extern const char * llama_mmq_enforcement_get_quant_type_name(int ggml_type);

/**
 * Get quantization category name
 * @param category Quantization category
 * @return Category name string
 */
extern const char * llama_mmq_enforcement_get_category_name(llama_quant_category_t category);

/**
 * Check if quantization type requires MMQ
 * @param ggml_type GGML type code
 * @return true if MMQ required for efficient decoding
 */
extern bool llama_mmq_enforcement_requires_mmq(int ggml_type);

#ifdef __cplusplus
}
#endif

