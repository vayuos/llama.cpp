#pragma once

/**
 * llama-probing-elimination.h
 *
 * REQUIREMENT #54: Complete elimination of capability detection from decode-critical path
 *
 * This header defines the probing elimination system that moves all runtime capability
 * detection from the decode path to startup, creating a static configuration that
 * guarantees zero feature probing during token generation.
 *
 * Key Design Principles:
 * - All capability detection happens at startup (before decode_start)
 * - Configuration immutable after initialization and locked at decode boundary
 * - Backend dispatch tables populated once, never re-evaluated
 * - Per-token decisions made without querying device/architecture capabilities
 * - Assembly validates absence of capability check branches
 * - Silent fallbacks eliminated in favor of hard preconditions
 * - Linear control flow in decode path (no branching on capabilities)
 *
 * Metrics Tracked:
 * - Runtime probing calls during decode (target: 0)
 * - Per-op backend selection decisions (target: 0)
 * - cudaGetDeviceProperties calls in decode (target: 0)
 * - Feature support checks per iteration (target: 0)
 * - Configuration modification attempts during decode (target: 0)
 */

#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>
#include <map>
#include <atomic>

// Forward declarations
struct llama_context;
struct ggml_cgraph;
struct ggml_tensor;

// ============================================================================
// SECTION 1: Probing Patterns and Detection
// ============================================================================

/**
 * Probing pattern enumeration - patterns to be eliminated
 * These represent runtime checks that must be moved to startup
 */
enum class llama_probing_pattern {
    PATTERN_CUDA_AVAILABLE = 0,           // if (cuda_available)
    PATTERN_DEVICE_TENSOR_CORES = 1,      // if (device_has_tensor_cores)
    PATTERN_MMQ_SUPPORTED = 2,            // if (mmq_supported)
    PATTERN_FLASH_ATTN_SUPPORTED = 3,     // if (flash_attn_supported)
    PATTERN_BACKEND_CAN_HANDLE = 4,       // if (backend_can_handle(op))
    PATTERN_TENSOR_IS_GPU = 5,            // if (tensor_is_gpu)
    PATTERN_ARCH_CHECK = 6,               // if (arch >= X)
    PATTERN_BACKEND_SUPPORTS = 7,         // if (ggml_backend_supports(op))
    PATTERN_DEVICE_PROPS_QUERY = 8,       // cudaGetDeviceProperties()
    PATTERN_FEATURE_FLAG_READ = 9,        // if (feature_flag_enabled)
};

/**
 * Lifecycle stage for probing elimination
 * Enforces strict ordering: STARTUP -> INIT -> DECODE_ACTIVE -> FROZEN
 */
enum class llama_probing_elimination_stage {
    UNINITIALIZED = 0,
    CAPABILITY_DETECTION = 1,    // Startup: probe hardware and populate cache
    CONFIGURATION_LOCKED = 2,    // Config immutable, dispatch tables bound
    DECODE_ACTIVE = 3,           // Decode in progress - no probing allowed
    VALIDATION_COMPLETE = 4,     // Assembly validation finished
    FROZEN_FINAL = 5,            // Permanent lock engaged
};

/**
 * Result of probing detection in assembly/bytecode
 */
struct llama_probing_detection_result {
    uint32_t patterns_found;
    uint32_t pattern_count[10];   // Count of each pattern type
    uint64_t instruction_count;
    uint64_t conditional_branches;
    uint64_t device_query_calls;
    bool decode_path_clean;       // true if zero probing in decode
    const char * violation_details;
};

// ============================================================================
// SECTION 2: Capability Cache Structure (Startup-Populated)
// ============================================================================

/**
 * GPU Architecture capabilities - detected at startup only
 * Immutable after initialization
 */
struct llama_gpu_capabilities {
    int32_t compute_capability_major;
    int32_t compute_capability_minor;
    int32_t max_threads_per_block;
    int32_t max_blocks_per_grid;
    uint64_t total_memory;
    uint64_t shared_memory_per_block;
    bool has_tensor_cores;
    bool supports_flash_attention;
    bool supports_mmq;
    bool supports_sm_copy_async;
    bool supports_cooperative_groups;
    int32_t device_id;
    const char * device_name;
};

/**
 * Backend availability structure - detected at startup
 * No runtime re-evaluation allowed
 */
struct llama_backend_availability {
    bool cuda_available;
    bool cuda_enabled_at_compile;
    bool opencl_available;
    bool sycl_available;
    bool metal_available;
    bool vulkan_available;
    bool kompute_available;
    bool cpu_available;

    // Selected backend (immutable after init)
    int32_t selected_backend;
    bool backend_validated;
    int64_t backend_validation_time_us;
};

/**
 * Feature support matrix - detected once at startup
 */
struct llama_feature_support_matrix {
    // Attention mechanisms
    bool flash_attention_available;
    bool dense_attention_fallback_available;

    // Quantization modes
    bool int8_quantization_supported;
    bool int4_quantization_supported;
    bool fp16_supported;
    bool fp8_supported;

    // Memory capabilities
    bool unified_memory_supported;
    bool pinned_memory_supported;
    bool kv_cache_compression_supported;

    // Graph optimization
    bool graph_capture_supported;
    bool graph_reuse_supported;
    bool kernel_fusion_supported;

    // Sampling
    bool deterministic_sampling_supported;
    bool complex_sampling_supported;

    // Validation timestamp
    int64_t detection_timestamp_us;
};

/**
 * Kernel capability matrix per operation type
 */
struct llama_kernel_capability_matrix {
    std::map<uint32_t, bool> op_cuda_capable;      // op_id -> is_cuda_capable
    std::map<uint32_t, bool> op_mfma_capable;      // op_id -> supports_mfma
    std::map<uint32_t, bool> op_fallback_available; // op_id -> has_fallback
};

// ============================================================================
// SECTION 3: Backend Dispatch Table (No Probing)
// ============================================================================

/**
 * Backend operation dispatch - selected at startup
 * No capability checks during execution
 */
typedef int (*llama_backend_compute_fn)(
    llama_context * ctx,
    const void * compute_params
);

typedef int (*llama_backend_init_fn)(
    llama_context * ctx,
    int device_id
);

typedef int (*llama_backend_cleanup_fn)(void);

/**
 * Complete backend dispatch table - bound at startup
 * No dynamic selection or fallback during decode
 */
struct llama_backend_dispatch_table {
    llama_backend_compute_fn compute;
    llama_backend_init_fn init;
    llama_backend_cleanup_fn cleanup;
    const char * backend_name;
    int32_t backend_id;
    bool validated;
    int64_t bind_timestamp_us;
};

/**
 * Per-operation backend dispatch - precompiled graph
 */
struct llama_op_backend_binding {
    uint32_t op_id;
    int32_t backend_id;
    void * kernel_ptr;      // Pre-selected kernel function
    bool supports_fallback;
    int64_t binding_timestamp_us;
};

// ============================================================================
// SECTION 4: Decode Configuration with No Probing
// ============================================================================

/**
 * Immutable decode configuration - locked at startup
 * Contains all decisions that would otherwise require capability queries
 */
struct llama_decode_config_immutable {
    // -------------------------------------------------------
    // Lifecycle and Locking
    // -------------------------------------------------------
    llama_probing_elimination_stage current_stage;
    bool configuration_locked;
    bool decode_active;
    std::atomic<uint64_t> probing_calls_detected_during_decode;
    int64_t lock_timestamp_us;

    // -------------------------------------------------------
    // Cached Capabilities (from Startup)
    // -------------------------------------------------------
    llama_gpu_capabilities gpu_capabilities;
    llama_backend_availability backend_availability;
    llama_feature_support_matrix feature_matrix;
    llama_kernel_capability_matrix kernel_matrix;

    // -------------------------------------------------------
    // Backend Dispatch (Selected at Startup)
    // -------------------------------------------------------
    llama_backend_dispatch_table backend_dispatch;
    std::vector<llama_op_backend_binding> op_bindings;

    // -------------------------------------------------------
    // Attention Configuration (Frozen)
    // -------------------------------------------------------
    struct {
        bool use_flash_attention;     // Decision made at startup
        bool use_dense_fallback;      // Only if flash unavailable
        bool kernel_preselected;
        void * attention_kernel_ptr;
    } attention_config;

    // -------------------------------------------------------
    // Memory Configuration (Frozen)
    // -------------------------------------------------------
    struct {
        uint64_t kv_cache_size;
        uint64_t sampling_workspace_size;
        bool use_pinned_memory;
        bool use_unified_memory;
        int64_t allocation_timestamp_us;
    } memory_config;

    // -------------------------------------------------------
    // Sampling Configuration (Frozen)
    // -------------------------------------------------------
    struct {
        int32_t sampling_mode;        // Preselected at startup
        int32_t top_k;
        float top_p;
        float temperature;
        bool deterministic;
        void * sampling_kernel_ptr;   // Pre-selected kernel
    } sampling_config;

    // -------------------------------------------------------
    // Feature Flags (Frozen)
    // -------------------------------------------------------
    struct {
        bool flash_attention_enabled;
        bool graph_reuse_enabled;
        bool quantization_enabled;
        bool streaming_enabled;
        bool deterministic_mode;
        uint32_t feature_mask;
        int64_t resolve_timestamp_us;
    } features;

    // -------------------------------------------------------
    // Validation and Metrics
    // -------------------------------------------------------
    struct {
        uint64_t runtime_probing_attempts;      // Target: 0
        uint64_t backend_selection_in_decode;   // Target: 0
        uint64_t device_property_queries;       // Target: 0
        uint64_t feature_checks_per_iteration;  // Target: 0
        uint64_t config_modification_attempts;  // Target: 0
        bool zero_probing_confirmed;
    } metrics;
};

// ============================================================================
// SECTION 5: Startup Capability Detection
// ============================================================================

/**
 * Phase 1: Probe GPU capabilities once at startup
 * Called exactly once before any decode operations
 *
 * @param gpu_caps Output structure to populate
 * @param device_id GPU device to query
 * @return 0 on success, negative on error
 */
int llama_probing_detect_gpu_capabilities(
    llama_gpu_capabilities * gpu_caps,
    int32_t device_id
);

/**
 * Phase 2: Check backend availability at startup
 * Validates which backends are available, selects primary
 *
 * @param backend_avail Output structure to populate
 * @return 0 on success, negative on error
 */
int llama_probing_detect_backend_availability(
    llama_backend_availability * backend_avail
);

/**
 * Phase 3: Build feature support matrix at startup
 * Determines which features are available on this hardware
 *
 * @param features Output matrix to populate
 * @param gpu_caps GPU capabilities from phase 1
 * @param backend_avail Backend availability from phase 2
 * @return 0 on success, negative on error
 */
int llama_probing_build_feature_matrix(
    llama_feature_support_matrix * features,
    const llama_gpu_capabilities * gpu_caps,
    const llama_backend_availability * backend_avail
);

/**
 * Phase 4: Bind backend dispatch functions
 * Selects and validates backend compute/init/cleanup functions
 *
 * @param config Configuration to populate
 * @param backend_avail Backend availability from phase 2
 * @param gpu_caps GPU capabilities from phase 1
 * @return 0 on success, negative if backend unavailable
 */
int llama_probing_bind_backend_dispatch(
    llama_decode_config_immutable * config,
    const llama_backend_availability * backend_avail,
    const llama_gpu_capabilities * gpu_caps
);

/**
 * Phase 5: Pre-select attention kernel at startup
 * No runtime check for flash attention during decode
 *
 * @param config Configuration to update
 * @return 0 on success, negative on error
 */
int llama_probing_bind_attention_kernel(
    llama_decode_config_immutable * config
);

/**
 * Phase 6: Pre-select sampling kernel at startup
 * No runtime sampling mode switching during decode
 *
 * @param config Configuration to update
 * @return 0 on success, negative on error
 */
int llama_probing_bind_sampling_kernel(
    llama_decode_config_immutable * config
);

/**
 * Phase 7: Build per-operation backend bindings
 * Pre-compiles which backend each operation uses
 *
 * @param config Configuration to populate
 * @param graph Computation graph to analyze
 * @return 0 on success, negative on error
 */
int llama_probing_build_op_backend_bindings(
    llama_decode_config_immutable * config,
    ggml_cgraph * graph
);

// ============================================================================
// SECTION 6: Configuration Lock and Enforcement
// ============================================================================

/**
 * Lock configuration at decode start
 * Prevents any capability probing and enables violation detection
 *
 * @param config Configuration to lock
 * @return 0 on success, negative if already locked
 */
int llama_probing_lock_configuration(
    llama_decode_config_immutable * config
);

/**
 * Check if configuration is locked (decode active)
 *
 * @param config Configuration to check
 * @return true if locked, false otherwise
 */
bool llama_probing_is_locked(const llama_decode_config_immutable * config);

/**
 * Detect probing attempt during decode
 * Called by any code attempting capability checks in decode path
 *
 * @param config Configuration
 * @param pattern_type Type of probing attempt
 * @return -1 if probing detected during decode, 0 if allowed
 */
int llama_probing_detect_violation(
    llama_decode_config_immutable * config,
    llama_probing_pattern pattern_type
);

/**
 * Guard against runtime capability check in decode path
 * Fails if decode is active
 *
 * @param config Configuration
 * @param pattern_name Name of pattern (e.g., "cuda_available")
 * @return 0 if allowed, negative if decode active
 */
int llama_probing_guard_capability_check(
    const llama_decode_config_immutable * config,
    const char * pattern_name
);

// ============================================================================
// SECTION 7: Assembly and Bytecode Validation
// ============================================================================

/**
 * Scan assembly for probing patterns in decode function
 * Validates absence of capability checks in generated code
 *
 * @param decode_fn Pointer to decode function
 * @param fn_size Size of function in bytes
 * @return Detection result with pattern counts
 */
llama_probing_detection_result llama_probing_scan_assembly(
    const void * decode_fn,
    size_t fn_size
);

/**
 * Validate no device property queries in decode path
 * Confirms cudaGetDeviceProperties and similar calls absent
 *
 * @param config Configuration to validate
 * @return 0 if clean, negative if violations found
 */
int llama_probing_validate_no_device_queries(
    const llama_decode_config_immutable * config
);

/**
 * Validate linear control flow in decode
 * Confirms no branching on capabilities
 *
 * @param config Configuration to validate
 * @return 0 if linear, negative if branching detected
 */
int llama_probing_validate_linear_control_flow(
    const llama_decode_config_immutable * config
);

/**
 * Instrument decode to detect runtime probing
 * Tracks all capability checks during execution
 *
 * @param config Configuration to monitor
 * @param decode_fn Decode function to instrument
 * @return Number of probing calls detected
 */
uint64_t llama_probing_instrument_decode(
    llama_decode_config_immutable * config,
    int (*decode_fn)(llama_context * ctx)
);

// ============================================================================
// SECTION 8: Static Dispatch Execution (Zero Probing)
// ============================================================================

/**
 * Execute preselected backend compute (no probing)
 * Must only use values from config, never query device
 *
 * @param config Frozen configuration
 * @param ctx Context to compute
 * @param params Compute parameters
 * @return Compute result
 */
int llama_probing_execute_backend_compute(
    const llama_decode_config_immutable * config,
    llama_context * ctx,
    const void * params
);

/**
 * Execute preselected attention kernel (no probing)
 * No runtime flash attention availability check
 *
 * @param config Frozen configuration
 * @param graph Computation graph
 * @param q Query tensor
 * @param k Key tensor
 * @param v Value tensor
 * @return Attention result
 */
int llama_probing_execute_attention(
    const llama_decode_config_immutable * config,
    ggml_cgraph * graph,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v
);

/**
 * Execute preselected sampling kernel (no probing)
 * No runtime sampling mode switching
 *
 * @param config Frozen configuration
 * @param logits Logit values
 * @param n_logits Number of logits
 * @param sampled_token Output token
 * @return Sampling result
 */
int llama_probing_execute_sampling(
    const llama_decode_config_immutable * config,
    float * logits,
    int32_t n_logits,
    int32_t * sampled_token
);

// ============================================================================
// SECTION 9: Validation and Metrics
// ============================================================================

/**
 * Verify configuration completeness
 * Checks that all dispatch functions are bound and valid
 *
 * @param config Configuration to validate
 * @return 0 if complete, negative if missing
 */
int llama_probing_validate_complete(
    const llama_decode_config_immutable * config
);

/**
 * Verify zero probing during decode execution
 * Confirms metrics show no capability checks occurred
 *
 * @param config Configuration with metrics
 * @return 0 if zero probing confirmed, non-zero otherwise
 */
int llama_probing_verify_zero_probing(
    const llama_decode_config_immutable * config
);

/**
 * Get detailed probing violation metrics
 *
 * @param config Configuration with metrics
 * @return Count of detected probing violations
 */
uint64_t llama_probing_get_violation_count(
    const llama_decode_config_immutable * config
);

/**
 * Print configuration for debugging
 * Shows which dispatch functions are locked and when
 *
 * @param config Configuration to print
 */
void llama_probing_print_config(
    const llama_decode_config_immutable * config
);

/**
 * Generate detailed probing elimination report
 *
 * @param config Configuration with metrics
 * @return Formatted string (must be freed by caller)
 */
char * llama_probing_generate_report(
    const llama_decode_config_immutable * config
);

// ============================================================================
// SECTION 10: Initialization and Cleanup
// ============================================================================

/**
 * Initialize decode configuration structure
 * Sets all fields to defaults and safe values
 *
 * @param config Configuration to initialize
 * @return 0 on success
 */
int llama_probing_config_init(
    llama_decode_config_immutable * config
);

/**
 * Cleanup configuration and free resources
 *
 * @param config Configuration to cleanup
 * @return 0 on success
 */
int llama_probing_config_cleanup(
    llama_decode_config_immutable * config
);

/**
 * Create a new configuration from defaults
 *
 * @return Allocated and initialized config, or nullptr on failure
 */
llama_decode_config_immutable * llama_probing_config_new(void);

/**
 * Destroy a configuration
 *
 * @param config Configuration to destroy
 */
void llama_probing_config_free(
    llama_decode_config_immutable * config
);

// ============================================================================
// SECTION 11: Helper Functions for Precondition Checks
// ============================================================================

/**
 * Convert capability check to hard precondition
 * Used to convert fallback paths to fatal errors
 *
 * @param config Configuration
 * @param feature Feature being checked
 * @param operation Operation requiring feature
 * @return 0 if feature available, negative (abort) if not
 */
int llama_probing_enforce_precondition(
    const llama_decode_config_immutable * config,
    const char * feature,
    const char * operation
);

/**
 * Assert decode path integrity (no probing allowed)
 * Called at decode start to lock configuration
 *
 * @param config Configuration to validate
 * @return 0 if safe, aborts if violations detected
 */
int llama_probing_assert_decode_integrity(
    const llama_decode_config_immutable * config
);

/**
 * Assert no configuration modification during decode
 *
 * @param config Configuration
 * @param field_name Name of field being modified
 * @return 0 if safe, negative if decode active
 */
int llama_probing_assert_no_reconfig(
    const llama_decode_config_immutable * config,
    const char * field_name
);

#endif // LLAMA_PROBING_ELIMINATION_H
