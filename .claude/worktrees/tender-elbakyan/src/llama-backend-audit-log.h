#pragma once

/**
 * Backend Usage Audit Logging for LLAMA Decode
 *
 * Single, authoritative backend audit report that runs exactly once per decode
 * session and proves which backend owns every decode-critical operation.
 *
 * This is not continuous logging. This is a one-time structural verification
 * report that converts backend correctness from assumption into verifiable fact.
 *
 * Audit Trigger Point:
 * - After decode graph construction
 * - After backend binding
 * - After graph freeze
 * - Before first token execution
 *
 * Audit Output:
 * - Deterministic backend report
 * - Kernel variant selection (MMQ vs cuBLAS, fused vs unfused, etc.)
 * - Backend ownership enumeration
 * - Zero CPU ownership guarantee
 * - Immediate abort on CPU-owned decode-critical ops
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <sstream>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AUDIT_PHASE_UNINITIALIZED = 0,
    AUDIT_PHASE_SETUP = 1,
    AUDIT_PHASE_GRAPH_BUILT = 2,
    AUDIT_PHASE_ENUMERATION = 3,
    AUDIT_PHASE_REPORTING = 4,
    AUDIT_PHASE_VALIDATION = 5,
    AUDIT_PHASE_COMPLETE = 6
} backend_audit_phase;

typedef enum {
    BACKEND_OWNERSHIP_CUDA = 0,
    BACKEND_OWNERSHIP_METAL = 1,
    BACKEND_OWNERSHIP_VULKAN = 2,
    BACKEND_OWNERSHIP_OPENCL = 3,
    BACKEND_OWNERSHIP_CPU = 4,
    BACKEND_OWNERSHIP_UNKNOWN = 5
} backend_ownership;

typedef enum {
    KERNEL_VARIANT_UNKNOWN = 0,
    KERNEL_VARIANT_MMQ = 1,
    KERNEL_VARIANT_CUBLAS = 2,
    KERNEL_VARIANT_FUSED = 3,
    KERNEL_VARIANT_UNFUSED = 4,
    KERNEL_VARIANT_FLASH_ATTENTION = 5,
    KERNEL_VARIANT_DENSE_ATTENTION = 6,
    KERNEL_VARIANT_QUANTIZED = 7,
    KERNEL_VARIANT_FP32 = 8
} kernel_variant;

typedef struct {
    const char * op_name;
    const char * tensor_shape;
    backend_ownership backend;
    kernel_variant kernel_type;
    uint64_t op_index;
    bool is_decode_critical;
    bool is_fused;
    const char * additional_info;
} backend_audit_node;

typedef struct {
    uint32_t total_nodes;
    uint32_t total_decode_critical;
    uint32_t cuda_owned;
    uint32_t metal_owned;
    uint32_t vulkan_owned;
    uint32_t opencl_owned;
    uint32_t cpu_owned;
    uint32_t unknown_owned;

    uint32_t mmq_kernels;
    uint32_t cublas_kernels;
    uint32_t fused_kernels;
    uint32_t unfused_kernels;
    uint32_t flash_attention_kernels;
    uint32_t dense_attention_kernels;

    bool cpu_ownership_detected;
    bool all_critical_ops_gpu;
    bool audit_passed;
    uint64_t audit_timestamp_ns;
} backend_audit_summary;

typedef struct {
    const char * violation_description;
    uint32_t cpu_owned_op_count;
    const char * first_cpu_op_name;
    uint64_t violation_timestamp_ns;
    bool abort_triggered;
} backend_audit_violation;

class backend_usage_audit_logger {
private:
    backend_audit_phase current_phase;
    std::vector<backend_audit_node> audit_nodes;
    backend_audit_summary audit_summary;
    std::vector<backend_audit_violation> violations;
    std::string audit_report;

    std::atomic<bool> audit_performed;
    std::atomic<bool> audit_passed;
    std::atomic<uint64_t> audit_timestamp;

    std::map<std::string, uint32_t> op_type_counts;
    std::map<backend_ownership, uint32_t> backend_counts;
    std::map<kernel_variant, uint32_t> kernel_counts;

public:
    backend_usage_audit_logger();

    bool initialize();
    bool mark_graph_built();
    bool begin_enumeration();

    bool enumerate_decode_node(const char * op_name,
                               const char * tensor_shape,
                               backend_ownership backend,
                               kernel_variant kernel_type,
                               bool is_decode_critical);

    bool finalize_enumeration();
    bool generate_audit_report();
    bool validate_audit_results();
    bool record_cpu_ownership_violation(const char * op_name);

    backend_audit_phase get_current_phase() const { return current_phase; }
    bool is_audit_complete() const { return audit_performed.load(); }
    bool did_audit_pass() const { return audit_passed.load(); }

    const backend_audit_summary & get_summary() const { return audit_summary; }
    const std::string & get_report() const { return audit_report; }
    std::vector<backend_audit_node> get_audit_nodes() const { return audit_nodes; }
    std::vector<backend_audit_violation> get_violations() const { return violations; }

    // Report generation
    std::string format_backend_name(backend_ownership backend) const;
    std::string format_kernel_variant(kernel_variant kernel) const;
    std::string generate_json_report() const;

    // Validation functions
    bool verify_no_cpu_ownership() const;
    bool verify_kernel_selection_consistency() const;
    bool verify_decode_critical_coverage() const;
    bool verify_backend_consistency() const;

    // Statistics
    size_t get_node_count() const { return audit_nodes.size(); }
    size_t get_decode_critical_count() const { return audit_summary.total_decode_critical; }
    size_t get_gpu_owned_count() const {
        return audit_summary.cuda_owned + audit_summary.metal_owned +
               audit_summary.vulkan_owned + audit_summary.opencl_owned;
    }
    size_t get_cpu_owned_count() const { return audit_summary.cpu_owned; }
    size_t get_violation_count() const { return violations.size(); }
};

class backend_audit_guard {
private:
    bool guard_active;
    backend_usage_audit_logger * logger;

public:
    backend_audit_guard(backend_usage_audit_logger * logger_ptr);
    ~backend_audit_guard();

    bool is_guard_active() const { return guard_active; }
};

extern backend_usage_audit_logger * g_backend_usage_audit_logger;

bool llama_init_backend_audit_logger();
bool llama_mark_graph_built();
bool llama_begin_backend_enumeration();

bool llama_enumerate_decode_node(const char * op_name,
                                 const char * tensor_shape,
                                 int backend_enum,
                                 int kernel_variant_enum,
                                 bool is_decode_critical);

bool llama_finalize_backend_enumeration();
bool llama_generate_backend_audit_report();
bool llama_validate_backend_audit();
bool llama_record_cpu_ownership_violation(const char * op_name);

bool llama_is_backend_audit_complete();
bool llama_did_backend_audit_pass();

const backend_audit_summary * llama_get_backend_audit_summary();
const char * llama_get_backend_audit_report();

void llama_print_backend_audit_report();
void llama_print_backend_audit_summary();
void llama_print_backend_audit_violations();
void llama_print_backend_audit_node_details();
void llama_export_backend_audit_json(const char * filename);

// Macro-based guards for audit integration
#define ASSERT_AUDIT_PASSED() \
    do { \
        if (g_backend_usage_audit_logger && !llama_did_backend_audit_pass()) { \
            return -1; \
        } \
    } while(0)

#define ENUMERATE_DECODE_BACKEND_NODE(op_name, shape, backend, variant, critical) \
    do { \
        if (g_backend_usage_audit_logger) { \
            if (!llama_enumerate_decode_node(op_name, shape, backend, variant, critical)) { \
                return -1; \
            } \
        } \
    } while(0)

#define RECORD_CPU_VIOLATION(op_name) \
    do { \
        if (g_backend_usage_audit_logger) { \
            llama_record_cpu_ownership_violation(op_name); \
        } \
    } while(0)

#define VERIFY_BACKEND_AUDIT_PASSED() \
    do { \
        if (g_backend_usage_audit_logger && !llama_did_backend_audit_pass()) { \
            llama_print_backend_audit_report(); \
            FATAL("Backend audit failed: CPU-owned decode-critical ops detected"); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
