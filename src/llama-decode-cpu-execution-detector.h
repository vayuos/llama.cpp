#pragma once

/**
 * Decode-Path CPU Execution Detector for LLAMA
 *
 * Implement a hard runtime detector that immediately identifies and aborts
 * if any decode-critical operation executes on the CPU during the decode phase.
 *
 * This enforces the invariant:
 * CPU must never participate in the decode-critical path.
 *
 * The detector must be structural, not heuristic.
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
    DECODE_CPU_DETECTOR_UNINITIALIZED = 0,
    DECODE_CPU_DETECTOR_SETUP = 1,
    DECODE_CPU_DETECTOR_ARMED = 2,
    DECODE_CPU_DETECTOR_MONITORING = 3,
    DECODE_CPU_DETECTOR_LOCKED = 4
} decode_cpu_detector_phase;

typedef enum {
    DECODE_CRITICAL_ATTENTION_MATMUL = 0,
    DECODE_CRITICAL_MLP_MATMUL = 1,
    DECODE_CRITICAL_KV_CACHE_UPDATE = 2,
    DECODE_CRITICAL_LOGITS = 3,
    DECODE_CRITICAL_SAMPLING = 4,
    DECODE_CRITICAL_ARGMAX = 5,
    DECODE_CRITICAL_QUANTIZED_MATMUL = 6,
    DECODE_CRITICAL_SOFTMAX = 7,
    DECODE_CRITICAL_RMSNORM = 8,
    DECODE_CRITICAL_FUSED_OPS = 9,
    DECODE_CRITICAL_OTHER = 10
} decode_critical_op_type;

typedef enum {
    EXEC_BACKEND_CPU = 0,
    EXEC_BACKEND_CUDA = 1,
    EXEC_BACKEND_METAL = 2,
    EXEC_BACKEND_VULKAN = 3,
    EXEC_BACKEND_OPENCL = 4,
    EXEC_BACKEND_UNKNOWN = 5
} execution_backend;

typedef struct {
    const char * op_name;
    decode_critical_op_type op_type;
    execution_backend expected_backend;
    execution_backend actual_backend;
    bool is_decode_critical;
    bool violation_detected;
} op_binding_record;

typedef struct {
    const char * op_name;
    const char * tensor_name;
    decode_critical_op_type op_type;
    execution_backend cpu_attempted_backend;
    execution_backend gpu_tensor_backend;
    bool was_during_decode;
    bool violation_logged;
    uint64_t violation_timestamp_ns;
} cpu_execution_violation_record;

typedef struct {
    bool decode_in_progress;
    bool detector_armed;
    bool backend_binding_frozen;
    bool sampling_authority_gpu_only;
    bool cpu_tensor_access_blocked;
    bool all_ops_backend_verified;
    uint64_t detector_arm_timestamp_ns;
} decode_cpu_execution_detector_config;

typedef struct {
    size_t total_ops_monitored;
    size_t gpu_ops_executed;
    size_t cpu_ops_attempted;
    size_t cpu_violations_blocked;
    size_t backend_mismatches;
    size_t tensor_access_violations;
    bool detection_active;
} decode_cpu_execution_validation_result;

class decode_cpu_execution_detector {
private:
    decode_cpu_execution_detector_config immutable_config;
    std::vector<op_binding_record> op_bindings;
    std::vector<cpu_execution_violation_record> violation_log;
    std::map<const char *, op_binding_record> op_registry;

    std::atomic<decode_cpu_detector_phase> current_phase;
    std::atomic<bool> detector_armed;
    std::atomic<bool> decode_in_progress;
    std::atomic<bool> detection_active;

    std::atomic<uint32_t> monitored_ops;
    std::atomic<uint32_t> gpu_ops_count;
    std::atomic<uint32_t> cpu_attempts;
    std::atomic<uint32_t> violations_blocked;

public:
    decode_cpu_execution_detector();

    bool initialize();
    bool enable_strict_mode(bool enable);

    bool define_decode_phase_flag();
    bool tag_decode_critical_ops();
    bool arm_detector();
    bool begin_decode_phase();
    bool end_decode_phase();

    bool register_op_binding(const char * op_name, decode_critical_op_type op_type,
                           bool is_decode_critical);
    bool set_op_expected_backend(const char * op_name, execution_backend backend);

    bool detect_cpu_op_execution(const char * op_name, execution_backend actual_backend);
    bool verify_op_backend_binding(const char * op_name, execution_backend actual_backend);
    bool verify_tensor_backend_access(const char * tensor_name, execution_backend tensor_backend);
    bool verify_sampling_authority(execution_backend sampling_backend);

    bool attempt_cpu_execution(const char * op_name, decode_critical_op_type op_type);
    bool attempt_cpu_tensor_access(const char * tensor_name, execution_backend tensor_backend);

    const decode_cpu_execution_detector_config & get_config() const { return immutable_config; }
    bool is_detector_armed() const { return detector_armed.load(); }
    bool is_decode_in_progress() const { return decode_in_progress.load(); }
    bool is_detection_active() const { return detection_active.load(); }
    decode_cpu_detector_phase get_current_phase() const { return current_phase.load(); }

    void record_op_execution(const char * op_name, execution_backend backend);
    void record_violation(const char * op_name, const char * tensor_name,
                         decode_critical_op_type op_type, execution_backend cpu_backend);
    void record_backend_mismatch(const char * op_name, execution_backend expected,
                               execution_backend actual);

    size_t get_monitored_ops() const { return monitored_ops.load(); }
    size_t get_gpu_ops() const { return gpu_ops_count.load(); }
    size_t get_cpu_attempts() const { return cpu_attempts.load(); }
    size_t get_violations_blocked() const { return violations_blocked.load(); }

    std::vector<op_binding_record> get_op_bindings() const { return op_bindings; }
    std::vector<cpu_execution_violation_record> get_violations() const { return violation_log; }

    decode_cpu_execution_validation_result validate_decode_cpu_purity() const;
    bool verify_no_cpu_execution() const;
    bool verify_backend_binding_stable() const;
    bool verify_sampling_gpu_exclusive() const;
};

class decode_cpu_detector_guard {
private:
    bool guard_active;
    bool decode_phase_active;

public:
    decode_cpu_detector_guard();
    ~decode_cpu_detector_guard();

    bool is_guard_active() const;
};

extern decode_cpu_execution_detector * g_decode_cpu_execution_detector;

bool llama_init_decode_cpu_execution_detector();
bool llama_enable_cpu_detector_strict_mode(bool enable);

bool llama_define_decode_phase_flag();
bool llama_tag_decode_critical_ops();
bool llama_arm_cpu_detector();
bool llama_begin_decode_phase_detection();
bool llama_end_decode_phase_detection();

bool llama_register_op_binding(const char * op_name, int op_type, bool is_decode_critical);
bool llama_set_op_expected_backend(const char * op_name, int backend);

bool llama_detect_cpu_op_execution(const char * op_name, int actual_backend);
bool llama_verify_op_backend_binding(const char * op_name, int actual_backend);
bool llama_verify_tensor_backend_access(const char * tensor_name, int tensor_backend);
bool llama_verify_sampling_authority(int sampling_backend);

bool llama_attempt_cpu_execution(const char * op_name, int op_type);
bool llama_attempt_cpu_tensor_access(const char * tensor_name, int tensor_backend);

bool llama_is_cpu_detector_armed();
bool llama_is_decode_detection_active();

void llama_record_op_execution(const char * op_name, int backend);
void llama_record_cpu_violation(const char * op_name, const char * tensor_name,
                              int op_type, int cpu_backend);
void llama_record_backend_mismatch(const char * op_name, int expected, int actual);

bool llama_validate_decode_cpu_purity();
bool llama_verify_no_cpu_execution();
bool llama_verify_backend_binding_stable();
bool llama_verify_sampling_gpu_exclusive();

void llama_print_cpu_detector_status();
void llama_print_op_binding_summary();
void llama_print_cpu_execution_violations();
void llama_print_decode_purity_report();

#define ASSERT_DECODE_GPU_EXECUTION(op_name, backend) \
    do { \
        if (g_decode_cpu_execution_detector && !llama_detect_cpu_op_execution(op_name, backend)) { \
            abort(); \
        } \
    } while(0)

#define ASSERT_OP_BACKEND_BINDING(op_name, backend) \
    do { \
        if (g_decode_cpu_execution_detector && !llama_verify_op_backend_binding(op_name, backend)) { \
            abort(); \
        } \
    } while(0)

#define GUARD_CPU_OP_EXECUTION(op_name, op_type) \
    do { \
        if (g_decode_cpu_execution_detector && !llama_attempt_cpu_execution(op_name, op_type)) { \
            abort(); \
        } \
    } while(0)

#define GUARD_CPU_TENSOR_ACCESS(tensor_name, backend) \
    do { \
        if (g_decode_cpu_execution_detector && !llama_attempt_cpu_tensor_access(tensor_name, backend)) { \
            abort(); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif
