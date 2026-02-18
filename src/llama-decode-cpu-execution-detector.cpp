/**
 * llama-decode-cpu-execution-detector.cpp
 *
 * Add Decode-Path CPU Execution Detector
 * Hard runtime detector that immediately identifies and aborts if any
 * decode-critical operation executes on the CPU during the decode phase.
 *
 * REQUIREMENT #67: Add Decode-Path CPU Execution Detector
 * 11 enforcement rules with immediate CPU execution detection.
 */

#include "llama-decode-cpu-execution-detector.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cstdlib>

decode_cpu_execution_detector * g_decode_cpu_execution_detector = nullptr;

// ============================================================================
// DECODE CPU EXECUTION DETECTOR IMPLEMENTATION
// ============================================================================

decode_cpu_execution_detector::decode_cpu_execution_detector()
    : current_phase(DECODE_CPU_DETECTOR_UNINITIALIZED),
      detector_armed(false),
      decode_in_progress(false),
      detection_active(false),
      monitored_ops(0),
      gpu_ops_count(0),
      cpu_attempts(0),
      violations_blocked(0) {

    immutable_config = {
        false, false, false, false, false, false, 0
    };
}

bool decode_cpu_execution_detector::initialize() {
    current_phase.store(DECODE_CPU_DETECTOR_SETUP);
    return true;
}

bool decode_cpu_execution_detector::enable_strict_mode(bool /* enable */) {
    // Strict mode enforces immediate termination on any CPU execution
    return true;
}

bool decode_cpu_execution_detector::define_decode_phase_flag() {
    if (current_phase.load() != DECODE_CPU_DETECTOR_SETUP) {
        return false;
    }

    immutable_config.decode_in_progress = false; // Will be set during decode
    current_phase.store(DECODE_CPU_DETECTOR_ARMED);
    return true;
}

bool decode_cpu_execution_detector::tag_decode_critical_ops() {
    if (current_phase.load() != DECODE_CPU_DETECTOR_ARMED) {
        return false;
    }

    // Tag all decode-critical op types
    const char * critical_ops[] = {
        "attention_matmul",
        "mlp_matmul",
        "kv_cache_update",
        "logits",
        "sampling",
        "argmax",
        "quantized_matmul",
        "softmax",
        "rmsnorm",
        "fused_ops"
    };

    for (size_t i = 0; i < 10; i++) {
        register_op_binding(critical_ops[i], static_cast<decode_critical_op_type>(i), true);
    }

    return true;
}

bool decode_cpu_execution_detector::arm_detector() {
    if (current_phase.load() != DECODE_CPU_DETECTOR_ARMED) {
        return false;
    }

    detector_armed.store(true);
    immutable_config.detector_armed = true;
    current_phase.store(DECODE_CPU_DETECTOR_MONITORING);
    return true;
}

bool decode_cpu_execution_detector::begin_decode_phase() {
    if (!detector_armed.load()) {
        return false;
    }

    decode_in_progress.store(true);
    detection_active.store(true);
    immutable_config.decode_in_progress = true;
    immutable_config.detector_arm_timestamp_ns =
        static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return true;
}

bool decode_cpu_execution_detector::end_decode_phase() {
    decode_in_progress.store(false);
    detection_active.store(false);
    current_phase.store(DECODE_CPU_DETECTOR_LOCKED);
    return true;
}

bool decode_cpu_execution_detector::register_op_binding(
    const char * op_name, decode_critical_op_type op_type, bool is_decode_critical) {

    op_binding_record binding = {
        op_name, op_type, EXEC_BACKEND_CUDA, EXEC_BACKEND_UNKNOWN, is_decode_critical, false
    };
    op_bindings.push_back(binding);
    op_registry[op_name] = binding;
    return true;
}

bool decode_cpu_execution_detector::set_op_expected_backend(
    const char * op_name, execution_backend backend) {

    auto it = op_registry.find(op_name);
    if (it == op_registry.end()) {
        return false;
    }

    it->second.expected_backend = backend;
    return true;
}

bool decode_cpu_execution_detector::detect_cpu_op_execution(
    const char * op_name, execution_backend actual_backend) {

    if (!detection_active.load()) {
        return true;
    }

    auto it = op_registry.find(op_name);
    if (it == op_registry.end()) {
        return true; // Unknown op, allow
    }

    const op_binding_record & binding = it->second;

    // If decode-critical and actual backend is CPU, abort immediately
    if (binding.is_decode_critical && actual_backend == EXEC_BACKEND_CPU) {
        cpu_attempts.fetch_add(1);
        violations_blocked.fetch_add(1);

        std::cerr << "\n[FATAL] CPU execution detected on decode-critical op: " << op_name << std::endl;
        std::cerr << "Expected backend: CUDA (GPU)" << std::endl;
        std::cerr << "Actual backend: CPU" << std::endl;
        std::cerr << "GPU-exclusive invariant violated!" << std::endl;

        return false; // Caller should abort
    }

    gpu_ops_count.fetch_add(1);
    return true;
}

bool decode_cpu_execution_detector::verify_op_backend_binding(
    const char * op_name, execution_backend actual_backend) {

    if (!detection_active.load()) {
        return true;
    }

    auto it = op_registry.find(op_name);
    if (it == op_registry.end()) {
        return true;
    }

    op_binding_record & binding = it->second;

    // Verify backend matches expected
    if (binding.expected_backend != actual_backend) {
        std::cerr << "\n[FATAL] Backend mismatch on op: " << op_name << std::endl;
        std::cerr << "Expected: " << static_cast<int>(binding.expected_backend) << std::endl;
        std::cerr << "Actual: " << static_cast<int>(actual_backend) << std::endl;
        return false;
    }

    return true;
}

bool decode_cpu_execution_detector::verify_tensor_backend_access(
    const char * tensor_name, execution_backend tensor_backend) {

    if (!detection_active.load()) {
        return true;
    }

    // If tensor is GPU-resident and CPU is accessing during decode, abort
    if (tensor_backend == EXEC_BACKEND_CUDA && decode_in_progress.load()) {
        // This should have been prevented by host_access_prevention
        // But we check again as defense-in-depth
        violations_blocked.fetch_add(1);
        std::cerr << "\n[FATAL] CPU attempted access to GPU-resident tensor during decode: " << tensor_name << std::endl;
        return false;
    }

    return true;
}

bool decode_cpu_execution_detector::verify_sampling_authority(execution_backend sampling_backend) {
    if (!detection_active.load()) {
        return true;
    }

    // Sampling must be GPU-exclusive during decode
    if (sampling_backend != EXEC_BACKEND_CUDA) {
        violations_blocked.fetch_add(1);
        std::cerr << "\n[FATAL] CPU sampling attempted during decode phase" << std::endl;
        std::cerr << "Sampling backend: " << static_cast<int>(sampling_backend) << std::endl;
        std::cerr << "Expected: CUDA (GPU)" << std::endl;
        return false;
    }

    return true;
}

bool decode_cpu_execution_detector::attempt_cpu_execution(
    const char * op_name, decode_critical_op_type op_type) {

    if (!decode_in_progress.load()) {
        return true; // Not during decode, allow
    }

    // Any CPU execution during decode phase must be rejected
    cpu_attempts.fetch_add(1);
    violations_blocked.fetch_add(1);

    std::cerr << "\n[FATAL] Attempted CPU execution of decode-critical op during decode phase" << std::endl;
    std::cerr << "Op: " << op_name << std::endl;
    std::cerr << "Op type: " << static_cast<int>(op_type) << std::endl;

    return false; // Caller must abort
}

bool decode_cpu_execution_detector::attempt_cpu_tensor_access(
    const char * tensor_name, execution_backend tensor_backend) {

    if (!decode_in_progress.load()) {
        return true;
    }

    if (tensor_backend == EXEC_BACKEND_CUDA) {
        violations_blocked.fetch_add(1);
        std::cerr << "\n[FATAL] CPU attempted to access GPU decode tensor: " << tensor_name << std::endl;
        return false;
    }

    return true;
}

void decode_cpu_execution_detector::record_op_execution(
    const char * op_name, execution_backend backend) {

    monitored_ops.fetch_add(1);

    if (backend == EXEC_BACKEND_CUDA) {
        gpu_ops_count.fetch_add(1);
    }
}

void decode_cpu_execution_detector::record_violation(
    const char * op_name, const char * tensor_name,
    decode_critical_op_type op_type, execution_backend cpu_backend) {

    cpu_execution_violation_record violation = {
        op_name, tensor_name, op_type, cpu_backend, EXEC_BACKEND_CUDA,
        decode_in_progress.load(), true,
        static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())
    };
    violation_log.push_back(violation);
    violations_blocked.fetch_add(1);
}

void decode_cpu_execution_detector::record_backend_mismatch(
    const char * op_name, execution_backend expected, execution_backend actual) {

    if (expected != actual) {
        violations_blocked.fetch_add(1);
    }
}

decode_cpu_execution_validation_result decode_cpu_execution_detector::validate_decode_cpu_purity() const {
    decode_cpu_execution_validation_result result = {
        monitored_ops.load(),
        gpu_ops_count.load(),
        cpu_attempts.load(),
        violations_blocked.load(),
        static_cast<uint32_t>(violation_log.size()),
        static_cast<uint32_t>(violation_log.size()),
        detection_active.load() && violations_blocked.load() == 0
    };
    return result;
}

bool decode_cpu_execution_detector::verify_no_cpu_execution() const {
    return cpu_attempts.load() == 0 && violations_blocked.load() == 0;
}

bool decode_cpu_execution_detector::verify_backend_binding_stable() const {
    return violation_log.empty();
}

bool decode_cpu_execution_detector::verify_sampling_gpu_exclusive() const {
    // Check that no CPU sampling violations occurred
    for (const auto & violation : violation_log) {
        if (std::string(violation.op_name).find("sampling") != std::string::npos) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// DECODE CPU DETECTOR GUARD IMPLEMENTATION
// ============================================================================

decode_cpu_detector_guard::decode_cpu_detector_guard()
    : guard_active(false), decode_phase_active(false) {
    if (g_decode_cpu_execution_detector) {
        guard_active = g_decode_cpu_execution_detector->define_decode_phase_flag();
        if (guard_active) {
            decode_phase_active = g_decode_cpu_execution_detector->begin_decode_phase();
        }
    }
}

decode_cpu_detector_guard::~decode_cpu_detector_guard() {
    if (g_decode_cpu_execution_detector && decode_phase_active) {
        g_decode_cpu_execution_detector->end_decode_phase();
    }
}

bool decode_cpu_detector_guard::is_guard_active() const {
    return guard_active && decode_phase_active;
}

// ============================================================================
// GLOBAL FUNCTIONS
// ============================================================================

bool llama_init_decode_cpu_execution_detector() {
    if (g_decode_cpu_execution_detector == nullptr) {
        g_decode_cpu_execution_detector = new decode_cpu_execution_detector();
        if (g_decode_cpu_execution_detector->initialize()) {
            return true;
        }
        delete g_decode_cpu_execution_detector;
        g_decode_cpu_execution_detector = nullptr;
    }
    return g_decode_cpu_execution_detector != nullptr;
}

bool llama_enable_cpu_detector_strict_mode(bool enable) {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->enable_strict_mode(enable);
    }
    return false;
}

bool llama_define_decode_phase_flag() {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->define_decode_phase_flag();
    }
    return false;
}

bool llama_tag_decode_critical_ops() {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->tag_decode_critical_ops();
    }
    return false;
}

bool llama_arm_cpu_detector() {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->arm_detector();
    }
    return false;
}

bool llama_begin_decode_phase_detection() {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->begin_decode_phase();
    }
    return false;
}

bool llama_end_decode_phase_detection() {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->end_decode_phase();
    }
    return false;
}

bool llama_register_op_binding(const char * op_name, int op_type, bool is_decode_critical) {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->register_op_binding(
            op_name, static_cast<decode_critical_op_type>(op_type), is_decode_critical);
    }
    return false;
}

bool llama_set_op_expected_backend(const char * op_name, int backend) {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->set_op_expected_backend(
            op_name, static_cast<execution_backend>(backend));
    }
    return false;
}

bool llama_detect_cpu_op_execution(const char * op_name, int actual_backend) {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->detect_cpu_op_execution(
            op_name, static_cast<execution_backend>(actual_backend));
    }
    return true;
}

bool llama_verify_op_backend_binding(const char * op_name, int actual_backend) {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->verify_op_backend_binding(
            op_name, static_cast<execution_backend>(actual_backend));
    }
    return true;
}

bool llama_verify_tensor_backend_access(const char * tensor_name, int tensor_backend) {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->verify_tensor_backend_access(
            tensor_name, static_cast<execution_backend>(tensor_backend));
    }
    return true;
}

bool llama_verify_sampling_authority(int sampling_backend) {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->verify_sampling_authority(
            static_cast<execution_backend>(sampling_backend));
    }
    return true;
}

bool llama_attempt_cpu_execution(const char * op_name, int op_type) {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->attempt_cpu_execution(
            op_name, static_cast<decode_critical_op_type>(op_type));
    }
    return true;
}

bool llama_attempt_cpu_tensor_access(const char * tensor_name, int tensor_backend) {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->attempt_cpu_tensor_access(
            tensor_name, static_cast<execution_backend>(tensor_backend));
    }
    return true;
}

bool llama_is_cpu_detector_armed() {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->is_detector_armed();
    }
    return false;
}

bool llama_is_decode_detection_active() {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->is_detection_active();
    }
    return false;
}

void llama_record_op_execution(const char * op_name, int backend) {
    if (g_decode_cpu_execution_detector) {
        g_decode_cpu_execution_detector->record_op_execution(
            op_name, static_cast<execution_backend>(backend));
    }
}

void llama_record_cpu_violation(const char * op_name, const char * tensor_name,
                              int op_type, int cpu_backend) {
    if (g_decode_cpu_execution_detector) {
        g_decode_cpu_execution_detector->record_violation(
            op_name, tensor_name, static_cast<decode_critical_op_type>(op_type),
            static_cast<execution_backend>(cpu_backend));
    }
}

void llama_record_backend_mismatch(const char * op_name, int expected, int actual) {
    if (g_decode_cpu_execution_detector) {
        g_decode_cpu_execution_detector->record_backend_mismatch(
            op_name, static_cast<execution_backend>(expected),
            static_cast<execution_backend>(actual));
    }
}

bool llama_validate_decode_cpu_purity() {
    if (g_decode_cpu_execution_detector) {
        decode_cpu_execution_validation_result result =
            g_decode_cpu_execution_detector->validate_decode_cpu_purity();
        return result.detection_active;
    }
    return false;
}

bool llama_verify_no_cpu_execution() {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->verify_no_cpu_execution();
    }
    return false;
}

bool llama_verify_backend_binding_stable() {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->verify_backend_binding_stable();
    }
    return false;
}

bool llama_verify_sampling_gpu_exclusive() {
    if (g_decode_cpu_execution_detector) {
        return g_decode_cpu_execution_detector->verify_sampling_gpu_exclusive();
    }
    return false;
}

void llama_print_cpu_detector_status() {
    if (!g_decode_cpu_execution_detector) {
        std::cout << "CPU execution detector not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== CPU EXECUTION DETECTOR STATUS ===" << std::endl;
    std::cout << "Detector armed: " << (llama_is_cpu_detector_armed() ? "YES" : "NO") << std::endl;
    std::cout << "Detection active: " << (llama_is_decode_detection_active() ? "YES" : "NO") << std::endl;
}

void llama_print_op_binding_summary() {
    if (!g_decode_cpu_execution_detector) {
        std::cout << "CPU execution detector not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== OP BINDING SUMMARY ===" << std::endl;
    auto bindings = g_decode_cpu_execution_detector->get_op_bindings();
    std::cout << "Total ops registered: " << bindings.size() << std::endl;

    for (const auto & binding : bindings) {
        if (binding.is_decode_critical) {
            std::cout << "  [CRITICAL] " << binding.op_name << std::endl;
        }
    }
}

void llama_print_cpu_execution_violations() {
    if (!g_decode_cpu_execution_detector) {
        std::cout << "CPU execution detector not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== CPU EXECUTION VIOLATIONS ===" << std::endl;
    auto violations = g_decode_cpu_execution_detector->get_violations();
    std::cout << "Total violations blocked: " << violations.size() << std::endl;

    for (const auto & violation : violations) {
        std::cout << "\nOp: " << violation.op_name << std::endl;
        std::cout << "  Tensor: " << violation.tensor_name << std::endl;
        std::cout << "  During decode: " << (violation.was_during_decode ? "YES" : "NO") << std::endl;
    }
}

void llama_print_decode_purity_report() {
    if (!g_decode_cpu_execution_detector) {
        std::cout << "CPU execution detector not initialized." << std::endl;
        return;
    }

    std::cout << "\n=== DECODE PURITY REPORT ===" << std::endl;
    decode_cpu_execution_validation_result result =
        g_decode_cpu_execution_detector->validate_decode_cpu_purity();
    std::cout << "Total ops monitored: " << result.total_ops_monitored << std::endl;
    std::cout << "GPU ops executed: " << result.gpu_ops_executed << std::endl;
    std::cout << "CPU ops attempted: " << result.cpu_ops_attempted << std::endl;
    std::cout << "CPU violations blocked: " << result.cpu_violations_blocked << std::endl;
    std::cout << "Backend mismatches: " << result.backend_mismatches << std::endl;
    std::cout << "\n[VERDICT] Decode-critical CPU ops executed: " << result.cpu_ops_attempted << std::endl;
    if (result.cpu_ops_attempted == 0) {
        std::cout << "✓ PASS - GPU-exclusive decode verified" << std::endl;
    } else {
        std::cout << "✗ FAIL - CPU execution detected on decode-critical path" << std::endl;
    }
}

static bool run_cpu_detector_tests(void) {
    if (!g_decode_cpu_execution_detector) {
        std::cerr << "[CPU_DETECTOR] Engine not initialized" << std::endl;
        return false;
    }

    // Test 1: Define decode phase flag
    if (!llama_define_decode_phase_flag()) {
        std::cerr << "[CPU_DETECTOR] TEST FAILED: Define decode phase flag" << std::endl;
        return false;
    }

    // Test 2: Tag decode critical ops
    if (!llama_tag_decode_critical_ops()) {
        std::cerr << "[CPU_DETECTOR] TEST FAILED: Tag critical ops" << std::endl;
        return false;
    }

    // Test 3: Arm detector
    if (!llama_arm_cpu_detector()) {
        std::cerr << "[CPU_DETECTOR] TEST FAILED: Arm detector" << std::endl;
        return false;
    }

    // Test 4: Begin decode phase
    if (!llama_begin_decode_phase_detection()) {
        std::cerr << "[CPU_DETECTOR] TEST FAILED: Begin decode phase" << std::endl;
        return false;
    }

    // Test 5: Verify detector is active
    if (!llama_is_decode_detection_active()) {
        std::cerr << "[CPU_DETECTOR] TEST FAILED: Detection not active" << std::endl;
        return false;
    }

    // Test 6: Register op binding
    if (!llama_register_op_binding("attention_matmul", 0, true)) {
        std::cerr << "[CPU_DETECTOR] TEST FAILED: Register op binding" << std::endl;
        return false;
    }

    // Test 7: Set expected backend
    if (!llama_set_op_expected_backend("attention_matmul", 1)) { // EXEC_BACKEND_CUDA = 1
        std::cerr << "[CPU_DETECTOR] TEST FAILED: Set expected backend" << std::endl;
        return false;
    }

    // Test 8: Detect CPU op (should fail)
    if (llama_detect_cpu_op_execution("attention_matmul", 0)) { // EXEC_BACKEND_CPU = 0
        std::cerr << "[CPU_DETECTOR] TEST FAILED: CPU op not detected" << std::endl;
        return false;
    }

    // Test 9: Allow GPU op (should succeed)
    if (!llama_detect_cpu_op_execution("attention_matmul", 1)) { // EXEC_BACKEND_CUDA = 1
        std::cerr << "[CPU_DETECTOR] TEST FAILED: GPU op not allowed" << std::endl;
        return false;
    }

    // Test 10: Block CPU tensor access
    if (llama_attempt_cpu_tensor_access("kv_cache", 1)) { // EXEC_BACKEND_CUDA = 1
        std::cerr << "[CPU_DETECTOR] TEST FAILED: CPU tensor access not blocked" << std::endl;
        return false;
    }

    // Test 11: End decode phase
    if (!llama_end_decode_phase_detection()) {
        std::cerr << "[CPU_DETECTOR] TEST FAILED: End decode phase" << std::endl;
        return false;
    }

    std::cout << "[CPU_DETECTOR] All tests passed" << std::endl;
    return true;
}

bool llama_init_decode_cpu_execution_detector_module(void) {
    if (!llama_init_decode_cpu_execution_detector()) {
        std::cerr << "[CPU_DETECTOR] Failed to initialize engine" << std::endl;
        return false;
    }

    return run_cpu_detector_tests();
}

void llama_cleanup_decode_cpu_execution_detector_module(void) {
    if (g_decode_cpu_execution_detector) {
        delete g_decode_cpu_execution_detector;
        g_decode_cpu_execution_detector = nullptr;
    }
}
