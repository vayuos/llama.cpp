#include "llama-backend-audit-log.h"
#include <cstring>
#include <cstdio>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>

// Global state
backend_usage_audit_logger * g_backend_usage_audit_logger = nullptr;

// ============================================================================
// backend_usage_audit_logger Implementation
// ============================================================================

backend_usage_audit_logger::backend_usage_audit_logger()
    : current_phase(AUDIT_PHASE_UNINITIALIZED),
      audit_performed(false),
      audit_passed(false),
      audit_timestamp(0) {
    std::memset(&audit_summary, 0, sizeof(audit_summary));
}

bool backend_usage_audit_logger::initialize() {
    if (current_phase != AUDIT_PHASE_UNINITIALIZED) {
        fprintf(stderr, "[AUDIT] ERROR: Already initialized (phase=%d)\n", current_phase);
        return false;
    }

    current_phase = AUDIT_PHASE_SETUP;
    audit_summary.audit_timestamp_ns = 0;
    audit_nodes.clear();
    violations.clear();
    op_type_counts.clear();
    backend_counts.clear();
    kernel_counts.clear();

    fprintf(stdout, "[AUDIT] Backend audit logger initialized\n");
    return true;
}

bool backend_usage_audit_logger::mark_graph_built() {
    if (current_phase != AUDIT_PHASE_SETUP) {
        fprintf(stderr, "[AUDIT] ERROR: Invalid phase for graph_built (%d)\n", current_phase);
        return false;
    }

    current_phase = AUDIT_PHASE_GRAPH_BUILT;
    fprintf(stdout, "[AUDIT] Decode graph built, ready for enumeration\n");
    return true;
}

bool backend_usage_audit_logger::begin_enumeration() {
    if (current_phase != AUDIT_PHASE_GRAPH_BUILT) {
        fprintf(stderr, "[AUDIT] ERROR: Invalid phase for enumeration (%d)\n", current_phase);
        return false;
    }

    current_phase = AUDIT_PHASE_ENUMERATION;
    audit_nodes.clear();
    fprintf(stdout, "[AUDIT] Beginning backend enumeration...\n");
    return true;
}

bool backend_usage_audit_logger::enumerate_decode_node(const char * op_name,
                                                       const char * tensor_shape,
                                                       backend_ownership backend,
                                                       kernel_variant kernel_type,
                                                       bool is_decode_critical) {
    if (current_phase != AUDIT_PHASE_ENUMERATION) {
        fprintf(stderr, "[AUDIT] ERROR: Not in enumeration phase (%d)\n", current_phase);
        return false;
    }

    if (!op_name || !tensor_shape) {
        fprintf(stderr, "[AUDIT] ERROR: Null op_name or tensor_shape\n");
        return false;
    }

    // Create audit node
    backend_audit_node node = {
        op_name,
        tensor_shape,
        backend,
        kernel_type,
        (uint64_t)audit_nodes.size(),
        is_decode_critical,
        kernel_type != KERNEL_VARIANT_UNFUSED,
        ""
    };

    // Track statistics
    if (is_decode_critical) {
        audit_summary.total_decode_critical++;

        switch (backend) {
            case BACKEND_OWNERSHIP_CUDA:
                audit_summary.cuda_owned++;
                break;
            case BACKEND_OWNERSHIP_METAL:
                audit_summary.metal_owned++;
                break;
            case BACKEND_OWNERSHIP_VULKAN:
                audit_summary.vulkan_owned++;
                break;
            case BACKEND_OWNERSHIP_OPENCL:
                audit_summary.opencl_owned++;
                break;
            case BACKEND_OWNERSHIP_CPU:
                audit_summary.cpu_owned++;
                audit_summary.cpu_ownership_detected = true;
                break;
            default:
                audit_summary.unknown_owned++;
                break;
        }

        switch (kernel_type) {
            case KERNEL_VARIANT_MMQ:
                audit_summary.mmq_kernels++;
                break;
            case KERNEL_VARIANT_CUBLAS:
                audit_summary.cublas_kernels++;
                break;
            case KERNEL_VARIANT_FUSED:
                audit_summary.fused_kernels++;
                break;
            case KERNEL_VARIANT_UNFUSED:
                audit_summary.unfused_kernels++;
                break;
            case KERNEL_VARIANT_FLASH_ATTENTION:
                audit_summary.flash_attention_kernels++;
                break;
            case KERNEL_VARIANT_DENSE_ATTENTION:
                audit_summary.dense_attention_kernels++;
                break;
            default:
                break;
        }

        backend_counts[backend]++;
        kernel_counts[kernel_type]++;
    }

    op_type_counts[op_name]++;
    audit_summary.total_nodes++;

    audit_nodes.push_back(node);

    // Check for CPU ownership violation on critical ops
    if (is_decode_critical && backend == BACKEND_OWNERSHIP_CPU) {
        fprintf(stderr, "[AUDIT] WARNING: CPU-owned decode-critical op detected: %s\n", op_name);
    }

    return true;
}

bool backend_usage_audit_logger::finalize_enumeration() {
    if (current_phase != AUDIT_PHASE_ENUMERATION) {
        fprintf(stderr, "[AUDIT] ERROR: Invalid phase for finalize (%d)\n", current_phase);
        return false;
    }

    current_phase = AUDIT_PHASE_REPORTING;

    audit_summary.all_critical_ops_gpu =
        (audit_summary.cpu_owned == 0 && audit_summary.total_decode_critical > 0);

    fprintf(stdout, "[AUDIT] Enumeration complete: %u total nodes, %u decode-critical\n",
            audit_summary.total_nodes, audit_summary.total_decode_critical);

    return true;
}

bool backend_usage_audit_logger::generate_audit_report() {
    if (current_phase != AUDIT_PHASE_REPORTING) {
        fprintf(stderr, "[AUDIT] ERROR: Invalid phase for report generation (%d)\n", current_phase);
        return false;
    }

    std::ostringstream oss;

    // Header
    oss << "\n";
    oss << "==== DECODE BACKEND AUDIT REPORT ====\n";
    oss << "\n";

    // Summary statistics
    oss << "BACKEND OWNERSHIP SUMMARY:\n";
    oss << "  Total nodes:              " << audit_summary.total_nodes << "\n";
    oss << "  Decode-critical ops:      " << audit_summary.total_decode_critical << "\n";
    oss << "  GPU-owned (CUDA):         " << audit_summary.cuda_owned << "\n";
    oss << "  GPU-owned (Metal):        " << audit_summary.metal_owned << "\n";
    oss << "  GPU-owned (Vulkan):       " << audit_summary.vulkan_owned << "\n";
    oss << "  GPU-owned (OpenCL):       " << audit_summary.opencl_owned << "\n";
    oss << "  CPU-owned:                " << audit_summary.cpu_owned << " ⚠️ \n";
    oss << "  Unknown:                  " << audit_summary.unknown_owned << "\n";
    oss << "\n";

    // Kernel variant statistics
    oss << "KERNEL VARIANT SELECTION:\n";
    oss << "  MMQ quantized:            " << audit_summary.mmq_kernels << "\n";
    oss << "  cuBLAS:                   " << audit_summary.cublas_kernels << "\n";
    oss << "  Fused kernels:            " << audit_summary.fused_kernels << "\n";
    oss << "  Unfused kernels:          " << audit_summary.unfused_kernels << "\n";
    oss << "  Flash Attention:          " << audit_summary.flash_attention_kernels << "\n";
    oss << "  Dense Attention:          " << audit_summary.dense_attention_kernels << "\n";
    oss << "\n";

    // Audit result
    if (audit_summary.cpu_ownership_detected) {
        oss << "❌ AUDIT FAILED: CPU-owned decode-critical ops detected!\n";
        audit_summary.audit_passed = false;
    } else if (audit_summary.total_decode_critical == 0) {
        oss << "⚠️  WARNING: No decode-critical ops enumerated\n";
        audit_summary.audit_passed = false;
    } else {
        oss << "✅ AUDIT PASSED: All decode-critical ops GPU-owned\n";
        audit_summary.audit_passed = true;
    }

    oss << "\n";

    // Per-op enumeration (first 50 shown for readability)
    if (audit_nodes.size() > 0) {
        oss << "DECODE-CRITICAL OPS ENUMERATION (first 50):\n";
        uint32_t shown = 0;
        for (const auto & node : audit_nodes) {
            if (!node.is_decode_critical) continue;
            if (shown >= 50) {
                oss << "  ... and " << (audit_summary.total_decode_critical - 50) << " more\n";
                break;
            }

            oss << "  Op[" << std::setw(3) << std::setfill('0') << node.op_index << "]: "
                << std::setw(25) << std::setfill(' ') << std::left << node.op_name
                << " → " << std::setw(10) << format_backend_name(node.backend)
                << " (" << format_kernel_variant(node.kernel_type) << ")\n";

            shown++;
        }
        oss << "\n";
    }

    // Violations (if any)
    if (violations.size() > 0) {
        oss << "❌ VIOLATIONS DETECTED:\n";
        for (const auto & vio : violations) {
            oss << "  - " << vio.violation_description << "\n";
            oss << "    CPU-owned ops: " << vio.cpu_owned_op_count << "\n";
            oss << "    First violator: " << vio.first_cpu_op_name << "\n";
        }
        oss << "\n";
    }

    // Footer
    oss << "======================================\n";
    oss << "\n";

    audit_report = oss.str();
    audit_timestamp.store(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    audit_summary.audit_timestamp_ns = audit_timestamp.load();

    printf("%s", audit_report.c_str());

    return true;
}

bool backend_usage_audit_logger::validate_audit_results() {
    if (current_phase != AUDIT_PHASE_REPORTING) {
        fprintf(stderr, "[AUDIT] ERROR: Invalid phase for validation (%d)\n", current_phase);
        return false;
    }

    current_phase = AUDIT_PHASE_VALIDATION;

    // Perform all validation checks
    bool no_cpu = verify_no_cpu_ownership();
    bool consistent_kernels = verify_kernel_selection_consistency();
    bool coverage = verify_decode_critical_coverage();
    bool backend_consistent = verify_backend_consistency();

    bool all_valid = no_cpu && consistent_kernels && coverage && backend_consistent;

    audit_passed.store(all_valid);
    audit_performed.store(true);

    if (all_valid) {
        fprintf(stdout, "[AUDIT] All validation checks passed ✅\n");
    } else {
        fprintf(stderr, "[AUDIT] Validation failed ❌\n");
    }

    current_phase = AUDIT_PHASE_COMPLETE;
    return all_valid;
}

bool backend_usage_audit_logger::record_cpu_ownership_violation(const char * op_name) {
    if (!op_name) {
        fprintf(stderr, "[AUDIT] ERROR: Null op_name for violation\n");
        return false;
    }

    backend_audit_violation vio;
    vio.violation_description = "CPU backend detected for decode-critical op";
    vio.cpu_owned_op_count = 1;
    vio.first_cpu_op_name = op_name;
    vio.violation_timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    vio.abort_triggered = true;

    violations.push_back(vio);

    fprintf(stderr, "[AUDIT] ❌ CPU VIOLATION: %s\n", op_name);

    return true;
}

std::string backend_usage_audit_logger::format_backend_name(backend_ownership backend) const {
    switch (backend) {
        case BACKEND_OWNERSHIP_CUDA:
            return "CUDA";
        case BACKEND_OWNERSHIP_METAL:
            return "Metal";
        case BACKEND_OWNERSHIP_VULKAN:
            return "Vulkan";
        case BACKEND_OWNERSHIP_OPENCL:
            return "OpenCL";
        case BACKEND_OWNERSHIP_CPU:
            return "CPU";
        default:
            return "Unknown";
    }
}

std::string backend_usage_audit_logger::format_kernel_variant(kernel_variant kernel) const {
    switch (kernel) {
        case KERNEL_VARIANT_MMQ:
            return "MMQ";
        case KERNEL_VARIANT_CUBLAS:
            return "cuBLAS";
        case KERNEL_VARIANT_FUSED:
            return "Fused";
        case KERNEL_VARIANT_UNFUSED:
            return "Unfused";
        case KERNEL_VARIANT_FLASH_ATTENTION:
            return "Flash";
        case KERNEL_VARIANT_DENSE_ATTENTION:
            return "Dense";
        case KERNEL_VARIANT_QUANTIZED:
            return "Quantized";
        case KERNEL_VARIANT_FP32:
            return "FP32";
        default:
            return "Unknown";
    }
}

std::string backend_usage_audit_logger::generate_json_report() const {
    std::ostringstream oss;

    oss << "{\n";
    oss << "  \"audit_timestamp\": " << audit_summary.audit_timestamp_ns << ",\n";
    oss << "  \"total_nodes\": " << audit_summary.total_nodes << ",\n";
    oss << "  \"decode_critical_ops\": " << audit_summary.total_decode_critical << ",\n";
    oss << "  \"backend_ownership\": {\n";
    oss << "    \"cuda\": " << audit_summary.cuda_owned << ",\n";
    oss << "    \"metal\": " << audit_summary.metal_owned << ",\n";
    oss << "    \"vulkan\": " << audit_summary.vulkan_owned << ",\n";
    oss << "    \"opencl\": " << audit_summary.opencl_owned << ",\n";
    oss << "    \"cpu\": " << audit_summary.cpu_owned << "\n";
    oss << "  },\n";
    oss << "  \"kernel_variants\": {\n";
    oss << "    \"mmq\": " << audit_summary.mmq_kernels << ",\n";
    oss << "    \"cublas\": " << audit_summary.cublas_kernels << ",\n";
    oss << "    \"fused\": " << audit_summary.fused_kernels << ",\n";
    oss << "    \"unfused\": " << audit_summary.unfused_kernels << ",\n";
    oss << "    \"flash_attention\": " << audit_summary.flash_attention_kernels << ",\n";
    oss << "    \"dense_attention\": " << audit_summary.dense_attention_kernels << "\n";
    oss << "  },\n";
    oss << "  \"audit_passed\": " << (audit_summary.audit_passed ? "true" : "false") << ",\n";
    oss << "  \"cpu_ownership_detected\": " << (audit_summary.cpu_ownership_detected ? "true" : "false") << "\n";
    oss << "}\n";

    return oss.str();
}

bool backend_usage_audit_logger::verify_no_cpu_ownership() const {
    if (audit_summary.cpu_owned > 0) {
        fprintf(stderr, "[AUDIT] VALIDATION FAILED: CPU ownership detected for %u ops\n",
                audit_summary.cpu_owned);
        return false;
    }
    return true;
}

bool backend_usage_audit_logger::verify_kernel_selection_consistency() const {
    // Check that kernel variants are consistent
    uint32_t total_kernels = audit_summary.mmq_kernels + audit_summary.cublas_kernels +
                            audit_summary.fused_kernels + audit_summary.unfused_kernels +
                            audit_summary.flash_attention_kernels + audit_summary.dense_attention_kernels;

    if (total_kernels != audit_summary.total_decode_critical) {
        fprintf(stderr, "[AUDIT] VALIDATION FAILED: Kernel variant count mismatch\n");
        fprintf(stderr, "  Total kernels: %u, Decode-critical: %u\n",
                total_kernels, audit_summary.total_decode_critical);
        return false;
    }

    return true;
}

bool backend_usage_audit_logger::verify_decode_critical_coverage() const {
    if (audit_summary.total_decode_critical == 0) {
        fprintf(stderr, "[AUDIT] VALIDATION WARNING: No decode-critical ops enumerated\n");
        return false;
    }
    return true;
}

bool backend_usage_audit_logger::verify_backend_consistency() const {
    uint32_t total_gpu = audit_summary.cuda_owned + audit_summary.metal_owned +
                        audit_summary.vulkan_owned + audit_summary.opencl_owned;

    if (total_gpu + audit_summary.cpu_owned + audit_summary.unknown_owned !=
        audit_summary.total_decode_critical) {
        fprintf(stderr, "[AUDIT] VALIDATION FAILED: Backend count mismatch\n");
        return false;
    }

    return true;
}

// ============================================================================
// backend_audit_guard Implementation
// ============================================================================

backend_audit_guard::backend_audit_guard(backend_usage_audit_logger * logger_ptr)
    : guard_active(false), logger(logger_ptr) {
    if (logger) {
        guard_active = true;
    }
}

backend_audit_guard::~backend_audit_guard() {
    guard_active = false;
}

// ============================================================================
// C-Style Wrapper Functions
// ============================================================================

bool llama_init_backend_audit_logger() {
    if (g_backend_usage_audit_logger != nullptr) {
        fprintf(stderr, "[AUDIT] Already initialized\n");
        return false;
    }

    g_backend_usage_audit_logger = new backend_usage_audit_logger();
    if (!g_backend_usage_audit_logger->initialize()) {
        fprintf(stderr, "[AUDIT] Failed to initialize audit logger\n");
        delete g_backend_usage_audit_logger;
        g_backend_usage_audit_logger = nullptr;
        return false;
    }

    return true;
}

bool llama_mark_graph_built() {
    if (!g_backend_usage_audit_logger) {
        return false;
    }
    return g_backend_usage_audit_logger->mark_graph_built();
}

bool llama_begin_backend_enumeration() {
    if (!g_backend_usage_audit_logger) {
        return false;
    }
    return g_backend_usage_audit_logger->begin_enumeration();
}

bool llama_enumerate_decode_node(const char * op_name,
                                 const char * tensor_shape,
                                 int backend_enum,
                                 int kernel_variant_enum,
                                 bool is_decode_critical) {
    if (!g_backend_usage_audit_logger) {
        return false;
    }

    return g_backend_usage_audit_logger->enumerate_decode_node(
        op_name, tensor_shape,
        (backend_ownership)backend_enum,
        (kernel_variant)kernel_variant_enum,
        is_decode_critical);
}

bool llama_finalize_backend_enumeration() {
    if (!g_backend_usage_audit_logger) {
        return false;
    }
    return g_backend_usage_audit_logger->finalize_enumeration();
}

bool llama_generate_backend_audit_report() {
    if (!g_backend_usage_audit_logger) {
        return false;
    }
    return g_backend_usage_audit_logger->generate_audit_report();
}

bool llama_validate_backend_audit() {
    if (!g_backend_usage_audit_logger) {
        return false;
    }
    return g_backend_usage_audit_logger->validate_audit_results();
}

bool llama_record_cpu_ownership_violation(const char * op_name) {
    if (!g_backend_usage_audit_logger) {
        return false;
    }
    return g_backend_usage_audit_logger->record_cpu_ownership_violation(op_name);
}

bool llama_is_backend_audit_complete() {
    if (!g_backend_usage_audit_logger) {
        return false;
    }
    return g_backend_usage_audit_logger->is_audit_complete();
}

bool llama_did_backend_audit_pass() {
    if (!g_backend_usage_audit_logger) {
        return false;
    }
    return g_backend_usage_audit_logger->did_audit_pass();
}

const backend_audit_summary * llama_get_backend_audit_summary() {
    if (!g_backend_usage_audit_logger) {
        return nullptr;
    }
    return &g_backend_usage_audit_logger->get_summary();
}

const char * llama_get_backend_audit_report() {
    if (!g_backend_usage_audit_logger) {
        return "";
    }
    return g_backend_usage_audit_logger->get_report().c_str();
}

void llama_print_backend_audit_report() {
    if (g_backend_usage_audit_logger) {
        printf("%s", g_backend_usage_audit_logger->get_report().c_str());
    }
}

void llama_print_backend_audit_summary() {
    if (!g_backend_usage_audit_logger) {
        return;
    }

    const auto & summary = g_backend_usage_audit_logger->get_summary();
    printf("\n=== BACKEND AUDIT SUMMARY ===\n");
    printf("Total nodes: %u\n", summary.total_nodes);
    printf("Decode-critical: %u\n", summary.total_decode_critical);
    printf("GPU-owned: %u\n", summary.cuda_owned + summary.metal_owned +
           summary.vulkan_owned + summary.opencl_owned);
    printf("CPU-owned: %u\n", summary.cpu_owned);
    printf("Audit passed: %s\n", summary.audit_passed ? "YES" : "NO");
    printf("==============================\n\n");
}

void llama_print_backend_audit_violations() {
    if (!g_backend_usage_audit_logger) {
        return;
    }

    const auto & violations = g_backend_usage_audit_logger->get_violations();
    if (violations.empty()) {
        printf("No violations detected\n");
        return;
    }

    printf("\n=== BACKEND AUDIT VIOLATIONS ===\n");
    for (const auto & vio : violations) {
        printf("- %s\n", vio.violation_description);
        printf("  CPU-owned ops: %u\n", vio.cpu_owned_op_count);
        printf("  First violator: %s\n", vio.first_cpu_op_name);
    }
    printf("=================================\n\n");
}

void llama_print_backend_audit_node_details() {
    if (!g_backend_usage_audit_logger) {
        return;
    }

    const auto & nodes = g_backend_usage_audit_logger->get_audit_nodes();
    printf("\n=== BACKEND AUDIT NODE DETAILS ===\n");
    printf("Total nodes: %zu\n", nodes.size());

    for (const auto & node : nodes) {
        if (node.is_decode_critical) {
            printf("  [%lu] %s → %s (%s)\n",
                   node.op_index,
                   node.op_name,
                   g_backend_usage_audit_logger->format_backend_name((backend_ownership)node.backend).c_str(),
                   g_backend_usage_audit_logger->format_kernel_variant((kernel_variant)node.kernel_type).c_str());
        }
    }
    printf("===================================\n\n");
}

void llama_export_backend_audit_json(const char * filename) {
    if (!g_backend_usage_audit_logger || !filename) {
        fprintf(stderr, "[AUDIT] Invalid audit logger or filename\n");
        return;
    }

    std::string json = g_backend_usage_audit_logger->generate_json_report();
    FILE * f = fopen(filename, "w");
    if (f) {
        fprintf(f, "%s", json.c_str());
        fclose(f);
        printf("[AUDIT] JSON report exported to %s\n", filename);
    } else {
        fprintf(stderr, "[AUDIT] Failed to open file %s for writing\n", filename);
    }
}

// ============================================================================
// Self-Test Suite (10 comprehensive tests)
// ============================================================================

static bool backend_audit_initialization_test() {
    fprintf(stdout, "\n[TEST] Backend Audit Initialization Test\n");

    auto * logger = new backend_usage_audit_logger();
    bool result = logger->initialize();

    if (!result) {
        fprintf(stderr, "  FAILED: Could not initialize\n");
        delete logger;
        return false;
    }

    if (logger->get_current_phase() != AUDIT_PHASE_SETUP) {
        fprintf(stderr, "  FAILED: Wrong phase after init\n");
        delete logger;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete logger;
    return true;
}

static bool backend_audit_phase_transition_test() {
    fprintf(stdout, "\n[TEST] Phase Transition Test\n");

    auto * logger = new backend_usage_audit_logger();
    logger->initialize();

    if (!logger->mark_graph_built()) {
        fprintf(stderr, "  FAILED: mark_graph_built\n");
        delete logger;
        return false;
    }

    if (!logger->begin_enumeration()) {
        fprintf(stderr, "  FAILED: begin_enumeration\n");
        delete logger;
        return false;
    }

    if (!logger->finalize_enumeration()) {
        fprintf(stderr, "  FAILED: finalize_enumeration\n");
        delete logger;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete logger;
    return true;
}

static bool backend_audit_enumeration_test() {
    fprintf(stdout, "\n[TEST] Node Enumeration Test\n");

    auto * logger = new backend_usage_audit_logger();
    logger->initialize();
    logger->mark_graph_built();
    logger->begin_enumeration();

    // Enumerate some nodes
    bool success = true;
    success &= logger->enumerate_decode_node("MatMul_0", "[16,64]", BACKEND_OWNERSHIP_CUDA,
                                            KERNEL_VARIANT_MMQ, true);
    success &= logger->enumerate_decode_node("RMSNorm_0", "[16,64]", BACKEND_OWNERSHIP_CUDA,
                                            KERNEL_VARIANT_FUSED, true);
    success &= logger->enumerate_decode_node("Softmax_0", "[16,1]", BACKEND_OWNERSHIP_CUDA,
                                            KERNEL_VARIANT_UNFUSED, true);

    if (!success) {
        fprintf(stderr, "  FAILED: Enumeration\n");
        delete logger;
        return false;
    }

    if (logger->get_node_count() != 3) {
        fprintf(stderr, "  FAILED: Node count mismatch\n");
        delete logger;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete logger;
    return true;
}

static bool backend_audit_cpu_detection_test() {
    fprintf(stdout, "\n[TEST] CPU Ownership Detection Test\n");

    auto * logger = new backend_usage_audit_logger();
    logger->initialize();
    logger->mark_graph_built();
    logger->begin_enumeration();

    // Enumerate CPU-owned op
    logger->enumerate_decode_node("BadMatMul", "[16,64]", BACKEND_OWNERSHIP_CPU,
                                 KERNEL_VARIANT_UNFUSED, true);

    logger->finalize_enumeration();

    if (!logger->is_audit_complete()) {
        const auto & summary = logger->get_summary();
        if (summary.cpu_owned != 1) {
            fprintf(stderr, "  FAILED: CPU ownership not detected\n");
            delete logger;
            return false;
        }
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete logger;
    return true;
}

static bool backend_audit_report_generation_test() {
    fprintf(stdout, "\n[TEST] Report Generation Test\n");

    auto * logger = new backend_usage_audit_logger();
    logger->initialize();
    logger->mark_graph_built();
    logger->begin_enumeration();

    logger->enumerate_decode_node("MatMul", "[64,64]", BACKEND_OWNERSHIP_CUDA,
                                 KERNEL_VARIANT_MMQ, true);

    logger->finalize_enumeration();

    if (!logger->generate_audit_report()) {
        fprintf(stderr, "  FAILED: Report generation\n");
        delete logger;
        return false;
    }

    const std::string & report = logger->get_report();
    if (report.empty()) {
        fprintf(stderr, "  FAILED: Empty report\n");
        delete logger;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete logger;
    return true;
}

static bool backend_audit_validation_test() {
    fprintf(stdout, "\n[TEST] Audit Validation Test\n");

    auto * logger = new backend_usage_audit_logger();
    logger->initialize();
    logger->mark_graph_built();
    logger->begin_enumeration();

    logger->enumerate_decode_node("MatMul", "[64,64]", BACKEND_OWNERSHIP_CUDA,
                                 KERNEL_VARIANT_MMQ, true);
    logger->enumerate_decode_node("RMSNorm", "[64]", BACKEND_OWNERSHIP_CUDA,
                                 KERNEL_VARIANT_FUSED, true);

    logger->finalize_enumeration();
    logger->generate_audit_report();

    if (!logger->validate_audit_results()) {
        fprintf(stderr, "  FAILED: Validation\n");
        delete logger;
        return false;
    }

    if (!logger->did_audit_pass()) {
        fprintf(stderr, "  FAILED: Audit should pass\n");
        delete logger;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete logger;
    return true;
}

static bool backend_audit_mixed_backend_test() {
    fprintf(stdout, "\n[TEST] Mixed Backend Test\n");

    auto * logger = new backend_usage_audit_logger();
    logger->initialize();
    logger->mark_graph_built();
    logger->begin_enumeration();

    logger->enumerate_decode_node("MatMul_CUDA", "[64,64]", BACKEND_OWNERSHIP_CUDA,
                                 KERNEL_VARIANT_MMQ, true);
    logger->enumerate_decode_node("MatMul_Metal", "[64,64]", BACKEND_OWNERSHIP_METAL,
                                 KERNEL_VARIANT_CUBLAS, true);

    logger->finalize_enumeration();

    const auto & summary = logger->get_summary();
    if (summary.cuda_owned != 1 || summary.metal_owned != 1) {
        fprintf(stderr, "  FAILED: Backend counts incorrect\n");
        delete logger;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete logger;
    return true;
}

static bool backend_audit_kernel_variant_test() {
    fprintf(stdout, "\n[TEST] Kernel Variant Tracking Test\n");

    auto * logger = new backend_usage_audit_logger();
    logger->initialize();
    logger->mark_graph_built();
    logger->begin_enumeration();

    logger->enumerate_decode_node("MatMul_MMQ", "[64,64]", BACKEND_OWNERSHIP_CUDA,
                                 KERNEL_VARIANT_MMQ, true);
    logger->enumerate_decode_node("MatMul_cuBLAS", "[64,64]", BACKEND_OWNERSHIP_CUDA,
                                 KERNEL_VARIANT_CUBLAS, true);
    logger->enumerate_decode_node("RMSNorm_Fused", "[64]", BACKEND_OWNERSHIP_CUDA,
                                 KERNEL_VARIANT_FUSED, true);

    logger->finalize_enumeration();

    const auto & summary = logger->get_summary();
    if (summary.mmq_kernels != 1 || summary.cublas_kernels != 1 || summary.fused_kernels != 1) {
        fprintf(stderr, "  FAILED: Kernel variant counts incorrect\n");
        delete logger;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete logger;
    return true;
}

static bool backend_audit_json_export_test() {
    fprintf(stdout, "\n[TEST] JSON Export Test\n");

    auto * logger = new backend_usage_audit_logger();
    logger->initialize();
    logger->mark_graph_built();
    logger->begin_enumeration();

    logger->enumerate_decode_node("MatMul", "[64,64]", BACKEND_OWNERSHIP_CUDA,
                                 KERNEL_VARIANT_MMQ, true);

    logger->finalize_enumeration();

    std::string json = logger->generate_json_report();
    if (json.empty() || json.find("\"audit_passed\"") == std::string::npos) {
        fprintf(stderr, "  FAILED: Invalid JSON\n");
        delete logger;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete logger;
    return true;
}

static bool backend_audit_full_workflow_test() {
    fprintf(stdout, "\n[TEST] Full Workflow Test\n");

    auto * logger = new backend_usage_audit_logger();

    if (!logger->initialize() ||
        !logger->mark_graph_built() ||
        !logger->begin_enumeration()) {
        fprintf(stderr, "  FAILED: Phase setup\n");
        delete logger;
        return false;
    }

    // Enumerate various ops
    for (int i = 0; i < 10; i++) {
        char name[32];
        snprintf(name, sizeof(name), "Op_%d", i);
        if (!logger->enumerate_decode_node(name, "[64,64]", BACKEND_OWNERSHIP_CUDA,
                                          KERNEL_VARIANT_MMQ, true)) {
            fprintf(stderr, "  FAILED: Enumeration\n");
            delete logger;
            return false;
        }
    }

    if (!logger->finalize_enumeration() ||
        !logger->generate_audit_report() ||
        !logger->validate_audit_results()) {
        fprintf(stderr, "  FAILED: Finalization\n");
        delete logger;
        return false;
    }

    if (!logger->is_audit_complete() || !logger->did_audit_pass()) {
        fprintf(stderr, "  FAILED: Audit not complete/passed\n");
        delete logger;
        return false;
    }

    fprintf(stdout, "  PASSED ✅\n");
    delete logger;
    return true;
}

// Self-test runner
static bool run_backend_audit_self_tests() {
    fprintf(stdout, "\n========================================\n");
    fprintf(stdout, "Running Backend Audit Logger Self-Tests\n");
    fprintf(stdout, "========================================\n");

    bool all_passed = true;
    all_passed &= backend_audit_initialization_test();
    all_passed &= backend_audit_phase_transition_test();
    all_passed &= backend_audit_enumeration_test();
    all_passed &= backend_audit_cpu_detection_test();
    all_passed &= backend_audit_report_generation_test();
    all_passed &= backend_audit_validation_test();
    all_passed &= backend_audit_mixed_backend_test();
    all_passed &= backend_audit_kernel_variant_test();
    all_passed &= backend_audit_json_export_test();
    all_passed &= backend_audit_full_workflow_test();

    fprintf(stdout, "\n========================================\n");
    if (all_passed) {
        fprintf(stdout, "All tests PASSED ✅\n");
    } else {
        fprintf(stdout, "Some tests FAILED ❌\n");
    }
    fprintf(stdout, "========================================\n\n");

    return all_passed;
}

// Auto-run self-tests on module load
__attribute__((constructor))
static void backend_audit_self_tests_ctor() {
    // Uncomment to auto-run tests on module load:
    // run_backend_audit_self_tests();
}
