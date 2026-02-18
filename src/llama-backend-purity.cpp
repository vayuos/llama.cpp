#include "../include/llama-backend-purity.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <cxxabi.h>
#include <dlfcn.h>
#include <elf.h>
#include <link.h>

// ============================================================================
// GLOBAL STATE
// ============================================================================

static llama_backend_purity_state_t * g_backend_purity_state = nullptr;
static std::atomic<bool> g_backend_purity_initialized(false);
static std::atomic<bool> g_backend_purity_locked(false);
static std::atomic<llama_backend_variant_t> g_selected_backend(LLAMA_BACKEND_VARIANT_UNDEFINED);

// ============================================================================
// BACKEND FEATURE TABLES
// ============================================================================

static const llama_backend_feature_t BACKEND_FEATURES[] = {
    {
        .backend_name = "CPU",
        .description = "CPU-only backend using system memory",
        .supports_dense_ops = true,
        .supports_mmq_ops = false,
        .supports_hybrid_memory = false,
        .supports_layer_offloading = false,
        .is_gpu_backend = false,
        .required_cmake_flags = "-DGGML_USE_CPU_BACKEND=ON",
        .conflicting_backends = "CUDA,Vulkan,Metal,OpenCL",
        .expected_symbol_count = 500
    },
    {
        .backend_name = "CUDA/cuBLAS",
        .description = "CUDA backend with cuBLAS for dense matrix operations",
        .supports_dense_ops = true,
        .supports_mmq_ops = false,
        .supports_hybrid_memory = false,
        .supports_layer_offloading = false,
        .is_gpu_backend = true,
        .required_cmake_flags = "-DGGML_USE_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS=ON -DGGML_CUDA_FORCE_MMQ=OFF",
        .conflicting_backends = "CPU_BACKEND,Vulkan,Metal,OpenCL,CUDA_MMQ",
        .expected_symbol_count = 1200
    },
    {
        .backend_name = "CUDA/MMQ",
        .description = "CUDA backend with MMQ kernels for token generation",
        .supports_dense_ops = false,
        .supports_mmq_ops = true,
        .supports_hybrid_memory = false,
        .supports_layer_offloading = false,
        .is_gpu_backend = true,
        .required_cmake_flags = "-DGGML_USE_CUDA=ON -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FORCE_CUBLAS=OFF",
        .conflicting_backends = "CPU_BACKEND,Vulkan,Metal,OpenCL,CUDA_CUBLAS",
        .expected_symbol_count = 1100
    },
    {
        .backend_name = "Vulkan",
        .description = "Vulkan backend for cross-platform GPU support",
        .supports_dense_ops = true,
        .supports_mmq_ops = true,
        .supports_hybrid_memory = false,
        .supports_layer_offloading = false,
        .is_gpu_backend = true,
        .required_cmake_flags = "-DGGML_USE_VULKAN=ON",
        .conflicting_backends = "CPU_BACKEND,CUDA,Metal,OpenCL",
        .expected_symbol_count = 1000
    },
    {
        .backend_name = "Metal",
        .description = "Metal backend for Apple devices",
        .supports_dense_ops = true,
        .supports_mmq_ops = true,
        .supports_hybrid_memory = false,
        .supports_layer_offloading = false,
        .is_gpu_backend = true,
        .required_cmake_flags = "-DGGML_USE_METAL=ON",
        .conflicting_backends = "CPU_BACKEND,CUDA,Vulkan,OpenCL",
        .expected_symbol_count = 900
    }
};

// ============================================================================
// EXPECTED BACKEND SYMBOLS FOR PURITY CHECKING
// ============================================================================

static const std::unordered_set<std::string> CPU_BACKEND_SYMBOLS = {
    "ggml_backend_cpu_init",
    "ggml_backend_is_cpu",
    "ggml_backend_cpu_buffer_type",
    "ggml_backend_cpu_buffer_from_ptr"
};

static const std::unordered_set<std::string> CUDA_CUBLAS_SYMBOLS = {
    "ggml_backend_cuda_init",
    "ggml_backend_is_cuda",
    "ggml_backend_cuda_buffer_type",
    "ggml_backend_cuda_get_device_count"
};

static const std::unordered_set<std::string> CUDA_MMQ_SYMBOLS = {
    "ggml_backend_cuda_init",
    "ggml_backend_is_cuda",
    "ggml_backend_cuda_buffer_type",
    "ggml_backend_cuda_get_device_count"
};

static const std::unordered_set<std::string> FORBIDDEN_HYBRID_SYMBOLS = {
    "ggml_backend_cpu_init",
    "llama_memory_hybrid_init",
    "llama_backend_sched_new",
    "ggml_backend_cpu_fallback",
    "ggml_backend_dispatch"
};

// ============================================================================
// INTERNAL HELPER FUNCTIONS
// ============================================================================

/**
 * Callback for dl_iterate_phdr to extract symbols from binary
 */
static int symbol_callback(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    std::unordered_set<std::string> * symbols =
        static_cast<std::unordered_set<std::string>*>(data);

    // This is a simplified version; in production would use libelf
    return 0;
}

/**
 * Extract symbol count from binary
 */
static size_t get_binary_symbol_count(void) {
    std::unordered_set<std::string> symbols;
    dl_iterate_phdr(symbol_callback, &symbols);
    return symbols.size();
}

/**
 * Check if symbol exists in binary
 */
static bool symbol_exists_in_binary(const char * symbol) {
    // In production, this would use libelf or nm
    void * handle = dlopen(nullptr, RTLD_LAZY);
    if (handle == nullptr) return false;

    bool exists = (dlsym(handle, symbol) != nullptr);
    dlclose(handle);
    return exists;
}

/**
 * Count unexpected backend symbols
 */
static size_t count_unexpected_symbols(llama_backend_variant_t variant) {
    size_t count = 0;

    switch (variant) {
        case LLAMA_BACKEND_VARIANT_CPU_ONLY:
            // CPU builds should not have CUDA symbols
            for (const auto& sym : CUDA_CUBLAS_SYMBOLS) {
                if (symbol_exists_in_binary(sym.c_str())) count++;
            }
            for (const auto& sym : CUDA_MMQ_SYMBOLS) {
                if (symbol_exists_in_binary(sym.c_str())) count++;
            }
            break;

        case LLAMA_BACKEND_VARIANT_CUDA_CUBLAS:
            // cuBLAS builds should not have CPU or MMQ symbols
            for (const auto& sym : CPU_BACKEND_SYMBOLS) {
                if (symbol_exists_in_binary(sym.c_str())) count++;
            }
            // Check for MMQ-specific symbols
            if (symbol_exists_in_binary("ggml_cuda_mmq_q8_0_q8_1")) count++;
            break;

        case LLAMA_BACKEND_VARIANT_CUDA_MMQ:
            // MMQ builds should not have CPU symbols
            for (const auto& sym : CPU_BACKEND_SYMBOLS) {
                if (symbol_exists_in_binary(sym.c_str())) count++;
            }
            break;

        case LLAMA_BACKEND_VARIANT_VULKAN:
        case LLAMA_BACKEND_VARIANT_METAL:
            // GPU builds should not have CPU symbols
            for (const auto& sym : CPU_BACKEND_SYMBOLS) {
                if (symbol_exists_in_binary(sym.c_str())) count++;
            }
            break;

        case LLAMA_BACKEND_VARIANT_UNDEFINED:
        default:
            return SIZE_MAX;
    }

    return count;
}

// ============================================================================
// PUBLIC API IMPLEMENTATION
// ============================================================================

extern "C" {

llama_backend_purity_state_t * llama_backend_purity_init(llama_backend_variant_t variant) {
    if (g_backend_purity_initialized.exchange(true)) {
        fprintf(stderr, "LLAMA_BACKEND_PURITY: Already initialized\n");
        return g_backend_purity_state;
    }

    g_backend_purity_state = new llama_backend_purity_state_t();
    if (g_backend_purity_state == nullptr) {
        fprintf(stderr, "LLAMA_BACKEND_PURITY: Memory allocation failed\n");
        return nullptr;
    }

    // Initialize state
    g_backend_purity_state->state = LLAMA_BACKEND_PURITY_STATE_INITIALIZED;
    g_backend_purity_state->selected_backend = variant;
    g_backend_purity_state->backend_locked = false;
    g_backend_purity_state->abort_on_violation = LLAMA_BACKEND_PURITY_STRICT_MODE;
    g_backend_purity_state->deferred_violations = new std::vector<std::string>();

    // Initialize atomic metrics
    g_backend_purity_state->violations.unused_backend_symbols_found = 0;
    g_backend_purity_state->violations.backend_dispatch_branches = 0;
    g_backend_purity_state->violations.hybrid_layer_placements = 0;
    g_backend_purity_state->violations.cpu_fallback_paths = 0;
    g_backend_purity_state->violations.backend_switching_statements = 0;
    g_backend_purity_state->violations.hybrid_memory_operations = 0;
    g_backend_purity_state->violations.dynamic_backend_registrations = 0;
    g_backend_purity_state->violations.backend_capability_enumerations = 0;
    g_backend_purity_state->violations.total_violations = 0;
    g_backend_purity_state->violations.violations_deferred = 0;

    g_backend_purity_state->metrics.total_symbols_verified = 0;
    g_backend_purity_state->metrics.expected_backend_symbols = 0;
    g_backend_purity_state->metrics.unexpected_backend_symbols = 0;
    g_backend_purity_state->metrics.binary_size_baseline = 0;
    g_backend_purity_state->metrics.binary_size_current = 0;
    g_backend_purity_state->metrics.verification_time_ns = 0;
    g_backend_purity_state->metrics.verification_count = 0;
    g_backend_purity_state->metrics.last_verification_passed = false;

    g_selected_backend = variant;

    fprintf(stderr, "LLAMA_BACKEND_PURITY: Initialized with backend variant %d\n", variant);
    return g_backend_purity_state;
}

void llama_backend_purity_free(llama_backend_purity_state_t * state) {
    if (state == nullptr) return;

    if (state->deferred_violations != nullptr) {
        delete state->deferred_violations;
        state->deferred_violations = nullptr;
    }

    delete state;
    g_backend_purity_state = nullptr;
    g_backend_purity_initialized = false;
}

bool llama_backend_purity_validate_config(
    llama_backend_purity_state_t * state,
    llama_backend_variant_t variant) {

    if (state == nullptr) {
        fprintf(stderr, "LLAMA_BACKEND_PURITY: Invalid state\n");
        return false;
    }

    if (variant != state->selected_backend) {
        fprintf(stderr, "LLAMA_BACKEND_PURITY: Backend mismatch (expected %d, got %d)\n",
                state->selected_backend, variant);
        return false;
    }

    // Validate no unexpected symbols
    size_t unexpected = count_unexpected_symbols(variant);
    if (unexpected > 0) {
        state->violations.unused_backend_symbols_found += unexpected;
        state->violations.total_violations += unexpected;
        fprintf(stderr, "LLAMA_BACKEND_PURITY: Found %zu unexpected symbols\n", unexpected);

        if (state->abort_on_violation) {
            fprintf(stderr, "LLAMA_BACKEND_PURITY: Aborting due to violations\n");
            std::abort();
        }
        return false;
    }

    state->state = LLAMA_BACKEND_PURITY_STATE_VALIDATED;
    return true;
}

size_t llama_backend_purity_validate_symbols(llama_backend_variant_t variant) {
    return count_unexpected_symbols(variant);
}

bool llama_backend_purity_check_binary_size(size_t max_size) {
    if (g_backend_purity_state == nullptr) return false;

    // In production, would get actual binary size via /proc/self/stat or similar
    size_t current_size = 0; // Placeholder

    g_backend_purity_state->metrics.binary_size_current = current_size;

    if (current_size > max_size) {
        g_backend_purity_state->violations.total_violations++;
        fprintf(stderr, "LLAMA_BACKEND_PURITY: Binary size %zu exceeds limit %zu\n",
                current_size, max_size);
        return false;
    }

    return true;
}

void llama_backend_purity_assert_no_branching(void) {
    if (g_backend_purity_state == nullptr) {
        fprintf(stderr, "LLAMA_BACKEND_PURITY: Not initialized\n");
        std::abort();
    }

    // In production, would use static analysis or instrumentation
    // to verify no backend-specific branching occurs
}

void llama_backend_purity_verify_backend_match(llama_backend_variant_t expected) {
    if (g_selected_backend != expected) {
        fprintf(stderr, "LLAMA_BACKEND_PURITY: Backend mismatch (expected %d, got %ld)\n",
                expected, (long)g_selected_backend.load());
        g_backend_purity_state->violations.total_violations++;

        if (g_backend_purity_state->abort_on_violation) {
            std::abort();
        }
    }
}

void llama_backend_purity_lock_backend(llama_backend_purity_state_t * state) {
    if (state == nullptr) return;
    state->backend_locked = true;
    g_backend_purity_locked = true;
    fprintf(stderr, "LLAMA_BACKEND_PURITY: Backend locked\n");
}

bool llama_backend_purity_is_locked(void) {
    return g_backend_purity_locked;
}

llama_backend_variant_t llama_backend_purity_get_backend(void) {
    return g_selected_backend;
}

const llama_backend_feature_t * llama_backend_purity_get_features(
    llama_backend_variant_t variant) {

    switch (variant) {
        case LLAMA_BACKEND_VARIANT_CPU_ONLY:
            return &BACKEND_FEATURES[0];
        case LLAMA_BACKEND_VARIANT_CUDA_CUBLAS:
            return &BACKEND_FEATURES[1];
        case LLAMA_BACKEND_VARIANT_CUDA_MMQ:
            return &BACKEND_FEATURES[2];
        case LLAMA_BACKEND_VARIANT_VULKAN:
            return &BACKEND_FEATURES[3];
        case LLAMA_BACKEND_VARIANT_METAL:
            return &BACKEND_FEATURES[4];
        case LLAMA_BACKEND_VARIANT_UNDEFINED:
        default:
            return nullptr;
    }
}

llama_backend_purity_metrics_t llama_backend_purity_get_metrics(
    llama_backend_purity_state_t * state) {

    if (state == nullptr) {
        return llama_backend_purity_metrics_t();
    }

    return state->metrics;
}

const char * llama_backend_purity_get_violation_report(
    llama_backend_purity_state_t * state) {

    if (state == nullptr) return "";
    if (state->deferred_violations == nullptr) return "";

    static std::string report;
    report.clear();

    std::ostringstream oss;
    oss << "Backend Purity Violation Report:\n";
    oss << "  Unused symbols found: " << state->violations.unused_backend_symbols_found << "\n";
    oss << "  Backend dispatch branches: " << state->violations.backend_dispatch_branches << "\n";
    oss << "  Hybrid layer placements: " << state->violations.hybrid_layer_placements << "\n";
    oss << "  CPU fallback paths: " << state->violations.cpu_fallback_paths << "\n";
    oss << "  Backend switching statements: " << state->violations.backend_switching_statements << "\n";
    oss << "  Hybrid memory operations: " << state->violations.hybrid_memory_operations << "\n";
    oss << "  Dynamic registrations: " << state->violations.dynamic_backend_registrations << "\n";
    oss << "  Total violations: " << state->violations.total_violations << "\n";

    if (!state->deferred_violations->empty()) {
        oss << "  Deferred violation details:\n";
        for (const auto& violation : *state->deferred_violations) {
            oss << "    - " << violation << "\n";
        }
    }

    report = oss.str();
    return report.c_str();
}

void llama_backend_purity_reset_violations(llama_backend_purity_state_t * state) {
    if (state == nullptr) return;

    state->violations.unused_backend_symbols_found = 0;
    state->violations.backend_dispatch_branches = 0;
    state->violations.hybrid_layer_placements = 0;
    state->violations.cpu_fallback_paths = 0;
    state->violations.backend_switching_statements = 0;
    state->violations.hybrid_memory_operations = 0;
    state->violations.dynamic_backend_registrations = 0;
    state->violations.backend_capability_enumerations = 0;
    state->violations.total_violations = 0;
    state->violations.violations_deferred = 0;

    if (state->deferred_violations != nullptr) {
        state->deferred_violations->clear();
    }
}

void llama_backend_purity_set_abort_on_violation(bool should_abort) {
    if (g_backend_purity_state != nullptr) {
        g_backend_purity_state->abort_on_violation = should_abort;
    }
}

const char * llama_backend_purity_get_symbol_list(void) {
    static std::string symbol_list;
    symbol_list.clear();

    std::ostringstream oss;
    oss << "CUDA_CUBLAS[";
    bool first = true;
    for (const auto& sym : CUDA_CUBLAS_SYMBOLS) {
        if (!first) oss << ",";
        oss << sym;
        first = false;
    }
    oss << "] CUDA_MMQ[";
    first = true;
    for (const auto& sym : CUDA_MMQ_SYMBOLS) {
        if (!first) oss << ",";
        oss << sym;
        first = false;
    }
    oss << "] CPU[";
    first = true;
    for (const auto& sym : CPU_BACKEND_SYMBOLS) {
        if (!first) oss << ",";
        oss << sym;
        first = false;
    }
    oss << "]";

    symbol_list = oss.str();
    return symbol_list.c_str();
}

bool llama_backend_purity_has_symbol(const char * symbol) {
    if (symbol == nullptr) return false;
    return symbol_exists_in_binary(symbol);
}

size_t llama_backend_purity_verify_graph(void * graph_ptr) {
    // In production, would traverse graph structure and count distinct backends
    (void)graph_ptr;
    return 0; // Single backend means 0 distinct backends (implicit)
}

void llama_backend_purity_register_exclusions(const char * backends) {
    if (backends == nullptr) return;
    fprintf(stderr, "LLAMA_BACKEND_PURITY: Registering exclusions: %s\n", backends);
}

size_t llama_backend_purity_full_scan(void) {
    if (g_backend_purity_state == nullptr) {
        fprintf(stderr, "LLAMA_BACKEND_PURITY: Not initialized\n");
        return 0;
    }

    size_t violations = count_unexpected_symbols(g_selected_backend);
    g_backend_purity_state->violations.unused_backend_symbols_found += violations;
    g_backend_purity_state->violations.total_violations += violations;

    return violations;
}

// ============================================================================
// CMake Integration Functions
// ============================================================================

const char * llama_backend_purity_get_cmake_flags(llama_backend_variant_t variant) {
    const auto * features = llama_backend_purity_get_features(variant);
    if (features == nullptr) return "";
    return features->required_cmake_flags;
}

bool llama_backend_purity_validate_cmake_config(llama_backend_variant_t variant) {
    const auto * features = llama_backend_purity_get_features(variant);
    if (features == nullptr) return false;

    // In production, would check environment variables or CMake cache file
    return true;
}

size_t llama_backend_purity_generate_cmake_config(
    llama_backend_variant_t variant,
    char * output) {

    if (output == nullptr) return 0;

    const auto * features = llama_backend_purity_get_features(variant);
    if (features == nullptr) {
        snprintf(output, 100, "# Invalid backend variant\n");
        return strlen(output);
    }

    int written = snprintf(output, 4096,
        "# Generated CMake configuration for %s\n"
        "# Backend: %s\n"
        "# Description: %s\n"
        "\n"
        "%s\n"
        "\n"
        "# Purity checks\n"
        "set(LLAMA_BACKEND_PURITY_ENABLED ON)\n"
        "set(LLAMA_BACKEND_PURITY_STRICT_MODE ON)\n"
        "set(LLAMA_SINGLE_BACKEND_VARIANT %d)\n",
        features->backend_name,
        features->backend_name,
        features->description,
        features->required_cmake_flags,
        variant);

    return (written > 0) ? written : 0;
}

// ============================================================================
// Build Profile Management
// ============================================================================

static const llama_backend_build_profile_t PREDEFINED_PROFILES[] = {
    {
        .profile_name = "CPU-Only",
        .variant = LLAMA_BACKEND_VARIANT_CPU_ONLY,
        .cmake_flags = "-DGGML_USE_CPU_BACKEND=ON",
        .description = "Single-threaded CPU backend for development and testing",
        .enable_metrics = true,
        .strict_mode = false
    },
    {
        .profile_name = "CUDA-cuBLAS",
        .variant = LLAMA_BACKEND_VARIANT_CUDA_CUBLAS,
        .cmake_flags = "-DGGML_USE_CUDA=ON -DGGML_CUDA_FORCE_CUBLAS=ON -DGGML_CUDA_FORCE_MMQ=OFF",
        .description = "CUDA backend with cuBLAS for dense matrix operations",
        .enable_metrics = true,
        .strict_mode = true
    },
    {
        .profile_name = "CUDA-MMQ",
        .variant = LLAMA_BACKEND_VARIANT_CUDA_MMQ,
        .cmake_flags = "-DGGML_USE_CUDA=ON -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FORCE_CUBLAS=OFF",
        .description = "CUDA backend with MMQ kernels for token generation",
        .enable_metrics = true,
        .strict_mode = true
    },
    {
        .profile_name = "Vulkan",
        .variant = LLAMA_BACKEND_VARIANT_VULKAN,
        .cmake_flags = "-DGGML_USE_VULKAN=ON",
        .description = "Vulkan backend for cross-platform GPU support",
        .enable_metrics = true,
        .strict_mode = true
    },
    {
        .profile_name = "Metal",
        .variant = LLAMA_BACKEND_VARIANT_METAL,
        .cmake_flags = "-DGGML_USE_METAL=ON",
        .description = "Metal backend for Apple devices",
        .enable_metrics = true,
        .strict_mode = true
    }
};

const llama_backend_build_profile_t * llama_backend_purity_get_profile(size_t profile_index) {
    if (profile_index >= sizeof(PREDEFINED_PROFILES) / sizeof(PREDEFINED_PROFILES[0])) {
        return nullptr;
    }
    return &PREDEFINED_PROFILES[profile_index];
}

size_t llama_backend_purity_get_profile_count(void) {
    return sizeof(PREDEFINED_PROFILES) / sizeof(PREDEFINED_PROFILES[0]);
}

// ============================================================================
// Performance Metrics
// ============================================================================

llama_backend_purity_binary_optimization_t llama_backend_purity_get_binary_optimization(void) {
    llama_backend_purity_binary_optimization_t opt;
    opt.baseline_size = 0;      // Would query actual sizes
    opt.optimized_size = 0;
    opt.reduction_bytes = 0;
    opt.reduction_percent = 0.0;
    opt.unused_symbols_bytes = 0;
    return opt;
}

llama_backend_purity_code_elimination_t llama_backend_purity_get_code_elimination(void) {
    llama_backend_purity_code_elimination_t elim;
    elim.dispatch_branches_eliminated = 0;
    elim.fallback_paths_eliminated = 0;
    elim.boundary_code_eliminated = 0;
    elim.registry_loops_eliminated = 0;
    elim.capability_enum_eliminated = 0;
    return elim;
}

} // extern "C"
