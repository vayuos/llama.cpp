# CONFIG_FREEZE Integration Guide

## Quick Start

This guide shows how to integrate the frozen configuration system into llama.cpp.

## File Locations

- **Header**: `//wsl.localhost/Ubuntu-24.04/home/viren/source/llama.cpp/llama.cpp/src/llama-config-freeze.h`
- **Implementation**: `//wsl.localhost/Ubuntu-24.04/home/viren/source/llama.cpp/llama.cpp/src/llama-config-freeze.cpp`
- **Documentation**: `//wsl.localhost/Ubuntu-24.04/home/viren/source/llama.cpp/llama.cpp/src/CONFIG_FREEZE.md`

## Integration Checklist

### Phase 1: Build System Integration

Add frozen config files to CMakeLists.txt:

```cmake
# In llama.cpp/src/CMakeLists.txt
set(LLAMA_SOURCES
    # ... existing sources ...
    llama-config-freeze.cpp
    # ... rest of sources ...
)

set(LLAMA_HEADERS
    # ... existing headers ...
    llama-config-freeze.h
    # ... rest of headers ...
)
```

### Phase 2: Header Integration

Include frozen config header in llama-context.h:

```cpp
// In llama-context.h (after existing includes)
#include "llama-config-freeze.h"

struct llama_context {
    // ... existing fields ...

    // NEW: Frozen configuration for decode
    // This structure is initialized at context creation and
    // locked when decode begins. It ensures zero runtime
    // flag evaluation during the decode-critical path.
    llama_frozen_config * frozen_config = nullptr;

    // ... rest of structure ...
};
```

### Phase 3: Context Constructor Integration

Initialize frozen config in llama_context constructor:

```cpp
// In llama-context.cpp constructor
llama_context::llama_context(const llama_model & model, llama_context_params params)
    : model(model)
    , cparams(params)
{
    // ... existing initialization code ...

    // ========== NEW: Initialize frozen configuration ==========
    frozen_config = llama_config_freeze_new();
    if (!frozen_config) {
        LLAMA_LOG_ERROR("Failed to allocate frozen configuration\n");
        throw std::runtime_error("Configuration freeze allocation failed");
    }

    // Phase 1: Parse CLI arguments (would come from application layer)
    // For now, use defaults via parse_cli(argc=0, argv=nullptr)
    int result = llama_config_freeze_parse_cli(frozen_config, 0, nullptr);
    if (result != 0) {
        LLAMA_LOG_ERROR("Failed to parse CLI configuration: %d\n", result);
        throw std::runtime_error("CLI configuration parsing failed");
    }

    // Phase 2: Resolve environment variables ONCE
    result = llama_config_freeze_resolve_env(frozen_config);
    if (result != 0) {
        LLAMA_LOG_ERROR("Failed to resolve environment: %d\n", result);
        throw std::runtime_error("Environment resolution failed");
    }

    // Phase 3: Bind backend dispatch
    result = llama_config_freeze_bind_backend(frozen_config, nullptr);
    if (result != 0) {
        LLAMA_LOG_ERROR("Failed to bind backend: %d\n", result);
        throw std::runtime_error("Backend binding failed");
    }

    // Phase 4: Bind sampling dispatch
    result = llama_config_freeze_bind_sampler(frozen_config);
    if (result != 0) {
        LLAMA_LOG_ERROR("Failed to bind sampler: %d\n", result);
        throw std::runtime_error("Sampler binding failed");
    }

    // Phase 5: Bind attention dispatch
    result = llama_config_freeze_bind_attention(frozen_config);
    if (result != 0) {
        LLAMA_LOG_ERROR("Failed to bind attention: %d\n", result);
        throw std::runtime_error("Attention binding failed");
    }

    // Phase 6: Allocate memory
    result = llama_config_freeze_allocate_memory(frozen_config, this);
    if (result != 0) {
        LLAMA_LOG_ERROR("Failed to allocate memory: %d\n", result);
        throw std::runtime_error("Memory allocation failed");
    }

    // Phase 7: Mark startup complete
    result = llama_config_freeze_startup_complete(frozen_config);
    if (result != 0) {
        LLAMA_LOG_ERROR("Startup completion failed: %d\n", result);
        throw std::runtime_error("Startup completion failed");
    }

    // Validate configuration is complete
    result = llama_config_freeze_validate_complete(frozen_config);
    if (result != 0) {
        LLAMA_LOG_ERROR("Configuration validation failed: %d\n", result);
        throw std::runtime_error("Configuration validation failed");
    }

    LLAMA_LOG_INFO("Frozen configuration initialized successfully\n");
    // ========== END: Frozen configuration initialization ==========

    // ... rest of existing constructor code ...
}
```

### Phase 4: Context Destructor Integration

Clean up frozen config in destructor:

```cpp
// In llama-context.cpp destructor
llama_context::~llama_context() {
    // ... existing cleanup code ...

    // NEW: Cleanup frozen configuration
    if (frozen_config) {
        llama_config_freeze_free(frozen_config);
        frozen_config = nullptr;
        LLAMA_LOG_DEBUG("Frozen configuration cleaned up\n");
    }

    // ... rest of existing cleanup code ...
}
```

### Phase 5: Decode Lock Integration

Add configuration lock at decode start:

```cpp
// In llama_context::decode()
int llama_context::decode(const llama_batch & batch_inp) {
    // NEW: Engage hard lock at decode start
    // This must happen exactly once per context before any decode operation
    if (frozen_config && !frozen_config->configuration_locked) {
        int lock_result = llama_config_freeze_lock_decode(frozen_config);
        if (lock_result != 0) {
            LLAMA_LOG_ERROR("Failed to lock configuration at decode start\n");
            return lock_result;
        }
        LLAMA_LOG_DEBUG("Configuration locked for decode\n");
    }

    // ... rest of existing decode implementation ...
}
```

### Phase 6: Static Dispatch Integration in Attention

Replace runtime branching with static dispatch:

```cpp
// BEFORE (in graph execution or attention computation)
if (ctx->flash_attention) {
    compute_flash_attention(...);
} else {
    compute_standard_attention(...);
}

// AFTER
llama_config_freeze_execute_attention(
    ctx->frozen_config,
    graph,
    q, k, v
);
```

### Phase 7: Static Dispatch Integration in Sampling

Replace sampling mode branching with static dispatch:

```cpp
// BEFORE (in sampler)
if (ctx->sampling_mode == TOP_K) {
    apply_top_k(logits, ...);
} else if (ctx->sampling_mode == TOP_P) {
    apply_top_p(logits, ...);
} else {
    apply_greedy(logits, ...);
}

// AFTER
llama_config_freeze_execute_sampler(
    ctx->frozen_config,
    logits,
    n_logits,
    sampled_tokens,
    n_samples
);
```

### Phase 8: Configuration Modification Guards

Add guards to all APIs that modify configuration:

```cpp
// In set_n_threads()
int llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    // NEW: Reject modifications during decode
    if (llama_config_freeze_reject_reconfig(frozen_config, "n_threads") != 0) {
        LLAMA_LOG_ERROR("Cannot modify thread count during decode\n");
        return -EINVAL;
    }

    // ... rest of implementation ...
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads_batch;
    return 0;
}

// In set_embeddings()
void llama_context::set_embeddings(bool value) {
    // NEW: Guard feature toggle
    if (llama_config_freeze_reject_reconfig(frozen_config, "embeddings") != 0) {
        LLAMA_LOG_ERROR("Cannot modify embeddings setting during decode\n");
        return;
    }

    // ... rest of implementation ...
}

// In set_causal_attn()
void llama_context::set_causal_attn(bool value) {
    // NEW: Guard attention setting
    if (llama_config_freeze_reject_reconfig(frozen_config, "causal_attn") != 0) {
        LLAMA_LOG_ERROR("Cannot modify causal attention during decode\n");
        return;
    }

    // ... rest of implementation ...
}
```

### Phase 9: Verification and Testing

Add verification after decode:

```cpp
// After decode completes
int verify_result = llama_config_freeze_verify_zero_reads(ctx->frozen_config);
if (verify_result != 0) {
    LLAMA_LOG_WARNING("Configuration verification failed: detected runtime flag reads\n");
}

// Print status report for debugging
char * report = llama_config_freeze_status_report(ctx->frozen_config);
if (report) {
    LLAMA_LOG_INFO("Config Freeze Status:\n%s\n", report);
    free(report);
}
```

## Implementation Patterns

### Pattern 1: Function Pointer Resolution

```cpp
// Startup: Resolve function pointer once
int llama_config_freeze_bind_backend(llama_frozen_config * config, llama_context * ctx) {
    if (config->cli_config.gpu_offload) {
        config->backend_dispatch.compute_fn = &cuda_backend_compute;
    } else {
        config->backend_dispatch.compute_fn = &cpu_backend_compute;
    }
    // Function pointer is now FIXED - no selection happens again
    return 0;
}

// Decode: Direct function invocation
int llama_config_freeze_execute_backend(
    const llama_frozen_config * config,
    llama_context * ctx,
    const void * params
) {
    // No branching - direct call to preselected function
    return config->backend_dispatch.compute_fn(ctx, params);
}
```

### Pattern 2: Precomputed Parameters

```cpp
// Startup: Compute parameters once
int llama_config_freeze_bind_sampler(llama_frozen_config * config) {
    config->sampling_params.top_k = 40;
    config->sampling_params.top_p = 0.95f;
    config->sampling_params.temperature = 0.8f;
    // Parameters are now FIXED
    return 0;
}

// Decode: Use precomputed values
int llama_config_freeze_execute_sampler(
    const llama_frozen_config * config,
    float * logits,
    int32_t n_logits,
    int32_t * sampled_tokens,
    uint32_t n_samples
) {
    // No parameter lookup - use precomputed values
    return config->sampler_dispatch.sample_fn(
        logits, n_logits, sampled_tokens, n_samples,
        (const void *)&config->sampling_params  // Already computed
    );
}
```

### Pattern 3: Modification Guards

```cpp
// Shared pattern for all configuration modifications
int modify_configuration(
    llama_context * ctx,
    const char * param_name,
    const char * old_value,
    const char * new_value
) {
    // Check if locked
    if (llama_config_freeze_reject_reconfig(ctx->frozen_config, param_name) != 0) {
        LLAMA_LOG_ERROR("Cannot modify %s during decode (was: %s, requested: %s)\n",
                       param_name, old_value, new_value);
        return -EINVAL;
    }

    // Safe to modify
    // ... apply modification ...
    return 0;
}
```

## Testing Integration

### Test 1: Configuration Completeness

```cpp
void test_frozen_config_complete() {
    llama_context * ctx = create_test_context();

    int result = llama_config_freeze_validate_complete(ctx->frozen_config);
    assert(result == 0);

    llama_free(ctx);
}
```

### Test 2: Lock Enforcement

```cpp
void test_lock_enforcement() {
    llama_context * ctx = create_test_context();

    // Before decode, modifications should work
    int result = ctx->set_n_threads(8, 8);
    // Should succeed or fail based on stage, not lock

    // Simulate decode
    llama_batch batch = llama_batch_init(512, 0, 1);
    ctx->decode(batch);

    // After decode, modifications should fail
    result = ctx->set_n_threads(4, 4);
    assert(result == -EINVAL);  // Locked

    llama_batch_free(batch);
    llama_free(ctx);
}
```

### Test 3: Zero Runtime Reads

```cpp
void test_zero_runtime_reads() {
    llama_context * ctx = create_test_context();

    llama_batch batch = llama_batch_init(512, 0, 1);
    ctx->decode(batch);

    int result = llama_config_freeze_verify_zero_reads(ctx->frozen_config);
    assert(result == 0);
    assert(ctx->frozen_config->metrics.runtime_flag_reads_during_decode == 0);

    llama_batch_free(batch);
    llama_free(ctx);
}
```

## Performance Validation

### Profile Decode Loop

```bash
# Before frozen config
perf record -e branches,branch-misses ./llama-main -m model.gguf -p "Hello world"
perf report

# After frozen config
# Should show:
# - Fewer branch misses
# - Higher branch prediction accuracy
# - Faster decode iterations
```

### Benchmark Decode Throughput

```cpp
void benchmark_decode() {
    llama_context * ctx = create_test_context();
    llama_batch batch = llama_batch_init(512, 0, 1);

    int64_t start = get_time_us();
    for (int i = 0; i < 100; ++i) {
        ctx->decode(batch);
    }
    int64_t end = get_time_us();

    uint64_t total_us = end - start;
    uint64_t avg_us = total_us / 100;

    printf("Average decode time: %" PRIu64 " µs\n", avg_us);

    llama_batch_free(batch);
    llama_free(ctx);
}
```

## Troubleshooting

### Issue: Configuration not locked during decode

**Solution**: Ensure `llama_config_freeze_lock_decode()` is called at decode start.

### Issue: Memory allocation fails

**Solution**: Check that `llama_config_freeze_allocate_memory()` completes successfully.

### Issue: Backend dispatch returns error

**Solution**: Verify backend was validated with `llama_config_freeze_bind_backend()`.

### Issue: Configuration lock violations during decode

**Solution**: Check for API calls that modify configuration during decode. Add guards to those APIs.

## Metrics to Monitor

### During Integration

1. **Configuration Lock Violations**: Should be 0
2. **Reconfiguration Attempts During Decode**: Should be 0
3. **Runtime Flag Reads**: Should be 0
4. **Dispatch Function Pointer Validity**: Should all be non-null

### After Integration

1. **Decode Throughput**: Should be equal or faster
2. **Branch Misses**: Should be reduced
3. **Memory Access Latency**: Should be reduced
4. **CPU Utilization**: Should be improved

## Validation Checklist

- [ ] Header file included in build system
- [ ] Implementation file compiled and linked
- [ ] Frozen config allocated in context constructor
- [ ] All phases execute in order
- [ ] Configuration validated at startup
- [ ] Configuration locked at decode start
- [ ] Static dispatch replaces runtime branching
- [ ] Modification guards added to all config APIs
- [ ] Verification tests pass
- [ ] Performance metrics meet targets
- [ ] No crashes or assertions during decode
- [ ] Lock violations are properly rejected and tracked

## Summary

The frozen configuration system eliminates runtime flag evaluation through:

1. One-time resolution at startup (CLI + environment)
2. Static dispatch function binding
3. Precomputed parameters stored in config
4. Hard lock preventing modifications during decode
5. Verification of zero runtime reads

This ensures optimal decode performance with zero configuration overhead.
