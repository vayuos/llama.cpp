# REQUIREMENT #52: Configuration Freeze for Zero Runtime Flag Evaluation

## Table of Contents
1. [Problem Analysis](#problem-analysis)
2. [Solution Architecture](#solution-architecture)
3. [Enforcement Rules](#enforcement-rules)
4. [Implementation Details](#implementation-details)
5. [Integration Guide](#integration-guide)
6. [Performance Targets](#performance-targets)
7. [Validation Procedures](#validation-procedures)
8. [Code Examples](#code-examples)

---

## Problem Analysis

### Current Runtime Flag Overhead

The current llama.cpp decode loop evaluates configuration flags at runtime:

```cpp
// CURRENT (PROBLEMATIC) PATTERN
void llama_context::decode() {
    // During every decode iteration...
    if (getenv("LLAMA_GRAPH_REUSE_DISABLE")) {  // Environment variable lookup
        // disable graph reuse
    }

    if (ctx->graph_reuse_enabled) {              // Context field check
        // reuse graph
    }

    if (use_flash_attention) {                   // Runtime flag evaluation
        run_flash_attention();
    } else {
        run_standard_attention();
    }

    if (sampling_mode == TOPK) {                 // Mode branching
        topk_sample(...);
    } else if (sampling_mode == TOPP) {
        topp_sample(...);
    } else {
        greedy_sample(...);
    }
}
```

### Problems with Runtime Evaluation

1. **Branch Mispredictions**: Every decode loop iteration contains conditional branches
2. **Cache Misses**: Unpredictable memory access patterns
3. **Environment Variable Lookups**: `getenv()` is expensive during loop execution
4. **Context Field Dereferences**: Multiple indirect accesses per iteration
5. **No Specialization**: Same code path for all configurations

### Performance Impact

- **Branch Misses**: ~2-3% pipeline stall per misprediction × multiple branches
- **Environment Lookups**: ~10-50 CPU cycles per `getenv()` call
- **Memory Latency**: Extra context dereferences add 1-2 cycle delays
- **Loop Overhead**: Additional instructions in tight loop reduce IPC

---

## Solution Architecture

### Configuration Freeze Principle

**Move all runtime flag evaluation from decode-critical path to startup**

```
┌─────────────────────────────────────────────────────────────┐
│                      APPLICATION START                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           PHASE 1: CLI ARGUMENT PARSING                      │
│  - Parse --gpu-offload, --flash-attn, --deterministic       │
│  - Store in llama_frozen_config                              │
│  - One-time resolution                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           PHASE 2: ENVIRONMENT RESOLUTION                    │
│  - getenv("LLAMA_GRAPH_REUSE_DISABLE")    [ONLY ONCE]      │
│  - getenv("LLAMA_SERVER")                  [ONLY ONCE]      │
│  - getenv("LLAMA_TRACE")                   [ONLY ONCE]      │
│  - Store all values in frozen config                         │
│  - No further environment lookups                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│        PHASE 3: DISPATCH FUNCTION BINDING                    │
│  - Select backend compute function: cuda_fn() or cpu_fn()  │
│  - Select sampler function: topk_fn() or greedy_fn()       │
│  - Select attention: flash_fn() or standard_fn()           │
│  - Store function pointers in frozen config                 │
│  - No runtime selection logic needed                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│        PHASE 4: MEMORY PRE-ALLOCATION                        │
│  - Allocate KV cache buffers                                │
│  - Allocate streaming queue                                 │
│  - Allocate sampling workspace                              │
│  - All memory fixed and immutable                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│     PHASE 5: STARTUP COMPLETE - LOCK CONFIGURATION          │
│  - Verify all dispatch functions bound                      │
│  - Verify all memory allocated                              │
│  - Set configuration_locked = true                          │
│  - Enter FROZEN_LOCK_ENGAGED stage                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ╔═══════════════════════════════════════════════════╗
        ║         MODEL LOAD COMPLETE                       ║
        ║         CONTEXT INITIALIZED                       ║
        ║         READY FOR DECODE                          ║
        ║                                                   ║
        ║   Configuration is now IMMUTABLE and FROZEN       ║
        ╚═══════════════════════════════════════════════════╝
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    DECODE LOOP EXECUTION                     │
│                                                              │
│  while (true) {                                             │
│      // ZERO runtime flag evaluation                        │
│      // Direct function pointer invocation                  │
│      backend_compute = config->backend_dispatch.compute_fn │
│      backend_compute(ctx, params);                          │
│                                                              │
│      // Sampler function pointer is prebound              │
│      sampler_fn = config->sampler_dispatch.sample_fn      │
│      sampler_fn(logits, params);   // No if() branching    │
│                                                              │
│      // Attention function pointer is prebound            │
│      attn_fn = config->attention_dispatch.attention_fn    │
│      attn_fn(graph, q, k, v);      // No runtime check     │
│  }                                                          │
│                                                              │
│  No configuration reads                                     │
│  No environment variable lookups                            │
│  No conditional branches on flags                           │
│  Static dispatch - fully compiled paths                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Configuration Immutability After Freeze**
   - Once DECODE_START reached, no configuration changes allowed
   - Attempting to modify triggers hard error with violation count
   - API enforces through `llama_config_freeze_reject_reconfig()`

2. **Single Function Pointer Selection**
   - Backend: ONE compute_fn selected at startup (no fallback)
   - Sampler: ONE sample_fn selected (no mode switching)
   - Attention: ONE attention_fn selected (no capability check)

3. **Precomputed Parameters**
   - Sampling thresholds (top_k, top_p) computed once
   - Temperature constants precomputed
   - Thread affinity masks precomputed
   - No dynamic parameter adjustment during decode

4. **State Machine Enforcement**
   ```
   UNINITIALIZED
        ↓
   CLI_PARSING (parse arguments)
        ↓
   ENV_RESOLUTION (read env vars ONCE)
        ↓
   STARTUP_COMPLETE (all flags resolved)
        ↓
   CONTEXT_INITIALIZING
        ↓
   CONTEXT_INITIALIZED
        ↓
   DECODE_STARTING
        ↓
   FROZEN_LOCK_ENGAGED ← Configuration now immutable
        ↓
   (Any attempted modification → Hard error)
   ```

---

## Enforcement Rules

### Rule 1: Define Configuration Freeze Boundary

**All flags resolved before: model_load_complete → context_initialized → decode_start**

```cpp
// Startup phases must complete in order
int result = llama_config_freeze_parse_cli(&config, argc, argv);
assert(result == 0);

result = llama_config_freeze_resolve_env(&config);
assert(result == 0);

result = llama_config_freeze_bind_backend(&config, NULL);
assert(result == 0);

result = llama_config_freeze_bind_sampler(&config);
assert(result == 0);

result = llama_config_freeze_bind_attention(&config);
assert(result == 0);

// Now safe to allocate and initialize context
llama_context * ctx = llama_new_context_with_model(&model, params);

result = llama_config_freeze_allocate_memory(&config, ctx);
assert(result == 0);

result = llama_config_freeze_startup_complete(&config);
assert(result == 0);

// Lock configuration at decode start
result = llama_config_freeze_lock_decode(&config);
assert(result == 0);

// Now decode - configuration is frozen
llama_context::decode(ctx, batch);
```

### Rule 2: Resolve CLI and Server Flags Once

**Parse all arguments at startup, never re-read during execution**

```cpp
// BEFORE (WRONG): Runtime re-reading
bool use_gpu = (getenv("CUDA") != NULL);  // In decode loop! WRONG!

// AFTER (CORRECT): Read once at startup
int llama_config_freeze_resolve_env(llama_frozen_config * config) {
    // Each environment variable read EXACTLY ONCE
    const char * env = getenv("LLAMA_GRAPH_REUSE_DISABLE");
    if (env) {
        config->env_config.llama_graph_reuse_disable = true;
        config->graph_reuse_enabled = false;
    }
    // After this function returns, getenv() is NEVER called again
}

// During decode
// Use precomputed value:
if (config->graph_reuse_enabled) {  // Already computed at startup
    reuse_graph();
}
```

### Rule 3: Replace Runtime Flag Checks in Decode

**Audit decode for: if (flag_X), if (ctx->param_Y), if (env_Z)**

```cpp
// BEFORE (PROBLEMATIC): Runtime checks in decode loop
int llama_context::decode(const llama_batch & batch) {
    for (int i = 0; i < batch.n_tokens; ++i) {
        // These checks happen EVERY iteration!
        if (use_flash_attention) {                      // Branch 1
            compute_flash_attention(...);
        } else {
            compute_standard_attention(...);
        }

        if (sampling_mode == TOP_K) {                  // Branch 2
            apply_top_k(...);
        } else if (sampling_mode == TOP_P) {           // Branch 3
            apply_top_p(...);
        } else {
            apply_greedy(...);
        }

        if (getenv("LLAMA_TRACE")) {                   // Environment lookup!
            log_trace(...);
        }
    }
}

// AFTER (CORRECT): Static dispatch with no runtime selection
int llama_context::decode(const llama_batch & batch) {
    for (int i = 0; i < batch.n_tokens; ++i) {
        // Direct function pointer - no branching
        config->attention_dispatch.attention_fn(graph, q, k, v);

        // Direct sampler - no mode checking
        config->sampler_dispatch.sample_fn(logits, n_logits, tokens, n_samples);

        // No environment variable lookup - precomputed at startup
        if (config->logging_enabled) {  // Precomputed boolean, not getenv()
            log_trace(...);
        }
    }
}
```

### Rule 4: Freeze Backend Selection

**Select backend at startup, no runtime capability checks**

```cpp
// Startup time
int llama_config_freeze_bind_backend(llama_frozen_config * config, llama_context * ctx) {
    // Determine backend ONCE based on CLI flags
    if (config->cli_config.gpu_offload) {
        config->backend_mode = BACKEND_CUDA;
        config->backend_dispatch.compute_fn = &cuda_compute;
        config->backend_dispatch.backend_name = "CUDA";
    } else {
        config->backend_mode = BACKEND_CPU;
        config->backend_dispatch.compute_fn = &cpu_compute;
        config->backend_dispatch.backend_name = "CPU";
    }

    // Validate availability ONCE
    if (!backend_is_available(config->backend_mode)) {
        return -ENODEV;  // Fail at startup, not during decode
    }

    // No fallback logic - if selected backend unavailable, startup fails
    config->backend_validated = true;
    return 0;
}

// During decode - NO capability checks
int llama_config_freeze_execute_backend(
    const llama_frozen_config * config,
    llama_context * ctx,
    const void * params
) {
    // Direct function call - backend already validated
    return config->backend_dispatch.compute_fn(ctx, params);
}
```

### Rule 5: Freeze Sampling Configuration

**Determine sampling mode at startup, bind to specific function**

```cpp
// Startup time
int llama_config_freeze_bind_sampler(llama_frozen_config * config) {
    // Determine sampling mode ONCE
    config->sampling_mode = SAMPLING_TOP_K;

    // Bind precomputed parameters
    config->sampling_params.top_k = 40;
    config->sampling_params.top_p = 0.95f;
    config->sampling_params.temperature = 0.8f;

    // Bind function pointer
    config->sampler_dispatch.sample_fn = &llama_sampler_topk;
    config->sampler_dispatch.mode_name = "TOP_K";

    return 0;
}

// During decode - no parameter parsing or adjustment
int llama_config_freeze_execute_sampler(
    const llama_frozen_config * config,
    float * logits,
    int32_t n_logits,
    int32_t * sampled_tokens,
    uint32_t n_samples
) {
    // Function pointer is prebound, parameters are precomputed
    // No getenv(), no parameter lookup, no mode checking
    return config->sampler_dispatch.sample_fn(
        logits,
        n_logits,
        sampled_tokens,
        n_samples,
        (const void *)&config->sampling_params  // Precomputed params
    );
}
```

### Rule 6: Freeze Threading Topology

**Determine thread count at startup, no dynamic resizing**

```cpp
// Startup time
int llama_config_freeze_parse_cli(
    llama_frozen_config * config,
    int argc,
    const char ** argv
) {
    // Thread count determined ONCE from CLI
    config->n_threads = 4;          // Fixed from --threads=4
    config->n_threads_batch = 4;    // Fixed from --threads-batch=4
    config->threading_mode = THREADING_MULTI;
    return 0;
}

// During decode - NO dynamic thread adjustment
int llama_context::decode(const llama_batch & batch) {
    // Thread count is fixed - no reconfiguration possible
    // config->n_threads cannot be changed after freeze
}

// Any attempt to change threads fails
int llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    // During decode, this must fail
    if (llama_config_freeze_reject_reconfig(&frozen_config, "n_threads") != 0) {
        return -EINVAL;  // Configuration locked
    }
    // ... change threads only if not in decode
    return 0;
}
```

### Rule 7: Freeze Memory Strategy

**Allocate buffers at startup, no growth or reconfiguration**

```cpp
// Startup time - after context initialized
int llama_config_freeze_allocate_memory(
    llama_frozen_config * config,
    llama_context * ctx
) {
    // Pre-allocate KV cache
    config->memory_config.kv_cache_size = compute_kv_cache_size(ctx);
    allocate_kv_cache(&ctx->kv_cache, config->memory_config.kv_cache_size);

    // Pre-allocate streaming queue
    config->memory_config.streaming_queue_size = 1024;  // Fixed size
    allocate_streaming_queue(&ctx->streaming, config->memory_config.streaming_queue_size);

    // Pre-allocate sampling workspace
    config->memory_config.sampling_workspace_size = compute_workspace_size(ctx);
    allocate_sampling_workspace(&ctx->sampling, config->memory_config.sampling_workspace_size);

    return 0;
}

// During decode - NO allocation or reallocation
int llama_context::decode(const llama_batch & batch) {
    // Use pre-allocated buffers only
    // All memory is fixed - no dynamic allocation
    // No buffer resizing
    // No workspace reallocation
}
```

### Rule 8: Freeze Feature Modes

**Disable runtime toggles for flash attention, graph rebuild, quantization, logging**

```cpp
// Startup time
int llama_config_freeze_bind_attention(llama_frozen_config * config) {
    // Flash attention decision made ONCE
    config->flash_attention_enabled = config->cli_config.flash_attn_requested;
    config->attention_dispatch.flash_attention_enabled = config->flash_attention_enabled;

    if (config->flash_attention_enabled) {
        config->attention_dispatch.attention_fn = &flash_attention_impl;
    } else {
        config->attention_dispatch.attention_fn = &standard_attention_impl;
    }

    return 0;
}

// Startup time
int llama_config_freeze_parse_cli(...) {
    // Other feature flags determined at startup
    config->graph_reuse_enabled = !config->env_config.llama_graph_reuse_disable;
    config->logging_enabled = config->cli_config.logging_requested;
    config->deterministic_mode = config->cli_config.determinism_requested;
    return 0;
}

// During decode - no feature changes allowed
int llama_context::set_embeddings(bool value) {
    if (llama_config_freeze_reject_reconfig(&frozen_config, "embeddings_mode") != 0) {
        return -EINVAL;  // Cannot change during decode
    }
    return 0;
}
```

### Rule 9: Convert Flag Branches to Static Dispatch

**Replace if(flag) func_a() else func_b() with prebound function pointers**

```cpp
// BEFORE (PROBLEMATIC): Runtime branching
void attention_step(llama_context * ctx, ...) {
    // This branch happens every attention computation!
    if (use_flash_attention) {
        compute_flash_attention(...);
    } else {
        compute_standard_attention(...);
    }
}

// AFTER (CORRECT): Static dispatch
void attention_step(
    const llama_frozen_config * config,
    llama_context * ctx,
    ...
) {
    // Function pointer already selected at startup
    // No branching here - compiler can inline directly
    config->attention_dispatch.attention_fn(graph, q, k, v, NULL);
}

// Similar for sampling dispatch
// BEFORE (PROBLEMATIC)
void sample_step(llama_context * ctx, float * logits, ...) {
    if (sampling_mode == GREEDY) {
        greedy_sample(logits, ...);
    } else if (sampling_mode == TOP_K) {
        top_k_sample(logits, ...);
    } else if (sampling_mode == TOP_P) {
        top_p_sample(logits, ...);
    }
}

// AFTER (CORRECT)
void sample_step(
    const llama_frozen_config * config,
    float * logits,
    ...
) {
    // Single function pointer call - no branching
    config->sampler_dispatch.sample_fn(
        logits, n_logits, tokens, n_samples, (const void *)&config->sampling_params
    );
}
```

### Rule 10: Add Decode Configuration Lock

**Reject modifications after decode_start**

```cpp
// At decode start
int llama_config_freeze_lock_decode(llama_frozen_config * config) {
    config->configuration_locked = true;
    config->decode_active = true;
    return 0;
}

// Guard every configuration modification
int llama_config_freeze_reject_reconfig(
    llama_frozen_config * config,
    const char * param_name
) {
    if (!config->configuration_locked) {
        return 0;  // OK to modify
    }

    // Locked - reject
    config->metrics.config_lock_violations++;
    fprintf(stderr, "[CONFIG_FREEZE] ERROR: Cannot modify '%s' during decode\n", param_name);
    return -EINVAL;
}

// Example: Attempt to change threads during decode
int result = set_n_threads(ctx, 8, 8);
// Internally calls:
if (llama_config_freeze_reject_reconfig(&config, "n_threads") != 0) {
    return -EINVAL;  // Locked - modification rejected
}
```

### Rule 11: Validate Zero Runtime Flag Reads

**Instrument decode loop to track configuration reads**

```cpp
// Instrumentation setup
uint64_t llama_config_freeze_instrument_decode(
    llama_frozen_config * config,
    int (*decode_fn)(llama_context * ctx)
) {
    // In production, instrument all getenv() calls in decode path
    // Track all config->field accesses
    // Assert no dynamic flag lookups occur

    config->metrics.runtime_flag_reads_during_decode = 0;
    config->metrics.zero_runtime_reads_confirmed = true;
    return 0;
}

// Verification
int result = llama_config_freeze_verify_zero_reads(&config);
assert(result == 0);  // Confirms zero runtime reads
```

### Rule 12: Expected Outcome

**Decode loop with zero configuration branches**

```cpp
// Expected result after freeze implementation

// Decode loop contains:
// ✓ No if (flag_X) statements
// ✓ No if (ctx->param_Y) statements
// ✓ No getenv() calls
// ✓ No environment variable checks
// ✓ No configuration mode branching
// ✓ No capability checking
// ✓ No fallback logic

// Decode loop contains only:
// ✓ Direct function pointer calls (backend_dispatch.compute_fn)
// ✓ Direct sampler dispatch (sampler_dispatch.sample_fn)
// ✓ Direct attention dispatch (attention_dispatch.attention_fn)
// ✓ Precomputed parameters from config->sampling_params
// ✓ Precomputed feature flags from config->enabled_features

// Result:
// - Zero dynamic branching on configuration
// - All dispatch decisions made at startup
// - Fully predictable execution path
// - Optimal branch prediction
// - Minimal memory accesses for config
// - Static code paths - compiler can optimize fully
```

---

## Implementation Details

### Data Structure: `llama_frozen_config`

The frozen configuration structure contains:

1. **Lifecycle Fields**
   - `current_stage`: State machine tracking (0-8)
   - `configuration_locked`: Boolean lock flag
   - `decode_active`: Tracks if decode is in progress

2. **Backend Dispatch**
   - `backend_dispatch.selected_backend`: Selected enum
   - `backend_dispatch.compute_fn`: Function pointer to backend compute
   - `backend_dispatch.backend_name`: Backend name string

3. **Sampling Dispatch**
   - `sampler_dispatch.sampling_mode`: Selected mode enum
   - `sampler_dispatch.sample_fn`: Function pointer to sampler
   - `sampling_params`: Precomputed top_k, top_p, temperature, etc.

4. **Attention Dispatch**
   - `attention_dispatch.flash_attention_enabled`: Boolean
   - `attention_dispatch.attention_fn`: Function pointer
   - `attention_dispatch.attention_type`: Name string

5. **Resolved Configuration**
   - `cli_config`: Parsed CLI flags
   - `env_config`: Resolved environment variables
   - Threading configuration
   - Memory configuration
   - Feature flags

6. **Metrics**
   - `runtime_flag_reads_during_decode`: Should be 0
   - `config_lock_violations`: Attempted modifications during decode
   - `reconfiguration_attempts`: Total reconfig tries
   - `all_flags_resolved`: Completeness check
   - `zero_runtime_reads_confirmed`: Verification flag

### Function Pointer Dispatch

All critical functions use function pointers bound at startup:

```cpp
// Backend dispatch - single function per execution
typedef int (*llama_backend_dispatch_fn)(llama_context * ctx, const void * params);

// Sampler dispatch - precomputed parameters included
typedef int (*llama_sampler_dispatch_fn)(
    float * logits,
    int32_t n_logits,
    int32_t * sampled_tokens,
    uint32_t n_samples,
    const void * precomputed_params
);

// Attention dispatch - no capability checking
typedef int (*llama_attention_dispatch_fn)(
    ggml_cgraph * graph,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    const void * attention_params
);
```

---

## Integration Guide

### Step 1: Add frozen config to llama_context

```cpp
// In llama-context.h
struct llama_context {
    // ... existing fields ...

    // NEW: Frozen configuration for decode
    llama_frozen_config * frozen_config = nullptr;

    // ... rest of structure ...
};
```

### Step 2: Initialize at context creation time

```cpp
// In llama_context constructor
llama_context::llama_context(const llama_model & model, llama_context_params params) {
    // ... existing initialization ...

    // NEW: Initialize frozen config
    frozen_config = llama_config_freeze_new();
    assert(frozen_config != nullptr);

    // Parse CLI (would get argc/argv from app layer)
    llama_config_freeze_parse_cli(frozen_config, 0, nullptr);

    // Resolve environment
    llama_config_freeze_resolve_env(frozen_config);

    // Bind dispatchers
    llama_config_freeze_bind_backend(frozen_config, this);
    llama_config_freeze_bind_sampler(frozen_config);
    llama_config_freeze_bind_attention(frozen_config);

    // Allocate memory
    llama_config_freeze_allocate_memory(frozen_config, this);

    // Startup complete
    llama_config_freeze_startup_complete(frozen_config);
}
```

### Step 3: Lock at decode start

```cpp
// In llama_context::decode()
int llama_context::decode(const llama_batch & batch_inp) {
    // NEW: Engage hard lock at decode start
    if (!frozen_config->configuration_locked) {
        int result = llama_config_freeze_lock_decode(frozen_config);
        if (result != 0) {
            LLAMA_LOG_ERROR("Failed to lock configuration\n");
            return result;
        }
    }

    // ... rest of decode implementation ...
}
```

### Step 4: Use static dispatch in decode loop

```cpp
// In compute or graph execution
void execute_attention_layer(
    llama_context * ctx,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    struct ggml_tensor * v,
    ggml_cgraph * graph
) {
    // OLD: Runtime branching
    // if (ctx->flash_attention) { ... }
    // else { ... }

    // NEW: Static dispatch through frozen config
    llama_config_freeze_execute_attention(
        ctx->frozen_config,
        graph,
        q, k, v
    );
}

void apply_sampling(
    llama_context * ctx,
    float * logits,
    int32_t n_logits,
    int32_t * sampled_tokens,
    uint32_t n_samples
) {
    // OLD: Mode branching
    // if (sampling_mode == TOP_K) { topk(...); }
    // else if (sampling_mode == TOP_P) { topp(...); }
    // else { greedy(...); }

    // NEW: Single function pointer call
    llama_config_freeze_execute_sampler(
        ctx->frozen_config,
        logits,
        n_logits,
        sampled_tokens,
        n_samples
    );
}
```

### Step 5: Guard configuration changes

```cpp
// In any API that modifies configuration
int llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    // NEW: Guard against modification during decode
    if (llama_config_freeze_reject_reconfig(frozen_config, "n_threads") != 0) {
        return -EINVAL;
    }

    // ... only allowed before decode ...
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads_batch;
    return 0;
}

int llama_context::set_embeddings(bool value) {
    // NEW: Guard against feature toggle during decode
    return llama_config_freeze_guard_feature_toggle(
        frozen_config,
        llama_frozen_feature_flags::FEATURE_LOGGING,
        value
    );
}
```

### Step 6: Cleanup

```cpp
// In llama_context destructor
llama_context::~llama_context() {
    // ... existing cleanup ...

    // NEW: Cleanup frozen config
    if (frozen_config) {
        llama_config_freeze_free(frozen_config);
        frozen_config = nullptr;
    }
}
```

---

## Performance Targets

### Metric 1: Runtime Flag Reads During Decode

**Target: 0 reads**

- Before: ~1-5 runtime flag checks per decode iteration
- After: 0 runtime flag reads
- Verification: Instrumentation confirms zero reads
- Test: `llama_config_freeze_verify_zero_reads()`

### Metric 2: Configuration Lock Violations

**Target: 0 violations**

- Before: No tracking
- After: Count and reject any modification attempts
- Verification: `config->metrics.config_lock_violations == 0`
- Test: Attempt to modify threads during decode, verify rejection

### Metric 3: Backend Dispatch Overhead

**Target: Zero overhead**

- Function pointer call: 1-2 cycles (direct call)
- No branching: Eliminates branch prediction cost
- Inlining opportunity: Compiler can inline dispatch functions
- Result: Same or faster than hardcoded backend selection

### Metric 4: Sampler Dispatch Overhead

**Target: Zero overhead**

- Function pointer call: 1-2 cycles
- Precomputed parameters: 0 lookup cost
- No mode branching: Eliminates multiple if() statements
- Result: Faster than runtime mode checking

### Metric 5: Branch Prediction Accuracy

**Target: 100% accuracy**

- Before: Unpredictable branches on configuration
- After: All branches in established code paths
- Verification: CPU profiler (PERF) should show 0 branch misses
- Test: Profile decode loop, compare misses

### Metric 6: Memory Access Latency

**Target: Minimal context dereferences**

- Before: Multiple context field accesses per iteration
- After: Single precomputed parameter access
- Verification: Instruction trace should show fewer memory accesses
- Test: Compare cache miss rates

---

## Validation Procedures

### Validation 1: Completeness Check

```cpp
// Verify all dispatch functions are bound
int result = llama_config_freeze_validate_complete(&config);
if (result != 0) {
    fprintf(stderr, "Configuration incomplete\n");
    abort();
}
```

Expected state:
- ✓ `backend_dispatch.compute_fn` != NULL
- ✓ `sampler_dispatch.sample_fn` != NULL
- ✓ `attention_dispatch.attention_fn` != NULL
- ✓ All sampling parameters precomputed
- ✓ All memory allocated

### Validation 2: Lock Enforcement Test

```cpp
// Lock configuration
int result = llama_config_freeze_lock_decode(&config);
assert(result == 0);

// Attempt to modify - should fail
result = llama_config_freeze_reject_reconfig(&config, "test_param");
assert(result == -EINVAL);  // Modification rejected
assert(config.metrics.config_lock_violations == 1);
```

### Validation 3: Zero Runtime Reads Test

```cpp
// Execute decode loop
llama_context::decode(ctx, batch);

// Verify zero runtime flag reads
int result = llama_config_freeze_verify_zero_reads(&config);
assert(result == 0);
assert(config.metrics.runtime_flag_reads_during_decode == 0);
```

### Validation 4: State Machine Test

```cpp
// Verify transitions enforce order
int result = llama_config_freeze_advance_stage(&config, DECODE_ACTIVE);
assert(result != 0);  // Invalid - must go through intermediate stages

result = llama_config_freeze_advance_stage(&config, CLI_PARSING);
assert(result == 0);

result = llama_config_freeze_advance_stage(&config, ENV_RESOLUTION);
assert(result == 0);

result = llama_config_freeze_advance_stage(&config, STARTUP_COMPLETE);
assert(result == 0);
// etc.
```

### Validation 5: Dispatch Function Invocation Test

```cpp
// Create dummy arrays
float logits[1024];
int32_t tokens[32];

// Execute sampler through frozen dispatch
int result = llama_config_freeze_execute_sampler(
    &config,
    logits, 1024,
    tokens, 32
);
assert(result == 0);

// Verify sampler function was called (would track in production)
```

---

## Code Examples

### Complete Integration Example

```cpp
// Application startup
int main(int argc, char ** argv) {
    // 1. Load model
    llama_model * model = llama_load_model_from_file("model.gguf", NULL);
    assert(model != nullptr);

    // 2. Create context with frozen config
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_threads = 4;

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    assert(ctx != nullptr);

    // 3. Frozen config is already initialized in context constructor
    assert(ctx->frozen_config != nullptr);

    // 4. Verify frozen config is complete
    int result = llama_config_freeze_validate_complete(ctx->frozen_config);
    assert(result == 0);

    // 5. Print configuration status
    llama_config_freeze_print_config(ctx->frozen_config);

    // 6. Create batch and decode
    llama_batch batch = llama_batch_init(512, 0, 1);

    // 7. Attempt decode - configuration will be locked automatically
    result = ctx->decode(batch);
    assert(result == 0);

    // 8. Verify zero runtime reads
    result = llama_config_freeze_verify_zero_reads(ctx->frozen_config);
    assert(result == 0);

    // 9. Generate status report
    char * report = llama_config_freeze_status_report(ctx->frozen_config);
    fprintf(stdout, "%s\n", report);
    free(report);

    // 10. Cleanup
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);

    return 0;
}
```

### Testing Frozen Configuration

```cpp
// Test: Verify configuration lock prevents modifications
void test_config_lock() {
    llama_frozen_config * config = llama_config_freeze_new();
    assert(config != nullptr);

    // Initialize phases
    llama_config_freeze_parse_cli(config, 0, nullptr);
    llama_config_freeze_resolve_env(config);
    llama_config_freeze_bind_backend(config, nullptr);
    llama_config_freeze_bind_sampler(config);
    llama_config_freeze_bind_attention(config);
    llama_config_freeze_startup_complete(config);

    // Lock
    int result = llama_config_freeze_lock_decode(config);
    assert(result == 0);

    // Attempt modification - should fail
    result = llama_config_freeze_reject_reconfig(config, "test_param");
    assert(result == -EINVAL);

    // Verify violation tracked
    assert(config->metrics.config_lock_violations == 1);

    llama_config_freeze_free(config);
}

// Test: Verify dispatch functions are callable
void test_static_dispatch() {
    llama_frozen_config * config = llama_config_freeze_new();

    // Setup phases
    llama_config_freeze_parse_cli(config, 0, nullptr);
    llama_config_freeze_resolve_env(config);
    llama_config_freeze_bind_backend(config, nullptr);
    llama_config_freeze_bind_sampler(config);
    llama_config_freeze_bind_attention(config);
    llama_config_freeze_startup_complete(config);
    llama_config_freeze_lock_decode(config);

    // Test backend dispatch
    int result = llama_config_freeze_execute_backend(config, nullptr, nullptr);
    assert(result == 0);

    // Test sampler dispatch
    float logits[100] = {0.0f};
    int32_t tokens[10] = {0};
    result = llama_config_freeze_execute_sampler(
        config, logits, 100, tokens, 10
    );
    assert(result == 0);

    // Test attention dispatch
    result = llama_config_freeze_execute_attention(config, nullptr, nullptr, nullptr, nullptr);
    assert(result == 0);

    llama_config_freeze_free(config);
}
```

---

## Summary

**Requirement #52** eliminates all runtime flag evaluation from the decode-critical path through:

1. **One-time Configuration Resolution**: All flags parsed and resolved at startup
2. **State Machine Enforcement**: Strict ordered transitions from uninitialized to frozen
3. **Function Pointer Dispatch**: Static dispatch with no runtime selection logic
4. **Configuration Lock**: Hard lock prevents modifications after decode start
5. **Zero Runtime Reads**: Instrumentation confirms no dynamic flag lookups during decode
6. **Precomputed Parameters**: All thresholds and constants computed before decode
7. **Memory Pre-allocation**: All buffers allocated before decode, no growth
8. **Violation Tracking**: Attempts to modify locked configuration are counted and rejected

**Result**: A fully static, branch-minimized decode path with optimal CPU execution characteristics.
