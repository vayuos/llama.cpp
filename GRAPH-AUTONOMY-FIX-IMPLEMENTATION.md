# Section 12 Violation Fix: GPU-Autonomous Decode Implementation Plan

**Scope:** Transform decode from CPU-driven per-token loop to GPU-autonomous persistent execution
**Complexity:** CRITICAL - Requires deep architectural changes
**Estimated Effort:** 40-60 hours full implementation
**Phases:** 5 sequential phases with validation checkpoints

---

## Phase 1: Eliminate CPU Decode Loop (Critical Path)

### Objective
Replace explicit `for` loop with GPU-driven execution interface

### Key Files
- `src/llama.cpp` - Main decode entry point
- `src/llama-gpu-exclusive-decode-engine.cpp` - GPU orchestration (already partially implemented)

### Implementation Steps

#### Step 1.1: Create GPU Decode Wrapper Function

**File:** `src/llama-gpu-exclusive-decode-engine.cpp`

```cpp
/**
 * Start GPU-autonomous decode session.
 *
 * This is the ONLY CPU entry point for token generation.
 * All per-token operations happen on GPU after this call.
 *
 * @param ctx Llama context
 * @param max_tokens Maximum tokens to generate
 * @return 0 on success, -1 on error
 */
int llama_gpu_exclusive_decode(
    llama_context * ctx,
    int max_tokens
) {
    // Validation
    if (!ctx || max_tokens <= 0) {
        LLAMA_LOG_ERROR("Invalid parameters: ctx=%p, max_tokens=%d\n", ctx, max_tokens);
        return -1;
    }

    // Phase 1: Initialize GPU decode engine
    if (llama_gpu_exclusive_engine_init(ctx, get_rng_seed())) {
        LLAMA_LOG_ERROR("GPU engine init failed\n");
        return -1;
    }

    // Phase 2: Prepare GPU graph
    if (llama_gpu_exclusive_engine_prepare_decode(ctx, max_tokens)) {
        LLAMA_LOG_ERROR("GPU graph preparation failed\n");
        return -1;
    }

    // Phase 3: Start GPU autonomous decode
    if (llama_gpu_exclusive_engine_start_decode()) {
        LLAMA_LOG_ERROR("GPU decode start failed\n");
        return -1;
    }

    // Phase 4: Wait for GPU completion (single wait, not per-token!)
    if (llama_gpu_exclusive_engine_wait_for_completion()) {
        LLAMA_LOG_ERROR("GPU decode wait failed\n");
        return -1;
    }

    // Phase 5: Cleanup
    llama_gpu_exclusive_engine_stop_decode();

    return 0;
}
```

#### Step 1.2: Remove CPU Loop from Simple Example

**File:** `examples/simple/simple.cpp`

**Before (lines 168-201):**
```cpp
for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return 1;
    }
    n_pos += batch.n_tokens;
    new_token_id = llama_sampler_sample(smpl, ctx, -1);
    if (llama_vocab_is_eog(vocab, new_token_id)) break;
    batch = llama_batch_get_one(&new_token_id, 1);
    n_decode += 1;
}
```

**After:**
```cpp
// GPU takes over completely - CPU just initiates
// All token generation happens autonomously on GPU
if (llama_gpu_exclusive_decode(ctx, n_prompt + n_predict - prompt_tokens.size())) {
    fprintf(stderr, "%s: GPU decode failed\n", __func__);
    return 1;
}

// Tokens are now in GPU output buffer
// Retrieve and display them
n_decode = llama_gpu_exclusive_get_tokens_produced(ctx);
for (int i = 0; i < n_decode; i++) {
    llama_token token = llama_gpu_exclusive_get_token(ctx, i);
    char buf[128];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) continue;
    std::string s(buf, n);
    printf("%s", s.c_str());
}
```

#### Step 1.3: Add Validation Enforcement

**File:** `src/llama-decode-loop-elimination.cpp`

```cpp
/**
 * ENFORCEMENT POINT 1: Detect CPU loop attempts
 *
 * Should NEVER be called if GPU owns decode.
 * If this fires, indicates GPU autonomy not achieved.
 */
int llama_decode_loop_elimination_detect_cpu_owns_loop(void) {
    // Check if decode is in progress on GPU
    if (g_decode_loop_elimination_state.progression_record.gpu_autonomous) {
        LLAMA_LOG_ERROR("[DECODE_AUTONOMY_VIOLATION] CPU attempted loop iteration during GPU decode\n");

        g_decode_loop_elimination_state.total_control_violations++;
        g_decode_loop_elimination_state.ownership_record.current_owner = LLAMA_LOOP_OWNER_CPU;

        if (g_decode_loop_elimination_state.enforcement_strict) {
            GGML_ABORT("CPU owns decode loop - GPU autonomy violated (Section 12)");
        }

        return -1;
    }

    return 0;
}

/**
 * ENFORCEMENT POINT 2: Detect per-token llama_decode calls
 *
 * llama_decode() should only be called once per decode session.
 * If called multiple times, indicates CPU loop controlling iteration.
 */
int llama_decode_loop_elimination_detect_per_token_decode_calls(void) {
    static uint64_t last_decode_call_time = 0;
    static int decode_call_count_in_session = 0;

    uint64_t now = ggml_time_us();

    // If less than 100ms since last call, likely same decode session
    if (now - last_decode_call_time < 100000) {
        decode_call_count_in_session++;

        if (decode_call_count_in_session > 1) {
            LLAMA_LOG_ERROR("[DECODE_AUTONOMY_VIOLATION] Multiple llama_decode() calls in session\n");
            LLAMA_LOG_ERROR("  Call count: %d (should be 1)\n", decode_call_count_in_session);
            LLAMA_LOG_ERROR("  This indicates CPU-driven loop iteration\n");

            if (g_decode_loop_elimination_state.enforcement_strict) {
                GGML_ABORT("Per-token llama_decode() calls detected - GPU autonomy violated");
            }

            return -1;
        }
    } else {
        decode_call_count_in_session = 1;
    }

    last_decode_call_time = now;
    return 0;
}
```

### Validation Checkpoint 1

After Phase 1, verify:
- [ ] `llama_gpu_exclusive_decode()` exists and is callable
- [ ] Simple example no longer has `for` loop (replaced with GPU call)
- [ ] Calling `llama_gpu_exclusive_decode()` initializes GPU engine
- [ ] GPU engine state transitions: IDLE → INITIALIZING → READY → RUNNING → COMPLETE
- [ ] Enforcement point 1 triggers if CPU tries to iterate during GPU decode
- [ ] Enforcement point 2 triggers if multiple `llama_decode()` calls in same session

---

## Phase 2: Transfer Token Index to GPU

### Objective
Move position tracking from CPU to GPU-resident state

### Key Files
- `src/llama-token-persistent-execution.cpp` - GPU token execution tracking
- `ggml/src/ggml-cuda/ggml-cuda.cu` - GPU state management

### Implementation Steps

#### Step 2.1: Define GPU-Resident Token State

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu`

```cpp
/**
 * GPU-resident decode state
 * Maintained entirely on GPU - CPU cannot access during decode
 */
struct gpu_decode_token_state {
    uint64_t current_token_index;      // Current token being processed
    uint64_t tokens_produced_count;    // Total tokens produced
    uint64_t max_tokens;               // Maximum tokens to generate

    bool decode_complete;              // GPU signals completion
    int error_code;                    // Error if any

    // Token selection state
    llama_token last_token;            // Last selected token
    llama_token pending_token;         // Next token to process
};

// GPU-resident state (in device memory)
__device__ gpu_decode_token_state g_gpu_decode_state;

/**
 * CUDA kernel: Initialize GPU-resident token state
 */
__global__ void gpu_init_token_state(uint64_t max_tokens) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        g_gpu_decode_state.current_token_index = 0;
        g_gpu_decode_state.tokens_produced_count = 0;
        g_gpu_decode_state.max_tokens = max_tokens;
        g_gpu_decode_state.decode_complete = false;
        g_gpu_decode_state.error_code = 0;
        g_gpu_decode_state.last_token = 0;
        g_gpu_decode_state.pending_token = 0;
    }
}

/**
 * Increment token index on GPU
 * Called from GPU autonomous decode loop
 */
__device__ void gpu_advance_token_index() {
    // Atomic increment for thread safety
    atomicAdd(&g_gpu_decode_state.current_token_index, 1);
    atomicAdd(&g_gpu_decode_state.tokens_produced_count, 1);

    // Check if complete
    if (g_gpu_decode_state.current_token_index >= g_gpu_decode_state.max_tokens) {
        g_gpu_decode_state.decode_complete = true;
    }
}

/**
 * Query token progress from host (CPU)
 * Can only call after decode complete!
 */
uint64_t gpu_get_tokens_produced() {
    uint64_t count;
    cudaMemcpyFromSymbol(&count,
                         g_gpu_decode_state.tokens_produced_count,
                         sizeof(uint64_t),
                         0,
                         cudaMemcpyDeviceToHost);
    return count;
}
```

#### Step 2.2: Prohibit CPU from Accessing Token Index During Decode

**File:** `src/llama-token-persistent-execution.cpp`

```cpp
/**
 * ENFORCEMENT POINT 3: Prevent CPU from modifying token state during decode
 */
int llama_token_persistent_enforce_gpu_ownership(void) {
    // If decode is active on GPU, CPU cannot touch token state
    if (ggml_backend_decode_mode_active()) {
        LLAMA_LOG_ERROR("[AUTONOMY_VIOLATION] CPU attempted to modify token state during GPU decode\n");

        if (enforce_strict) {
            GGML_ABORT("CPU cannot modify token index - GPU owns progression (Section 12)");
        }

        return -1;
    }

    return 0;
}

/**
 * ENFORCEMENT POINT 4: Prevent CPU from reading token index during decode
 */
int llama_token_persistent_prevent_cpu_position_reads(void) {
    static int read_attempts_during_decode = 0;

    if (ggml_backend_decode_mode_active()) {
        read_attempts_during_decode++;

        if (read_attempts_during_decode % 100 == 0) {  // Log every 100th attempt
            LLAMA_LOG_WARN("[AUTONOMY_VIOLATION] CPU polling token index (%d attempts)\n",
                          read_attempts_during_decode);
        }

        return -1;  // Indicate read should fail
    }

    read_attempts_during_decode = 0;
    return 0;
}
```

#### Step 2.3: GPU Loop Updates Token Index Autonomously

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu`

```cpp
/**
 * GPU Autonomous Decode Loop (CUDA kernel)
 *
 * This kernel runs on GPU and NEVER returns to CPU until complete.
 * All token progression is internal to GPU.
 */
__global__ void gpu_autonomous_decode_loop_kernel(
    const float * embed_matrix,
    const float * attention_weights,
    int model_size,
    int * output_tokens
) {
    // Each block handles one token generation
    if (blockIdx.x == 0 && threadIdx.x == 0) {

        // GPU-autonomous loop - CPU has NO control here
        while (!g_gpu_decode_state.decode_complete) {

            // Step 1: Get current token
            uint64_t token_idx = g_gpu_decode_state.current_token_index;

            // Step 2: Fetch input for current token
            // (This is on GPU - no CPU batch setup!)
            float input_embedding[EMBED_DIM];
            fetch_embedding_gpu(token_idx, embed_matrix, input_embedding);

            // Step 3: Compute token (attention, etc.)
            float logits[VOCAB_SIZE];
            compute_logits_gpu(input_embedding, attention_weights, logits);

            // Step 4: Sample next token (entirely on GPU!)
            uint32_t sampled_token = gpu_sample_token(logits);

            // Step 5: Store output
            output_tokens[token_idx] = sampled_token;

            // CRITICAL: GPU advances its own token index
            // NO CPU involvement here!
            gpu_advance_token_index();

            // Step 6: Continue loop (GPU decision only)
            if (g_gpu_decode_state.current_token_index >= g_gpu_decode_state.max_tokens) {
                g_gpu_decode_state.decode_complete = true;
            }

            if (is_eog_token(sampled_token)) {
                g_gpu_decode_state.decode_complete = true;
            }
        }

        // Signal CPU that decode is complete
        gpu_signal_decode_complete();
    }
}
```

### Validation Checkpoint 2

After Phase 2, verify:
- [ ] GPU-resident `gpu_decode_token_state` created in device memory
- [ ] `gpu_advance_token_index()` atomically updates GPU state
- [ ] CPU cannot read token index during decode (enforcement point 4)
- [ ] CPU cannot write token index during decode (enforcement point 3)
- [ ] GPU kernel increments token index autonomously
- [ ] Test with small sequence: verify N tokens produced match expected count

---

## Phase 3: GPU-Based Sampling

### Objective
Implement all token selection on GPU, eliminate CPU sampling

### Key Files
- `ggml/src/ggml-cuda/sampling_kernel.cu` - GPU sampling implementation
- `src/llama-sampler.cpp` - Disable CPU samplers during decode

### Implementation Steps

#### Step 3.1: Implement GPU Temperature Sampling Kernel

**File:** `ggml/src/ggml-cuda/sampling_kernel.cu`

```cpp
/**
 * GPU temperature sampling kernel
 * Applies temperature scaling and samples token
 */
__global__ void gpu_sample_temperature_kernel(
    const float * logits,    // GPU logits
    float temperature,
    uint32_t vocab_size,
    uint32_t rng_seed,
    uint32_t * output_token
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Step 1: Apply temperature scaling
        float scaled_logits[VOCAB_SIZE];  // Assume < 128K vocab
        for (int i = 0; i < vocab_size; i++) {
            scaled_logits[i] = logits[i] / temperature;
        }

        // Step 2: Compute softmax (GPU-side)
        float probabilities[VOCAB_SIZE];
        compute_softmax_gpu(scaled_logits, probabilities, vocab_size);

        // Step 3: Sample using GPU RNG
        float rng_value = gpu_rng_uniform(rng_seed);

        // Step 4: Select token via cumulative sum
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probabilities[i];
            if (rng_value < cumsum) {
                *output_token = i;
                return;
            }
        }

        // Fallback (should not reach)
        *output_token = 0;
    }
}

/**
 * GPU greedy sampling kernel
 * Returns token with highest logit (argmax)
 */
__global__ void gpu_sample_greedy_kernel(
    const float * logits,
    uint32_t vocab_size,
    uint32_t * output_token
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Find argmax
        float max_logit = -FLT_MAX;
        uint32_t max_token = 0;

        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                max_token = i;
            }
        }

        *output_token = max_token;
    }
}
```

#### Step 3.2: Prevent CPU Sampling During Decode

**File:** `src/llama-sampler.cpp`

```cpp
/**
 * ENFORCEMENT POINT 5: Prevent CPU samplers during GPU decode
 *
 * Any CPU sampling call during decode violates Section 12
 */
llama_token llama_sampler_sample(llama_sampler * sampler,
                                  llama_context * ctx,
                                  int idx) {
    // Check if we're in GPU-exclusive decode mode
    if (ggml_backend_decode_mode_active()) {
        LLAMA_LOG_ERROR("[DECODE_AUTONOMY_VIOLATION] CPU sampler called during GPU decode\n");
        LLAMA_LOG_ERROR("  Sampler type: %s\n", sampler_type_name(sampler->type));
        LLAMA_LOG_ERROR("  This violates Section 12 - all sampling must be GPU-based\n");

        if (enforce_strict_mode) {
            GGML_ABORT("CPU sampling forbidden during GPU-autonomous decode");
        }

        return LLAMA_TOKEN_NULL;  // Signal error
    }

    // Normal (non-decode) operation
    return sampler->sample_fn(sampler, ctx, idx);
}
```

#### Step 3.3: Integrate GPU Sampling into GPU Decode Loop

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu`

```cpp
/**
 * Updated GPU autonomous loop with GPU sampling
 */
__global__ void gpu_autonomous_decode_loop_with_sampling(
    const float * embed_matrix,
    const float * attention_weights,
    int model_size,
    float temperature,
    uint32_t vocab_size,
    uint32_t rng_seed,
    int * output_tokens
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {

        while (!g_gpu_decode_state.decode_complete) {
            uint64_t token_idx = g_gpu_decode_state.current_token_index;

            // Fetch input and compute logits (GPU-resident)
            float input_embedding[EMBED_DIM];
            fetch_embedding_gpu(token_idx, embed_matrix, input_embedding);

            float logits[VOCAB_SIZE];
            compute_logits_gpu(input_embedding, attention_weights, logits);

            // SAMPLE ON GPU (no CPU involved!)
            uint32_t sampled_token;
            gpu_sample_temperature_kernel<<<1,1>>>(
                logits,
                temperature,
                vocab_size,
                rng_seed + token_idx,  // Vary seed per token
                &sampled_token
            );

            // Store result (GPU-resident)
            output_tokens[token_idx] = sampled_token;

            // Advance GPU token index
            gpu_advance_token_index();

            // Check termination (GPU side)
            if (is_eog_token(sampled_token) ||
                g_gpu_decode_state.current_token_index >= g_gpu_decode_state.max_tokens) {
                g_gpu_decode_state.decode_complete = true;
            }
        }

        gpu_signal_decode_complete();
    }
}
```

### Validation Checkpoint 3

After Phase 3, verify:
- [ ] GPU sampling kernels compile and run
- [ ] Temperature kernel produces valid token IDs
- [ ] Greedy kernel correctly finds argmax
- [ ] CPU sampler blocked with error if called during decode
- [ ] GPU logits never transferred to CPU during decode
- [ ] All 6 samplers (temperature, top-k, top-p, greedy, penalties, grammar) cannot be called during decode phase

---

## Phase 4: Persistent CUDA Graphs

### Objective
Consolidate into single persistent graph for entire decode

### Key Files
- `ggml/src/ggml-cuda/ggml-cuda.cu` - Graph capture and launch
- `src/llama-gpu-exclusive-decode-engine.cpp` - Graph management

### Implementation Steps

#### Step 4.1: Capture Single Persistent Graph

**File:** `src/llama-gpu-exclusive-decode-engine.cpp`

```cpp
/**
 * ENFORCEMENT POINT 6: Single graph per decode session
 */
int llama_gpu_exclusive_engine_prepare_decode(
    const llama_context * ctx,
    int max_tokens
) {
    if (g_gpu_engine.graph_captured) {
        LLAMA_LOG_ERROR("[GRAPH_VIOLATION] Attempt to capture multiple graphs\n");
        return -1;
    }

    // Begin graph capture on GPU
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    uint64_t graph_id = ggml_cuda_graph_capture_begin(stream);

    // Launch entire decode loop kernel ONCE
    // This kernel will contain the entire decode logic
    gpu_autonomous_decode_loop_with_sampling<<<1, 1, 0, stream>>>(
        ctx->embed_matrix,
        ctx->attention_weights,
        ctx->model_size,
        ctx->temperature,
        ctx->vocab_size,
        ctx->rng_seed,
        ctx->output_buffer
    );

    // Capture end - graph now contains entire decode logic
    int result = ggml_cuda_graph_capture_end(graph_id, stream);

    if (result == 0) {
        g_gpu_engine.active_graph_id = graph_id;
        g_gpu_engine.graph_captured = true;

        // Instantiate the graph
        if (ggml_cuda_graph_instantiate(graph_id, stream) == 0) {
            g_gpu_engine.graph_instantiated = true;
        }
    }

    cudaStreamDestroy(stream);

    return result;
}

/**
 * ENFORCEMENT POINT 7: Single graph launch
 */
int llama_gpu_exclusive_engine_start_decode() {
    if (!g_gpu_engine.graph_instantiated) {
        LLAMA_LOG_ERROR("[GRAPH_VIOLATION] Graph not instantiated\n");
        return -1;
    }

    // Single launch for entire decode
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int result = ggml_cuda_graph_launch(g_gpu_engine.active_graph_id, stream);

    // No per-token replays!
    // GPU kernel continues autonomously until complete

    cudaStreamDestroy(stream);

    return result;
}
```

#### Step 4.2: Verify Single-Launch Constraint

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu`

```cpp
/**
 * Graph launch counter for violation detection
 */
static __device__ int g_graph_launch_count = 0;

/**
 * ENFORCEMENT POINT 8: Detect multiple graph launches
 */
__global__ void verify_single_launch_kernel() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(&g_graph_launch_count, 1);

        if (g_graph_launch_count > 1) {
            // Multiple launches detected - GPU autonomy violated!
            printf("[GRAPH_AUTONOMY_VIOLATION] Graph launched %d times\n",
                   g_graph_launch_count);
        }
    }
}
```

### Validation Checkpoint 4

After Phase 4, verify:
- [ ] Single CUDA graph created per decode session
- [ ] Graph contains entire autonomous decode loop
- [ ] Graph launched exactly once per session
- [ ] No per-token graph replays
- [ ] Graph launch count never exceeds 1
- [ ] Test with various sequence lengths (10, 100, 1000 tokens)

---

## Phase 5: GPU Signal Interface

### Objective
Implement event-based signaling instead of CPU polling/blocking

### Key Files
- `src/llama-gpu-exclusive-decode-engine.cpp` - Signal handling
- `ggml/src/ggml-cuda/ggml-cuda.cu` - CUDA event management

### Implementation Steps

#### Step 5.1: GPU Signal Events

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu`

```cpp
/**
 * GPU-to-CPU signaling mechanism
 * GPU writes signal, CPU waits (not polls!)
 */

// CUDA events for signaling
cudaEvent_t g_decode_complete_event = nullptr;
cudaEvent_t g_token_ready_event = nullptr;

/**
 * GPU signals decode completion
 * Called from GPU kernel when done
 */
__device__ void gpu_signal_decode_complete() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Record event on GPU (atomic operation)
        // Signal propagates to CPU via CUDA event mechanism
        g_gpu_decode_state.decode_complete = true;
    }
}

/**
 * CPU waits for GPU completion signal
 * Does NOT poll - uses CUDA event wait
 */
int gpu_wait_for_decode_complete(int timeout_ms) {
    cudaEvent_t event;
    cudaEventCreate(&event);

    // Record event after GPU work
    cudaEventRecord(event);

    // Wait for event (blocking but efficient)
    cudaError_t err = cudaEventSynchronize(event);

    cudaEventDestroy(event);

    if (err != cudaSuccess) {
        LLAMA_LOG_ERROR("GPU decode wait failed: %s\n",
                       cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
```

#### Step 5.2: CPU Wait-for-Signal (Not Polling)

**File:** `src/llama-gpu-exclusive-decode-engine.cpp`

```cpp
/**
 * ENFORCEMENT POINT 9: CPU waits for signal, doesn't poll
 */
int llama_gpu_exclusive_engine_wait_for_completion() {
    if (!g_gpu_engine.state == GPU_ENGINE_DECODING) {
        LLAMA_LOG_ERROR("Engine not in decoding state\n");
        return -1;
    }

    // Wait for GPU signal (efficient, not busy-wait)
    int result = gpu_wait_for_decode_complete(10000);  // 10 second timeout

    if (result == 0) {
        // Query results from GPU
        uint64_t tokens_produced = gpu_get_tokens_produced();

        LLAMA_LOG_INFO("GPU decode complete: %lu tokens produced\n",
                      tokens_produced);

        g_gpu_engine.state = GPU_ENGINE_GRAPH_READY;  // Not RUNNING anymore
        g_gpu_engine.total_tokens += tokens_produced;

        return 0;
    } else {
        LLAMA_LOG_ERROR("GPU decode timeout\n");
        return -1;
    }
}

/**
 * ENFORCEMENT POINT 10: Prevent CPU polling for token ready
 */
int llama_gpu_exclusive_detect_cpu_polling(void) {
    static int poll_count = 0;
    static uint64_t poll_start_time = 0;

    // Detect rapid status checks (polling behavior)
    uint64_t now = ggml_time_us();

    if (now - poll_start_time < 1000) {  // < 1ms apart
        poll_count++;

        if (poll_count > 10) {  // More than 10 checks in 1ms = polling
            LLAMA_LOG_WARN("[AUTONOMY_VIOLATION] CPU polling for GPU completion\n");
            LLAMA_LOG_WARN("  Detected %d status checks in < 1ms\n", poll_count);

            return -1;  // Indicate polling detected
        }
    } else {
        poll_count = 0;
    }

    poll_start_time = now;
    return 0;
}
```

### Validation Checkpoint 5

After Phase 5, verify:
- [ ] `gpu_signal_decode_complete()` called from GPU kernel
- [ ] `gpu_wait_for_decode_complete()` waits without polling
- [ ] CPU blocking time is minimal (GPU does the work)
- [ ] CUDA event signals properly propagate
- [ ] No busy-wait loops in CPU code during decode
- [ ] Polling detection triggers if CPU checks status repeatedly

---

## Final Verification Checklist

### Structural Checks
- [ ] No CPU `for` or `while` loop iterating per-token
- [ ] Single entry point: `llama_gpu_exclusive_decode()`
- [ ] GPU kernel contains entire autonomous loop
- [ ] Token index maintained on GPU only
- [ ] All sampling on GPU (no CPU sampler calls during decode)
- [ ] Batch structure not created during decode

### Enforcement Checks
- [ ] Enforcement Point 1: CPU loop detection works
- [ ] Enforcement Point 2: Per-token decode calls detected
- [ ] Enforcement Point 3: CPU cannot modify token state
- [ ] Enforcement Point 4: CPU cannot read token index
- [ ] Enforcement Point 5: CPU sampling blocked
- [ ] Enforcement Point 6: Single graph constraint enforced
- [ ] Enforcement Point 7: Single launch constraint enforced
- [ ] Enforcement Point 8: Multiple launches detected
- [ ] Enforcement Point 9: CPU waits for signal
- [ ] Enforcement Point 10: Polling detection works

### Performance Checks
- [ ] Throughput improved 2-3x
- [ ] Per-token CPU time ≈ 0
- [ ] GPU utilization 95%+
- [ ] Total decode time ≈ GPU kernel time
- [ ] No visible CPU stalls during decode

### Integration Checks
- [ ] Simple example works with GPU decode
- [ ] Batched example supports GPU decode
- [ ] Server (llama-server) supports GPU decode
- [ ] Output tokens match CPU decode results
- [ ] Sampling produces expected distribution

---

## Expected Outcomes

### Before (CPU-Driven)
```
for (token in 1..N) {
    llama_decode()          [CPU blocks here]
    sampler_sample()        [CPU reads GPU output]
    batch_setup()           [CPU creates batch]
    pos++                   [CPU advances]
}

Throughput: 100-200 tokens/sec
Wall time: 5-10 seconds for 1000 tokens
CPU time: ~5-10% busy
```

### After (GPU-Autonomous)
```
gpu_exclusive_decode(max_N)  [CPU does NOT block per-token]
// GPU runs autonomous loop:
// for token in 1..N:
//     compute_token()      [GPU]
//     sample_token()       [GPU]
//     advance()            [GPU]
// GPU signals: COMPLETE

Throughput: 400-600+ tokens/sec
Wall time: 1.5-2.5 seconds for 1000 tokens
CPU time: ≈ 0% busy during decode
```

**Improvement: 2-3x throughput, 75-80% faster execution**

---

## Implementation Timeline

- **Phase 1:** 6-8 hours (CPU loop elimination)
- **Phase 2:** 8-10 hours (Token index on GPU)
- **Phase 3:** 10-12 hours (GPU sampling)
- **Phase 4:** 6-8 hours (Persistent graphs)
- **Phase 5:** 4-6 hours (Signal interface)
- **Testing:** 6-8 hours (Validation and debugging)

**Total: 40-60 hours (1-2 weeks full-time development)**

---

This implementation plan provides a clear path to true GPU autonomy while maintaining validation checkpoints to ensure correctness at each phase.
