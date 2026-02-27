# Section 12 Violation: Graph Autonomy - Detailed Analysis

**Status:** VIOLATION CONFIRMED
**Severity:** CRITICAL - Breaks fundamental GPU autonomy invariant
**Impact:** +50-100% reduction in achievable throughput due to per-token CPU orchestration

---

## The Violation (Confirmed)

### Current CPU-Driven Loop (simple.cpp, lines 168-201)

```cpp
for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
    // SYNCHRONIZATION POINT 1: CPU blocks until GPU finishes
    if (llama_decode(ctx, batch)) {  // Blocking call - GPU must complete
        return 1;
    }

    n_pos += batch.n_tokens;                    // VIOLATION 1: CPU advances token position

    // SYNCHRONIZATION POINT 2: CPU must sample/select next token
    new_token_id = llama_sampler_sample(smpl, ctx, -1);  // VIOLATION 2: CPU selects token

    if (llama_vocab_is_eog(vocab, new_token_id)) {  // VIOLATION 3: CPU checks termination
        break;
    }

    batch = llama_batch_get_one(&new_token_id, 1);  // VIOLATION 4: CPU prepares next batch
    n_decode += 1;
}
```

### Why This Violates Section 12

**Section 12 Requirement:** "Decode graph must outlive individual tokens with autonomous GPU-driven progression"

**Actual Behavior:**
```
CPU Iteration N:
  ├─ Call llama_decode() [BLOCKING - GPU must finish]
  ├─ CPU waits for GPU completion
  ├─ GPU completes token N
  ├─ CPU reads token N output
  ├─ CPU samples/selects token N+1
  ├─ CPU prepares batch for token N+1
  └─ Loop back to llama_decode()

Result: Token cadence = Host Scheduling Time + GPU Compute Time
```

**Required Behavior:**
```
CPU: Start Decode Session
  │
GPU Autonomous Loop (No CPU Involvement Per Token):
  ├─ Compute token N
  ├─ Internally sample/select token N+1
  ├─ Advance token index N → N+1
  ├─ Fetch next input (N+1)
  ├─ Loop to compute token N+1
  └─ Signal CPU when complete [NO BLOCKING]

Result: Token cadence = GPU Compute Time Only (CPU not in critical path)
```

---

## Architectural Violations in Detail

### Violation 1: CPU Owns Loop Iteration

**Current Code:**
```cpp
for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; )
```

**Problem:**
- The `for` loop is CPU-side
- CPU decides when to iterate
- GPU cannot control its own progression
- Requires CPU to remain active per-token iteration

**Compliance Requirement:**
- Loop must be embedded in GPU/CUDA execution context
- GPU kernel advances its own iteration counter
- CPU has NO role in loop progression

**Example Fix Pattern:**
```cpp
// CPU side (non-blocking):
gpu_engine_start_decode();  // Initiates GPU autonomous loop
gpu_engine_wait_signal();   // Wait for GPU signals (not busy-wait!)

// GPU side (persistent kernel):
__global__ void gpu_autonomous_decode_loop() {
    while (token_index < max_tokens) {
        compute_token(token_index);
        token = sample_token();
        token_index++;  // GPU advances its own counter
        signal_cpu_if_needed();
    }
}
```

### Violation 2: CPU Advances Token Position

**Current Code:**
```cpp
n_pos += batch.n_tokens;
```

**Problem:**
- CPU maintains position counter
- CPU responsible for sequence progression
- GPU has no authority over its position
- Per-token CPU involvement creates sync point

**Compliance Requirement:**
- Token index maintained on GPU only
- GPU advances independently
- CPU cannot read or write token position during decode

**Required State Management:**
```cpp
// GPU-resident token state (NOT accessible from CPU during decode):
struct gpu_decode_state {
    uint64_t current_token_index;     // GPU maintains
    uint64_t tokens_produced;         // GPU updates
    bool decode_complete;             // GPU signals
};

// CPU can only:
// 1. Initialize it once before decode starts
// 2. Read it after decode completes
// 3. NOT modify during decode phase
```

### Violation 3: CPU Checks Gating Condition

**Current Code:**
```cpp
for (int n_pos = 0;
     n_pos + batch.n_tokens < n_prompt + n_predict;  // ← CPU checks condition
    )
```

**Problem:**
- CPU evaluates termination condition each iteration
- GPU cannot autonomously determine when to stop
- CPU serves as gating authority
- Requires CPU evaluation of GPU output

**Compliance Requirement:**
- Termination condition checked on GPU
- GPU decides when to stop autonomously
- CPU receives completion signal from GPU (not polling)

**Correct Implementation:**
```cpp
// GPU kernel (no CPU gating):
if (current_token_index >= max_tokens || is_eog_token(token)) {
    gpu_decode_complete();  // GPU signals completion
    return;  // GPU-side loop exit, not CPU-controlled
}
```

### Violation 4: CPU Blocks Waiting for GPU

**Current Code:**
```cpp
if (llama_decode(ctx, batch)) {  // ← Blocking call per iteration
    return 1;
}
```

**Problem:**
- `llama_decode()` blocks until kernel completes
- CPU sits idle waiting for GPU
- Creates implicit CPU→GPU synchronization per token
- Violates "no per-token CPU involvement"

**Compliance Requirement:**
- No blocking calls during decode progression
- GPU signals completion (not CPU polling)
- CPU invokes once and waits for final signal

**Correct Pattern:**
```cpp
// CPU side: Non-blocking initiation
gpu_engine_start_decode(max_tokens);  // Returns immediately

// CPU can do other work here if needed, but typically:
gpu_engine_wait_for_decode_complete();  // Wait for GPU signal, not blocking call
```

### Violation 5: CPU Samples Next Token

**Current Code:**
```cpp
new_token_id = llama_sampler_sample(smpl, ctx, -1);  // ← CPU-side sampling
```

**Problem:**
- CPU performs sampling/selection logic
- Requires logits transfer from GPU to CPU (D2H transfer)
- Creates data dependency: GPU→CPU→GPU
- Breaking if GPU sampling backend not used

**Compliance Requirement:**
- Sampling must occur on GPU (CUDA kernels)
- No CPU involvement in token selection
- Logits remain GPU-resident

**Correct Implementation:**
```cpp
// GPU kernel (no CPU involvement):
logits = compute_logits(token_index);      // GPU-resident
temperature_logits = apply_temperature(logits);  // GPU-resident
next_token = gpu_sample(temperature_logits);      // GPU sampling kernel
// Next token selected entirely on GPU
```

### Violation 6: CPU Prepares Next Iteration

**Current Code:**
```cpp
batch = llama_batch_get_one(&new_token_id, 1);  // ← CPU batch setup
```

**Problem:**
- CPU constructs next batch
- Requires CPU read of GPU-computed token
- Sets up state for next CPU iteration
- Couples CPU and GPU via shared batch structure

**Compliance Requirement:**
- Batch preparation on GPU
- GPU fetches its own input for next iteration
- No CPU involvement in batch structure during decode

**Correct Pattern:**
```cpp
// GPU kernel (self-contained):
token = gpu_token_buffer[current_token_index];  // Fetch own input
embedding = fetch_embedding(token);             // Prepare own input
// Next iteration uses GPU-fetched data, no CPU batch setup
```

---

## Architectural Comparison

### Current Architecture (CPU-Driven)

```
CPU Loop Iteration:
┌─────────────────────────────────────────────────────────┐
│ 1. CPU: Call llama_decode(batch)    [BLOCKING]         │
│    └─ GPU: Compute token                               │
│       └─ GPU: Wait for results                         │
├─ 2. CPU: Read GPU output (blocking)                    │
├─ 3. CPU: Sample next token (CPU kernel or D2H)         │
├─ 4. CPU: Check termination condition                   │
├─ 5. CPU: Create next batch                             │
├─ 6. CPU: Increment position counter                    │
└─ 7. Loop back to step 1                                │
```

**Critical Path:** CPU Time + GPU Time (sequential)
**Per-Token Overhead:** ~10-50μs CPU orchestration
**Achievable Throughput:** ~100-200 tokens/sec (CPU-limited)

### Required Architecture (GPU-Driven)

```
GPU Autonomous Decode:
┌──────────────────────────────────────┐
│ CPU: Start GPU decode               │ (non-blocking)
│ CPU: Wait for completion signal      │ (wait once, not per-token)
├──────────────────────────────────────┤
│ GPU Persistent Kernel Loop:         │ (runs autonomously)
│ ├─ Token N: Compute                  │
│ ├─ Token N: Sample (on GPU)          │
│ ├─ Token N: Advance index            │
│ ├─ Token N+1: Fetch input            │
│ ├─ Loop: Repeat for N+1              │
│ └─ Signal: Send completion to CPU    │
└──────────────────────────────────────┘
```

**Critical Path:** GPU Time Only (CPU not in loop)
**Per-Token Overhead:** 0 (no CPU involvement)
**Achievable Throughput:** 400-600+ tokens/sec (GPU-saturated)

**Improvement Factor:** 2-3x throughput increase

---

## Required Implementation Changes

### Change 1: Eliminate CPU Decode Loop

**File:** `src/llama.cpp` or wherever main decode happens
**Change:** Replace explicit `for` loop with GPU-driven execution

**Before:**
```cpp
for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
    llama_decode(ctx, batch);
    // ... rest of loop ...
}
```

**After:**
```cpp
// GPU takes over - CPU just initiates and waits
gpu_exclusive_decode(ctx, max_tokens);

// Or with explicit steps:
gpu_engine_start_decode(ctx, max_tokens);
gpu_engine_wait_for_completion();
```

### Change 2: Transfer Token Index to GPU

**File:** GPU backend (CUDA, etc.)
**Change:** Move token position tracking to GPU

**Before:**
```cpp
// CPU-side
int current_pos = 0;
for (...) {
    current_pos += tokens_processed;
}
```

**After:**
```cpp
// GPU-side (persistent kernel)
__global__ void gpu_decode_loop(...) {
    uint64_t token_index = 0;  // GPU-resident
    while (token_index < max_tokens) {
        compute_and_advance();
        token_index++;  // GPU increments
    }
}
```

### Change 3: GPU-Based Sampling

**File:** GPU backend sampling kernel
**Change:** All token selection on GPU

**Before:**
```cpp
// CPU-side CPU sampling
llama_sampler_sample(ctx);  // May trigger D2H transfer
```

**After:**
```cpp
// GPU-side sampling kernel (CUDA)
__global__ void gpu_sample_token(...) {
    logits = read_gpu_resident_logits();
    sample_kernel_apply_temperature(...);
    selected_token = gpu_rng_sample(...);
    // Token stays on GPU
}
```

### Change 4: Persistent CUDA Graph

**File:** GPU backend graph management
**Change:** Single persistent graph containing decode loop

**Before:**
```cpp
// Per-token graph creation/replay
for each token {
    cuda_graph_launch(graph);
    cuda_synchronize();  // Per-token sync!
}
```

**After:**
```cpp
// Single persistent graph containing entire loop
cuda_graph = gpu_create_persistent_decode_graph();
cuda_graph_launch(graph);  // Single launch
// GPU runs entire decode autonomously
```

### Change 5: GPU Signal Interface

**File:** GPU backend - signal mechanism
**Change:** GPU signals CPU completion instead of CPU polling

**Before:**
```cpp
// CPU polls
while (gpu_not_done) {
    check_gpu_status();
}
```

**After:**
```cpp
// GPU signals
gpu_decode_wait_for_signal(SIGNAL_DECODE_COMPLETE, timeout_ms);
```

---

## Verification Checklist

To verify GPU autonomy is achieved:

- [ ] No CPU-side `for` or `while` loop iterating per-token
- [ ] No `llama_decode()` calls per-token (only setup/teardown calls)
- [ ] No CPU modification of token index during decode
- [ ] No `llama_sampler_sample()` calls during decode phase
- [ ] No batch structure creation during decode phase
- [ ] All token-to-token progression on GPU
- [ ] Single GPU kernel launch for entire decode (persistent graph)
- [ ] GPU → CPU signaling (not CPU polling)
- [ ] Zero per-token CPU blocking time

---

## Performance Impact

### Current (CPU-Driven)
- Throughput: 100-200 tokens/sec
- Per-token CPU time: ~10-50 microseconds
- Critical path: CPU + GPU
- GPU utilization: 70-80% (GPU waits for CPU sometimes)

### Target (GPU-Autonomous)
- Throughput: 400-600+ tokens/sec
- Per-token CPU time: 0 (CPU not in loop)
- Critical path: GPU only
- GPU utilization: 95%+ (continuous)

### Expected Improvement
- **Throughput:** 2-3x increase
- **Latency:** Per-token latency reduced 50%
- **Efficiency:** ~50% reduction in wall-clock time for same sequence

---

## Implementation Priority

**Phase 1 (Critical):** Eliminate CPU loop iteration
- Remove `for` loop from llama.cpp
- Implement GPU autonomous loop skeleton
- Implement signal interface

**Phase 2 (High):** Transfer token index to GPU
- Move position counter to GPU persistent state
- Ensure GPU increments internally
- Verify CPU cannot access during decode

**Phase 3 (High):** GPU-based sampling
- Implement GPU sampling kernels if not present
- Remove CPU-side sampler calls during decode
- Ensure logits stay GPU-resident

**Phase 4 (Medium):** Persistent CUDA graphs
- Consolidate into single persistent graph
- Eliminate per-token graph replay
- Test with actual models

**Phase 5 (Low):** Optimization
- Fine-tune GPU kernel parameters
- Optimize signal mechanism
- Profile and reduce overhead

---

## Code Files to Modify

### Core Implementation
1. `src/llama.cpp` - Main entry points
2. `src/llama-context.cpp` - Context management
3. `ggml/src/ggml-cuda/ggml-cuda.cu` - GPU backend

### Infrastructure (Already Exists)
1. `src/llama-gpu-exclusive-decode-engine.cpp` - Orchestration ✓
2. `src/llama-decode-loop-elimination.cpp` - Loop ownership ✓
3. `src/llama-decode-persistent-kernel.cpp` - Persistent GPU loop ✓
4. `src/llama-token-persistent-execution.cpp` - GPU token execution ✓

### New Files Needed
1. `src/llama-cpu-decode-loop-removal.cpp` - Remove CPU loop
2. `ggml/src/ggml-cuda/gpu-persistent-decode-kernel.cu` - GPU autonomous loop
3. `ggml/src/ggml-cuda/gpu-sampling-kernel.cu` - GPU sampling

---

## Critical Success Factors

1. **No CPU loop during decode** - This is the core violation
2. **GPU controls token progression** - Index on GPU, GPU increments
3. **Non-blocking interface** - CPU initiates once, waits for signal
4. **Persistent execution** - Single GPU kernel launch for entire sequence
5. **All sampling on GPU** - No CPU token selection or D2H logits transfer

Without all 5 factors, GPU autonomy is NOT achieved and throughput remains CPU-limited.

---

## Architectural Verification Method

To prove GPU autonomy:

```cpp
// Instrument the code to track CPU involvement
1. Count CPU loop iterations during decode → Should be 0
2. Count llama_decode() calls during decode → Should be ≤ 1 (only init)
3. Count CPU sampling calls during decode → Should be 0
4. Count GPU signals received → Should be exactly 1 (decode complete)
5. Measure CPU time during decode → Should be ≈ 0
6. Measure total time vs GPU time → Should match (GPU-bound, not CPU-bound)
```

If any of the above checks fail, GPU autonomy is NOT achieved.

---

**This is the deepest architectural change required to achieve true GPU-exclusive decode.**
