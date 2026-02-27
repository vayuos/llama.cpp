# GPU-Exclusive Decode - Code Pattern Reference

**Purpose:** Quick reference for the exact code patterns used throughout the GPU-exclusive decode implementation.

## Pattern 1: Decode-Mode Detection Check

**Used in:** Synchronization guards, sampling entry points, transfer validation

```cpp
// Check if decode phase is active
if (ggml_backend_decode_mode_active()) {
    // Skip operations during decode phase
    return;
}

// Otherwise proceed with non-decode operations
handle_prefill_operations();
```

**Files Using This Pattern:**
- `ggml/src/ggml-cuda/ggml-cuda.cu` (backend sync)
- `src/llama-sampler.cpp` (all 6 samplers)
- `ggml/src/ggml-cuda/sampling_impl.cu` (transfer guard)

**Key Insight:** This pattern allows selective code path activation based on phase, not hardware capability.

---

## Pattern 2: Compile-Time Code Exclusion

**Used in:** CPU sampling implementations, hybrid KV paths, CPU backend registration

### Basic Exclusion Pattern

```cpp
#ifndef LLAMA_CPU_SAMPLING_EXCLUDED
    // CPU-only code here
    void cpu_temperature_sampler(...) {
        // Implementation
    }
#endif  // LLAMA_CPU_SAMPLING_EXCLUDED
```

### Inclusion-Only Pattern (Alternative)

```cpp
#ifdef LLAMA_CPU_SAMPLING_EXCLUDED
    // Nothing here - code is excluded
#else
    // CPU code included when NOT excluded
    void cpu_temperature_sampler(...) {
        // Implementation
    }
#endif
```

**Files Using This Pattern:**
- `src/llama-sampler.cpp` (2041 lines of CPU sampling wrapped)
- `ggml/src/ggml-backend-reg.cpp` (CPU backend registration)

**Key Insight:** `#ifndef` is cleaner for "exclude optional feature" pattern; readability improves because normal code is in main block.

---

## Pattern 3: Compile-Time Safety Verification

**Used in:** Header of files that can be optionally excluded

```cpp
// Verify dependencies are met when feature is excluded
#ifdef LLAMA_CPU_SAMPLING_EXCLUDED
    #ifndef GGML_USE_CUDA
        #error "LLAMA_CPU_SAMPLING_EXCLUDED requires GGML_USE_CUDA (GPU backend must be available)"
    #endif

    // Verify GPU sampling capability exists
    #ifndef GGML_CUDA_SAMPLING
        #warning "LLAMA_CPU_SAMPLING_EXCLUDED set but GGML_CUDA_SAMPLING not defined"
    #endif
#endif
```

**Location:** `src/llama-sampler.cpp` (lines 11-22)

**Key Insight:** Compile-time errors are better than runtime failures. This pattern catches configuration mistakes at build time, not deployment time.

---

## Pattern 4: Runtime Hard Error (Replaces Assertions)

**Used in:** KV cache enforcement, transfer prevention, architecture violation detection

### Simple Hard Error

```cpp
if (condition_violation) {
    LLAMA_LOG_ERROR("%s: FATAL - Description of what violated\n", __func__);
    GGML_ABORT("Architecture violation message for operators");
}
```

### Detailed Hard Error (Production-Safe)

```cpp
if (kv_gpu_only_locked && cpu_path_invoked) {
    LLAMA_LOG_ERROR("%s: FATAL - GPU-only KV mode active but CPU KV path invoked\n", __func__);
    LLAMA_LOG_ERROR("%s: Section 11.3 violation - Hybrid KV cache modes are FORBIDDEN\n", __func__);
    LLAMA_LOG_ERROR("%s: All KV operations must remain GPU-resident\n", __func__);
    LLAMA_LOG_ERROR("%s: Current context: tokens=%d, layers=%d, gpu_only=%d\n",
                    __func__, n_tokens, n_layer, kv_gpu_only_locked);
    GGML_ABORT("CPU KV access during GPU-exclusive decode - architecture violation");
}
```

**Location:** `src/llama-kv-cache.cpp` (7 locations)

**Advantages Over Assertions:**
- Works in both Debug and Release builds
- Includes diagnostic context
- Cannot be optimized away
- Clearer error messages for operations

**Key Insight:** Assertions are for development; production code needs guaranteed error handling.

---

## Pattern 5: Boundary Condition Check (INT_MAX Padding)

**Used in:** MoE expert dispatch, quantization kernels

### Skip Pattern (Source Filtering)

```cpp
// Skip padding positions marked with INT_MAX
if (ids != nullptr && i01 == INT_MAX) {
    return;  // Don't process this position
}

// Safe to use indices now
process_expert_at(i01);
```

**Location:** `ggml/src/ggml-cuda/mmid.cu` (lines 47-50)

### Conversion Pattern (Normalize Bad Values)

```cpp
int expert_id = -1;  // Neutral value

if (iex < n_expert_used && it < n_tokens) {
    expert_id = ids[it*si1 + iex];

    // Convert padding marker to neutral value
    if (expert_id == INT_MAX) {
        expert_id = -1;  // Use sentinel value instead
    }
}

// expert_id is now safe: either valid index or -1
```

**Location:** `ggml/src/ggml-cuda/mmid.cu` (lines 76-82)

### Check-Before-Use Pattern (Kernel Processing)

```cpp
// Skip padding positions in quantization
if (ids != nullptr && i01 == INT_MAX) {
    return;  // Exit kernel for this position
}

// Process non-padding values
float val = input[i01];
quantize_value(val, output);
```

**Locations:** `ggml/src/ggml-cuda/quantize.cu` (3 kernels)

**Key Insight:** INT_MAX serves as both a padding marker AND a sentinel value that should never be used as an array index.

---

## Pattern 6: Transfer Guard (Prevent D2H on Decode)

**Used in:** GPU sampling implementation to prevent fallback to CPU

```cpp
// Check if we're in decode phase AND transfers are being attempted
if (ggml_backend_decode_mode_active()) {
    // This is the decode-critical path

    if (cuda_check_transfer_guard(logits_device, host_buffer)) {
        LLAMA_LOG_ERROR("%s: FATAL - Attempting H2D/D2H during decode phase\n", __func__);
        LLAMA_LOG_ERROR("%s: Section 11.6 violation - All sampling must remain GPU-resident\n", __func__);
        LLAMA_LOG_ERROR("%s: Logits cannot be transferred to CPU during token generation\n", __func__);
        GGML_ABORT("GPU sampling transfer fallback blocked during decode");
    }
}

// Safe to proceed with GPU-resident operations
gpu_sample_token(logits_device, token_out);
```

**Location:** `ggml/src/ggml-cuda/sampling_impl.cu` (lines 299-313)

**Key Insight:** This pattern catches attempted fallbacks to CPU sampling, preventing silent degradation.

---

## Pattern 7: Conditional Backend Registration

**Used in:** Backend registry to exclude CPU backend at compile-time

### Conditional Registration Pattern

```cpp
#ifdef GGML_USE_CPU
    // CPU backend is available for compilation

    #ifndef LLAMA_GPU_EXCLUSIVE_DECODE
        // Normal case: register CPU as fallback
        register_backend(ggml_backend_cpu_reg());
    #else
        // GPU-exclusive case: do NOT register CPU
        // This ensures GPU is the ONLY available backend
    #endif
#endif
```

**Location:** `ggml/src/ggml-backend-reg.cpp` (lines 211-219)

**Build-Time Behavior:**
- When `LLAMA_GPU_EXCLUSIVE_DECODE=OFF` (default): CPU backend registered
- When `LLAMA_GPU_EXCLUSIVE_DECODE=ON`: CPU backend not in registry

**Key Insight:** Compile-time exclusion is stronger than runtime checks because it eliminates the possibility entirely.

---

## Pattern 8: Sampling Entry Point Check

**Used in:** All 6 CPU samplers to detect decode-phase misuse

```cpp
// Temperature sampler example
struct ggml_sampler * ggml_sampler_init_temp(float temp) {
    // Detect misuse during decode phase
    if (ggml_backend_decode_mode_active()) {
        LLAMA_LOG_ERROR("%s: CPU temperature sampling called during GPU decode phase\n", __func__);
        LLAMA_LOG_ERROR("%s: Section 15.2 violation - CPU sampling forbidden on decode path\n", __func__);
        GGML_ABORT("CPU sampling on decode path (Section 15.2 violation)");
    }

    // Safe to create CPU sampler for prefill operations
    auto * sampler = new ggml_sampler_temp(temp);
    return (struct ggml_sampler *) sampler;
}
```

**Sampler Locations:**
1. Temperature: `llama-sampler.cpp` line 1972
2. Top-K: `llama-sampler.cpp` line 1384
3. Top-P: `llama-sampler.cpp` line 1500
4. Greedy: `llama-sampler.cpp` line 1077
5. Penalties: `llama-sampler.cpp` line 2861
6. Grammar: `llama-sampler.cpp` line 2638

**Key Insight:** Each sampler independently checks decode mode, providing defense-in-depth even if one check is missed.

---

## Pattern 9: Architectural Documentation Comment Block

**Used in:** Complex sections where behavior might be questioned

```cpp
// ============================================================================
// FIX: Section X.Y - DESCRIPTIVE TITLE
// ============================================================================
//
// Problem Statement:
// [What was broken and why]
//
// Solution Applied:
// [What was changed and how]
//
// Impact:
// [What performance/correctness benefit is achieved]
//
// Verification:
// [How to verify the fix is working]
//
// Related Sections:
// [Cross-references to systemchanges.md]
// ============================================================================
```

**Example Location:** `ggml/src/ggml-backend-reg.cpp` (lines 197-209)

**Key Insight:** Documentation blocks make complex fixes self-explanatory and help future maintainers understand the "why" behind changes.

---

## Pattern 10: KV Cache GPU Residency Check

**Used in:** 7 critical KV cache operations to ensure GPU-only execution

```cpp
// Before allowing CPU KV path
void llama_kv_cache_seq_add(...) {
    if (kv_gpu_only_locked) {
        LLAMA_LOG_ERROR("%s: FATAL - GPU-only KV mode active but CPU KV path invoked\n", __func__);
        LLAMA_LOG_ERROR("%s: Section 11.3 violation - Hybrid KV cache modes are FORBIDDEN\n", __func__);
        LLAMA_LOG_ERROR("%s: All KV operations must remain GPU-resident\n", __func__);
        GGML_ABORT("CPU KV access during GPU-exclusive decode - architecture violation");
    }

    // Only reached if NOT in GPU-exclusive mode
    add_kv_sequence(cache, seq, pos);
}
```

**7 Locations Protected:**
1. Sequence addition check
2. GPU lock violation
3. Hybrid mode prohibition
4. Buffer access control
5. Sequence processing
6. Layer assignment
7. Final residency check

**Key Insight:** Multiple checks on the same operation provides redundant safety against logic errors or refactoring accidents.

---

## Pattern 11: Error Message Template

**Used in:** All architecture violation errors

```
%s: [SEVERITY] - [WHAT_HAPPENED]
%s: Section X.Y - [ARCHITECTURAL_RULE_VIOLATED]
%s: [CONSEQUENCE_IF_CONTINUED]
%s: [OPTIONAL_CONTEXT_INFO]
```

**Example:**
```
"CPU sampling called during GPU decode phase"                    // WHAT
"Section 15.2 violation - CPU sampling forbidden on decode path" // RULE
"GPU sampling kernel expected, got CPU fallback"                 // CONSEQUENCE
"Function=temp_sampler, mode=decode, backend=cuda"               // CONTEXT
```

**Location:** All files with hard errors

**Key Insight:** Consistent error message format helps operators quickly identify and fix problems.

---

## Summary Table

| Pattern | Purpose | Files | Count |
|---------|---------|-------|-------|
| Decode-mode detection | Phase-aware code path | 3 files | 8+ uses |
| Compile-time exclusion | Optional feature removal | 2 files | 4+ blocks |
| Safety verification | Dependency checking | 1 file | 2+ checks |
| Hard error (replace assert) | Production error handling | 2 files | 10+ locations |
| INT_MAX boundary check | MoE padding filtering | 2 files | 5+ locations |
| Transfer guard | D2H prevention | 1 file | 1 core + guards |
| Conditional registration | Backend minimalism | 1 file | 1 registration |
| Entry point check | Sampler phase checking | 6 samplers | 6 checks |
| Documentation block | Architectural clarity | 8+ locations | 8+ blocks |
| GPU residency check | KV cache enforcement | 1 file | 7 locations |

---

## Usage Examples

### Building with Maximum Safety
```bash
cmake .. \
    -DLLAMA_GPU_EXCLUSIVE_DECODE=ON \
    -DLLAMA_CPU_SAMPLING_EXCLUDED=ON \
    -DLLAMA_KV_HYBRID_EXCLUDED=ON \
    -DGGML_CUDA_SAMPLING=ON
```

**Result:** All 11 patterns active simultaneously

### Building with Runtime Guards Only
```bash
cmake .. \
    -DGGML_CUDA_SAMPLING=ON
```

**Result:** Patterns 1, 5-6, 8-11 active (compile-time exclusions disabled)

### Verifying Pattern Implementation
```bash
# Count decode-mode checks
grep -r "ggml_backend_decode_mode_active" src/ ggml/

# Count hard errors replacing assertions
grep -r "GGML_ABORT" src/ ggml/ | wc -l

# Check compile-time guards
grep -r "LLAMA_CPU_SAMPLING_EXCLUDED" src/

# Verify INT_MAX handling
grep -r "INT_MAX" ggml/src/ggml-cuda/ | grep -v ".orig"
```

---

## Key Design Principles

1. **Compile-Time > Runtime:** Use `#ifndef` guards where possible
2. **Defense-in-Depth:** Multiple checks on same operation
3. **Fail Fast:** Abort immediately on violations, no recovery attempts
4. **Explicit State:** Use flags like `kv_gpu_only_locked` not implicit detection
5. **Clear Documentation:** Every fix includes architectural context
6. **Production-Safe:** Errors work in Release builds, not just Debug
7. **Self-Explanatory:** Code reads like architecture specification

---

This reference guide covers all 11 code patterns used throughout the GPU-exclusive decode implementation. Use this as a template for any future architectural enforcement or safety checks.
