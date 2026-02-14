# Decode-Critical Op Boundary Splitting Prevention - Implementation Summary

## Overview

This implementation enforces strict structural changes to prevent any single logical decode operation from being split across CPU and GPU backends. The enforcement is multi-layered, operating at different levels of the inference pipeline.

## Implementation Components

### 1. Core Invariant Definition (`llama-decode-structure.h/cpp`)

**Purpose:** Define and enforce the fundamental invariant that decode-critical ops execute on a single backend (GPU).

**Key Functions:**

- `llama_decode_validate_single_backend_ownership()` - Validates that a node has GPU backend ownership
- `llama_decode_freeze_ownership()` - Freezes backend ownership state to prevent reassignments
- `llama_decode_tensor_gpu_resident_check()` - Ensures tensors scheduled for GPU remain GPU-resident
- `llama_decode_validate_all_gpu_resident()` - BFS traversal to verify entire decode graph is GPU-resident
- `llama_decode_check_backend_match_strict()` - Prevents implicit backend bridging on mismatch
- `llama_decode_audit_no_implicit_copies()` - Audits tensor for implicit device transfers
- `llama_decode_reject_cpu_fallbacks()` - Blocks CPU fallback micro-ops during decode
- `llama_decode_dump_backend_assignments()` - Logs backend assignments for audit/debugging
- `llama_decode_alloc_ownership()` / `llama_decode_free_ownership()` - Ownership tracking allocation

**Data Structures:**

```c
typedef struct {
    int32_t node_id;         // Index in graph
    int32_t backend_owner;   // Single backend (0=GPU)
    bool is_locked;          // Immutable flag
    bool is_composite;       // Composite op flag
} llama_decode_node_owner;

typedef struct {
    llama_decode_node_owner * nodes;
    int32_t n_nodes;
    int32_t primary_backend; // Expected GPU backend
    uint64_t graph_hash;     // Graph integrity check
    bool is_frozen;          // Frozen state flag
} llama_decode_graph_ownership;
```

### 2. Implicit Device Transfer Prevention (`ggml-backend.cpp`)

**Purpose:** Block implicit device transfers across CPU↔GPU boundaries during decode execution.

**Enforcement Point:** `ggml_backend_tensor_copy()` function

**Check Added:**
```cpp
if (ggml_get_decode_mode()) {
    bool src_is_host = ggml_backend_buffer_is_host(src->buffer);
    bool dst_is_host = ggml_backend_buffer_is_host(dst->buffer);
    
    if (src_is_host != dst_is_host) {
        GGML_ABORT("DECODE STRUCTURE VIOLATION: Implicit device transfer during decode mode");
    }
}
```

**Guarantees:**
- No mid-operation tensor materialization on CPU
- No automatic host↔device memory migration
- All intermediate tensors remain on their allocated device

### 3. Composite Op Enforcement (`llama-decode-composite.h/cpp`)

**Purpose:** Ensure composite decode operations (attention, matmul, norm, softmax, etc.) execute entirely on GPU with all sub-kernels present.

**Composite Ops Tracked:**
- `mul_mat`, `mul_mat_q`, `mul_mat_id_q` - Quantized/dense matrix multiply
- `norm`, `rms_norm`, `group_norm` - Normalization layers
- `soft_max` - Attention softmax
- `gelu`, `silu`, `relu` - Activation functions
- `sum_rows`, `repeat` - Tensor manipulation

**Key Functions:**

- `ggml_composite_op_enforce_gpu_only()` - Enforces GPU-only execution of composite ops
  - Verifies GPU backend supports the operation
  - Validates all input tensors are GPU-resident
  - Aborts if CPU fallback would be required

- `ggml_audit_no_cpu_fallbacks_in_decode()` - Audits graph for fallback risks
  - Scans for softmax fallback patterns
  - Detects norm decomposition fallbacks
  - Identifies quantization decompose operations
  - Warns of attention fallback risks

- `ggml_validate_decode_graph_all_gpu()` - Validates all-GPU assignment
  - Ensures every node assigned to backend 0 (GPU)
  - Aborts on non-GPU assignments during decode

- `ggml_validate_decode_graph_immutable()` - Validates immutability after freeze
  - Compares current vs. frozen backend assignments
  - Aborts on any divergence from frozen state

### 4. Scheduler Enforcement (`ggml-backend.cpp`)

**Purpose:** Prevent backend reassignment, mixed execution, and dynamic graph splitting during decode.

**Key Enforcement Points:**

1. **Single Backend Mode (`ggml_backend_sched_set_single_backend()`)**
   - Forces all nodes to a specific backend
   - No fallback or dynamic selection
   - GPU unsupported ops cause allocation failure

2. **Backend Locking (`ggml_backend_sched_lock_backends()`)**
   - After freeze, backend selection becomes immutable
   - Any attempt to modify backends aborts

3. **Graph Splitting Guard (`ggml_backend_sched_split_graph()`)**
   - Checks `is_backend_locked` flag
   - Aborts if split requested while locked
   - Prevents per-token replan

4. **Backend Cache (`backend_cache` array)**
   - Populated during first plan pass
   - Reused on subsequent decode invocations
   - Ensures consistent node→backend mapping

**Implementation:**
```cpp
// Enforce single backend for decode graphs
if (sched->single_backend_id != -1) {
    for (int i = 0; i < graph->n_nodes; i++) {
        tensor_backend_id(node) = sched->single_backend_id;
        
        // Strict check: backend must support op, no fallback
        if (!ggml_backend_supports_op(sched->backends[id], node)) {
            GGML_ABORT("Node '%s' not supported by single bound backend");
        }
    }
}

// Validate immutability after freeze
if (sched->is_backend_locked && should_replan) {
    GGML_ABORT("Graph re-planning forbidden while scheduler is locked");
}
```

### 5. Context-Level Integration (`llama-context.cpp`)

**Purpose:** Orchestrate enforcement at the decode/prefill boundary.

**Key Integration Points:**

1. **Graph Type Detection**
   ```cpp
   if (gtype == LLM_GRAPH_TYPE_DECODER) {
       ggml_backend_sched_set_single_backend(sched.get(), 0);
       // Enable all checks below
   }
   ```

2. **Post-Allocation Validation**
   ```cpp
   if (gtype == LLM_GRAPH_TYPE_DECODER) {
       ggml_audit_no_cpu_fallbacks_in_decode(gf);
       llama_decode_validate_all_gpu_resident(gf->nodes[last]);
   }
   ```

3. **Graph Freezing**
   ```cpp
   if (gtype == LLM_GRAPH_TYPE_DECODER) {
       active_decode_graph = gf;  // Pin pointer
       ggml_backend_sched_lock_backends(sched.get(), true);
       ggml_backend_set_decode_mode(true);
   }
   ```

4. **Decode Mode State Management**
   - **Entry:** Set when decode graph allocated
   - **Execution:** All ops must be GPU-resident
   - **Exit:** Only on context shutdown or explicit reset

## Execution Guarantees

After enforcement, the following properties hold during autonomous token decode:

### Structural Invariants
1. **Single Backend Ownership** - Every node has exactly one backend owner (GPU:0)
2. **No Boundary Crossing** - No tensor crosses CPU↔GPU boundary mid-operation
3. **GPU Exclusivity** - CPU cannot be pacing resource or execution bottleneck
4. **Immutability** - Graph topology and backend assignments are frozen

### Operational Guarantees
1. **No Implicit Transfers** - No automatic `ggml_backend_tensor_copy` on mismatch
2. **No Mixed Execution** - Composite ops execute entirely on GPU or fail
3. **No Fallbacks** - Unsupported ops abort rather than fallback to CPU
4. **No Dynamic Replanning** - Graph remains static from freeze until decode complete

### Performance Implications
1. **Reduced Idle Gaps** - GPU executes continuously without CPU intervention
2. **Predictable Latency** - No dynamic backend selection or fallback overhead
3. **Streamlined Memory** - All intermediate tensors remain device-resident
4. **Efficient PCIe Usage** - Only critical data transferred pre-decode

## Error Detection

Violations are detected and reported at multiple points:

| Check | Location | Aborts On |
|-------|----------|-----------|
| Device Transfer | `ggml_backend_tensor_copy()` | CPU↔GPU crossing mid-op |
| Single Backend | Scheduler split | Non-GPU nodes in decode |
| Composite Ops | Post-allocation | Missing GPU kernels |
| Fallback Risk | Graph audit | Softmax/norm/quantize/attention CPU paths |
| Immutability | Graph replan | Backend reassignment post-freeze |
| GPU Residency | Graph validation | Host-resident decode tensors |
| Scheduler Lock | `split_graph()` | Replan attempt while locked |

## Testing Strategy

### Unit Tests
- Backend ownership validation
- Tensor residency checks
- Implicit copy detection
- Fallback rejection

### Integration Tests
- Decode graph freeze/lock behavior
- Multi-token generation validation
- Reuse of frozen graphs
- Composite op execution

### Regression Tests
- Performance comparison (frozen vs. dynamic)
- Memory layout consistency
- Backend assignment stability

## Migration Path

For existing code:
1. Decode graphs automatically detected at graph build time
2. First execution triggers freezing
3. Subsequent invocations reuse frozen state
4. Non-decode (prefill/generic) graphs unaffected

For model implementations:
1. All decode-critical ops must have GPU kernels
2. No CPU-only ops in decode paths
3. Pre-allocate GPU memory for entire decode
4. Use GPU softmax/norm/matmul implementations

## References

- Specification: "Prevent CPU↔GPU Op Boundary Splitting (Decode-Critical Isolation)"
- Related: GPU_TOPK_IMPLEMENTATION.md, OPTIMIZATION_SUMMARY.md
