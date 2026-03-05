# Decode Critical Boundary Enforcement

## Overview

This implementation enforces strict structural isolation to prevent CPU↔GPU op boundary splitting during autonomous token decode. The goal is to ensure that the GPU remains the continuous execution owner with zero implicit PCIe transfers.

## Problem Solved

### The GPU Idle Gap Issue

In typical decode implementations, several common patterns cause GPU idle periods:

1. **Mixed-backend execution within composite ops**
   - GPU matmul → CPU bias add
   - GPU attention → CPU softmax
   - GPU logits → CPU sampling penalties

2. **Implicit tensor transfers**
   - GPU compute produces output
   - Output copied to CPU automatically
   - CPU reads/modifies result
   - Result copied back to GPU
   - GPU continues execution

3. **Fallback micro-ops**
   - GPU kernel unavailable for operation X
   - Silently executes on CPU instead
   - Tensor transferred to/from device
   - Hidden PCIe latency

4. **Graph re-optimization mid-decode**
   - Scheduler reassigns backend mid-execution
   - Already-scheduled ops change backends
   - Tensors must be transferred to new location

## Enforcement Architecture

### 1. Invariant: Single Backend Ownership

**Every decode-critical logical operation executes entirely on GPU.**

Decode-critical operations:
- Attention blocks (mul_mat, mul_mat_q, soft_max)
- Normalization (rms_norm, norm, group_norm)
- Feedforward networks (gelu, silu, relu)
- KV cache operations (get_rows, set_rows)
- Token sampling (softmax, argmax, sample)

### 2. Backend Assignment Immutability

Once the decode graph is frozen:
- No node can change backends
- No re-scheduling allowed
- No dynamic optimization during execution
- Graph topology remains locked

### 3. GPU-Resident Tensor Validation

All tensors in the decode computation graph:
- Allocated on GPU buffer
- No host-aliasing
- No implicit materialization on CPU
- Never copied to host

### 4. Implicit Bridging Prevention

When a tensor is on backend A and operation runs on backend B:
- Normal mode: automatic copy (allowed)
- Decode mode: immediate abort (forbidden)

### 5. Fallback Rejection

If GPU backend lacks implementation for critical op:
- Graph construction fails
- No fallback to CPU
- Requires GPU implementation to exist

### 6. Graph Immutability After Freeze

After decoding starts:
- Backend assignments locked
- Tensor relocation forbidden
- Op re-scheduling forbidden
- Topology changes forbidden

## Key Components

### `llama-decode-boundary-enforce.h/.cpp`

Core enforcement state machine and validators:
- `llama_decode_boundary_activate()` - Start decode mode
- `llama_decode_boundary_freeze_graph()` - Lock graph topology
- `llama_decode_boundary_validate_immutable()` - Check no changes
- `llama_decode_boundary_enforce_all_gpu()` - Verify all nodes GPU-resident
- `llama_decode_boundary_check_no_bridging()` - Prevent implicit copies
- `llama_decode_boundary_audit_all_gpu_resident()` - Validate tensor allocation

### `llama-decode-boundary-integration.h/.cpp`

Context-aware wrappers that integrate with graph pipeline:
- `llama_decode_boundary_activate_context()` - Enable for context
- `llama_decode_boundary_pre_compute_validate()` - Pre-execution checks
- `llama_decode_boundary_post_compute()` - Post-execution cleanup
- `llama_decode_boundary_enforce_composite_op()` - Graph building hook

### `ggml-decode-enforcement.h/.cpp`

Low-level backend dispatch enforcement:
- `ggml_decode_check_implicit_copy()` - Block device transfers
- `ggml_decode_check_backend_immutable()` - Lock assignments
- `ggml_decode_check_tensor_buffer_residency()` - Validate allocation
- `ggml_decode_check_op_has_gpu_impl()` - Require GPU implementations
- `ggml_decode_check_no_cpu_nodes()` - Audit no CPU nodes

### `llama-decode-composite.h/.cpp`

Operation-level enforcement:
- `ggml_composite_op_enforce_gpu_only()` - Composite op validation
- `ggml_audit_no_cpu_fallbacks_in_decode()` - Fallback detection
- `ggml_validate_decode_graph_all_gpu()` - Graph-wide validation

## Integration Points

### Graph Building Phase
When constructing the decode computation graph:

```cpp
// 1. Mark op as decode-critical
ggml_composite_op_enforce_gpu_only(op, gpu_backend);

// 2. Validate all sub-components have GPU support
ggml_audit_no_cpu_fallbacks_in_decode(graph);
```

### Before Graph Execution
Pre-compute validation ensures invariants hold:

```cpp
// 1. Activate boundary enforcement
llama_decode_boundary_activate_context(ctx, GPU_BACKEND_ID);

// 2. Freeze graph topology (locked from this point)
llama_decode_boundary_freeze_graph(state, node_backends, n_nodes, graph_hash);

// 3. Validate all nodes GPU-assigned
llama_decode_boundary_enforce_all_gpu(state, node_backends, n_nodes);

// 4. Audit all tensors GPU-resident
llama_decode_boundary_audit_all_gpu_resident(state, root_tensor);
```

### During Backend Dispatch
When scheduler assigns ops to backends:

```cpp
// Reject any backend mismatch (implicit copy)
ggml_decode_check_implicit_copy(tensor_backend, op_backend, in_decode);

// Verify op has GPU implementation
ggml_decode_check_op_has_gpu_impl(op_name, backend, in_decode);

// Validate tensor is properly allocated
ggml_decode_check_tensor_buffer_residency(tensor, backend, in_decode);
```

### After Each Graph Compute
Maintains audit trail:

```cpp
// Validate immutability (no reassignments happened)
llama_decode_boundary_validate_immutable(state, current_backends, n_nodes);

// Log any violations for debugging
llama_decode_boundary_dump_context_assignments(ctx, graph, backends, n_nodes);
```

## Error Handling

All violations result in **immediate abort** with detailed diagnostic messages:

### Example: CPU Node in Decode Graph
```
DECODE BOUNDARY ERROR: Node 42 assigned to backend 1 (expected 0 - GPU only).
  Decode graphs must be 100% GPU-resident.
  No CPU ops, no fallbacks, no exceptions.
```

### Example: Implicit Backend Bridging
```
DECODE BOUNDARY: Implicit bridging detected!
  Tensor backend: 0 (GPU)
  Op backend: -1 (CPU)
  Decode forbids automatic device transfers.
```

### Example: GPU-Scheduled Tensor is CPU-Resident
```
DECODE BOUNDARY: Tensor 'logits' assigned to GPU but is CPU-resident!
  Implicit CPU materialization detected.
```

### Example: Backend Reassignment
```
DECODE STRUCTURE VIOLATION: Backend assignment changed for node 15 ('softmax').
  Previous backend: 0
  Current backend: -1
  Decode graphs are immutable after freeze.
```

## Performance Impact

### Benefits
- **Eliminates GPU idle gaps** from implicit device transfers
- **Prevents hidden PCIe latency** from mid-operation copies
- **Guarantees continuous GPU execution** during decode
- **No scheduler overhead** once graph is frozen

### Overhead
- One-time cost during graph freezing (validation passes)
- Negligible per-node cost (simple integer checks)
- No impact after freeze (assertions only, can be optimized out in release builds)

## Testing and Validation

### Unit Tests
Test boundary enforcement in isolation:
```cpp
// Test GPU-only assignment
test_enforce_all_gpu();

// Test immutability after freeze
test_validate_immutable();

// Test implicit copy rejection
test_check_no_bridging();

// Test tensor residency validation
test_validate_tensor_residency();

// Test CPU fallback rejection
test_reject_fallback_ops();
```

### Integration Tests
Test within full context and graph execution:
```cpp
// Test decode graph freezing works end-to-end
test_decode_boundary_integration();

// Test context-level enforcement
test_context_decode_boundary();

// Test mixed context scenarios
test_multi_context_isolation();
```

### Stress Tests
Validate under various architectures:
- Different model architectures
- Various batch sizes
- Different GPU backends (CUDA, Metal, etc.)
- Multi-GPU scenarios

## Debug Output

Enable detailed logging:
```cpp
// Log all boundary enforcement actions
LLAMA_LOG_DEBUG("DECODE BOUNDARY: ...");

// Log backend assignments for audit
llama_decode_boundary_dump_context_assignments(ctx, graph, backends, n_nodes);

// Inspect frozen graph state
llama_decode_boundary_dump_assignments(state, graph, backends, n_nodes);
```

## Future Enhancements

1. **Per-device enforcement** - Handle multi-GPU scenarios
2. **Dynamic kernel validation** - Verify GPU kernel exists before scheduling
3. **Tensor memory tracking** - Track allocation source and changes
4. **Performance metrics** - Count boundary checks and violations
5. **Adaptive fallback detection** - Identify operations with poor GPU support

## References

### Related Code
- `llama-context.h` - Context structure with boundary state
- `llama-graph.h` - Graph building and execution
- `llama-arch.cpp` - Model architecture and op registration
- `ggml-backend.cpp` - Backend dispatch layer

### Design Principles
1. **Fail fast** - Abort on first violation rather than silent errors
2. **Explicit over implicit** - No silent device transfers in decode
3. **Immutability** - Graph cannot change once frozen
4. **Audit trail** - Full logging of all enforcement actions
5. **GPU-exclusive** - CPU is never an option during decode

