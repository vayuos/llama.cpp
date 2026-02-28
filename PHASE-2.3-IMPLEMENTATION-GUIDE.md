# Phase 2.3: CUDA Stream Synchronization & Performance Realization

## Overview

Phase 2.3 adds actual CUDA stream event synchronization to realize the 15-25% throughput gains from async pipelining.

**Status:** Planning & architecture
**Expected Impact:** +15-25% throughput (6.67 → 8-10 tokens/sec)
**Estimated Effort:** 4-6 hours
**Risk Level:** Medium (GPU synchronization, threading)

---

## Architecture

### Pipeline Execution Model

```
Time →

Token N (CPU):
  [CPU compute layers 0-66 on CPU_COMPUTE stream] ──→ [Record event]
                                                              ↓
Token N+1 (CPU):
  [CPU compute layers 0-66 on CPU_COMPUTE stream] ──→ [Record event]
                          ↓ (parallel execution)

Token N (GPU):
                    [Wait for N's CPU event]
                          ↓
        [GPU compute layers 36-49 on GPU_COMPUTE stream] ──→ [Record event]
                                                                    ↓
Token N+1 (GPU):
                              [Wait for N+1's CPU event]
                                      ↓
                  [GPU compute layers 36-49 on GPU_COMPUTE stream] ──→ Done
```

### Key Synchronization Points

1. **After CPU compute (layers 0-66):**
   - Record cudaEvent on CPU_COMPUTE stream
   - Event signals "CPU compute done for this token"

2. **Before GPU compute (layers 36-49):**
   - Wait for CPU's recorded event
   - Ensures CPU output available before GPU uses it
   - Guarantees correct data dependency

3. **After GPU compute:**
   - Record cudaEvent on GPU_COMPUTE stream
   - Event signals "GPU compute done, output ready"

4. **On output:**
   - Wait for GPU's recorded event if needed
   - Ensures GPU finished before returning token to user

---

## Implementation Steps

### Step 1: Extend Scheduler with CUDA Event Handling

**File:** `src/llama-stream-scheduler.cpp`

Add CUDA event creation/recording/waiting:

```cpp
// In llama_stream_scheduler_init():
for (int i = 0; i < scheduler->num_events; i++) {
    // Create CUDA events (non-blocking, auto-reset)
    // scheduler->sync_events[i] = new cudaEvent_t;
    // cudaEventCreate((cudaEvent_t*)scheduler->sync_events[i],
    //                  cudaEventNonBlocking | cudaEventDisableTiming);
}

// In llama_stream_scheduler_record_event():
// cudaEventRecord(*(cudaEvent_t*)scheduler->sync_events[event_index],
//                 (cudaStream_t)scheduler->streams[stream_type].cuda_stream);

// In llama_stream_scheduler_wait_event():
// cudaEventSynchronize(*(cudaEvent_t*)scheduler->sync_events[event_index]);
// OR with timeout:
// cudaStreamWaitEvent((cudaStream_t)scheduler->streams[wait_on_stream].cuda_stream,
//                     *(cudaEvent_t*)scheduler->sync_events[event_index]);
```

### Step 2: Add Synchronization Hooks to Decode Engine

**File:** `src/llama-gpu-exclusive-decode-engine.cpp`

Modify `llama_gpu_exclusive_engine_decode_step()`:

```cpp
int llama_gpu_exclusive_engine_decode_step(int token) {
    ...

    // After CPU compute (would be called from compute loop):
    llama_stream_scheduler_record_event(g_stream_scheduler,
                                        LLAMA_STREAM_CPU_COMPUTE,
                                        token % 4);  // Reuse 4 events in cycle

    // Before GPU compute (would be called from compute loop):
    llama_stream_scheduler_wait_event(g_stream_scheduler,
                                      (token-1) % 4,  // Wait for prev token's CPU event
                                      5000);  // 5 second timeout

    // After GPU compute:
    llama_stream_scheduler_record_event(g_stream_scheduler,
                                        LLAMA_STREAM_GPU_COMPUTE,
                                        token % 4);
    ...
}
```

### Step 3: Integrate with Compute Loop

**File:** `src/llama-model.cpp` (compute loop)

Pseudo-code:

```cpp
// For each layer in forward pass:
for (int layer = 0; layer < n_layers; ++layer) {
    if (layer < CPU_GPU_BOUNDARY) {  // CPU layers
        // Run on CPU_COMPUTE stream
        stream = llama_gpu_exclusive_engine_get_cpu_stream();
        compute_layer_on_stream(layer, stream);

        if (layer == CPU_GPU_BOUNDARY - 1) {  // Last CPU layer
            llama_stream_scheduler_record_event(...);
        }
    } else {  // GPU layers
        if (layer == CPU_GPU_BOUNDARY) {  // First GPU layer
            llama_stream_scheduler_wait_event(...);  // Wait for CPU done
        }

        // Run on GPU_COMPUTE stream
        stream = llama_gpu_exclusive_engine_get_gpu_stream();
        compute_layer_on_stream(layer, stream);
    }
}
```

### Step 4: Add Validation & Measurement

**New file:** `src/llama-pipeline-validator.h/cpp`

```cpp
struct llama_pipeline_validation {
    bool outputs_match;           // Byte-for-byte match with baseline
    double baseline_tokens_sec;   // Single-stream throughput
    double pipelined_tokens_sec;  // Multi-stream throughput
    double improvement_percent;   // % improvement
    uint64_t gpu_stalls;          // Times GPU waited for CPU
    uint64_t cpu_stalls;          // Times CPU waited for GPU
};

llama_pipeline_validation llama_validate_pipeline(
    llama_context * ctx,
    const char * baseline_log,    // Baseline output for comparison
    int num_tokens_to_generate);
```

---

## Testing Strategy

### Test 1: Output Correctness
```bash
# Generate 100 tokens with baseline (Phase 2.2)
# Generate 100 tokens with Phase 2.3
# Compare outputs byte-for-byte (must match exactly)
```

### Test 2: Performance Measurement
```bash
# Measure tokens/sec (target: +15-25% improvement)
# Monitor GPU utilization (target: 50-70%, vs 6% before)
# Monitor CPU utilization (target: 70-80%, vs 50% before)
# Check for deadlocks/hangs (timeout after 60s)
```

### Test 3: Stress Test
```bash
# Generate 1000+ tokens continuously
# Monitor for memory leaks
# Check for GPU error recovery
# Validate no sync deadlocks
```

### Test 4: Edge Cases
```bash
# Single token (token 0)
# Last token in sequence
# Rapid token generation
# Large batch sizes
```

---

## Integration Checklist

- [ ] CUDA event creation in scheduler init
- [ ] CUDA event cleanup in scheduler cleanup
- [ ] Event recording in decode_step (after CPU compute)
- [ ] Event waiting in decode_step (before GPU compute)
- [ ] Compute loop stream awareness (use get_cpu_stream/get_gpu_stream)
- [ ] Output correctness validation (byte-for-byte match)
- [ ] Performance measurement infrastructure
- [ ] Deadlock detection (timeout guards)
- [ ] Memory leak testing
- [ ] Comprehensive CL description
- [ ] Git commit

---

## Known Risks & Mitigations

| Risk | Mitigation |
|---|---|
| GPU-CPU synchronization deadlock | Timeout guards on all waits (5-10s) |
| Data corruption from race conditions | Validate outputs match baseline exactly |
| Memory leaks from events | Proper cleanup in llama_stream_scheduler_cleanup() |
| Event handle exhaustion | Reuse 4 events in rotating buffer (token % 4) |
| GPU stall if CPU slower | Monitor metrics, can disable scheduler |
| Backward compatibility | Scheduler optional, fallback to single-stream |

---

## Expected Outcomes

### Before Phase 2.3:
```
Tokens/sec: 6.67
GPU util: 6%
CPU util: 50%
GPU stalls: ~150ms/token
```

### After Phase 2.3:
```
Tokens/sec: 8-10 (+15-25%)
GPU util: 50-70%
CPU util: 70-80%
GPU stalls: ~0-10ms/token
```

---

## File Changes Summary

| File | Changes | Lines |
|---|---|---|
| `llama-stream-scheduler.cpp` | Add CUDA event handling | +50-80 |
| `llama-stream-scheduler.h` | Add event creation/cleanup | +10-15 |
| `llama-gpu-exclusive-decode-engine.cpp` | Add sync hooks | +30-50 |
| `llama-model.cpp` | Use scheduler streams | +20-40 |
| `llama-pipeline-validator.h/cpp` | Add validation | +100-150 |
| Total | - | ~250-350 |

---

## Success Criteria

✅ **Must have:**
- Output matches baseline exactly (byte-for-byte)
- Compiles without errors/warnings
- No deadlocks (timeout guard prevents hanging)
- Throughput ≥ 8 tokens/sec (20% minimum improvement)

✅ **Should have:**
- GPU utilization 50%+ (vs 6% before)
- Comprehensive validation passing
- Performance metrics documented

✅ **Nice to have:**
- Zero GPU stalls in ideal case
- Memory usage optimized
- Backward compatibility maintained

---

## Next Steps

1. Implement CUDA event handling in scheduler
2. Add synchronization hooks to decode engine
3. Modify compute loop for stream awareness
4. Create validation infrastructure
5. Test & validate
6. Performance measurement
7. Comprehensive git commit with detailed CL

