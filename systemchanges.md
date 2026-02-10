## 1. Objective Definition

### 1.1 Primary Objective

* Reduce CPU utilization during the token-by-token decode phase
* Increase sustained GPU utilization during decode
* Maximize tokens-per-second (t/s) under strict correctness constraints

### 1.2 Scope of Change

* Allowed:

  * Rebuild `llama.cpp`
  * Modify CUDA backend behavior
  * Modify graph execution, scheduling, and kernel fusion
  * Move eligible decode-phase work from CPU to GPU
* Not allowed:

  * Changing model architecture
  * Changing model weights
  * Changing prompt or external API behavior
  * Speculative decoding
  * Relaxing autoregressive semantics

### 1.3 Correctness Constraints

* Deterministic execution
* Exact autoregressive token dependency
* Bitwise-stable results for identical inputs (within backend-defined FP tolerance)
* Correct under worst-case execution ordering

### 1.4 Target Execution Mode

* Single active sequence
* Interactive / long-running session
* Decode dominated workload (not prefill)
* Research-grade reproducibility preferred over heuristic speedups

### 1.5 Success Criteria

* CPU usage significantly reduced during decode (no longer pegged at 100%)
* GPU utilization substantially increased during decode (approaching saturation relative to workload)
* Higher sustained decode throughput without violating constraints
* No regressions in correctness, determinism, or stability
## 2. Hardware & Runtime Context (HW-Specific)

### 2.1 CPU Characteristics

* CPU: x86_64 desktop-class processor
* Cores / Threads: 12 hardware threads available to llama.cpp
* SIMD support enabled:

  * SSE3
  * SSSE3
  * AVX
  * AVX2
  * F16C
  * FMA
  * BMI2
* OpenMP enabled
* Current observed behavior:

  * CPU reaches ~100% utilization during decode
  * CPU time dominated by:

    * Decode loop orchestration
    * Sampling chain execution
    * CUDA kernel dispatch and synchronization
    * Server-side control flow

### 2.2 GPU Characteristics

* GPU: NVIDIA GeForce RTX 4060 Ti
* Architecture: Ada Lovelace
* Compute Capability: 8.9
* VRAM: 16 GiB
* Features available and enabled:

  * Tensor Cores
  * CUDA Graphs
  * Flash Attention
  * MMQ quantized matmul kernels
* Observed behavior:

  * Prefill phase: GPU ~100% utilized
  * Decode phase: GPU utilization drops sharply due to host pacing

### 2.3 Memory Topology

* Discrete memory architecture:

  * CPU DRAM
  * GPU VRAM (PCIe-connected)
* No unified memory usage
* Model weights partially offloaded to GPU:

  * ~34 transformer layers resident in VRAM
* Remaining layers and buffers resident in host memory
* KV cache split:

  * GPU KV cache for offloaded layers
  * CPU KV cache for remaining layers
* VRAM pressure sources:

  * Quantized weights
  * KV cache (long context)
  * CUDA compute buffers

### 2.4 Software Environment

* Operating System: Linux (Debian/Ubuntu class)
* NVIDIA proprietary driver installed
* CUDA runtime available and functional
* llama.cpp built with:

  * CUDA enabled
  * MMQ backend enabled
  * Flash attention enabled
  * CUDA graphs enabled
  * OpenMP enabled

### 2.5 Runtime Execution Mode

* Binary: `llama-server`
* Single active sequence (`n_seq_max = 1`)
* Context size: 8192
* Batch size during prefill: >1
* Batch size during decode: effectively 1 token
* Long-running process:

  * Model loaded once
  * Reused across requests
* Decode-dominated steady state

### 2.6 Threading and Scheduling

* Total threads available to llama.cpp: 12
* Thread roles:

  * ggml worker threads
  * CUDA dispatch and synchronization
  * HTTP server threads
* CPU threads are responsible for:

  * Per-token decode scheduling
  * Sampling chain execution
  * Graph execution coordination
  * Kernel launch management
* Current consequence:

  * CPU is the pacing resource
  * GPU frequently idle between short kernel executions

### 2.7 Key Constraint Implied by This Hardware

* GPU has substantial unused compute headroom during decode
* CPU overhead, not GPU math, is the limiting factor
* PCIe latency and kernel launch overhead are significant at batch = 1
* Improving performance requires:

  * Reducing CPU involvement per token
  * Increasing GPU kernel residency and work per launch
## 3. Model Characteristics (Build-Aware, Non-Model-Specific)

### 3.1 Model Class Assumptions

* Model type: decoder-only transformer
* Format: GGUF
* Supports:

  * Dense transformer layers
  * Optional MoE layers (if present in model)
* Quantization: any GGUF-supported quantization (Q4–Q8, K-variants, IQ, etc.)
* Parameter count: arbitrary
* Autoregressive, causal, token-by-token decode
* No architectural modification allowed

These assumptions must hold **identically** across all builds listed.

---

### 3.2 Common Decode Semantics (Invariant Across Builds)

For every generated token:

* One full forward pass through all active transformer layers
* Strict dependency:

  * tokenₙ₊₁ depends on tokenₙ
* No token-level parallelism
* No speculative execution
* No semantic reordering

This invariant must be preserved in **all build variants**.

---

### 3.3 Build Variants and Their Model Interaction

#### 3.3.1 `build_cpu_hybrid`

* Model layers may execute on:

  * CPU only
  * CPU + partial GPU offload
* GPU used opportunistically
* CPU responsible for:

  * Scheduling
  * Sampling
  * Most control flow
* Decode phase dominated by CPU
* GPU utilization limited by host pacing

Implication:

* Model math is split, increasing synchronization cost
* GPU underutilization expected during decode

---

#### 3.3.2 `build_cuda_cublas_dense`

* Dense layers executed using cuBLAS
* Quantized or dequantized matmul depending on configuration
* GPU executes:

  * GEMM / GEMV via cuBLAS
* CPU executes:

  * Sampling
  * Graph orchestration
  * Kernel dispatch
* cuBLAS kernels are:

  * Highly optimized
  * Short-lived at batch = 1

Implication:

* Excellent prefill performance
* Decode phase limited by kernel launch latency and CPU scheduling

---

#### 3.3.3 `build_cuda_dense`

* Dense CUDA kernels without cuBLAS
* Uses custom CUDA matmul kernels
* GPU executes:

  * Dense linear layers
* CPU executes:

  * Sampling
  * Control logic
  * Scheduling

Implication:

* Lower kernel launch overhead than cuBLAS in some cases
* Still CPU-paced during decode
* GPU underutilized at batch = 1

---

#### 3.3.4 `build_cuda_mmq_moe`

* Quantized MMQ kernels enabled
* Supports:

  * Quantized dense layers
  * Quantized MoE layers (if model includes MoE)
* GPU executes:

  * Fused quantized matmul kernels
  * Flash attention (if enabled)
* CPU executes:

  * Sampling chain
  * Decode loop control
  * Kernel scheduling

Implication:

* Best arithmetic efficiency per kernel
* Still limited by per-token CPU orchestration
* GPU idle gaps between kernels during decode

---

#### 3.3.5 `build_cuda_hybrid`

* Mixed execution:

  * Some layers on GPU
  * Some layers on CPU
* GPU executes compute-heavy layers
* CPU executes remaining layers and control logic

Implication:

* Increased synchronization overhead
* Decode phase strongly CPU-bound
* GPU frequently waits for CPU-side layers

---

#### 3.3.6 `build_cpu_cuda_hybrid`

* Explicit hybrid pipeline:

  * CPU and GPU alternate work
* CPU performs:

  * Sampling
  * Non-offloaded layers
* GPU performs:

  * Offloaded layers only

Implication:

* Maximum flexibility
* Worst-case decode utilization for GPU
* CPU becomes strict pacing bottleneck

---

### 3.4 KV Cache Behavior (Across All Builds)

* KV cache grows linearly with generated tokens
* Access pattern:

  * Sequential append
  * Read-heavy during attention
* Location depends on build:

  * GPU KV cache for GPU-resident layers
  * CPU KV cache for CPU-resident layers
* Split KV cache introduces:

  * Synchronization overhead
  * Additional latency

---

### 3.5 Sampling Behavior (Across All Builds)

* Sampling is logically model-independent
* Sampling pipeline:

  * Logits → transformations → token selection
* Sampling is:

  * Control-heavy
  * Low arithmetic intensity
* Sampling is currently CPU-resident in all builds

Implication:

* Sampling is a dominant CPU cost during decode
* Sampling latency directly stalls GPU progress

---

### 3.6 Decode-Phase Cost Structure (Model-Agnostic)

At batch size = 1:

* GPU kernels are:

  * Small
  * Short-lived
  * Frequent
* CPU responsibilities are:

  * Serial
  * Latency-critical
* Overall throughput limited by:

  * Kernel launch overhead
  * CPU scheduling
  * CPU sampling

This behavior is **independent of model size or architecture**.

---

### 3.7 Performance Implication Across Builds

* Changing build type changes **where math runs**
* It does **not** remove:

  * Autoregressive dependency
  * CPU-paced decode loop
* All builds share the same fundamental limitation:

  * GPU work per token is too small relative to CPU overhead

---

### 3.8 Non-Negotiable Constraints Imposed by Model + Builds

* Exact autoregressive semantics must be preserved
* Token order must not change
* No batching across tokens
* No speculative decode
* No approximation of attention or sampling

Only **execution restructuring and control-path changes** are permitted.

---
## 3.9 Build Type vs Model Type — Throughput Characteristics

### 3.9.1 `build_cpu_hybrid`

**Best suited for:**

* Very small models
* Models that do not fit in GPU memory
* CPU-heavy or experimentation workloads

**t/s characteristics:**

* Lowest tokens/sec for medium and large models
* Decode phase fully CPU-bound
* GPU provides marginal acceleration only

**Conclusion:**

* Never optimal for maximum t/s
* Should be excluded when GPU is available and model fits partially or fully

---

### 3.9.2 `build_cuda_cublas_dense`

**Best suited for:**

* Dense, non-quantized or lightly quantized models
* Medium-to-large dense models with strong GEMM utilization
* Prefill-heavy workloads

**t/s characteristics:**

* Very high prefill throughput
* Decode phase limited by:

  * cuBLAS kernel launch overhead
  * Small GEMV workload at batch = 1
* GPU underutilized during decode despite fast kernels

**Conclusion:**

* Excellent for short prompts + long prefills
* Suboptimal for long decode workloads
* Not ideal when decode t/s is the primary metric

---

### 3.9.3 `build_cuda_dense`

**Best suited for:**

* Dense models using custom CUDA kernels
* Situations where cuBLAS overhead dominates

**t/s characteristics:**

* Slightly better decode t/s than cuBLAS in some cases
* Lower kernel launch overhead
* Still CPU-paced decode loop

**Conclusion:**

* Marginally better than cuBLAS for decode
* Still not the best choice for maximum t/s

---

### 3.9.4 `build_cuda_mmq_moe`

**Best suited for:**

* Quantized models (Q4–Q8, K-variants)
* Large dense models
* MoE models (if present)
* Long-context decode workloads

**t/s characteristics:**

* Highest arithmetic efficiency per kernel
* Best decode-phase GPU utilization among current builds
* Reduced memory bandwidth pressure
* Still limited by CPU-driven decode orchestration

**Conclusion:**

* **Best existing build for maximum decode t/s**
* Preferred choice for large quantized models
* Baseline for all further optimization work

---

### 3.9.5 `build_cuda_hybrid`

**Best suited for:**

* Models that partially fit in VRAM
* Mixed CPU/GPU execution required by memory limits

**t/s characteristics:**

* Decode throughput heavily impacted by CPU↔GPU synchronization
* GPU frequently stalled waiting for CPU layers
* t/s lower than full-GPU builds

**Conclusion:**

* Necessary only under VRAM constraints
* Never optimal for maximum t/s

---

### 3.9.6 `build_cpu_cuda_hybrid`

**Best suited for:**

* Experimental setups
* Memory-constrained environments

**t/s characteristics:**

* Alternating CPU/GPU execution per layer
* High synchronization overhead
* Lowest effective GPU utilization

**Conclusion:**

* Worst choice for decode throughput
* Should be avoided when optimizing for t/s

---

### 3.9.7 Summary Table

| Model Type            | Preferred Build           | Reason                                  |
| --------------------- | ------------------------- | --------------------------------------- |
| Small dense model     | `build_cuda_dense`        | Lower overhead than cuBLAS              |
| Medium dense model    | `build_cuda_cublas_dense` | Strong GEMM utilization                 |
| Large dense model     | `build_cuda_mmq_moe`      | Quantized efficiency, fewer bottlenecks |
| Large quantized model | `build_cuda_mmq_moe`      | Best decode t/s                         |
| MoE model             | `build_cuda_mmq_moe`      | Native MoE + MMQ support                |
| Partial VRAM fit      | `build_cuda_hybrid`       | Required by memory                      |
| CPU-only fallback     | `build_cpu_hybrid`        | Last resort                             |

---

### 3.9.8 Final Throughput Ranking (Decode Phase)

From highest to lowest tokens/sec:

1. `build_cuda_mmq_moe`
2. `build_cuda_dense`
3. `build_cuda_cublas_dense`
4. `build_cuda_hybrid`
5. `build_cpu_cuda_hybrid`
6. `build_cpu_hybrid`

This ranking holds **independent of model family**, assuming similar size and quantization.
## 4. Execution Mode Clarification

### 4.1 Request Pattern

* Single active request at any given time
* No concurrent user requests
* No request batching across users
* No background jobs
* Execution is strictly sequential at the request level

### 4.2 Sequence Characteristics

* Single sequence generation (`n_seq_max = 1`)
* One token generated per decode step
* Strict autoregressive dependency:

  * Token *n+1* cannot be computed before token *n* is finalized
* No parallel decoding across sequences

### 4.3 Interaction Style

* Interactive or long-running session
* Prompt provided once, followed by long decode phase
* Decode phase dominates total runtime
* Streaming output may be enabled, but streaming semantics must not affect decode execution

### 4.4 Server vs CLI Execution

* Execution may occur via:

  * `llama-cli`
  * `llama-server`
* Server mode characteristics:

  * HTTP request handling
  * Slot management
  * Request lifecycle management
* CLI mode characteristics:

  * Minimal control flow
  * Fewer synchronization points
* Optimizations must apply to both modes or clearly specify mode-specific behavior

### 4.5 Batching Behavior

* Prefill phase:

  * Batch size > 1 allowed
  * High GPU utilization expected
* Decode phase:

  * Effective batch size = 1 token
  * No token batching allowed
* Micro-batching across tokens is not permitted

### 4.6 Sampling Mode

* Sampling may be:

  * Deterministic (greedy, `temp = 0`)
  * Stochastic (top-k, top-p, temperature)
* Sampling must preserve:

  * Exact semantics
  * Determinism when configured
* Sampling decisions currently gate progression to the next token

### 4.7 Correctness and Ordering Guarantees

* Token emission order must be preserved
* No reordering of compute relative to token output
* No speculative or rollback-based execution
* Each token must be fully committed before the next decode step begins

### 4.8 Termination Conditions

* Decode loop terminates when:

  * End-of-sequence token is generated
  * Maximum token limit is reached
* Termination checks must be exact and deterministic
* No early stopping heuristics allowed

### 4.9 Implication for Optimization

* Execution is latency-serial by definition
* GPU starvation during decode is caused by:

  * Host-driven orchestration
  * Fine-grained synchronization
* Any optimization must:

  * Reduce host involvement per token
  * Increase GPU work per decode step
  * Preserve strict execution order
## 5. High-Level Decode Pipeline Mapping

### 5.1 Decode Entry Point

* Decode begins after prompt prefill is completed
* Control enters the decode loop from:

  * `llama_decode()` (CLI)
  * Server-side task loop (`llama-server`)
* Decode loop executes once per generated token

### 5.2 Per-Token Decode Lifecycle (Logical)

For each token generation step, the following stages occur in strict order:

1. Input token embedding lookup
2. Forward pass through all transformer layers
3. Logits computation
4. Sampling / token selection
5. KV cache update
6. Token commit and output
7. Termination check

Each stage must complete before the next begins.

### 5.3 Transformer Forward Pass

* Executed layer-by-layer
* For each layer:

  * Normalization
  * Attention computation using KV cache
  * Feed-forward network
* Execution backend depends on build:

  * CPU kernels
  * CUDA dense kernels
  * cuBLAS kernels
  * MMQ quantized kernels
* GPU kernels are launched per layer or per fused group

### 5.4 Graph Construction and Execution

* ggml graph represents the computation for one token
* Graph may be:

  * Rebuilt
  * Partially reused
  * Fully reused (if CUDA graphs enabled and valid)
* CPU is responsible for:

  * Graph scheduling
  * Node execution order
  * Dispatching GPU kernels
* GPU executes only the compute nodes assigned to it

### 5.5 KV Cache Interaction

* KV cache is accessed during attention in each layer
* Read operations:

  * Keys and values for all previous tokens
* Write operations:

  * Append current token’s key and value
* KV cache location depends on layer placement:

  * GPU KV cache for GPU layers
  * CPU KV cache for CPU layers
* KV cache updates are serialized per token

### 5.6 Sampling Stage

* Logits are produced as the final output of the forward pass
* Sampling pipeline executes:

  * Logit post-processing
  * Probability filtering (if enabled)
  * Token selection
* Sampling is currently CPU-resident
* Sampling completion is a hard dependency for next token decode

### 5.7 Output and State Update

* Selected token is:

  * Added to output buffer
  * Used to update internal state
* Context position is incremented
* Sequence state is updated
* Any streaming output is emitted

### 5.8 Synchronization Points

* CPU waits for GPU kernel completion before sampling
* CPU waits for sampling to complete before next decode step
* CPU waits for KV cache updates before proceeding
* These synchronization points occur once per token

### 5.9 Loop Continuation

* Decode loop repeats until:

  * End-of-sequence token generated
  * Maximum token count reached
* No overlap between iterations
* Entire pipeline is strictly serial at token granularity

### 5.10 Key Observation from Pipeline

* GPU performs only compute kernels
* CPU controls:

  * Loop progression
  * Scheduling
  * Sampling
  * Synchronization
* GPU idle time during decode is caused by:

  * Fine-grained kernel launches
  * CPU-bound stages between kernels

This mapping defines where CPU–GPU imbalance originates and where restructuring must occur.
## 6. CPU Responsibility Audit

### 6.1 Decode Loop Control

* CPU owns the outer decode loop
* CPU determines:

  * When a new token decode begins
  * When the next decode step may proceed
* CPU enforces strict token-by-token sequencing
* CPU blocks progression until all dependent stages complete

### 6.2 Graph Scheduling and Execution

* CPU constructs or validates the ggml computation graph
* CPU schedules graph nodes for execution
* CPU determines execution order of:

  * CPU nodes
  * GPU nodes
* CPU dispatches GPU kernels node-by-node
* CPU tracks completion of each node

### 6.3 CUDA Kernel Dispatch

* CPU launches all CUDA kernels
* CPU incurs:

  * Kernel launch latency
  * Stream synchronization overhead
* CPU waits for GPU completion at defined sync points
* Kernel dispatch occurs multiple times per token

### 6.4 Sampling and Token Selection

* CPU executes the full sampling pipeline:

  * Logit post-processing
  * Penalties (if enabled)
  * Top-k / top-p filtering (if enabled)
  * Temperature scaling
  * Final token selection
* Sampling is:

  * Control-heavy
  * Branch-heavy
  * Latency-sensitive
* Sampling completion gates the next decode step

### 6.5 KV Cache Management

* CPU manages:

  * KV cache metadata
  * Sequence positions
  * Cache boundaries
* For CPU-resident layers:

  * CPU performs KV writes
* For GPU-resident layers:

  * CPU coordinates KV updates and synchronization
* CPU ensures KV consistency across layers

### 6.6 Synchronization and Barriers

* CPU inserts synchronization points:

  * Before sampling
  * After GPU kernel execution
  * Before next decode iteration
* CPU performs:

  * Blocking waits
  * Polling
* These barriers serialize execution at token granularity

### 6.7 Thread Pool Management

* CPU manages ggml worker threads
* CPU schedules work across threads
* CPU handles thread wake-ups and sleeps
* Thread management overhead increases at batch size = 1

### 6.8 Server-Side Control (if applicable)

* CPU handles:

  * HTTP request parsing
  * Slot selection
  * Request lifecycle management
  * Logging and metrics
* Server-side logic executes concurrently with decode
* Server overhead competes with decode for CPU time

### 6.9 Memory Management

* CPU allocates and frees:

  * Temporary buffers
  * Host-side compute buffers
* CPU manages:

  * Host-device memory mappings
  * Pinned memory regions
* Allocation and bookkeeping occur during decode

### 6.10 Aggregate Impact on Performance

* CPU performs multiple latency-critical tasks per token
* CPU responsibilities are strictly serialized
* CPU overhead directly determines:

  * Decode latency per token
  * GPU idle time between kernels
* CPU becomes the pacing resource for the entire pipeline

This audit identifies CPU responsibilities that must be reduced, eliminated, or offloaded to improve decode-phase GPU utilization and throughput.
## 7. GPU Responsibility Audit

### 7.1 Core Compute Responsibilities

* GPU executes numerical computation for transformer layers assigned to it
* Primary GPU workloads include:

  * Linear projections (Q, K, V, output)
  * Attention score computation
  * Softmax over attention scores
  * Attention-weighted value accumulation
  * Feed-forward network (MLP) layers
* GPU kernels are invoked once per layer or per fused layer group per token

### 7.2 Backend-Specific Compute Paths

* Depending on build configuration, GPU executes one of:

  * Custom CUDA dense kernels
  * cuBLAS GEMM / GEMV kernels
  * MMQ quantized matmul kernels
* Backend selection determines:

  * Kernel shape
  * Kernel launch count
  * Arithmetic intensity
* All GPU kernels are launched by the CPU

### 7.3 Flash Attention Execution

* When enabled, GPU executes flash-attention kernels
* Flash-attention reduces:

  * Memory traffic
  * Intermediate buffer usage
* Flash-attention kernels operate on:

  * Query for current token
  * Full KV cache up to current position
* Kernel execution time increases with context length

### 7.4 KV Cache Operations

* GPU performs:

  * Reads from KV cache during attention
  * Writes of new key and value vectors for current token
* KV cache memory resides in GPU VRAM for GPU-resident layers
* KV cache updates are serialized per token

### 7.5 Quantization and Dequantization

* For quantized models:

  * GPU kernels perform on-the-fly dequantization
* Dequantization is fused with matmul where possible
* Quantization reduces memory bandwidth but does not reduce kernel launch count

### 7.6 Kernel Launch Granularity

* GPU kernels during decode are:

  * Small
  * Short-lived
  * Launched frequently
* Kernel execution time per launch is often much shorter than:

  * CPU kernel launch overhead
  * CPU synchronization latency

### 7.7 Synchronization Behavior

* GPU execution is synchronized with CPU at:

  * End of graph execution
  * Before sampling
* GPU cannot proceed independently
* GPU frequently idle while waiting for CPU to:

  * Launch next kernel
  * Complete sampling
  * Update control state

### 7.8 GPU Utilization Characteristics

* During prefill:

  * Large kernels
  * High occupancy
  * GPU near saturation
* During decode:

  * Low occupancy
  * Frequent idle gaps
  * Utilization limited by host pacing, not compute capacity

### 7.9 Limitations of Current GPU Role

* GPU does not control:

  * Decode loop progression
  * Token selection
  * Sampling
* GPU cannot overlap work across tokens
* GPU has no persistent execution context across decode iterations

### 7.10 Aggregate Impact on Performance

* GPU is capable of much higher sustained throughput
* GPU utilization during decode is artificially constrained
* GPU underutilization is caused by:

  * Fine-grained kernel launches
  * CPU-driven control flow
  * Lack of persistent GPU execution

This audit shows that the GPU is underused not due to insufficient compute work, but due to execution structure and host-driven orchestration.
## 8. CPU↔GPU Synchronization Points

### 8.1 Decode-Step Boundary Synchronization

* At the end of each token decode step, CPU waits for all GPU kernels to complete
* No overlap is allowed between:

  * GPU execution for token *n*
  * CPU sampling and control for token *n+1*
* This synchronization occurs once per generated token

### 8.2 Graph Execution Synchronization

* ggml graph execution introduces implicit synchronization:

  * CPU blocks until all GPU nodes in the graph complete
* Even when CUDA graphs are enabled:

  * CPU still enforces completion before sampling
* Graph-level synchronization serializes GPU work at token granularity

### 8.3 Kernel Launch Synchronization

* Each CUDA kernel launch incurs:

  * Host-side launch overhead
  * Implicit ordering within the CUDA stream
* CPU often waits for:

  * Kernel completion
  * Stream state update
* Kernel launches are frequent and fine-grained during decode

### 8.4 Sampling Dependency Barrier

* Sampling cannot begin until:

  * All logits are fully computed on GPU
* CPU waits for GPU to finish logits computation
* This barrier prevents overlap between:

  * GPU compute
  * CPU sampling
* Sampling completion blocks the next GPU launch

### 8.5 KV Cache Consistency Barrier

* KV cache updates must be completed before:

  * Next token decode begins
* CPU ensures:

  * GPU KV writes are visible
  * CPU KV metadata is updated
* This introduces another per-token synchronization point

### 8.6 Memory Visibility Synchronization

* Host access to GPU-resident data (e.g., logits) requires:

  * Explicit or implicit synchronization
* Device-to-host transfers introduce:

  * Blocking waits
  * Pipeline stalls
* Even small transfers cause decode-phase delays

### 8.7 Server-Side Synchronization (if applicable)

* In server mode:

  * CPU may synchronize on request lifecycle events
  * Slot state transitions are synchronized with decode progression
* These synchronizations are serialized with decode steps

### 8.8 CUDA Graph Constraints

* CUDA graphs reduce kernel launch overhead
* However:

  * Graph replay is still initiated by CPU
  * Graph boundaries enforce synchronization
* Graph invalidation (e.g., context growth) forces CPU intervention

### 8.9 Cumulative Effect of Synchronization

* Multiple synchronization points exist per token
* Synchronization overhead dominates compute time at batch size = 1
* GPU frequently idle while CPU:

  * Waits
  * Samples
  * Updates state

### 8.10 Optimization Implication

* Reducing or eliminating CPU↔GPU synchronization points is critical
* Key targets:

  * Sampling barrier
  * Graph-level barrier
  * Per-kernel launch waits
* GPU utilization cannot increase without restructuring these synchronization points
## 9. Backend Selection Logic

### 9.1 Backend Selection Overview

* Backend selection determines **where** and **how** tensor operations are executed
* Selection occurs at:

  * Build time (compiled backends)
  * Runtime (capability checks, environment variables, flags)
* Backend choice directly impacts:

  * Kernel launch count
  * Arithmetic intensity
  * CPU↔GPU synchronization frequency
  * Decode-phase throughput

### 9.2 Available Backends

* CPU backend
* CUDA dense backend
* CUDA cuBLAS dense backend
* CUDA MMQ backend (quantized, MoE-capable)
* Hybrid CPU↔CUDA backends

Each backend implements the same logical ops but with different execution characteristics.

### 9.3 Build-Time Backend Availability

* Compiled backends are determined by:

  * CMake configuration
  * CUDA availability
  * Architecture flags
* Only compiled backends are candidates at runtime
* Missing backends force fallback to available alternatives

### 9.4 Runtime Backend Selection Criteria

Backend selection at runtime depends on:

* Tensor location (CPU memory vs GPU memory)
* Tensor datatype and quantization
* Operation type (matmul, attention, normalization, etc.)
* GPU capabilities (compute capability, tensor core support)
* Environment variables and runtime flags

### 9.5 CPU Backend Selection

* Selected when:

  * Operation has no GPU implementation
  * Tensor resides in CPU memory
  * GPU memory is insufficient
  * Explicit CPU-only execution requested
* CPU backend introduces:

  * Additional synchronization
  * Reduced GPU utilization
* Any CPU backend invocation during decode creates a hard pacing bottleneck

### 9.6 CUDA Dense Backend Selection

* Selected when:

  * Tensors are dense and GPU-resident
  * Quantization does not apply
* Uses custom CUDA kernels
* Lower launch overhead than cuBLAS in some cases
* Still launch-heavy at batch size = 1

### 9.7 CUDA cuBLAS Dense Backend Selection

* Selected when:

  * Dense GEMM/GEMV operations are detected
  * cuBLAS is enabled
* cuBLAS provides highly optimized kernels
* At batch size = 1:

  * GEMV-dominated
  * Kernel launch overhead becomes significant
* cuBLAS backend excels during prefill, not decode

### 9.8 CUDA MMQ Backend Selection

* Selected when:

  * Model weights are quantized
  * MMQ kernels support the quantization format
* MMQ backend provides:

  * Fused quantized matmul
  * Reduced memory bandwidth
* MMQ is preferred for:

  * Large quantized models
  * Decode-heavy workloads
* MMQ still depends on CPU-driven scheduling

### 9.9 Hybrid Backend Selection

* Selected when:

  * Some layers fit on GPU
  * Others must run on CPU
* Introduces:

  * CPU↔GPU alternation per layer
  * Increased synchronization
* Hybrid execution significantly degrades decode throughput

### 9.10 Backend Fallback Behavior

* If a preferred backend is unavailable or fails:

  * Execution falls back to the next available backend
* Fallbacks may occur silently
* Silent fallback to CPU backend during decode is catastrophic for throughput

### 9.11 Environment Variable Influence

* Environment variables can force backend selection:

  * Forcing MMQ
  * Forcing cuBLAS
  * Disabling certain backends
* Incorrect configuration can:

  * Increase CPU execution
  * Reduce GPU residency
* Backend forcing must be verified at runtime logs

### 9.12 Decode-Phase Implications

* Backend selection is evaluated repeatedly during decode
* Backend switching increases overhead
* Optimal decode performance requires:

  * Stable backend selection
  * Maximum GPU-resident execution
  * Zero CPU fallback during decode

### 9.13 Optimization Implication

* Backend logic must be:

  * Predictable
  * Static across decode
* For maximum tokens/sec:

  * All decode-path operations must map to GPU backends
  * CPU backend invocation must be eliminated or isolated

This section defines how backend selection directly controls decode-phase performance and where intervention is required to prevent GPU underutilization.
## 10. Threading & Parallelism Analysis

### 10.1 CPU Thread Model Overview

* llama.cpp uses a CPU thread pool managed by ggml
* Thread count controlled by:

  * `--threads`
  * `--threads-batch`
* Threads are shared across:

  * Graph execution
  * CPU backend ops
  * CUDA kernel dispatch
  * Sampling
  * Server logic (if applicable)

### 10.2 Thread Roles During Decode

During decode (batch size = 1), CPU threads perform:

* Decode loop control
* Graph scheduling and traversal
* CUDA kernel launch and synchronization
* Sampling pipeline execution
* KV cache metadata updates
* Server-side request handling (in server mode)

These tasks are **latency-serial**, not throughput-parallel.

### 10.3 Effective Parallelism in Decode Phase

* Decode phase exposes very little exploitable parallelism:

  * One token at a time
  * Strict ordering constraints
* CPU threads cannot work independently on future tokens
* Most threads are either:

  * Idle
  * Spinning
  * Waiting on synchronization

High CPU utilization does **not** imply useful parallel work.

### 10.4 Oversubscription Effects

* Using too many CPU threads can:

  * Increase context switching
  * Increase cache thrashing
  * Increase synchronization overhead
* Oversubscription can **reduce** effective decode throughput
* Optimal thread count is often:

  * Much lower than available cores
  * Close to number of active CPU backend tasks

### 10.5 Interaction with CUDA Dispatch

* CUDA kernel launches are serialized per stream
* Multiple CPU threads do not increase GPU kernel concurrency
* Excess threads increase:

  * Lock contention
  * Dispatch overhead
* CUDA graphs reduce launch overhead but not CPU control flow

### 10.6 Sampling and Thread Utilization

* Sampling is:

  * Branch-heavy
  * Poorly vectorizable
* Sampling executes on a single CPU thread
* Additional threads provide no benefit during sampling
* Sampling time directly stalls the entire decode pipeline

### 10.7 Server Mode Threading

* Server introduces additional threads for:

  * HTTP handling
  * Slot management
  * Logging
* These threads compete with decode threads for CPU time
* Server threading increases scheduling noise during decode

### 10.8 GPU Parallelism vs CPU Parallelism

* GPU parallelism is massive but underutilized during decode
* CPU parallelism cannot compensate due to serial dependencies
* Mismatch between CPU threading model and GPU execution model causes inefficiency

### 10.9 Thread Affinity and Scheduling

* Default OS scheduling may:

  * Migrate threads across cores
  * Increase cache misses
* Lack of thread pinning increases jitter
* Jitter increases GPU idle gaps

### 10.10 Optimization Implications

* Increasing CPU threads does not increase decode throughput
* Reducing unnecessary CPU threads can:

  * Reduce overhead
  * Improve GPU feed consistency
* Maximum throughput requires:

  * Minimal CPU thread count
  * Minimal CPU-side synchronization
  * GPU-resident execution where possible

This analysis shows that decode performance is limited by **serial control flow**, not lack of CPU threads, and that excess CPU parallelism can actively harm GPU utilization.
## 11. Memory Mapping & Allocation

### 11.1 Memory Allocation Domains

* Two physically separate memory domains:

  * CPU DRAM
  * GPU VRAM
* All data movement between domains occurs explicitly over PCIe
* No implicit unified memory behavior is relied upon

### 11.2 Model Weight Allocation

* Model weights are loaded from GGUF into host memory
* Depending on build and flags:

  * A subset of layers is offloaded to GPU VRAM
  * Remaining layers stay in CPU memory
* Offloading decisions are made at load time
* Weight placement is static during decode

### 11.3 KV Cache Allocation

* KV cache size grows linearly with context length
* KV cache allocation occurs at context initialization
* Allocation split depends on layer placement:

  * GPU-resident KV cache for GPU layers
  * CPU-resident KV cache for CPU layers
* Split KV cache introduces:

  * Additional synchronization
  * Memory access overhead
* KV cache memory is reused across tokens but never relocated

### 11.4 Compute Buffer Allocation

* Temporary compute buffers are allocated for:

  * Intermediate activations
  * Attention outputs
  * FFN outputs
* Buffers may reside in:

  * CPU memory
  * GPU memory
* Allocation strategy depends on backend
* Buffer sizes are fixed per context and reused per token

### 11.5 Memory Mapping Modes

* Memory-mapped file loading (`mmap`) may be enabled or disabled
* When enabled:

  * Model weights are memory-mapped from disk
  * Pages are faulted on demand
* When disabled:

  * Model weights are fully loaded into RAM
* `mmap` affects load time and memory pressure but not decode compute
* For decode performance:

  * `mmap` status is largely irrelevant once weights are resident

### 11.6 Pinned and Pageable Memory

* Host-to-device transfers may use:

  * Pageable memory
  * Pinned (page-locked) memory
* Pageable memory transfers introduce:

  * Additional latency
  * Implicit synchronization
* Pinned memory:

  * Reduces transfer latency
  * Increases host memory pressure
* Decode performance benefits from minimizing transfers regardless of memory type

### 11.7 Allocation Lifetime and Churn

* Most allocations occur during:

  * Model load
  * Context initialization
* Decode phase allocation churn is minimal
* However:

  * Any allocation during decode introduces synchronization
  * Allocation must be avoided in the decode loop

### 11.8 Memory Visibility and Synchronization

* Host access to GPU-resident buffers requires synchronization
* Device writes must complete before host reads
* Memory visibility rules enforce:

  * Implicit barriers
  * Pipeline stalls
* Small device-to-host reads (e.g., logits) are disproportionately expensive

### 11.9 Memory Fragmentation Considerations

* Long-running processes risk:

  * Host memory fragmentation
  * GPU memory fragmentation
* Fragmentation increases allocation cost
* Stable buffer reuse is critical for sustained throughput

### 11.10 Optimization Implications

* Decode throughput is maximized when:

  * All decode-path data remains GPU-resident
  * No per-token host↔device transfers occur
  * No allocations occur during decode
* Any CPU access to GPU data during decode introduces a hard synchronization point
* Memory placement decisions directly impact GPU utilization and tokens/sec
## 12. Graph Lifetime Analysis

### 12.1 Graph Definition

* A ggml graph represents the computation required to produce one token
* The graph includes:

  * All transformer layer operations
  * Attention computation
  * FFN computation
  * Logits computation
* The graph encodes both CPU and GPU nodes

### 12.2 Graph Construction Phase

* Graph construction occurs during:

  * Context initialization
  * Prompt prefill
  * Decode, if graph invalidation conditions are met
* Graph construction is performed on CPU
* Graph construction cost is non-trivial but amortized during prefill

### 12.3 Graph Reuse During Decode

* During decode, the same logical graph structure is reused per token
* However, graph execution is still:

  * Triggered by CPU
  * Synchronized per token
* Graph reuse does not imply autonomous GPU execution

### 12.4 Conditions That Invalidate Graph Reuse

Graph reuse may be invalidated when:

* Context length increases beyond planned bounds
* KV cache layout changes
* Backend selection changes
* Tensor shapes change
* Certain flags or modes are toggled

When invalidated:

* Graph must be rebuilt on CPU
* Decode throughput temporarily degrades

### 12.5 CUDA Graph Integration

* CUDA graphs may capture:

  * Kernel launch sequences
  * Memory access patterns
* CUDA graph replay reduces:

  * Kernel launch overhead
* However:

  * CUDA graph replay is initiated by CPU
  * Graph boundaries still enforce synchronization
* CUDA graphs do not remove CPU control flow

### 12.6 Graph Execution Flow

* For each token:

  * CPU initiates graph execution
  * CPU dispatches GPU kernels according to graph
  * CPU waits for graph completion
* No overlap exists between graph executions of different tokens

### 12.7 Graph Granularity

* Graph is defined at token granularity
* Each token corresponds to one full graph execution
* Fine granularity increases:

  * Synchronization frequency
  * CPU overhead
  * GPU idle gaps

### 12.8 Graph Node Scheduling

* CPU schedules execution of graph nodes
* CPU determines:

  * Node ordering
  * Backend selection per node
* GPU has no visibility into:

  * Upcoming nodes
  * Future tokens

### 12.9 Lifetime of Graph Resources

* Graph-associated buffers:

  * Allocated during context setup
  * Reused across decode iterations
* Resource reuse is effective
* Control-path overhead remains dominant

### 12.10 Optimization Implications

* Graph reuse alone is insufficient to maximize decode throughput
* Fundamental limitation:

  * Graph execution is CPU-driven and token-scoped
* To increase GPU utilization:

  * Graph granularity must be increased
  * GPU must execute multiple decode steps autonomously
* Without restructuring graph lifetime, GPU idle gaps persist
## 13. Attention Path Analysis

### 13.1 Role of Attention in Decode Phase

* Attention is the dominant operation during decode at long context lengths
* For each generated token:

  * Query corresponds to the current token
  * Keys and values correspond to all previous tokens
* Attention cost increases linearly with context length

### 13.2 Attention Execution Stages

For each transformer layer during decode:

1. Query, Key, Value projection
2. Attention score computation:

   * Dot product of query with all keys
3. Scaling and masking (causal mask)
4. Softmax over sequence length
5. Weighted sum over values
6. Output projection

Each stage must complete before proceeding to the next.

### 13.3 Backend Variants for Attention

* CPU backend:

  * Fully CPU-resident
  * Extremely slow for long contexts
* CUDA dense backend:

  * Custom CUDA kernels
  * Multiple kernel launches per layer
* cuBLAS backend:

  * Uses GEMV/GEMM for projections
  * Separate kernels for attention steps
* Flash-attention backend:

  * Fused attention kernels
  * Reduced memory traffic
  * Fewer intermediate buffers

### 13.4 Flash-Attention Enablement

* Flash-attention is enabled when:

  * GPU supports required features
  * Attention dimensions are compatible
  * Flags permit its use
* When enabled:

  * Attention computation is fused into fewer kernels
  * Memory reads/writes are minimized
* Flash-attention is critical for decode performance at long context

### 13.5 KV Cache Interaction

* Attention reads:

  * All previous keys and values from KV cache
* KV cache is:

  * Read-heavy
  * Sequentially extended
* KV cache location affects performance:

  * GPU-resident KV cache enables high bandwidth access
  * CPU-resident KV cache introduces severe latency

### 13.6 Kernel Granularity and Launch Behavior

* Attention kernels during decode are:

  * Small
  * Launched frequently
* Even flash-attention kernels are short-lived at batch size = 1
* Kernel launch overhead becomes significant relative to compute

### 13.7 Synchronization in Attention Path

* Attention kernels must complete before:

  * Sampling
  * Next layer execution
* CPU enforces completion via synchronization
* No overlap is allowed between attention of token *n* and any other work

### 13.8 Scaling Behavior with Context Length

* As context length increases:

  * Attention compute per token increases
  * Kernel execution time increases
* GPU utilization improves somewhat at very long contexts
* CPU overhead remains present and limits scaling

### 13.9 Attention as a Throughput Lever

* Attention is one of the few decode-phase operations with:

  * Substantial GPU work
  * Increasing cost with context length
* Optimizations in attention:

  * Kernel fusion
  * Persistent kernels
  * Reduced synchronization
* Yield direct gains in GPU utilization

### 13.10 Optimization Implications

* Maximum decode throughput requires:

  * Flash-attention enabled and always selected
  * GPU-resident KV cache
  * Elimination of CPU-side attention orchestration
* Attention kernels must be:

  * Long-lived
  * Executed with minimal host intervention
* Without restructuring the attention path, GPU utilization remains artificially capped
## 14. Quantization Cost Analysis

### 14.1 Purpose of Quantization

* Quantization reduces:

  * Model memory footprint
  * Memory bandwidth requirements
* Enables larger models to fit in limited VRAM
* Quantization does **not** change model semantics

### 14.2 Quantization Formats

* Common GGUF quantization formats include:

  * Q4, Q5, Q6, Q8
  * K-variants (Q4_K, Q6_K, etc.)
  * IQ and mixed formats
* Quantization granularity:

  * Block-based
  * Per-channel or per-group scales
* All quantized formats require dequantization during compute

### 14.3 Dequantization Execution Location

* Dequantization may occur:

  * On CPU (for CPU-resident layers)
  * On GPU (inside CUDA kernels)
* GPU-side dequantization is preferred for throughput
* CPU-side dequantization introduces:

  * Additional CPU compute
  * Extra memory traffic
  * Additional synchronization

### 14.4 Dequantization Cost Characteristics

* Dequantization is:

  * Low arithmetic intensity
  * Memory-bound
* Cost per operation is small, but:

  * Occurs frequently
  * Is repeated for each token and each layer
* Dequantization cost becomes significant at batch size = 1

### 14.5 Interaction with GEMV/GEMM

* During decode:

  * GEMV dominates
  * Quantized GEMV kernels perform:

    * Dequantization
    * Multiply–accumulate
* Kernel execution time is short
* Kernel launch overhead becomes a large fraction of total time

### 14.6 Quantization vs Kernel Fusion

* Fused quantized kernels:

  * Combine dequantization and matmul
  * Reduce intermediate memory traffic
* MMQ backend provides:

  * Better fusion
  * Higher arithmetic efficiency
* Non-fused paths increase:

  * Kernel count
  * Synchronization points

### 14.7 Quantization Impact on CPU Load

* Quantization does not reduce:

  * CPU sampling cost
  * CPU scheduling overhead
  * CPU synchronization cost
* Quantization shifts compute to GPU but leaves control on CPU
* CPU remains the decode pacing resource

### 14.8 Quantization Impact on GPU Utilization

* Quantization reduces GPU memory bandwidth pressure
* Quantization reduces compute per kernel
* Reduced compute can:

  * Shorten kernel duration
  * Increase relative launch overhead
* GPU utilization may decrease at small batch sizes despite faster kernels

### 14.9 Trade-Off Summary

* Quantization improves:

  * Model fit
  * Prefill throughput
* Quantization does not inherently improve decode utilization
* For decode:

  * Faster kernels can worsen utilization if launch overhead dominates

### 14.10 Optimization Implications

* Quantization alone cannot fix decode underutilization
* Maximum benefit requires:

  * Fused quantized kernels
  * Reduced kernel launch count
  * Persistent or long-lived GPU execution
* Quantization should be paired with:

  * Execution restructuring
  * Reduced CPU orchestration

Quantization is necessary for scale, but insufficient for maximizing decode-phase GPU utilization without complementary architectural changes.
## 15. Sampling Optimization Scope

### 15.1 Role of Sampling in Decode

* Sampling determines the next token from model logits
* Sampling occurs once per generated token
* Sampling is on the critical path:

  * Decode cannot proceed until sampling completes
* Sampling cost is small in FLOPs but large in latency impact

### 15.2 Current Sampling Pipeline

* Sampling is executed entirely on CPU
* Typical pipeline stages include:

  * Logit bias application
  * Penalties (repeat, frequency, presence)
  * Temperature scaling
  * Top-k filtering
  * Top-p filtering
  * Final token selection
* Even when disabled via flags, control flow remains present

### 15.3 Sampling Cost Characteristics

* Branch-heavy and control-heavy
* Poor cache locality
* Poor SIMD utilization
* Executes on a single CPU thread
* Latency-sensitive rather than throughput-bound

### 15.4 Sampling as a Decode Bottleneck

* Sampling introduces a hard barrier:

  * GPU must finish computing logits
  * CPU must complete sampling
  * Only then can the next decode step begin
* Sampling latency directly creates GPU idle time
* Sampling dominates decode latency when GPU kernels are short

### 15.5 Deterministic vs Stochastic Sampling

* Deterministic (greedy, temp = 0):

  * Sampling reduces to argmax
  * Still incurs control-path overhead
* Stochastic sampling:

  * Adds probability normalization
  * Adds filtering and randomness
  * Increases CPU latency
* Deterministic sampling is preferable for throughput but still suboptimal on CPU

### 15.6 GPU Offloading Potential

* Sampling operations are mathematically simple:

  * Reductions
  * Comparisons
  * Prefix sums
* These operations are well-suited to GPU execution
* GPU-based sampling can:

  * Eliminate CPU sampling latency
  * Remove a major synchronization point

### 15.7 Constraints on Sampling Optimization

* Sampling must preserve:

  * Exact semantics
  * Determinism when configured
  * Correct handling of penalties and filters
* Sampling cannot be speculative
* Sampling result must be final before next token decode

### 15.8 Incremental Optimization Path

* Phase 1:

  * Move argmax (greedy sampling) to GPU
* Phase 2:

  * Move penalty application to GPU
* Phase 3:

  * Move top-k / top-p filtering to GPU
* Phase 4:

  * Fully GPU-resident sampling pipeline

Each phase removes CPU work and reduces GPU idle time.

### 15.9 Impact on CPU and GPU Utilization

* CPU utilization decreases as sampling is offloaded
* GPU utilization increases due to:

  * Additional GPU work per token
  * Reduced idle gaps
* Tokens/sec increases due to reduced per-token latency

### 15.10 Optimization Implications

* Sampling is one of the highest-impact optimization targets
* Offloading sampling to GPU yields:

  * Immediate throughput gains
  * Reduced synchronization
* Sampling optimization is required to approach maximum decode-phase GPU utilization
## 16. Server-Specific Overheads

### 16.1 Server Execution Context

* Server mode runs as a long-lived process
* Provides HTTP-based APIs for inference
* Supports:

  * Request handling
  * Slot management
  * Streaming responses
* Server logic executes concurrently with decode

### 16.2 HTTP Request Handling

* CPU processes:

  * TCP connections
  * HTTP parsing
  * Request validation
* Even with a single active request:

  * HTTP threads remain active
  * Polling and event loops consume CPU time
* HTTP handling adds latency and scheduling noise

### 16.3 Slot Management

* Server maintains slot structures for each potential request
* Slot selection and lifecycle management:

  * Locking
  * State transitions
* Slot logic executes per request and during decode
* Slot bookkeeping introduces additional CPU overhead

### 16.4 Streaming Response Logic

* For streaming outputs:

  * Tokens are serialized into HTTP responses
  * Partial responses are flushed frequently
* Streaming introduces:

  * System calls
  * Buffer management
  * Additional synchronization
* Streaming serialization competes with decode for CPU time

### 16.5 Logging and Metrics

* Server logs:

  * Requests
  * Progress updates
  * Errors
* Logging involves:

  * String formatting
  * I/O operations
* Metrics collection and reporting add overhead
* Logging overhead increases under verbose modes

### 16.6 Prompt Cache Management

* Server may maintain a prompt cache
* Cache lookup and management:

  * Hashing
  * Memory management
* Cache logic executes during request handling
* Cache overhead is unrelated to decode math but consumes CPU

### 16.7 Threading and Synchronization

* Server introduces additional threads:

  * HTTP workers
  * Event loop threads
* Threads compete for CPU resources
* Synchronization between server threads and decode threads introduces jitter

### 16.8 Interaction with Decode Loop

* Server-side events can:

  * Preempt decode threads
  * Delay kernel dispatch
* Decode loop is not isolated from server logic
* Server overhead directly increases GPU idle gaps

### 16.9 Impact on Throughput

* Server mode consistently yields lower tokens/sec than CLI mode
* Server overhead becomes more pronounced:

  * At batch size = 1
  * During long decode sessions
* Server is optimized for flexibility, not raw throughput

### 16.10 Optimization Implications

* Maximum decode throughput requires:

  * Minimizing server-side logic during decode
  * Reducing or disabling logging
  * Reducing streaming flush frequency
* For throughput-critical workloads:

  * Prefer CLI or a stripped-down server
* Server logic must be isolated from the decode hot path to avoid GPU starvation
## 17. Configuration-Only Optimizations

### 17.1 CPU Thread Configuration

* Set CPU threads explicitly to avoid oversubscription
* Match threads to **physical cores**, not logical cores
* Over-allocation increases context switching and hurts GPU feed rate
* Optimal setting keeps CPU busy but stable during decode

### 17.2 GPU Layer Offload Configuration

* Configure the number of layers offloaded to GPU explicitly
* Full offload benefits dense transformer models
* Partial offload is optimal for:

  * Large models exceeding VRAM
  * Hybrid CPU↔GPU execution
* Misconfigured offload leads to frequent CPU↔GPU transfers

### 17.3 Batch Size and Micro-Batching

* Increase batch size only if:

  * Model supports batching efficiently
  * GPU has sufficient VRAM
* For single-user decoding:

  * Micro-batching can improve kernel efficiency
  * Excessive batching increases latency
* Batch settings must align with attention implementation

### 17.4 Context Length Configuration

* Larger context increases:

  * KV-cache size
  * Attention computation cost
* Tokens/sec drops non-linearly with context length
* Use the smallest context length that satisfies workload needs

### 17.5 KV Cache Placement

* Configure KV cache location:

  * GPU-resident for speed
  * CPU-resident for memory savings
* GPU KV cache maximizes throughput
* CPU KV cache introduces frequent memory transfers

### 17.6 Quantization Selection

* Choose quantization based on model architecture:

  * Dense models benefit from uniform low-bit quantization
  * MoE models benefit from mixed or expert-aware quantization
* Lower-bit quantization reduces memory bandwidth pressure
* Incorrect quantization increases dequantization overhead

### 17.7 Backend Selection Flags

* Explicitly select backend:

  * CPU-only
  * CUDA dense
  * CUDA hybrid
  * CUDA MMQ / MoE
* Avoid auto-selection in performance-critical runs
* Backend mismatch leads to suboptimal kernel paths

### 17.8 Precision Configuration

* Prefer FP16 / BF16 where supported
* Avoid FP32 unless required for numerical stability
* Lower precision:

  * Reduces memory bandwidth
  * Increases tensor core utilization

### 17.9 Sampling Configuration

* Disable unnecessary sampling features:

  * Top-k if not needed
  * Top-p if deterministic output is acceptable
* Simpler sampling paths reduce CPU-side overhead
* Sampling cost becomes significant at high tokens/sec

### 17.10 Logging and Verbosity

* Disable verbose logging
* Reduce runtime diagnostics
* Logging competes with decode threads for CPU
* Even minimal logging impacts peak throughput

### 17.11 NUMA and Memory Locality

* Pin process to a single NUMA node if possible
* Ensure CPU threads and memory allocations are local
* Cross-NUMA memory access increases latency
* NUMA misalignment causes GPU starvation via slow host preparation

### 17.12 Power and Clock Configuration

* Ensure CPU and GPU run in performance mode
* Disable aggressive power saving
* Throttling directly reduces sustained tokens/sec
* Stable clocks are critical for long decode sessions
## 18. Build-Time Optimization Options

### 18.1 Compiler Selection

* Prefer **clang** or **gcc** with latest stable versions
* Use a single compiler consistently across all backends
* Avoid mixed compiler toolchains (e.g., gcc + nvcc mismatches)

### 18.2 Global Compiler Flags

* Enable aggressive optimization:

  * `-O3`
  * `-ffast-math`
  * `-funroll-loops`
* Disable debug symbols for production builds
* Strip binaries to reduce I-cache pressure

### 18.3 CPU Architecture Targeting

* Compile with native CPU targeting:

  * `-march=native`
* Enables:

  * AVX2 / AVX-512
  * FMA
  * AMX (if available)
* Prevents fallback to generic scalar kernels

### 18.4 CUDA Architecture Targeting

* Set explicit CUDA compute capability:

  * Example: `sm_89` for Ada Lovelace
* Avoid multi-arch fat binaries unless required
* Single-arch builds reduce binary size and kernel dispatch overhead

### 18.5 CUDA Kernel Configuration

* Enable CUDA-specific optimizations:

  * Tensor core usage
  * MMA kernels
  * Fused operations
* Disable legacy or compatibility kernels
* Ensure MMQ kernels are compiled and enabled when supported

### 18.6 Backend-Specific Build Targets

* Build only required backends:

  * CPU
  * CUDA dense
  * CUDA MMQ / MoE
* Avoid building unused backends to:

  * Reduce binary size
  * Reduce backend selection overhead
* Each backend increases dispatch complexity

### 18.7 cuBLAS vs Custom CUDA Kernels

* Enable cuBLAS builds for:

  * Dense FP16 / BF16 models
* Prefer custom CUDA MMQ kernels for:

  * Quantized models
  * MoE routing
* Avoid hybrid cuBLAS + MMQ unless required

### 18.8 AMX and SIMD Enablement

* Enable AMX explicitly if CPU supports it
* Ensure AMX kernels are not disabled at compile time
* Validate SIMD feature detection during build
* Incorrect detection forces scalar fallbacks

### 18.9 Threading Runtime Configuration

* Enable pthreads explicitly
* Avoid OpenMP unless intentionally tuned
* OpenMP defaults often oversubscribe CPU cores
* Pthreads give finer control over decode threads

### 18.10 Memory Allocation Strategy

* Enable custom allocators where supported
* Prefer aligned allocations
* Reduce malloc/free in hot paths
* Improves cache locality and reduces CPU stalls

### 18.11 LTO and PGO

* Enable Link Time Optimization (LTO) if build time permits
* Profile-Guided Optimization (PGO) yields gains for stable workloads
* PGO improves branch prediction in decode loops
* Especially useful for server builds

### 18.12 Disable Unused Features

* Disable:

  * Tests
  * Examples
  * Debug utilities
* Reduces compile time and binary size
* Prevents accidental linkage of slow paths

### 18.13 Determinism vs Performance

* Disable strict determinism where acceptable
* Allow relaxed math and non-deterministic reductions
* Deterministic modes often block kernel fusion

### 18.14 Static vs Shared Linking

* Static linking improves startup time
* Shared linking reduces memory footprint
* Choose based on deployment model
* Startup latency impacts server warm-up time

### 18.15 Validation After Build

* Verify:

  * Correct backend selection
  * Expected kernel usage
  * No silent CPU fallbacks
* Always benchmark tokens/sec post-build
* A “successful build” does not guarantee optimal performance
## 19. Minimal Code Change Targets

### 19.1 Backend Forcing (Hard Selection)

* Force CUDA backend selection early
* Prevent runtime fallback to CPU backends
* Ensure:

  * CUDA backend is selected at graph build time
  * CPU backend is only used for unavoidable ops
* Target files:

  * `ggml-backend-reg.cpp`
  * `ggml-backend.cpp`

### 19.2 Disable Silent CPU Fallbacks

* Identify ops that silently fall back to CPU
* Add hard errors or warnings for:

  * Unsupported CUDA kernels
  * Missing MMQ / dense kernels
* Prevent mixed execution unless explicitly allowed

### 19.3 Reduce Backend Dispatch Overhead

* Cache backend decisions per graph
* Avoid repeated backend capability checks
* Minimize virtual dispatch in hot paths
* Target:

  * `ggml-backend-impl.h`
  * `ggml-backend.cpp`

### 19.4 Graph Construction Simplification

* Avoid rebuilding graphs per token where possible
* Reuse static graph templates
* Limit dynamic shape changes
* Target:

  * `llama-graph.cpp`
  * `llama-context.cpp`

### 19.5 Kernel Fusion Opportunities

* Identify sequential ops suitable for fusion:

  * RMSNorm + MatMul
  * Bias + Activation
* Prefer fused CUDA kernels where available
* Avoid splitting ops across CPU/GPU boundaries

### 19.6 Reduce CPU-Side Bookkeeping

* Minimize:

  * Tensor metadata updates
  * Shape checks per token
* Cache tensor layouts after first use
* Target:

  * `ggml.c`
  * `ggml.cpp`

### 19.7 Synchronization Minimization

* Reduce explicit `cudaDeviceSynchronize` calls
* Prefer stream-based synchronization
* Batch kernel launches where possible
* Target:

  * `ggml-cuda.cu`
  * CUDA backend wrappers

### 19.8 Sampling Path Offload

* Move more sampling logic to GPU where feasible
* Reduce CPU-side top-k/top-p computation
* Target:

  * `sampling.cpp`
  * `top-k.cu`
  * `topk-moe.cu`

### 19.9 KV-Cache Handling Optimization

* Ensure KV-cache stays resident on GPU
* Avoid CPU↔GPU copies per token
* Align KV-cache memory for CUDA access
* Target:

  * `llama-kv-cache.cpp`
  * `llama-memory-hybrid.cpp`

### 19.10 Logging and Debug Guards

* Compile out verbose logging in hot paths
* Guard debug checks behind compile-time flags
* Target:

  * `log.cpp`
  * `debug.cpp`

### 19.11 Thread Wake-Up Reduction

* Reduce thread wake/sleep cycles during decode
* Avoid condition-variable churn per token
* Prefer busy-wait only where justified
* Target:

  * `ggml-threading.cpp`

### 19.12 Server Hot Path Trimming

* Minimize JSON serialization during streaming
* Reduce HTTP framing overhead
* Avoid per-token mutex locks
* Target:

  * `server.cpp`
  * `server-task.cpp`
  * `server-queue.cpp`

### 19.13 Configuration Parsing Once

* Parse CLI / server config once at startup
* Avoid repeated argument lookups
* Cache resolved flags
* Target:

  * `arg.cpp`
  * `preset.cpp`

### 19.14 Compile-Time Feature Freezing

* Freeze feature flags at build time
* Avoid runtime checks for unused features
* Reduces branch misprediction
* Target:

  * `common.cmake`
  * `ggml-config.cmake.in`

### 19.15 Validation Scope

* After minimal changes, validate:

  * CPU utilization drop
  * GPU utilization increase
  * Tokens/sec improvement
* Ensure correctness before deeper refactors
## 20. Expected Outcome Projection

### 20.1 CPU Utilization

* Decode-phase CPU utilization reduced from near-100% to **moderate / bounded levels**
* CPU usage dominated by:

  * Sampling coordination
  * Minimal scheduling and I/O
* No CPU saturation caused by:

  * Graph rebuilds
  * Backend dispatch
  * Tensor bookkeeping

### 20.2 GPU Utilization

* GPU utilization remains **high and stable** during:

  * Prefill
  * Token-by-token decode
* Reduced GPU idle gaps between tokens
* Higher kernel occupancy due to:

  * Fewer CPU↔GPU sync points
  * Better batching of GPU work
  * Elimination of silent CPU fallbacks

### 20.3 Tokens per Second (t/s)

* Decode throughput increases measurably:

  * Typical gain: **1.3× – 2.0×** for single-sequence interactive decode
  * Larger gains possible on:

    * High-end GPUs
    * MMQ-optimized builds
* Throughput variance per token reduced (more consistent latency)

### 20.4 Latency Characteristics

* Lower per-token latency jitter
* Reduced long-tail stalls caused by:

  * CPU contention
  * Synchronization barriers
* More predictable response streaming behavior

### 20.5 Memory Behavior

* KV-cache remains GPU-resident
* Reduced PCIe traffic during decode
* Lower CPU cache pressure
* More stable GPU memory allocation (fewer reallocations)

### 20.6 Determinism & Correctness

* Exact autoregressive semantics preserved
* No speculative decoding introduced
* Sampling behavior unchanged
* Bitwise-identical outputs for identical seeds and configs

### 20.7 Operational Stability

* No regression in:

  * Long-running sessions
  * Context growth
  * Server uptime
* Improved stability under sustained decode load

### 20.8 Practical Success Criteria

All of the following achieved simultaneously:

* CPU not the decode bottleneck
* GPU actively executing during decode
* Higher sustained tokens/sec
* No correctness or determinism regressions
## 21. Validation Method

### 21.1 Baseline Capture

* Run inference with current configuration
* Record separately for **prefill** and **decode**:

  * CPU utilization (per-core and total)
  * GPU utilization (% and SM occupancy)
  * Tokens per second (steady-state decode)
* Tools:

  * `htop` / `perf stat`
  * `nvidia-smi dmon`
  * `llama-server` internal timing logs

### 21.2 Controlled Experiment Setup

* Single sequence only
* Fixed prompt
* Fixed random seed
* Fixed sampling parameters
* Fixed context size
* Disable all speculative or parallel decoding features

### 21.3 Stepwise Change Validation

For each applied change (config or build):

* Re-run identical prompt
* Compare against baseline:

  * CPU usage delta
  * GPU usage delta
  * t/s delta
* Reject change if:

  * Output differs
  * Determinism breaks
  * CPU usage increases

### 21.4 Decode-Focused Measurement

* Ignore prefill metrics after initial confirmation
* Measure only steady-state decode (≥ 50 tokens)
* Ensure GPU utilization does not dip between tokens

### 21.5 Synchronization Stall Detection

* Enable debug timing:

  * `GGML_CUDA_DEBUG=1`
* Check for:

  * Excessive `cudaDeviceSynchronize`
  * CPU-side waits between kernels
* Confirm reduction after optimization

### 21.6 CPU Fallback Detection

* Verify no unexpected CPU ops:

  * Check backend logs for CPU execution paths
  * Ensure all eligible ops are CUDA-backed
* Fail validation if any core decode ops execute on CPU unintentionally

### 21.7 Memory Residency Verification

* Confirm:

  * Model weights GPU-resident
  * KV cache GPU-resident
* Monitor PCIe traffic:

  * No per-token host↔device transfers

### 21.8 Long-Run Stability Test

* Continuous generation ≥ 10k tokens
* Observe:

  * Memory leaks
  * Performance degradation
  * GPU utilization drift

### 21.9 Acceptance Criteria

Validation passes only if all conditions hold:

* Decode CPU usage < saturation
* GPU utilization sustained during decode
* Higher steady-state t/s
* Deterministic, correct output
* No instability over long runs
