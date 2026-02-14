## 1. Objective Definition

### 1.1 Primary Objective

* Ensure tokens-per-second (t/s) is **never gated by CPU execution**
* Make the GPU the **exclusive execution resource** for all decode-critical computation
* Increase sustained GPU utilization during decode
* Allow CPU execution **only for latency-tolerant, non-pacing work**

### 1.2 Scope of Change

* Allowed:

  * Rebuild `llama.cpp`
  * Modify CUDA backend behavior
  * Modify graph execution, scheduling, and kernel fusion
  * Reclassify work into decode-critical (GPU-exclusive) and decode-non-critical (CPU-eligible)
  * Introduce explicit task allocation and admission control
* Not allowed:

  * CPU fallback for decode-critical execution
  * Dynamic per-op or per-token backend switching
  * Hybrid CPU↔GPU execution on the decode critical path
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
* No backend-dependent behavioral divergence

### 1.4 Target Execution Mode

* Single active decode sequence
* Interactive / long-running session
* Decode-dominated workload (prefill not part of optimization target)
* GPU executes **all decode-critical operations**
* CPU executes **only work that does not gate token emission**

### 1.5 Success Criteria

* Decode tokens/sec remains stable under load
* GPU utilization remains high during decode
* CPU usage does **not** correlate with t/s
* No decode-critical operation executes on CPU
* No silent backend fallback or hybrid execution
* No regressions in correctness, determinism, or stability

### 1.6 Backend Invariant (Canonical)

> **All decode-critical work has exactly one backend owner: the GPU; CPU execution is strictly non-pacing, non-blocking, and never part of the token-generation dependency chain.**


## 2. Hardware & Runtime Context (HW-Specific)

### 2.1 CPU Characteristics

* CPU: x86_64 desktop-class processor
* Cores / Threads: 12 hardware threads available to the runtime
* SIMD support enabled:

  * SSE3
  * SSSE3
  * AVX
  * AVX2
  * F16C
  * FMA
  * BMI2
* OpenMP **disabled for decode-critical execution**
* CPU role constrained to **non-pacing, non-critical work**
* Observed and intended behavior:

  * CPU may reach high utilization
  * CPU must **not** gate tokens-per-second
  * CPU time limited to:

    * Request parsing and scheduling
    * Tokenization and preprocessing
    * Sampling-independent control logic
    * Server-side I/O, logging, and metrics
    * Admission control and task classification

---

### 2.2 GPU Characteristics

* GPU: NVIDIA GeForce RTX 4060 Ti
* Architecture: Ada Lovelace
* Compute Capability: 8.9
* VRAM: 16 GiB
* Features available and selectively used:

  * Tensor Cores
  * Flash Attention
  * MMQ quantized matmul kernels
* Explicitly disabled or restricted:

  * CUDA Graphs for decode-critical execution
* Intended behavior:

  * GPU is the **exclusive execution backend** for all decode-critical computation
  * GPU utilization remains high during steady-state decode
  * GPU is the sole pacing resource for token emission

---

### 2.3 Memory Topology

* Discrete memory architecture:

  * CPU DRAM
  * GPU VRAM (PCIe-connected)
* No unified or managed memory usage
* Model weight placement:

  * Maximum feasible transformer layers resident in GPU VRAM
  * Remaining layers statically assigned to CPU **only if they are outside the decode-critical path**
* KV cache placement:

  * GPU-resident KV cache for all decode-critical layers
  * CPU KV cache permitted only for non-pacing or background-managed state
* VRAM pressure sources:

  * Quantized weights
  * KV cache for long context
  * CUDA compute and temporary buffers

---

### 2.4 Software Environment

* Operating System: Linux (Debian/Ubuntu class)
* NVIDIA proprietary driver installed and stable
* CUDA runtime available and functional
* llama.cpp built with:

  * CUDA enabled
  * MMQ backend enabled and forced for decode-critical paths
  * Flash attention enabled
  * CUDA graphs disabled for decode
  * OpenMP disabled for decode-critical execution

---

### 2.5 Runtime Execution Mode

* Binary: `llama-server`
* Single active decode sequence (`n_seq_max = 1`)
* Context size: 8192
* Batch size during prefill: >1
* Batch size during decode: exactly 1 token
* Long-running process:

  * Model loaded once
  * Reused across requests
* Decode-dominated steady state
* Prefill performance explicitly out of optimization scope

---

### 2.6 Threading and Scheduling

* Total threads available: 12
* Thread roles strictly separated:

  * GPU execution threads (decode-critical)
  * CPU scheduling and admission threads
  * HTTP server and I/O threads
* CPU threads are **not permitted** to execute:

  * Decode-critical graph nodes
  * Attention or MLP computation
  * Logits generation
  * Token-selection gating logic
* Intended consequence:

  * GPU executes uninterrupted decode loops
  * CPU never becomes the pacing resource
  * GPU idle gaps minimized by reduced host-side orchestration

---

### 2.7 Key Constraint Implied by This Hardware

* GPU has substantial unused compute headroom during decode
* Decode throughput is limited by host orchestration if not constrained
* PCIe latency and kernel launch overhead are significant at batch = 1
* Performance improvement requires:

  * Eliminating CPU participation in the token-generation dependency chain
  * Increasing GPU kernel residency and work density per decode step
  * Ensuring CPU work remains strictly orthogonal to t/s-critical execution

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

Only the following **GPU-first, decode-correct builds** are considered.

---

#### 3.3.1 `build_cuda_cublas_dense`

* Dense layers executed using cuBLAS
* Quantized or dequantized matmul depending on model format
* GPU executes (decode-critical):

  * All attention and MLP matmul
  * KV cache reads and writes
  * Logits computation
  * Token selection (argmax / deterministic sampling)
* CPU executes (non-pacing only):

  * Request handling
  * Tokenization and preprocessing
  * Scheduling and admission control
  * Logging, metrics, and I/O
* cuBLAS kernels are:

  * Highly optimized
  * Short-lived at batch = 1

**Implication:**

* GPU is the sole decode pacing resource
* CPU does not participate in the token-generation dependency chain
* Decode throughput limited by GPU math + kernel launch overhead only
* CPU utilization does not affect tokens/sec

---

#### 3.3.2 `build_cuda_mmq_moe`

* Quantized MMQ kernels enabled
* Supports:

  * Quantized dense layers
  * Quantized MoE layers (if model includes MoE)
* GPU executes (decode-critical):

  * Fused quantized matmul kernels
  * Flash attention (if enabled)
  * KV cache operations
  * Logits computation
  * Token selection
* CPU executes (non-pacing only):

  * Scheduling and control-plane logic
  * Server-side I/O and background tasks
* MMQ kernels are:

  * Long-lived relative to cuBLAS
  * Higher arithmetic density per launch

**Implication:**

* Highest achievable GPU residency per token
* Minimal kernel launch frequency
* GPU remains authoritative for decode pacing
* CPU load is orthogonal to t/s

---

### 3.4 KV Cache Behavior (Across Both Builds)

* KV cache grows linearly with generated tokens
* Access pattern:

  * Sequential append
  * Read-heavy during attention
* KV cache location:

  * GPU-resident for all decode-critical layers
* No semantic difference in cache behavior between builds
* CPU does not participate in KV cache mutation on the decode path

---

### 3.5 Sampling Behavior (Across Both Builds)

* Sampling is logically model-independent
* Pipeline:

  * Logits → transformations → token selection
* Sampling is:

  * Decode-critical
  * Latency-sensitive
* Sampling execution:

  * GPU-resident for decode-critical path
  * CPU involvement limited to non-blocking auxiliary logic only

**Implication:**

* Sampling latency does not stall GPU progress
* Token emission rate is not gated by CPU execution

---

### 3.6 Decode-Phase Cost Structure (Model-Agnostic)

At batch size = 1:

* GPU work:

  * Attention + MLP matmul
  * KV cache access
  * Logits + sampling
* CPU work:

  * Non-blocking orchestration
  * Control-plane logic
* Overall throughput limited by:

  * GPU kernel efficiency
  * Kernel launch overhead
  * Memory bandwidth and cache locality

CPU execution is **not** on the decode-critical path.

---

### 3.7 Performance Implication Across Builds

* Build choice determines:

  * Kernel fusion level
  * Arithmetic density per token
* Build choice does **not** alter:

  * Autoregressive dependency
  * Token ordering
  * Semantic correctness
* Fundamental improvement over baseline:

  * Removal of CPU from decode pacing
  * Stable tokens/sec under CPU load

---

### 3.8 Non-Negotiable Constraints Imposed by Model + Builds

* Exact autoregressive semantics preserved
* Token order preserved
* No batching across tokens
* No speculative decode
* No approximation of attention or sampling

Only **execution placement and control-path restructuring** are permitted.

---

## 3.9 Build Type vs Model Type — Throughput Characteristics

### 3.9.1 `build_cuda_cublas_dense`

**Best suited for:**

* Dense, non-quantized or lightly quantized models
* Medium-to-large dense models with strong GEMM utilization
* Stable, deterministic decode workloads

**t/s characteristics:**

* High prefill throughput
* Decode throughput governed by:

  * cuBLAS efficiency
  * Kernel launch overhead

**Conclusion:**

* GPU-paced decode with deterministic behavior
* Suitable when model is dense and VRAM pressure is manageable

---

### 3.9.2 `build_cuda_mmq_moe`

**Best suited for:**

* Quantized models (Q4–Q8, K-variants)
* Large dense models
* MoE models (if present)
* Long-context decode workloads

**t/s characteristics:**

* Highest sustained decode tokens/sec
* Reduced kernel launch count
* Best GPU occupancy under strict autoregressive constraints

**Conclusion:**

* **Preferred build for maximum decode throughput**
* Baseline for all further performance optimization

---

### 3.9.3 Summary Table

| Model Type         | Preferred Build           | Reason                           |
| ------------------ | ------------------------- | -------------------------------- |
| Medium dense model | `build_cuda_cublas_dense` | Strong GEMM, stable GPU pacing   |
| Large dense model  | `build_cuda_mmq_moe`      | Quantized efficiency, higher t/s |
| Large quantized    | `build_cuda_mmq_moe`      | Best decode GPU utilization      |
| MoE model          | `build_cuda_mmq_moe`      | Native MoE + MMQ kernel support  |

---

### 3.9.4 Final Throughput Ranking (Decode Phase)

From highest to lowest tokens/sec:

1. `build_cuda_mmq_moe`
2. `build_cuda_cublas_dense`

This ranking assumes **GPU-exclusive decode-critical execution** and holds independent of model family, given comparable size and quantization.


## 4. Execution Mode Clarification

### 4.1 Request Pattern

* Single active request at any given time
* No concurrent user requests
* No request batching across users
* No background jobs
* Execution is strictly sequential at the request level
* Request admission and scheduling are non-pacing and must not gate decode

---

### 4.2 Sequence Characteristics

* Single sequence generation (`n_seq_max = 1`)
* One token generated per decode step
* Strict autoregressive dependency:

  * Token *n+1* cannot be computed before token *n* is fully finalized
* No parallel decoding across sequences
* No overlap between decode-critical steps

---

### 4.3 Interaction Style

* Interactive or long-running session
* Prompt provided once, followed by a long decode phase
* Decode phase dominates total runtime
* Streaming output may be enabled
* Streaming semantics must not alter decode ordering or execution dependencies

---

### 4.4 Server vs CLI Execution

* Execution may occur via:

  * `llama-cli`
  * `llama-server`
* Server mode characteristics:

  * HTTP request handling
  * Slot management
  * Request lifecycle management
  * Background I/O and control-plane activity
* CLI mode characteristics:

  * Minimal control flow
  * Fewer synchronization points
* Decode-critical execution semantics must be identical in both modes
* Mode-specific logic must remain non-pacing

---

### 4.5 Batching Behavior

* Prefill phase:

  * Batch size > 1 allowed
  * High GPU utilization expected
* Decode phase:

  * Effective batch size = 1 token
  * No token batching permitted
* Micro-batching or speculative aggregation across tokens is not allowed

---

### 4.6 Sampling Mode

* Sampling may be:

  * Deterministic (greedy, `temp = 0`)
  * Stochastic (top-k, top-p, temperature)
* Sampling must preserve:

  * Exact semantics
  * Determinism when configured
* Sampling is decode-critical and must not introduce CPU pacing
* Sampling completion is required before advancing to the next token

---

### 4.7 Correctness and Ordering Guarantees

* Token emission order must be preserved
* No reordering of compute relative to token output
* No speculative, rollback, or predictive execution
* Each token must be fully committed before the next decode step begins
* Backend choice must not affect observable semantics

---

### 4.8 Termination Conditions

* Decode loop terminates only when:

  * End-of-sequence token is generated, or
  * Maximum token limit is reached
* Termination checks must be exact and deterministic
* No heuristic or early-stop shortcuts permitted

---

### 4.9 Implication for Optimization

* Execution is inherently latency-serial
* GPU starvation during decode originates from:

  * Host-driven orchestration
  * Fine-grained synchronization on the critical path
* Any optimization must:

  * Remove CPU execution from the token-generation dependency chain
  * Reduce host involvement per token
  * Increase GPU work density per decode step
  * Preserve strict execution order and correctness

## 5. High-Level Decode Pipeline Mapping

### 5.1 Decode Entry Point

* Decode begins after prompt prefill is completed
* Control enters the decode loop from:

  * `llama_decode()` (CLI)
  * Server-side decode task loop (`llama-server`)
* Decode loop executes once per generated token
* Entry into the decode loop marks the start of the **decode-critical phase**

---

### 5.2 Per-Token Decode Lifecycle (Logical)

For each token generation step, the following stages occur in strict order:

1. Input token embedding lookup
2. Forward pass through all transformer layers
3. Logits computation
4. Sampling / token selection
5. KV cache update
6. Token commit and output
7. Termination check

All stages are **decode-critical** and must complete before the next token begins.

---

### 5.3 Transformer Forward Pass

* Executed layer-by-layer
* For each layer:

  * Normalization
  * Attention computation using KV cache
  * Feed-forward network
* Execution backend is **GPU-exclusive** for decode-critical layers
* GPU kernels may be launched:

  * Per layer, or
  * As fused kernel groups (build-dependent)
* CPU must not execute any layer participating in the decode-critical path

---

### 5.4 Graph Construction and Execution

* A ggml graph represents the computation required to generate **one token**
* Graph characteristics:

  * Execution order is fixed and deterministic
  * Node dependencies encode strict autoregressive semantics
* Graph handling rules:

  * Graph construction, reuse, or validation may occur on CPU
  * Graph execution of decode-critical nodes is **GPU-exclusive**
* CPU must not:

  * Execute decode-critical graph nodes
  * Gate graph execution progress
* GPU executes all decode-critical compute nodes without CPU interposition

---

### 5.5 KV Cache Interaction

* KV cache is accessed during attention in every transformer layer
* Operations per token:

  * Read: keys and values for all previous tokens
  * Write: append current token’s key and value
* KV cache rules:

  * KV cache for decode-critical layers is GPU-resident
  * KV cache mutation on the decode path is GPU-exclusive
* KV cache updates are serialized per token but must not involve CPU pacing

---

### 5.6 Sampling Stage

* Logits are produced as the final output of the forward pass
* Sampling pipeline includes:

  * Logit post-processing
  * Probability filtering (if enabled)
  * Token selection
* Sampling is **decode-critical**
* Sampling completion is a hard dependency for advancing to the next token
* Sampling must not introduce CPU execution on the token-generation dependency chain

---

### 5.7 Output and State Update

* Selected token is:

  * Committed to the output buffer
  * Used to update internal sequence state
* Context position is incremented
* Sequence state is updated deterministically
* Streaming output, if enabled, is emitted asynchronously and must not gate decode

---

### 5.8 Synchronization Rules

* GPU executes decode-critical work without CPU-driven per-stage blocking
* CPU must not introduce synchronization points that gate token emission
* Required ordering constraints are enforced by:

  * Graph dependencies
  * GPU execution ordering
* Any CPU-side waits must be non-pacing and outside the decode-critical path

---

### 5.9 Loop Continuation

* Decode loop repeats until:

  * End-of-sequence token is generated, or
  * Maximum token count is reached
* No overlap between token iterations
* Token-level execution remains strictly serial and deterministic

---

### 5.10 Key Observation from the Pipeline

* Decode-critical computation must be **entirely GPU-owned**
* CPU responsibilities are limited to:

  * Control-plane logic
  * Task classification
  * Admission and scheduling
  * I/O, logging, and background work
* GPU idle time during decode must not be caused by CPU pacing
* Any restructuring must target:

  * Removing CPU participation from the token-generation dependency chain
  * Increasing GPU work density per decode step
  * Preserving strict execution order and correctness

## 6. CPU Responsibility Audit (Revised per GPU-Exclusive Decode Principle)

### 6.1 Decode Loop Control

* CPU may host the **control-plane loop structure**
* CPU must **not** gate progression of decode-critical work
* CPU responsibilities limited to:

  * Initiating decode requests
  * Handling termination conditions
* CPU must **not** block token progression based on CPU-side stages
* Token-by-token sequencing is enforced by **GPU execution dependencies**, not CPU waits

---

### 6.2 Graph Scheduling and Execution

* CPU may:

  * Construct or validate ggml graphs
  * Perform static dependency analysis
* CPU must **not**:

  * Schedule decode-critical nodes dynamically
  * Determine per-node execution order at runtime
  * Gate execution of GPU nodes
* Decode-critical graph execution is **GPU-owned**
* Graph execution order is enforced by:

  * Graph structure
  * GPU execution semantics

---

### 6.3 CUDA Kernel Dispatch

* CPU initiates kernel launches but must not:

  * Insert per-node synchronization
  * Poll for completion on the decode-critical path
* Kernel launch overhead must be amortized or reduced
* GPU execution must proceed without CPU-driven stalls
* CPU-side synchronization is permitted **only outside** the token-generation dependency chain

---

### 6.4 Sampling and Token Selection

* Sampling is **decode-critical**
* Sampling must **not** execute on CPU
* CPU must not participate in:

  * Logit post-processing
  * Probability filtering
  * Token selection
* Sampling completion must be driven by GPU execution flow
* CPU may only observe results after token commitment

---

### 6.5 KV Cache Management

* KV cache mutation on the decode path is **GPU-exclusive**
* CPU responsibilities limited to:

  * Non-critical metadata bookkeeping
  * Allocation outside the decode-critical path
* CPU must not:

  * Perform KV writes for decode-critical layers
  * Gate KV consistency checks per token
* KV correctness is enforced by GPU execution ordering

---

### 6.6 Synchronization and Barriers

* CPU must not insert synchronization points that:

  * Block GPU progress
  * Gate token emission
* Decode-critical synchronization must be:

  * Implicit
  * GPU-internal
* CPU-side waits, polling, or barriers are permitted **only** for non-pacing tasks

---

### 6.7 Thread Pool Management

* CPU thread pools may exist for:

  * Background tasks
  * I/O
  * Server control-plane logic
* Decode-critical execution must not depend on:

  * ggml worker thread availability
  * CPU thread wake/sleep cycles
* Thread scheduling overhead must be fully decoupled from decode pacing

---

### 6.8 Server-Side Control (if applicable)

* CPU handles:

  * HTTP request parsing
  * Slot management
  * Request lifecycle
  * Logging and metrics
* Server-side execution must be:

  * Asynchronous
  * Non-blocking
* Server activity must not contend with decode-critical GPU execution

---

### 6.9 Memory Management

* CPU may manage:

  * Long-lived allocations
  * Initialization-time buffers
* CPU must not perform:

  * Per-token allocation
  * Per-token deallocation
  * Per-token host-device bookkeeping
* Decode-critical memory usage must be preallocated and GPU-resident

---

### 6.10 Aggregate Impact on Performance (Revised)

* CPU performs **no latency-critical work per token**
* CPU responsibilities are:

  * Orthogonal
  * Asynchronous
  * Non-pacing
* GPU is the **sole pacing resource** for decode
* Decode throughput is determined by:

  * GPU kernel efficiency
  * Kernel fusion and residency
  * Memory bandwidth and cache locality

This revised audit defines the **target state**: CPU remains active but is **never on the token-generation dependency chain**, eliminating CPU-induced GPU idle time and preserving stable tokens/sec.

## 7. GPU Responsibility Audit

### 7.1 Core Decode-Critical Responsibilities (GPU-Exclusive)

* GPU is the **sole execution authority** for all decode-critical computation

* GPU executes the **entire token-generation dependency chain**

* Primary GPU responsibilities include:

  * Linear projections (Q, K, V, output)
  * Attention score computation
  * Softmax over attention scores
  * Attention-weighted value accumulation
  * Feed-forward network (MLP) layers
  * Logits computation
  * Token selection / sampling
  * KV cache read and write for current token

* All operations whose outputs determine the next token are **GPU-exclusive**

---

### 7.2 Backend-Specific Compute Paths (GPU-Owned)

* Depending on build configuration, GPU executes one of:

  * CUDA dense kernels
  * cuBLAS GEMM / GEMV kernels
  * MMQ quantized matmul kernels
* Backend selection determines:

  * Kernel fusion strategy
  * Arithmetic intensity
  * Kernel residency duration
* Backend choice **does not alter execution ownership**:

  * Decode-critical compute remains GPU-only in all cases

---

### 7.3 Flash Attention Execution

* When enabled, GPU executes flash-attention kernels
* Flash-attention operates fully on GPU and:

  * Reduces memory traffic
  * Eliminates intermediate buffers
* Operates on:

  * Query for current token
  * Entire KV cache up to current position
* Execution time scales with context length
* No CPU participation or gating is permitted

---

### 7.4 KV Cache Operations (GPU-Resident)

* GPU performs all KV cache interactions for decode-critical layers:

  * Reads during attention
  * Writes of new key/value vectors for current token
* KV cache resides in GPU VRAM for all decode-critical layers
* KV cache updates are:

  * Serialized per token
  * Executed entirely on GPU
* CPU does not participate in KV mutation or synchronization

---

### 7.5 Quantization and Dequantization

* For quantized models:

  * Dequantization is executed on GPU
  * Dequantization is fused with matmul where possible
* Quantization reduces memory bandwidth pressure
* Quantization does not introduce CPU involvement in decode

---

### 7.6 Execution Granularity and Residency

* GPU execution during decode is structured to maximize residency:

  * Persistent graphs
  * Fused kernels
  * Reduced launch boundaries
* Per-token execution minimizes host-visible transitions
* GPU performs the full per-token forward pass without CPU pacing

---

### 7.7 Synchronization Semantics

* Decode-critical ordering is enforced by:

  * GPU execution dependencies
  * Graph-level ordering guarantees
* CPU does **not** insert synchronization points on the decode path
* GPU does **not** wait for CPU between decode stages
* Token-level serialization is preserved **entirely within GPU execution**

---

### 7.8 GPU Utilization Characteristics (Target State)

* Prefill phase:

  * Large kernels
  * High occupancy
  * Near-saturation utilization
* Decode phase:

  * Sustained kernel residency
  * Minimal idle gaps
  * Utilization limited by model arithmetic, not host pacing

---

### 7.9 GPU Autonomy Guarantees

* GPU controls:

  * Decode loop progression
  * Token generation cadence
  * Sampling and commitment
* GPU maintains persistent execution context across decode iterations
* CPU has no authority to stall or gate token emission

---

### 7.10 Aggregate Impact on Performance

* GPU becomes the **sole pacing resource** for tokens/sec
* Decode throughput scales with GPU capability
* CPU load no longer affects decode t/s
* GPU underutilization caused by host-driven orchestration is eliminated

This audit defines the **required end state**: the GPU owns and executes the entire decode-critical path, with no CPU participation on the token-generation dependency chain.

## 8. CPU↔GPU Synchronization Points (Target-State, Post-Modification)

### 8.1 Decode-Step Boundary Synchronization

* **No CPU↔GPU synchronization exists on the decode-critical path**
* Token-to-token progression is enforced **entirely within GPU execution**
* CPU does **not** wait for GPU completion to advance decode
* GPU autonomously determines completion of token *n* and initiation of token *n+1*

---

### 8.2 Graph Execution Synchronization

* ggml graph execution for decode is:

  * GPU-resident
  * Persistently instantiated
* CPU does **not** block on graph completion during decode
* Graph execution ordering is enforced by:

  * GPU-side dependencies
  * CUDA execution semantics
* CPU has no visibility requirement into intermediate graph completion

---

### 8.3 Kernel Launch and Ordering Semantics

* Kernel launches for decode are:

  * Issued as part of persistent GPU execution
  * Not interleaved with CPU decision points
* CPU does **not** wait on individual kernel completions
* Kernel ordering is enforced by:

  * CUDA stream semantics
  * Graph-level dependencies on GPU

---

### 8.4 Sampling Dependency Barrier

* Sampling is executed **on GPU**
* No device-to-host transfer of logits occurs on the decode path
* No CPU-side sampling barrier exists
* Token selection completes on GPU and directly feeds the next decode step

---

### 8.5 KV Cache Consistency

* KV cache is fully GPU-resident for decode-critical layers
* KV cache reads and writes are:

  * Ordered by GPU execution
  * Serialized per token within GPU context
* CPU does **not** participate in KV cache mutation or visibility checks
* No CPU-enforced KV consistency barrier exists

---

### 8.6 Memory Visibility and Data Movement

* Decode-critical data remains GPU-resident:

  * Activations
  * Logits
  * KV cache
  * Sampling state
* No device-to-host transfers occur on the decode path
* CPU accesses decode outputs only **after** token commitment, asynchronously

---

### 8.7 Server-Side Synchronization

* Server-side CPU logic is decoupled from decode execution
* Request lifecycle events:

  * Do not gate decode progression
  * Do not introduce synchronization into GPU execution
* Slot state and streaming output are handled asynchronously

---

### 8.8 CUDA Graph Usage

* CUDA graphs are used to:

  * Eliminate per-kernel launch overhead
  * Maintain persistent GPU execution
* Graph replay is GPU-driven during decode
* Graph invalidation triggers:

  * A controlled pause **outside** the decode-critical path
  * Never mid-token or between tokens

---

### 8.9 Cumulative Synchronization State

* Zero CPU↔GPU synchronization points exist per token
* GPU execution proceeds continuously across decode iterations
* CPU activity cannot introduce GPU idle gaps
* Token throughput is invariant to CPU load

---

### 8.10 Optimization Implication (Final)

* Decode performance is no longer limited by:

  * CPU waits
  * Sampling barriers
  * Graph-level synchronization
  * Kernel launch overhead
* Tokens/sec is determined solely by:

  * GPU compute capability
  * Model arithmetic intensity
  * Context length effects

This section defines the **required invariant**:
**no CPU↔GPU synchronization is permitted on the token-generation dependency chain.**

## 9. Backend Selection Logic (Aligned with GPU-Exclusive Decode)

### 9.1 Backend Selection Objective

* Backend selection determines **where** decode-critical operations execute

* The primary invariant is:

  > **All decode-critical operations must resolve to a GPU backend, without exception**

* Backend choice directly defines:

  * Tokens/sec
  * GPU occupancy
  * Presence or absence of CPU pacing

---

### 9.2 Decode-Critical vs Non-Critical Classification

Backend selection is governed by **task classification**, not capability fallback.

* **Decode-critical operations**:

  * Must be GPU-exclusive
  * Backend choice is fixed before execution
* **Non-critical operations**:

  * May execute on CPU
  * Must not gate token emission

This classification is **static and irreversible** per task.

---

### 9.3 Available Backends (Logical)

* GPU backends (decode-critical eligible):

  * CUDA dense
  * CUDA cuBLAS dense
  * CUDA MMQ (quantized, MoE-capable)
* CPU backend (non-critical only)

Hybrid backends are **explicitly disallowed** on the decode path.

---

### 9.4 Build-Time Backend Availability

* Build configuration defines which GPU backends exist
* For decode:

  * At least one GPU backend **must** be available
  * Absence of a suitable GPU backend is a **hard error**, not a fallback condition
* CPU backend availability does **not** imply decode eligibility

---

### 9.5 Runtime Backend Resolution (Decode Path)

For decode-critical operations:

* Backend resolution occurs **once**, before decode begins
* Resolution rules:

  * Tensor must be GPU-resident
  * Operation must have a GPU implementation
  * Backend must remain constant across all decode steps
* If resolution fails:

  * Decode does not start
  * Execution aborts with an explicit error

No per-token or per-layer backend switching is permitted.

---

### 9.6 CPU Backend Usage Rules

* CPU backend is **never** selected for decode-critical operations
* CPU backend may be used only for:

  * Tokenization
  * Request handling
  * Logging and metrics
  * Scheduling and admission control
  * Other latency-tolerant tasks

Any CPU backend invocation on the decode dependency chain is forbidden.

---

### 9.7 CUDA Dense Backend Role

* Eligible for decode only if:

  * All decode-critical layers are GPU-resident
  * No CPU fallback paths exist
* Suitable primarily for:

  * Dense, non-quantized models
* Backend selection is fixed for the entire decode session

---

### 9.8 CUDA cuBLAS Backend Role

* Eligible for decode under strict conditions:

  * GPU-exclusive execution guaranteed
  * No GEMV → CPU fallback paths
* Optimized for prefill
* Decode use is allowed only when:

  * Backend remains GPU-resident
  * Kernel launch behavior is stable

---

### 9.9 CUDA MMQ Backend Role

* Preferred backend for decode-heavy workloads
* Selected when:

  * Model is quantized
  * MMQ supports the quantization format
* Advantages:

  * Fused kernels
  * Reduced launch count
  * Higher sustained GPU occupancy
* Backend is fixed for the entire decode lifecycle

---

### 9.10 Prohibited Hybrid Execution

* The following are explicitly forbidden during decode:

  * Layer-wise CPU↔GPU alternation
  * Partial layer execution on CPU
  * CPU fallback due to VRAM pressure
  * Dynamic backend switching

Hybrid execution is treated as a correctness violation, not an optimization.

---

### 9.11 Backend Fallback Policy (Decode)

* **No fallback exists on the decode path**
* If a decode-critical operation cannot be mapped to GPU:

  * Execution aborts
  * Error is surfaced immediately
* Silent fallback is prohibited

---

### 9.12 Environment Variable and Flag Constraints

* Environment variables may:

  * Restrict backend choice
  * Force a specific GPU backend
* They must **never**:

  * Enable CPU fallback for decode
  * Introduce backend instability across tokens
* Backend selection must be logged and verified at startup

---

### 9.13 Decode-Phase Invariant

> **Backend selection for decode is static, GPU-exclusive, and immutable for the lifetime of the decode session.**

This invariant guarantees:

* CPU never gates token emission
* GPU utilization is not disrupted by backend churn
* Tokens/sec is determined solely by GPU compute capability

## 10. Threading & Parallelism Analysis (Aligned with GPU-Exclusive Decode)

### 10.1 CPU Thread Model (Reinterpreted)

* llama.cpp exposes a CPU thread pool via ggml
* Thread count controlled by:

  * `--threads`
  * `--threads-batch`
* Under the **GPU-exclusive decode invariant**, CPU threads are **not part of the decode-critical path**
* CPU threads are permitted only for **non-blocking, non-pacing work**

---

### 10.2 Decode-Critical vs Non-Critical Thread Roles

All CPU thread activity must be classified **before execution**.

#### Decode-critical (forbidden on CPU)

* Any work that gates next-token emission
* Includes:

  * Graph traversal that blocks GPU progress
  * Sampling / argmax
  * Per-token decode loop control
  * Synchronization that delays GPU execution

➡ **CPU threads must never execute these**

#### Non-critical (CPU-eligible)

* Request parsing
* Tokenization
* Logging and metrics
* Server I/O
* Admission control
* Background housekeeping
* Prefetching future requests
* Memory reclamation
* Non-blocking orchestration

➡ These may freely use CPU threads

---

### 10.3 Effective Parallelism Under the New Model

* Decode phase remains **logically serial** at the token level
* However:

  * GPU executes the entire decode-critical graph
  * CPU threads do **not** participate in token-to-token sequencing
* Result:

  * CPU parallelism is decoupled from decode throughput
  * CPU load no longer determines tokens/sec

---

### 10.4 Oversubscription Reframed

* CPU oversubscription is harmful **only if CPU is on the critical path**
* Under GPU-exclusive decode:

  * CPU threads may saturate without impacting t/s
  * Oversubscription affects only background latency
* Therefore:

  * Decode performance is insensitive to CPU thread count
  * CPU thread tuning is no longer a throughput lever

---

### 10.5 Interaction with CUDA Dispatch (Corrected)

* CUDA kernel launches and sequencing are logically owned by the GPU decode engine
* CPU threads do **not**:

  * Pace kernel launches
  * Synchronize per token
  * Gate progression between kernels
* Multiple CPU threads do not improve decode
* CPU thread reduction is beneficial only to reduce noise, not to increase t/s

---

### 10.6 Sampling and Threading (Post-Change)

* Sampling is **decode-critical**
* Therefore:

  * Sampling must be GPU-resident
  * CPU threads must not execute sampling logic
* CPU thread count has **zero impact** on sampling latency once migrated

---

### 10.7 Server Mode Threading (Isolated)

* Server threads handle:

  * HTTP
  * Slot lifecycle
  * Logging
* These threads are isolated from decode execution
* Hard rule:

  * Server threads must never block GPU decode scheduling
* Server load may increase CPU usage but must not affect tokens/sec

---

### 10.8 GPU Parallelism vs CPU Parallelism (Final Model)

* GPU parallelism is the **only throughput determinant**
* CPU parallelism is auxiliary
* There is no attempt to “balance” work between CPU and GPU
* Instead:

  * GPU owns decode
  * CPU owns everything else

---

### 10.9 Thread Affinity and Scheduling (Secondary Concern)

* Thread pinning may reduce jitter
* However:

  * Jitter does not affect decode t/s once CPU is off the critical path
* Thread affinity is an optimization for stability, not throughput

---

### 10.10 Final Implications

* Increasing CPU threads does **not** increase decode throughput
* Decreasing CPU threads does **not** reduce decode throughput
* Decode performance is invariant to CPU scheduling once:

  > **CPU is removed from the token-generation dependency chain**

This section formally establishes that **threading is no longer a decode performance variable** once GPU-exclusive execution is enforced.

## 11. Memory Mapping & Allocation (Aligned with GPU-Exclusive Decode)

### 11.1 Memory Allocation Domains

* Two strictly separated memory domains:

  * CPU DRAM
  * GPU VRAM
* PCIe transfers are **explicit and controlled**
* No unified memory, no implicit migration
* Decode-critical execution relies **exclusively on GPU-resident memory**

---

### 11.2 Model Weight Allocation (Target State)

* Model weights are loaded once at initialization
* For decode-critical execution:

  * **All transformer layers participating in decode must reside in GPU VRAM**
* No CPU-resident layers are permitted on the decode path
* Weight placement is:

  * Static
  * Immutable during decode
* Any layer not fitting in VRAM must prevent decode start (admission control), not trigger hybrid execution

---

### 11.3 KV Cache Allocation (Target State)

* KV cache is allocated at context initialization
* KV cache properties:

  * Fully GPU-resident
  * Grows monotonically with sequence length
  * Never split across CPU and GPU
* KV cache updates are:

  * Per-token
  * Serialized
  * Executed entirely on GPU
* CPU does not:

  * Read KV
  * Write KV
  * Track KV metadata for decode

---

### 11.4 Compute Buffer Allocation

* All decode-critical compute buffers are:

  * Allocated in GPU VRAM
  * Pre-allocated before decode begins
  * Reused across tokens
* Buffers include:

  * Activations
  * Attention intermediates
  * FFN intermediates
  * Logits
  * Sampling state
* **No buffer allocation occurs during decode**

---

### 11.5 Memory Mapping Modes (Clarified)

* `mmap` affects only model load behavior
* Decode-phase behavior is invariant to `mmap` once weights are resident
* For decode performance:

  * `mmap` must not introduce page faults during decode
* All pages required for decode must be resident before token generation starts

---

### 11.6 Host↔Device Transfers (Forbidden on Decode Path)

* No host↔device transfers are allowed during decode-critical execution
* Specifically forbidden per token:

  * Logits transfer to CPU
  * Sampling data transfer
  * KV metadata transfer
* Device↔host transfers may occur only:

  * After token commitment
  * Asynchronously
  * Outside the decode dependency chain

---

### 11.7 Pinned and Pageable Memory (Non-Critical Only)

* Pinned memory may be used for:

  * Asynchronous output streaming
  * Logging
  * Metrics
* Pageable memory is acceptable for:

  * CPU-only tasks
* Decode-critical execution is **independent of host memory type** because no transfers occur

---

### 11.8 Allocation Lifetime and Churn (Invariant)

* All allocations required for decode are completed before first token
* Decode loop performs:

  * Zero allocations
  * Zero frees
* Any allocation during decode is considered a correctness violation of the execution model

---

### 11.9 Memory Visibility and Synchronization (Eliminated)

* CPU never reads GPU-resident decode data
* Therefore:

  * No device-to-host visibility barrier exists per token
  * No implicit synchronization is introduced by memory access
* GPU enforces all required ordering internally

---

### 11.10 Fragmentation and Long-Running Stability

* Stable allocation layout is mandatory
* Buffers are:

  * Fixed-size
  * Reused
* No dynamic growth except KV cache append within pre-reserved bounds
* Fragmentation must not evolve during decode

---

### 11.11 Final Memory Invariant

* **All decode-critical state, data, and computation remain GPU-resident for the entire decode phase**
* CPU memory is **never accessed** by operations on the token-generation dependency chain
* Memory placement is a **hard correctness constraint**, not a performance hint

This section establishes memory residency as a first-class enforcement mechanism ensuring that **CPU cannot re-enter the decode-critical path**.

## 12. Graph Lifetime Analysis (Aligned with GPU-Exclusive Decode)

### 12.1 Graph Definition (Reinterpreted)

* A ggml graph represents the **entire decode-critical computation** required to produce one token
* The graph includes:

  * All transformer layer operations
  * Attention computation
  * FFN computation
  * Logits computation
  * Sampling and token commitment
* Decode-critical graphs are **GPU-exclusive**
* CPU nodes are **not permitted** on the decode graph

---

### 12.2 Graph Construction Phase

* Graph construction occurs:

  * During context initialization
  * During prefill
* Graph construction is a **CPU-side setup activity**
* Graph construction must be completed **before decode begins**
* Graph construction is **forbidden** during active decode

---

### 12.3 Graph Reuse During Decode (Target State)

* Decode uses a **single, stable graph structure**
* Graph is:

  * Persistently instantiated
  * Reused across all decode iterations
* Graph execution is:

  * Autonomous on GPU
  * Not re-triggered by CPU per token
* CPU does not initiate, gate, or synchronize graph execution per token

---

### 12.4 Graph Invalidation Rules (Strict)

Graph reuse **must not** be invalidated during decode.

* The following are **disallowed during decode**:

  * Context growth beyond preallocated bounds
  * KV cache layout changes
  * Backend selection changes
  * Tensor shape changes
  * Mode or flag toggles

If any invalidation condition occurs:

* Decode must stop
* Control returns to CPU
* Graph may be rebuilt **only outside** the decode-critical phase

---

### 12.5 CUDA Graph Usage (Decode-Critical)

* CUDA graphs are used to:

  * Capture the full decode graph
  * Eliminate per-kernel launch overhead
* CUDA graph replay is:

  * Persistent
  * GPU-resident
  * Not initiated per token by CPU
* CUDA graph boundaries do **not** introduce synchronization on the decode path

---

### 12.6 Graph Execution Flow (Corrected)

* Decode execution model:

  * GPU enters decode loop once
  * GPU executes graph iterations internally
  * GPU advances token index autonomously
* CPU is not involved in:

  * Per-token graph execution
  * Kernel dispatch
  * Completion checks
* Token-level ordering is enforced entirely on GPU

---

### 12.7 Graph Granularity (Required)

* Graph granularity is **decode-loop–level**, not token-trigger–level
* One persistent graph handles:

  * Multiple decode iterations
  * Internal token sequencing
* Eliminates per-token CPU↔GPU round trips

---

### 12.8 Graph Node Scheduling

* Node ordering is:

  * Static
  * Encoded in the graph
* Backend selection for nodes is:

  * Fixed
  * GPU-only
* GPU has full visibility into:

  * All nodes
  * All decode iterations
* CPU has no role in node scheduling during decode

---

### 12.9 Lifetime of Graph Resources

* All graph-associated resources are:

  * Allocated before decode
  * GPU-resident
  * Reused across all tokens
* No graph-related allocation, deallocation, or mutation occurs during decode

---

### 12.10 Optimization Implication (Final)

* Graph reuse is **necessary but insufficient** unless paired with:

  * GPU-autonomous execution
  * Persistent graph lifetime
* Maximum decode throughput requires:

  * Zero per-token CPU graph interaction
  * No graph invalidation during decode
  * GPU-controlled decode loop

This section defines the **mandatory invariant**:

> **The decode graph must outlive individual tokens and execute autonomously on the GPU for the entire decode phase.**

Without this invariant, CPU pacing and GPU idle gaps inevitably reappear.

## 13. Attention Path Analysis (Aligned with GPU-Exclusive Decode)

### 13.1 Role of Attention in Decode Phase

* Attention is the **dominant decode-critical operation** at long context lengths
* For each generated token:

  * Query corresponds to the current token
  * Keys and values correspond to all previous tokens
* Attention cost scales linearly with context length
* Attention lies **directly on the token-generation dependency chain**

---

### 13.2 Attention Execution Stages (Decode-Critical)

For each transformer layer during decode, the following stages occur in strict order:

1. Query, Key, Value projection
2. Attention score computation (query × all keys)
3. Scaling and causal masking
4. Softmax over sequence length
5. Weighted sum over values
6. Output projection

All stages are **decode-critical** and must execute **entirely on GPU**.

---

### 13.3 Backend Eligibility for Attention

* **CPU backend**:

  * Forbidden for decode
  * Introduces catastrophic latency at long context
* **CUDA dense / cuBLAS backends**:

  * GPU-resident
  * Acceptable only if fully GPU-exclusive
* **Flash-attention backend**:

  * Preferred and mandatory when supported
  * Provides maximal kernel fusion and minimal memory traffic

Backend choice must be **fixed before decode** and must not change across tokens.

---

### 13.4 Flash-Attention Requirement

* Flash-attention must be enabled when:

  * GPU supports required instructions
  * Attention dimensions are compatible
* Flash-attention properties:

  * Fused attention kernels
  * Reduced intermediate buffers
  * High arithmetic intensity
* Flash-attention is a **hard requirement** for sustained decode throughput at long context

---

### 13.5 KV Cache Interaction (Attention-Critical)

* Attention reads:

  * All previous keys and values from KV cache
* KV cache properties:

  * Fully GPU-resident
  * Sequential append per token
* CPU-resident KV cache is **forbidden** on the decode path
* KV cache access ordering is enforced by GPU execution

---

### 13.6 Kernel Granularity and Residency

* Attention kernels must be structured to maximize GPU residency:

  * Fused kernels
  * Persistent execution where possible
* Per-token attention execution must avoid:

  * Per-stage kernel launches
  * CPU-visible boundaries
* Kernel launch overhead must be amortized across decode iterations

---

### 13.7 Synchronization Semantics

* Attention execution must not introduce CPU↔GPU synchronization
* Ordering constraints are enforced by:

  * GPU execution dependencies
  * Graph structure
* CPU must not wait for attention completion
* Attention completion must directly feed the next decode stage on GPU

---

### 13.8 Scaling with Context Length (Target Behavior)

* As context length increases:

  * Attention compute per token increases
  * GPU kernel duration increases
* GPU utilization **naturally improves** with longer context
* CPU overhead remains **constant and non-pacing**

---

### 13.9 Attention as the Primary Throughput Lever

* Attention provides:

  * The largest per-token GPU workload
  * The greatest opportunity for increasing GPU occupancy
* Key levers:

  * Flash-attention
  * Kernel fusion
  * Persistent execution
  * Elimination of host intervention
* Improvements in attention directly translate to higher tokens/sec

---

### 13.10 Final Optimization Invariant

* For decode:

  * Attention must be **GPU-exclusive**
  * KV cache must be **GPU-resident**
  * Flash-attention must be **always selected when available**
  * No CPU orchestration or synchronization is permitted

Without enforcing these invariants, attention becomes the dominant source of GPU idle time and decode throughput collapses.

## 14. Quantization Cost Analysis (Aligned with GPU-Exclusive Decode)

### 14.1 Purpose of Quantization

* Quantization reduces:

  * Model memory footprint
  * Memory bandwidth requirements
* Enables larger models to fit within fixed VRAM limits
* Quantization does **not** alter model semantics or autoregressive behavior
* Quantization is a **capacity enabler**, not a control-path optimization

---

### 14.2 Quantization Formats (Decode-Relevant)

* Supported GGUF quantization formats include:

  * Q4, Q5, Q6, Q8
  * K-variants (Q4_K, Q6_K, etc.)
  * IQ and mixed formats
* Quantization granularity:

  * Block-based
  * Per-group scaling
* All formats require dequantization during compute
* Quantization format must remain **fixed across decode**

---

### 14.3 Dequantization Execution Policy (Hard Rule)

* **All decode-path dequantization must occur on GPU**
* CPU-side dequantization during decode is **forbidden**
* GPU-side dequantization must be:

  * Embedded inside compute kernels
  * Invisible to the CPU control path

Any CPU-visible dequantization immediately introduces a decode-critical bottleneck.

---

### 14.4 Dequantization Cost Characteristics

* Dequantization is:

  * Low arithmetic intensity
  * Memory-bound
* Cost per operation is small, but:

  * Repeated per layer
  * Repeated per token
* At batch size = 1, dequantization cost is dominated by:

  * Kernel launch overhead
  * Synchronization, not arithmetic

---

### 14.5 Interaction with Decode-Time GEMV

* Decode is GEMV-dominated
* Quantized GEMV kernels must perform:

  * Dequantization
  * Multiply–accumulate
* Kernel execution time per token is short
* Without fusion, launch overhead dominates total latency

---

### 14.6 Quantization and Kernel Fusion Requirement

* Quantization must be paired with **fused kernels**
* Required properties:

  * Dequantization + matmul in a single kernel
  * No intermediate buffers
  * No intermediate synchronization
* MMQ backend is preferred because it:

  * Maximizes fusion
  * Minimizes kernel count
  * Reduces memory traffic

Non-fused quantized paths are decode-hostile.

---

### 14.7 Quantization Impact on CPU Involvement

* Quantization does **not** reduce:

  * CPU sampling cost
  * CPU scheduling cost
  * CPU synchronization cost
* Quantization shifts arithmetic to GPU but leaves control-path unchanged
* Without architectural changes, CPU remains the decode pacing resource

Quantization **must not be misinterpreted** as a CPU offload mechanism.

---

### 14.8 Quantization Impact on GPU Utilization

* Quantization effects:

  * Reduces memory bandwidth pressure
  * Reduces arithmetic work per kernel
* Side effect:

  * Shorter kernel runtimes
  * Higher relative kernel launch overhead
* At batch size = 1, faster kernels can **reduce effective GPU utilization**

GPU underutilization is structural, not arithmetic.

---

### 14.9 Trade-Off Summary (Decode Phase)

* Quantization improves:

  * Model capacity
  * VRAM fit
* Quantization alone does **not** improve:

  * Decode throughput
  * GPU utilization
* For decode:

  * Arithmetic speedups are secondary
  * Control-path elimination is primary

---

### 14.10 Optimization Invariants for Quantized Decode

* For decode:

  * All quantized compute must be GPU-exclusive
  * Dequantization must be kernel-fused
  * No CPU-side quantization logic permitted
  * No backend fallback allowed
* Quantization must be paired with:

  * Reduced kernel count
  * Persistent GPU execution
  * Zero CPU involvement in decode-critical stages

Quantization is **necessary for scale**, but **insufficient for throughput** unless combined with GPU-exclusive execution and elimination of CPU orchestration.

## 15. Sampling Optimization Scope (Aligned with GPU-Exclusive Decode)

### 15.1 Role of Sampling in Decode

* Sampling determines the next token from model logits
* Sampling occurs once per generated token
* Sampling lies **directly on the decode-critical path**
* Decode **must not** proceed until sampling is complete
* Any CPU participation in sampling immediately makes CPU the pacing resource

---

### 15.2 Sampling Execution Invariant (Hard Rule)

> **All decode-path sampling must execute on GPU.**

* CPU-based sampling during decode is **forbidden**
* CPU may not:

  * Read logits
  * Apply penalties
  * Perform argmax
  * Perform filtering
* CPU must not observe intermediate sampling state

---

### 15.3 Current Sampling Pipeline (Baseline)

* Sampling stages typically include:

  * Logit bias application
  * Penalties (repeat, frequency, presence)
  * Temperature scaling
  * Top-k filtering
  * Top-p filtering
  * Final token selection
* These stages are:

  * Branch-heavy
  * Latency-sensitive
  * Serial
* When executed on CPU, they introduce a **hard synchronization barrier**

---

### 15.4 Sampling Cost Characteristics

* Sampling FLOPs are negligible
* Sampling latency impact is dominant because:

  * GPU must idle while CPU samples
  * Sampling gates the next decode step
* At batch size = 1:

  * Sampling latency directly caps tokens/sec

---

### 15.5 Determinism and Correctness Requirements

Sampling must preserve:

* Exact semantic equivalence to CPU implementation
* Determinism when configured (e.g., `temp = 0`)
* Correct handling of:

  * Penalties
  * Filters
  * Randomness (when enabled)
* Sampling result must be **final and committed** before next token decode

GPU execution must enforce these guarantees intrinsically.

---

### 15.6 GPU Suitability for Sampling

* Sampling operations map naturally to GPU primitives:

  * Reductions (argmax)
  * Elementwise transforms
  * Prefix sums
  * Comparisons
* Sampling kernels are:

  * Small
  * Deterministic
  * Easily fused
* GPU-based sampling eliminates:

  * Device→host transfer of logits
  * CPU-side control-path latency
  * Per-token synchronization barrier

---

### 15.7 Required Sampling Architecture

* Sampling must be:

  * GPU-resident
  * Graph-embedded
  * Executed as part of the decode graph
* Sampling output (token ID) must remain on GPU
* Token commitment and position advance must occur on GPU

CPU is notified **after** token commitment, asynchronously.

---

### 15.8 Incremental Migration Plan (Non-Speculative)

* Phase 1:

  * GPU argmax for deterministic sampling (`temp = 0`)
* Phase 2:

  * GPU penalty application
* Phase 3:

  * GPU top-k / top-p filtering
* Phase 4:

  * Fully GPU-resident stochastic sampling

Each phase **removes a decode-critical CPU dependency**.

---

### 15.9 Impact on CPU and GPU Utilization

* CPU utilization:

  * Drops sharply during decode
  * Becomes non-pacing
* GPU utilization:

  * Increases due to added per-token work
  * Eliminates idle gaps between kernels
* Tokens/sec increases due to:

  * Reduced per-token latency
  * Removal of synchronization barriers

---

### 15.10 Final Sampling Invariant

* For decode:

  * Sampling must be GPU-exclusive
  * CPU must not gate token progression
  * No logits or sampling data may cross to CPU
* Sampling optimization is **mandatory**, not optional, to achieve:

  * Stable tokens/sec
  * High GPU utilization
  * Elimination of CPU as decode bottleneck

Sampling is the **single highest-impact control-path optimization** once compute and memory are GPU-resident.

## 16. Server-Specific Overheads (Aligned with GPU-Exclusive Decode)

### 16.1 Server Execution Context (Reinterpreted)

* Server mode is a long-lived control plane
* Provides HTTP-based ingress/egress only
* Decode execution must be **logically and temporally isolated** from server control logic
* Server responsibilities must never intersect the decode-critical path

---

### 16.2 HTTP Request Handling (Non-Critical Only)

* CPU handles:

  * TCP connections
  * HTTP parsing
  * Request validation
* These operations are **pre-decode only**
* Once decode begins:

  * No HTTP parsing
  * No request mutation
  * No control-path interaction with decode
* HTTP handling must execute **entirely outside** the decode dependency chain

---

### 16.3 Slot Management (Admission-Time Only)

* Slot allocation and lifecycle management occur:

  * Before decode starts
  * After decode completes
* During decode:

  * Slot state must be immutable
  * No locking
  * No transitions
* Slot logic must not execute concurrently with decode-critical GPU execution

---

### 16.4 Streaming Response Logic (Asynchronous Only)

* Token streaming must be:

  * Asynchronous
  * Post-commit
  * Non-blocking
* Streaming operations may include:

  * Serialization
  * Network I/O
  * Buffer flushing
* Streaming must not:

  * Stall decode
  * Introduce synchronization
  * Delay next token generation
* GPU decode must proceed independently of client consumption rate

---

### 16.5 Logging and Metrics (Strictly Non-Critical)

* Logging and metrics collection are:

  * CPU-only
  * Latency-tolerant
* During decode:

  * Logging must be minimal or disabled
  * Metrics must be aggregated asynchronously
* No logging or metrics operation may block or preempt decode execution

---

### 16.6 Prompt Cache Management (Decode-External)

* Prompt cache lookup occurs:

  * Before decode
* Cache insertion occurs:

  * After decode
* Cache management must not:

  * Run during decode
  * Touch decode-resident memory
  * Interact with GPU state

---

### 16.7 Threading Model Separation

* Server threads and decode execution must be isolated:

  * Separate thread pools
  * Separate scheduling domains
* Server threads must not:

  * Preempt decode control
  * Interfere with GPU scheduling
* Decode execution must assume **exclusive access** to its required CPU control thread(s)

---

### 16.8 Decode Isolation Requirement

* Once decode begins:

  * Server logic becomes read-only observer
  * No server-side events may gate decode progression
* Decode loop must be immune to:

  * HTTP traffic
  * Slot management
  * Logging
  * Metrics
  * Streaming backpressure

---

### 16.9 Throughput Implications (Corrected)

* Server mode must not reduce tokens/sec relative to CLI mode
* Any throughput delta indicates:

  * Improper isolation
  * Decode-path CPU contamination
* Proper architecture yields:

  * Identical decode throughput
  * Independent control-plane overhead

---

### 16.10 Final Server Invariant

* **Server logic is control-plane only**
* **Decode execution is GPU-exclusive and control-plane isolated**
* Server responsibilities must never:

  * Block
  * Pace
  * Synchronize with
  * Or otherwise influence decode-critical execution

This section enforces the rule that **server flexibility must not compromise decode throughput**, ensuring GPU utilization and tokens/sec remain invariant regardless of server mode.

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

## 18. Build-Time Optimization Options (Aligned with GPU-Exclusive Decode)

### 18.1 Compiler Selection (Invariant)

* Use a **single, modern compiler toolchain** consistently:

  * `gcc` (latest stable) **or**
  * `clang` (latest stable)
* CUDA host compiler must match the chosen C/C++ compiler
* Mixed toolchains are **forbidden** due to ABI drift and backend inconsistency
* Compiler choice must be fixed and reproducible

---

### 18.2 Global Compiler Flags (Decode-Critical)

* Enable aggressive optimization for all targets:

  * `-O3`
  * `-ffast-math`
  * `-funroll-loops`
* Disable debug symbols in production builds
* Strip binaries after build to reduce:

  * I-cache pressure
  * Instruction fetch overhead
* No debug or instrumentation code may exist on the decode path

---

### 18.3 CPU Architecture Targeting (Control-Plane Only)

* Compile CPU code with:

  * `-march=native`
* Enables:

  * AVX2 / AVX-512
  * FMA
  * AMX (if present)
* **Important constraint**:

  * CPU optimizations apply **only** to non-decode-critical code
  * Decode-critical execution must not depend on CPU performance

CPU tuning is for control-plane efficiency, not throughput.

---

### 18.4 CUDA Architecture Targeting (Hard Requirement)

* Compile CUDA code for a **single explicit architecture**:

  * Example: `sm_89` (Ada Lovelace)
* Multi-arch fat binaries are **disallowed**
* Benefits:

  * Smaller binaries
  * Faster kernel dispatch
  * Predictable kernel selection
* CUDA architecture must match the deployment GPU exactly

---

### 18.5 CUDA Kernel Configuration (Decode-Critical)

* Enable all relevant CUDA optimizations:

  * Tensor Core usage
  * MMA pipelines
  * Fused kernels
* Disable:

  * Legacy kernels
  * Compatibility fallbacks
* MMQ kernels must be:

  * Compiled
  * Enabled
  * Preferred when quantization is used

Any missing kernel must fail build-time validation, not fall back at runtime.

---

### 18.6 Backend Compilation Policy (Strict)

* Compile **only** the backends required for the target execution model:

  * CPU backend (control-plane only)
  * CUDA dense backend **or**
  * CUDA MMQ / MoE backend
* Hybrid backend configurations are **forbidden**
* Every additional backend increases:

  * Selection logic
  * Fallback risk
  * Decode instability

Backend minimalism is a correctness requirement.

---

### 18.7 cuBLAS vs MMQ (Mutual Exclusivity)

* cuBLAS builds:

  * Allowed only for dense FP16/BF16 models
  * Prefill-optimized, decode-hostile
* MMQ builds:

  * Mandatory for quantized models
  * Mandatory for decode-heavy workloads
* **cuBLAS + MMQ in the same binary is forbidden** for decode-critical execution

Backend choice must be singular and final.

---

### 18.8 AMX and SIMD Enablement (Non-Critical)

* AMX and SIMD detection must be correct at build time
* Mis-detection causing scalar fallbacks is unacceptable
* These optimizations apply only to:

  * CPU preprocessing
  * Server logic
  * Non-decode workloads
* Decode correctness and throughput must not depend on AMX availability

---

### 18.9 Threading Runtime Configuration (Decode Isolation)

* Prefer pthread-based threading
* OpenMP is **disallowed** on decode-critical paths
* OpenMP may be used only if:

  * Strictly confined to non-decode tasks
  * Explicitly capped and isolated
* Decode execution must assume:

  * Minimal CPU thread usage
  * No thread oversubscription

---

### 18.10 Memory Allocation Strategy (Decode Invariant)

* Use aligned, preallocated buffers
* All decode-critical allocations must occur:

  * At initialization
  * Before first token
* No `malloc`, `free`, or allocator interaction is permitted during decode
* Custom allocators are allowed only if:

  * Allocation phase is strictly pre-decode

---

### 18.11 LTO and PGO (Control-Plane Only)

* LTO is recommended if build time permits
* PGO may improve:

  * Server control flow
  * Sampling (if still partially CPU-side during transition)
* LTO/PGO must not:

  * Introduce backend ambiguity
  * Alter kernel selection
* Decode-critical GPU execution must remain unaffected

---

### 18.12 Feature Elimination (Mandatory)

* Disable at build time:

  * Tests
  * Examples
  * Debug utilities
  * Profiling hooks
* Prevents:

  * Accidental linkage
  * Hidden slow paths
* Production binary must contain **only execution-relevant code**

---

### 18.13 Determinism Policy (Decode-Critical)

* Determinism requirements are enforced at the **algorithmic level**
* Build flags may allow:

  * Relaxed math
  * Non-associative reductions
* As long as:

  * Output semantics are preserved
  * Deterministic modes behave deterministically when enabled
* Determinism must not block kernel fusion or GPU residency

---

### 18.14 Linking Strategy

* Static vs shared linking is a deployment choice
* Decode throughput is unaffected if:

  * All code paths are resolved at load time
* Startup latency is secondary to sustained decode throughput
* Choose linking based on operational constraints

---

### 18.15 Post-Build Validation (Non-Negotiable)

A build is **invalid** unless all are verified:

* Single backend selected and locked
* No CPU backend invocation during decode
* No backend fallback paths reachable
* Sampling executes on GPU
* KV cache fully GPU-resident
* Tokens/sec benchmark meets expectation

> **A build that runs is not a correct build.
> A correct build is one that cannot violate decode invariants.**

This section establishes build-time configuration as a **hard enforcement layer** that prevents CPU re-entry into the decode-critical path by construction.

## 19. Minimal Code Change Targets (Aligned with GPU-Exclusive Decode)

### 19.1 Backend Forcing (Hard Selection)

* Force **CUDA-only backend selection** before decode begins
* Disallow runtime backend switching during decode
* Enforce that:

  * Decode graph is built with a single GPU backend
  * CPU backend is **never selectable** for decode-critical ops
* CPU backend may exist **only** for non-decode control-plane code
* Target files:

  * `ggml-backend-reg.cpp`
  * `ggml-backend.cpp`

---

### 19.2 Eliminate Silent CPU Fallbacks (Mandatory)

* Identify all ops that can silently fall back to CPU
* Replace silent fallback with:

  * Hard error, or
  * Explicit decode abort
* Any unsupported GPU op during decode must **fail fast**
* Mixed CPU↔GPU execution on decode path is forbidden

---

### 19.3 Backend Decision Caching

* Resolve backend selection **once per decode graph**
* Cache backend decisions at graph build time
* Remove repeated:

  * Capability checks
  * Virtual dispatch
* Backend resolution must not occur inside the decode loop
* Target:

  * `ggml-backend-impl.h`
  * `ggml-backend.cpp`

---

### 19.4 Graph Construction Freezing

* Build decode graph once, before decode starts
* Prohibit graph rebuild during decode
* Freeze:

  * Tensor shapes
  * Backend assignments
  * Memory layout
* Dynamic graph mutation during decode is forbidden
* Target:

  * `llama-graph.cpp`
  * `llama-context.cpp`

---

### 19.5 Kernel Fusion Enforcement

* Prefer fused CUDA kernels for sequential ops:

  * RMSNorm + MatMul
  * Bias + Activation
* Avoid emitting fine-grained ops that:

  * Increase kernel count
  * Increase synchronization
* Fusion decisions must be static and backend-specific
* CPU/GPU op boundaries are forbidden on decode path

---

### 19.6 CPU Bookkeeping Elimination

* Remove per-token CPU-side:

  * Tensor metadata updates
  * Shape validation
  * Layout checks
* Cache all tensor metadata after graph construction
* Decode loop must not touch tensor descriptors
* Target:

  * `ggml.c`
  * `ggml.cpp`

---

### 19.7 Synchronization Reduction (Decode-Critical)

* Remove all explicit `cudaDeviceSynchronize` calls on decode path
* Use stream-ordered execution only
* No CPU-visible synchronization per token
* Synchronization must be implicit and GPU-internal
* Target:

  * `ggml-cuda.cu`
  * CUDA backend wrappers

---

### 19.8 Sampling Path GPU Migration

* Treat sampling as decode-critical
* Incrementally move sampling to GPU:

  * Argmax first
  * Penalties
  * Top-k / top-p
* CPU must not:

  * Read logits
  * Gate next-token progression
* Target:

  * `sampling.cpp`
  * `top-k.cu`
  * `topk-moe.cu`

---

### 19.9 KV Cache GPU Residency Enforcement

* Enforce fully GPU-resident KV cache for decode
* Prohibit:

  * CPU-resident KV
  * CPU KV metadata updates per token
* KV cache layout must be frozen pre-decode
* Target:

  * `llama-kv-cache.cpp`
  * `llama-memory-hybrid.cpp`

---

### 19.10 Logging and Debug Path Removal

* Compile out logging from decode path
* Guard all debug checks behind compile-time flags
* No runtime logging conditionals allowed in hot paths
* Target:

  * `log.cpp`
  * `debug.cpp`

---

### 19.11 Thread Wake-Up Suppression

* Prevent per-token thread wake/sleep cycles
* Eliminate condition-variable churn during decode
* Decode loop must assume:

  * Fixed thread state
  * No scheduler interaction
* Target:

  * `ggml-threading.cpp`

---

### 19.12 Server Hot-Path Isolation

* Server logic must not execute on decode threads
* Remove per-token:

  * JSON serialization
  * Mutex locks
* Streaming must be asynchronous and non-blocking
* Target:

  * `server.cpp`
  * `server-task.cpp`
  * `server-queue.cpp`

---

### 19.13 One-Time Configuration Resolution

* Parse CLI / server flags once at startup
* Cache resolved configuration
* No argument or preset lookup during decode
* Target:

  * `arg.cpp`
  * `preset.cpp`

---

### 19.14 Compile-Time Feature Freezing

* Disable unused features at build time
* Remove runtime feature checks for disabled paths
* Reduces:

  * Branch misprediction
  * Control-path noise
* Target:

  * `common.cmake`
  * `ggml-config.cmake.in`

---

### 19.15 Validation Scope (Non-Negotiable)

After minimal changes, validate:

* CPU is not on decode dependency chain
* GPU utilization increases during decode
* Tokens/sec increases without variance
* No CPU backend invocation during decode
* Correctness and determinism preserved

> **Minimal code changes are acceptable only if they enforce decode invariants.
> Any change that preserves CPU pacing is insufficient.**

## 20. Expected Outcome Projection (Aligned with GPU-Exclusive Decode)

### 20.1 CPU Utilization (Revised)

* Decode-phase CPU utilization reduced from near-100% to **strictly non-pacing levels**
* CPU usage during decode is limited to:

  * Asynchronous notification handling
  * Control-plane bookkeeping
  * Output streaming (non-blocking)
* CPU is **never** responsible for:

  * Sampling
  * Graph execution
  * Backend dispatch
  * Tensor or KV bookkeeping
* CPU load may fluctuate, but **cannot affect tokens/sec**

---

### 20.2 GPU Utilization (Revised)

* GPU utilization remains **high, stable, and authoritative** during:

  * Prefill
  * Entire decode phase
* GPU execution is:

  * Continuous
  * Self-paced
  * Free of host-induced gaps
* No GPU idle time caused by:

  * CPU synchronization
  * Sampling barriers
  * Backend selection logic
  * Server-side interference

---

### 20.3 Tokens per Second (t/s)

* Decode throughput increases structurally, not heuristically
* Expected gains:

  * **1.5× – 2.5×** for single-sequence interactive decode
  * Higher gains at:

    * Long context lengths
    * Quantized MMQ builds
    * High-end GPUs
* Token latency variance is minimized due to elimination of CPU gating

---

### 20.4 Latency Characteristics

* Per-token latency becomes:

  * Predictable
  * GPU-dominated
* No long-tail stalls caused by:

  * CPU scheduling
  * Thread contention
  * Synchronization barriers
* Streaming output latency decoupled from decode execution

---

### 20.5 Memory Behavior

* All decode-critical memory remains GPU-resident
* Zero per-token host↔device transfers
* PCIe traffic during decode reduced to:

  * Asynchronous notifications only
* CPU cache pressure significantly reduced
* GPU memory layout remains stable across long-running decode sessions

---

### 20.6 Determinism & Correctness (Invariant)

* Exact autoregressive semantics preserved
* No speculative execution
* No token reordering
* Sampling semantics unchanged
* Deterministic configurations produce **bitwise-identical outputs**
* GPU execution enforces ordering intrinsically

---

### 20.7 Operational Stability

* Stable behavior under:

  * Long-running decode sessions
  * Maximum context lengths
  * Sustained load
* No regressions in:

  * Context growth
  * KV cache behavior
  * Server uptime
* Fewer failure modes due to removal of hybrid execution paths

---

### 20.8 Practical Success Criteria (Final)

All of the following must be true simultaneously:

* CPU is **not** on the decode dependency chain
* GPU is the sole pacing resource for token generation
* Decode-phase GPU utilization is consistently high
* Sustained tokens/sec is higher and more stable
* No correctness, determinism, or stability regressions

> **Success is defined not by higher CPU efficiency,
> but by the complete removal of CPU from decode-critical execution.**

## 21. Validation Method (Aligned with GPU-Exclusive Decode)

### 21.1 Baseline Capture (Decode-Focused)

* Run inference with the **current reference build**
* Measure **prefill** and **decode separately**
* Record during steady-state decode:

  * CPU utilization (per-core and total)
  * GPU utilization (% and SM activity)
  * Tokens per second (after warm-up)
* Tooling (non-intrusive):

  * `htop` / `perf stat`
  * `nvidia-smi dmon`
  * llama.cpp internal timing counters

Baseline establishes **CPU-paced decode behavior**.

---

### 21.2 Controlled Experiment Setup (Invariant)

All validation runs must use:

* Single active sequence (`n_seq_max = 1`)
* Fixed prompt
* Fixed random seed
* Fixed sampling configuration
* Fixed context size
* No speculative decoding
* No parallel or batched decode
* Identical runtime flags across runs

Any deviation invalidates comparison.

---

### 21.3 Stepwise Change Validation (Hard Gate)

For **each individual change** (build or code):

* Re-run the exact same workload
* Compare against baseline:

e.g.:

* CPU utilization change
* GPU utilization change
* Decode t/s change

Immediately **reject** the change if **any** of the following occur:

* Output differs (semantic mismatch)
* Determinism breaks
* CPU re-enters decode dependency chain
* CPU usage increases on decode path

---

### 21.4 Decode-Only Measurement Discipline

* Ignore prefill metrics after first confirmation
* Measure only:

  * Steady-state decode
  * ≥ 50 consecutive tokens
* Ensure:

  * GPU utilization does not dip between tokens
  * No per-token idle gaps caused by host waits

Decode behavior, not prefill, defines success.

---

### 21.5 Synchronization Stall Detection

* Enable CUDA debug instrumentation when available:

  * `GGML_CUDA_DEBUG=1`
* Inspect for:

  * Explicit `cudaDeviceSynchronize` calls
  * Host-side waits between kernel launches
  * Graph-level synchronization per token
* Validation requires:

  * Elimination or strict reduction of decode-path synchronization

Any remaining per-token host barrier is a failure.

---

### 21.6 CPU Fallback Detection (Mandatory)

* Verify backend logs and traces confirm:

  * No CPU backend execution during decode
* Validation fails if **any** decode-critical op executes on CPU, including:

  * Matmul
  * Attention
  * Sampling
  * KV updates

Silent fallback is treated as a correctness violation.

---

### 21.7 Memory Residency Verification

* Confirm all decode-critical data is GPU-resident:

  * Model weights
  * KV cache
  * Activations
  * Sampling state
* Monitor PCIe traffic:

  * No per-token device↔host transfers
* Any decode-phase host access to GPU data invalidates the result

---

### 21.8 Long-Run Stability Validation

* Run continuous generation for ≥ 10,000 tokens
* Observe:

  * GPU utilization stability
  * CPU utilization stability
  * Memory growth or leaks
* Reject if:

  * Performance degrades over time
  * GPU utilization drifts downward
  * CPU begins pacing decode

Stability is as important as peak throughput.

---

### 21.9 Final Acceptance Criteria (Non-Negotiable)

Validation passes **only if all conditions hold simultaneously**:

* CPU is not on the decode dependency chain
* GPU is the sole pacing resource during decode
* Decode-phase GPU utilization is sustained and stable
* Steady-state tokens/sec is higher than baseline
* Output is correct and deterministic
* No regressions under long-running decode

> **A change is valid only if it makes CPU irrelevant to decode throughput.**

---

### 21.10 Implementation Reference: Runtime Invariants

The objectives defined in this document are enforced via the following runtime mechanisms:

1.  **Node Tagging**: All tensors in the decode graph (`LLM_GRAPH_TYPE_DECODER`) are tagged with `GGML_TENSOR_FLAG_DECODE_CRITICAL`.
2.  **Strict Backend Ownership**: The `ggml_backend_sched` enforces that any node tagged as `DECODE_CRITICAL` must be assigned to a non-CPU (GPU) backend. Violation results in a fatal `GGML_ASSERT`.
3.  **Deterministic Scheduling**: Backend decisions are "frozen" after the first successful decode graph allocation to prevent runtime variance.
4.  **In-Graph Sampling**: Sampling operations (Argmax, Penalties) are integrated into the primary decode graph, ensuring they remain on the GPU and reside on the high-performance dependency chain.
