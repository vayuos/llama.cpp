# SystemAddOnChanges.md

## Phase II — MoE & Persistent Decode Extensions

### 1. Purpose of This Document

This document defines the **Phase II architectural extensions** to the system described in `systemchanges.md` .

Phase I established the foundational invariant:

> **All decode-critical computation must be GPU-exclusive, with zero CPU participation on the token-generation dependency chain.**

Phase II extends this invariant to cover:

* Mixture-of-Experts (MoE) execution
* Q8-class quantized expert models
* Persistent decode graph execution
* Elimination of structural GPU underutilization
* Full GPU autonomy across routing, scheduling, sampling, and token commitment

This document introduces only **additive constraints and structural reinforcements**.
No previously defined invariant is relaxed or modified.

---

### 2. Phase Boundary Definition

Phase I solved:

* CPU removal from decode pacing
* GPU-exclusive dense execution
* GPU-resident KV cache
* GPU-based sampling migration
* Static backend enforcement
* Deterministic decode semantics

Phase II addresses the remaining structural bottlenecks that persist in:

* MoE-based architectures
* Sparse expert execution paths
* High-precision Q8 quantized models
* Kernel-launch–dominated batch=1 decode workloads

Phase II assumes all Phase I guarantees are already enforced.

---

### 3. Architectural Scope

The additions defined in this file apply only to:

* Single-sequence decode (`n_seq_max = 1`)
* Batch size = 1 token during decode
* Long-running interactive sessions
* GPU-exclusive execution model
* Deterministic autoregressive semantics
* No speculative decoding
* No hybrid CPU↔GPU decode participation

No changes to:

* Model architecture
* Model weights
* Tokenization semantics
* Autoregressive ordering
* External API behavior

are permitted under this phase.

---

### 4. Primary Objective of Phase II

The objective of Phase II is to achieve:

* **Sustained maximum GPU residency during decode**
* Elimination of MoE-induced GPU bubbles
* Zero host orchestration inside decode loop
* Kernel-launch amortization across tokens
* Stable and load-invariant tokens-per-second (t/s)
* Deterministic GPU-side routing and sampling

The target end state is:

> **A fully GPU-autonomous decode engine in which routing, expert selection, attention, sampling, and token commitment execute inside a persistent device-resident graph without CPU synchronization per token.**

---

### 5. Performance Philosophy

Phase II is guided by a strict performance model:

1. Decode is latency-serial.
2. At batch=1, kernel launch overhead dominates arithmetic.
3. MoE multiplies small-kernel pressure.
4. Any CPU-visible boundary collapses GPU utilization.
5. Arithmetic improvements are secondary to control-path elimination.

Therefore:

* Control-path removal takes precedence over FLOP optimization.
* Kernel fusion takes precedence over micro-optimizing GEMV.
* Graph persistence takes precedence over backend diversity.
* Admission control takes precedence over hybrid execution.

---

### 6. Determinism Preservation

All Phase II changes must preserve:

* Bitwise-stable results within backend tolerance
* Stable expert routing order
* Stable tie-breaking behavior
* No race-dependent nondeterminism
* Strict token-by-token ordering

Performance gains must never alter decode semantics.

---

### 7. Enforcement Model

Phase II introduces additional enforcement layers:

* Compile-time elimination of forbidden paths
* Runtime admission checks for expert residency
* Hard-fail policy for unsupported GPU mappings
* Static graph freezing before decode begins

A system that “runs” but violates GPU autonomy is considered incorrect.

---

### 8. Relationship to Phase I

Phase I removed CPU from the decode path.

Phase II removes structural GPU underutilization caused by:

* Sparse expert execution
* Per-token routing overhead
* Kernel fragmentation
* Host-scheduled expert dispatch

Together, Phase I + Phase II define the complete decode engine architecture.

---

# TABLE OF CONTENTS

## Phase II — MoE & Persistent Decode Extensions

---

## 1. Introduction

### 1.1 Purpose of Phase II

### 1.2 Relationship to Phase I

### 1.3 Architectural Assumptions

### 1.4 Scope and Non-Scope

### 1.5 Terminology and Definitions

### 1.6 Determinism Preservation Statement

---

## 2. Phase II Objective Definition

### 2.1 Primary Throughput Objective

### 2.2 GPU Residency Objective

### 2.3 MoE Structural Objective

### 2.4 Persistent Graph Objective

### 2.5 Admission-Control Objective

### 2.6 Hard Invariants Introduced in Phase II

---

## 3. MoE Execution Model Extension

### 3.1 MoE Layer Characteristics

### 3.2 Routing on the Decode-Critical Path

### 3.3 Sparse Expert Activation Semantics

### 3.4 Deterministic Routing Requirements

### 3.5 Expert Execution Ordering Guarantees

### 3.6 Expert Isolation Model

---

## 4. GPU-Resident MoE Routing

### 4.1 Router Projection Requirements

### 4.2 Top-K Expert Selection on GPU

### 4.3 Deterministic Tie-Breaking Rules

### 4.4 Prohibited CPU Participation

### 4.5 Routing Kernel Fusion Requirements

### 4.6 Routing Memory Residency Constraints

---

## 5. Fused Router → Dispatch Architecture

### 5.1 Kernel Fusion Model

### 5.2 Dispatch Index Generation

### 5.3 Intermediate Tensor Elimination

### 5.4 Launch Minimization Strategy

### 5.5 Synchronization Elimination Requirements

---

## 6. GPU-Owned Expert Scheduling

### 6.1 Removal of CPU Expert Loops

### 6.2 Device-Side Conditional Execution

### 6.3 GPU Scheduling Authority

### 6.4 Expert Invocation Semantics

### 6.5 Enforcement of Host Isolation

---

## 7. Persistent Decode Graph Extension

### 7.1 Graph Lifetime Across Tokens

### 7.2 Embedding Experts Inside Persistent Graph

### 7.3 Conditional Expert Execution in Graph

### 7.4 Graph Replay Autonomy

### 7.5 Graph Invalidation Prohibitions

### 7.6 Decode Loop Ownership Model

---

## 8. Expert-Local KV Cache Architecture

### 8.1 Expert-Affine KV Layout

### 8.2 GPU-Resident KV Enforcement

### 8.3 KV Mutation Ordering

### 8.4 KV Isolation Across Experts

### 8.5 Metadata Elimination on CPU

### 8.6 Preallocation Requirements

---

## 9. Q8_K_XL Expert Kernel Specialization

### 9.1 GEMV-Dominated Decode Behavior

### 9.2 Fused Dequantization + GEMV

### 9.3 Tensor Core Utilization Strategy

### 9.4 Kernel Residency Optimization

### 9.5 Removal of Generic Quantized Fallbacks

### 9.6 Arithmetic vs Launch Overhead Analysis

---

## 10. Sparse Expert Optimization Layer

### 10.1 Empty Expert Elision

### 10.2 Zero-Weight Expert Skipping

### 10.3 Kernel Launch Coalescing

### 10.4 Expert Execution Reordering

### 10.5 GPU-Side Load Balancing

### 10.6 Cache Locality Optimization

---

## 11. Sampling Integration Inside MoE Graph

### 11.1 Sampling as Decode-Critical Node

### 11.2 Embedding Sampling into Persistent Graph

### 11.3 Removal of Device→Host Transfers

### 11.4 Deterministic GPU Sampling

### 11.5 Token Commitment on GPU

### 11.6 CPU Notification Model

---

## 12. Synchronization Elimination for MoE

### 12.1 Removal of Per-Token Host Barriers

### 12.2 Stream-Ordered Execution Model

### 12.3 Device-Only Dependency Enforcement

### 12.4 Prohibition of Mid-Graph Sync

### 12.5 Decode Boundary Guarantees

---

## 13. Admission Control & Residency Enforcement

### 13.1 Expert Weight Residency Verification

### 13.2 VRAM Capacity Validation

### 13.3 Hard-Fail Policy

### 13.4 Prevention of Hybrid Execution

### 13.5 Decode Eligibility Checks

---

## 14. Build-Time Enforcement Extensions

### 14.1 MoE-Only Backend Compilation Policy

### 14.2 Removal of Unused Backends

### 14.3 Compile-Time Stripping of CPU MoE Paths

### 14.4 Determinism Enforcement Flags

### 14.5 Kernel Availability Verification

---

## 15. Runtime Validation & Invariant Auditing

### 15.1 Decode-Critical CPU Invocation Audit

### 15.2 Expert Routing Validation

### 15.3 Persistent Graph Stability Check

### 15.4 GPU Residency Monitoring

### 15.5 Tokens/sec Stability Verification

### 15.6 Determinism Regression Testing

---

## 16. Performance Modeling for MoE Decode

### 16.1 Kernel Launch Dominance Model

### 16.2 Sparse Execution Cost Model

### 16.3 Q8 Arithmetic Intensity Model

### 16.4 Context-Length Scaling Behavior

### 16.5 Expected Throughput Envelope

### 16.6 Bottleneck Identification Framework

---

## 17. Server-Mode Isolation (MoE Context)

### 17.1 Expert Execution Isolation from Control Plane

### 17.2 Streaming Non-Blocking Guarantee

### 17.3 Logging Elimination on Decode Path

### 17.4 Slot Immutability During Decode

---

## 18. Memory Stability & Long-Run Behavior

### 18.1 Allocation Freeze Before Decode

### 18.2 Fragmentation Prevention

### 18.3 Expert Weight Pinning

### 18.4 Long-Context Stability

### 18.5 VRAM Pressure Monitoring

---

## 19. Failure Modes & Prohibited Behaviors

### 19.1 CPU Routing Fallback

### 19.2 Graph Rebuild During Decode

### 19.3 Hybrid Expert Execution

### 19.4 Device→Host Logit Transfers

### 19.5 Dynamic Backend Switching

### 19.6 Runtime Feature Mutation

---

## 20. Final Phase II Invariants

### 20.1 GPU-Autonomous Decode Statement

### 20.2 MoE Routing Exclusivity Statement

### 20.3 Persistent Graph Statement

### 20.4 Deterministic Execution Statement

### 20.5 No-CPU-Decode Guarantee

### 20.6 Throughput Determinism Guarantee

---

## 1. Introduction

### 1.1 Purpose of Phase II

Phase II extends the architectural guarantees established in Phase I by addressing structural decode bottlenecks specific to:

* Mixture-of-Experts (MoE) models
* High-precision quantized models (e.g., Q8_K_XL)
* Sparse expert activation patterns
* Kernel-launch–dominated batch=1 decode execution

Phase I removed CPU participation from the decode-critical path.
Phase II removes **structural GPU underutilization** that persists even after CPU elimination.

The objective of Phase II is to achieve:

* Sustained maximum GPU residency during decode
* Elimination of MoE-induced GPU idle gaps
* Fully GPU-autonomous routing, scheduling, and sampling
* Persistent decode graph execution across tokens
* Stable and load-invariant tokens-per-second (t/s)

Phase II is strictly additive. No Phase I invariant is relaxed.

---

### 1.2 Relationship to Phase I

Phase I established the foundational invariant:

> All decode-critical work has exactly one backend owner: the GPU.

Phase I guarantees:

* No CPU fallback on decode path
* GPU-resident KV cache
* GPU-based sampling
* Static backend selection
* No per-token CPU↔GPU synchronization

Phase II assumes all Phase I guarantees are already enforced and focuses on:

* MoE routing and expert execution autonomy
* Kernel fragmentation caused by sparse execution
* Persistent decode graph lifetime across tokens
* Elimination of host-scheduled expert orchestration

Phase I removed CPU pacing.
Phase II removes residual GPU structural inefficiencies.

Together, Phase I + Phase II define the complete decode execution model.

---

### 1.3 Architectural Assumptions

Phase II assumes the following execution environment:

* Single active decode sequence (`n_seq_max = 1`)
* Batch size = 1 during decode
* Long-running interactive session
* GPU-exclusive decode-critical execution
* No speculative decoding
* No hybrid CPU↔GPU decode execution
* Deterministic autoregressive semantics
* Persistent CUDA graph support available

Hardware assumptions:

* Discrete GPU with fixed VRAM
* All decode-critical layers and experts fit in GPU memory
* No unified or managed memory usage
* No dynamic weight paging

Phase II does not alter these assumptions.

---

### 1.4 Scope and Non-Scope

#### In Scope

* GPU-resident MoE routing
* Fused router → dispatch execution
* GPU-owned expert scheduling
* Persistent decode graph extensions
* Expert-local KV cache layout
* Q8_K_XL expert kernel specialization
* Sparse expert launch minimization
* Embedding sampling inside MoE graph
* Admission control for expert residency

#### Explicitly Out of Scope

* Changing model architecture
* Changing model weights
* Altering autoregressive semantics
* Token reordering
* Speculative decoding
* Multi-sequence batching
* Tensor parallelism
* CPU fallback paths
* API behavior changes

Phase II modifies execution placement and structure only.
Model semantics remain unchanged.

---

### 1.5 Terminology and Definitions

**Decode-Critical Path**
All computation required to generate the next token.

**GPU-Autonomous Execution**
Execution model in which decode progression occurs entirely on GPU without host synchronization per token.

**Persistent Decode Graph**
A device-resident execution graph that remains valid and replayable across all decode iterations.

**MoE Routing**
The process of selecting top-k experts for a given token during forward pass.

**Expert Scheduling**
Execution ordering and invocation of selected experts.

**Expert-Affine KV Cache**
KV layout where each expert maintains its own GPU-resident KV region without CPU metadata arbitration.

**Kernel Fragmentation**
Performance loss caused by excessive small kernel launches during batch=1 decode.

**Structural GPU Underutilization**
GPU idle time caused by execution structure rather than insufficient arithmetic workload.

**Admission Control**
Pre-decode validation ensuring that all decode-critical resources are GPU-resident and invariant-compliant.

---

### 1.6 Determinism Preservation Statement

All Phase II extensions must preserve:

* Exact autoregressive token dependency
* Stable expert routing decisions
* Deterministic tie-breaking rules
* Identical observable output for identical inputs (within backend FP tolerance)
* Stable execution under worst-case scheduling order

Parallel GPU routing and expert execution must not introduce:

* Race-dependent nondeterminism
* Backend-dependent behavioral divergence
* Token-order instability

Performance gains must not alter model semantics.

Phase II strengthens performance guarantees without modifying correctness guarantees.

## 2. Phase II Objective Definition

### 2.1 Primary Throughput Objective

The primary throughput objective of Phase II is to maximize sustained tokens-per-second (t/s) during batch=1 decode for MoE and Q8-class models, under the strict constraints defined in Phase I.

Throughput must:

* Be determined solely by GPU compute capability and memory bandwidth
* Remain stable under CPU load
* Remain stable under server control-plane activity
* Remain stable across long-running decode sessions
* Degrade only as a function of context length and model arithmetic

Throughput must not be limited by:

* CPU-side routing
* CPU expert scheduling
* Per-token kernel launch overhead
* Host-driven synchronization
* Graph rebuilds or backend churn

Phase II targets elimination of structural overhead that remains after CPU removal from the decode-critical path.

---

### 2.2 GPU Residency Objective

Phase II requires maximal GPU residency during decode.

GPU residency is defined as:

* Continuous device-side execution during decode iterations
* Minimal idle gaps between kernels
* No per-token host-driven stalls
* Persistent execution context across tokens

The GPU must:

* Own routing
* Own expert selection
* Own expert execution
* Own sampling
* Own token commitment
* Own progression to the next decode step

The CPU must not:

* Gate decode progression
* Observe intermediate decode state
* Schedule expert execution
* Synchronize per token

GPU utilization must be structurally enforced, not statistically observed.

---

### 2.3 MoE Structural Objective

MoE introduces structural inefficiencies due to:

* Sparse expert activation
* Increased kernel count
* Conditional execution paths
* Routing overhead

The MoE structural objective is to:

* Make routing fully GPU-resident
* Fuse router, selection, and dispatch
* Remove CPU-side expert orchestration
* Minimize expert kernel launch count
* Eliminate empty-expert execution
* Ensure expert execution remains inside persistent device execution

MoE must behave as a first-class GPU-native subsystem rather than a host-scheduled extension of dense execution.

Sparse execution must not reintroduce host visibility into the decode-critical path.

---

### 2.4 Persistent Graph Objective

Phase II formalizes the persistent decode graph as a hard architectural requirement.

The decode graph must:

* Be constructed before decode begins
* Remain valid for the entire decode session
* Contain all transformer layers and experts
* Contain sampling and token commitment
* Encode execution ordering entirely on device
* Be replayable without host intervention per token

Graph invalidation during decode is prohibited.

Graph replay must:

* Not require per-token CPU dispatch
* Not depend on host scheduling
* Not introduce synchronization boundaries

The decode loop must be logically device-owned.

---

### 2.5 Admission-Control Objective

Admission control ensures that decode begins only if structural invariants are satisfiable.

Before decode:

* All expert weights must be GPU-resident
* All decode-critical buffers must be preallocated
* KV cache must be GPU-resident
* Backend selection must be immutable
* Persistent graph must be constructible
* No CPU fallback paths may be reachable

If any condition fails:

* Decode must not start
* Execution must fail deterministically
* Hybrid execution is not permitted

Admission control prevents runtime structural violations.

---

### 2.6 Hard Invariants Introduced in Phase II

Phase II introduces the following non-negotiable invariants:

1. **GPU-Autonomous MoE Routing**
   All expert routing and selection execute exclusively on GPU.

2. **GPU-Owned Expert Scheduling**
   No CPU involvement in expert invocation or ordering.

3. **Persistent Device-Resident Decode Graph**
   Graph must outlive individual tokens and execute without per-token host interaction.

4. **Zero Per-Token Host Synchronization**
   No device→host barriers during decode-critical execution.

5. **Expert-Affine GPU-Resident KV Cache**
   KV mutation and visibility are device-enforced.

6. **No Hybrid Execution Under Any Condition**
   Unsupported GPU mapping results in hard failure.

7. **Deterministic GPU Routing and Execution**
   Parallel device execution must preserve deterministic behavior.

8. **Allocation Freeze Before Decode**
   No memory allocation or layout mutation during decode.

These invariants extend Phase I from:

> GPU-exclusive decode

to:

> Fully GPU-autonomous MoE decode with persistent execution and structurally enforced maximal residency.

## 3. MoE Execution Model Extension

### 3.1 MoE Layer Characteristics

Mixture-of-Experts (MoE) layers replace the dense feed-forward block with a sparse expert selection mechanism.

An MoE layer consists of:

* A router projection producing expert scores
* Top-k expert selection per token
* Conditional execution of selected experts
* Aggregation of expert outputs

Decode-phase characteristics:

* Batch size = 1
* Top-k typically small relative to total experts
* Expert activation varies per token
* Arithmetic intensity per expert is lower than dense MLP
* Kernel count increases relative to dense layers

Structural implications:

* Routing lies on the decode-critical path
* Expert invocation is conditional and token-dependent
* Sparse activation multiplies kernel-launch pressure
* Poor scheduling creates GPU idle gaps

MoE layers must be treated as decode-critical compute units equivalent in importance to attention.

---

### 3.2 Routing on the Decode-Critical Path

Routing determines which experts are executed for the current token.

Routing includes:

* Router linear projection
* Score normalization (if applicable)
* Top-k selection
* Tie-breaking

Routing is decode-critical because:

* Expert selection directly influences token logits
* Next-token decode cannot begin until routing completes
* Expert scheduling depends on routing output

Therefore:

* Routing must be GPU-exclusive
* No CPU inspection of routing scores
* No host-visible expert masks
* No device→host routing metadata transfer

Routing completion must feed expert execution directly within device context.

---

### 3.3 Sparse Expert Activation Semantics

Sparse expert activation implies:

* Only k experts (k ≪ total experts) execute per token
* Activation set varies per token
* Some experts may remain unused for long spans

Sparse semantics must satisfy:

* Exact mathematical equivalence to dense MoE definition
* No approximation or pruning beyond defined top-k
* Stable output aggregation

Structural constraints:

* Empty or zero-weight experts must not incur kernel launches
* Sparse execution must not introduce host conditionals
* Expert selection must not require CPU arbitration

Sparse activation must reduce compute without introducing structural stalls.

---

### 3.4 Deterministic Routing Requirements

GPU-parallel routing must preserve determinism.

Determinism requirements:

* Identical expert selection for identical inputs
* Stable tie-breaking rules
* No race-dependent top-k behavior
* No backend-dependent selection divergence

Top-k selection must:

* Use a deterministic algorithm
* Avoid nondeterministic atomic patterns
* Guarantee stable ordering of equal scores

Parallel execution must not affect observable routing decisions.

Routing nondeterminism is considered a correctness violation.

---

### 3.5 Expert Execution Ordering Guarantees

Expert execution ordering must satisfy:

* Semantic equivalence regardless of execution order
* Stable aggregation behavior
* Deterministic output accumulation

GPU may reorder expert execution internally for efficiency, provided:

* All selected experts execute exactly once
* Aggregation is associative and deterministic
* Final combined output matches reference semantics

Prohibited behaviors:

* CPU-imposed expert loops
* Order-dependent floating-point accumulation without control
* Execution paths dependent on host scheduling

Execution ordering must be device-controlled and invariant-preserving.

---

### 3.6 Expert Isolation Model

Each expert must be logically isolated during decode.

Isolation requirements:

* Expert weights are GPU-resident
* Expert compute buffers are GPU-resident
* Expert KV interactions (if any) are device-enforced
* No shared host-managed metadata per token

Isolation prevents:

* CPU arbitration between experts
* Cross-expert synchronization on host
* Host-driven expert dispatch

Experts must function as independent GPU execution units embedded inside the persistent decode graph.

The MoE layer, as a whole, must behave as:

> A device-resident conditional compute block with no host-visible intermediate state.

This extension formalizes MoE as a first-class GPU-native component in the decode-critical architecture.
