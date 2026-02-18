#pragma once

#include "ggml-cpp.h"
#include "ggml-opt.h"
#include "llama-adapter.h"
#include "llama-cparams.h"
#include "llama-decode-admission-control.h"
#include "llama-decode-boundary-enforce.h"
#include "llama-decode-cpu-hard-failure.h"
#include "llama-decode-invariant-enforce.h"
#include "llama-task-taxonomy.h"
#include "llama-token-dependency-assert.h"
#include "llama-backend-immutability-enforce.h"
#include "llama-graph-backend-binding.h"
#include "llama-fallback-elimination.h"
#include "llama-cuda-support-enforce.h"
#include "llama-decode-backend-lock.h"
#include "llama-graph-freeze-enforce.h"
#include "llama-decode-rebuild-prohibition.h"
#include "llama-graph-backend-cache.h"
#include "llama-graph-schedule-elimination.h"
#include "llama-token-persistent-execution.h"
#include "llama-decode-loop-elimination.h"
#include "llama-token-step-gating-elimination.h"
#include "llama-sampling-elimination.h"
#include "llama-kvcache-elimination.h"
#include "llama-tensor-metadata-elimination.h"
#include "llama-greedy-sampling-gpu.h"
#include "llama-penalty-gpu.h"
#include "llama-topk-gpu.h"
#include "llama-topp-gpu.h"
#include "llama-logits-gpu.h"
#include "llama-token-selection-authority.h"
#include "llama-kvcache-position-gpu.h"
#include "llama-context-position-gpu.h"
#include "llama-kv-metadata-gpu.h"
#include "llama-kv-layout-freeze.h"
#include "llama-transfer-prohibition-gpu.h"
#include "llama-token-buffer-gpu.h"
#include "llama-attention-state-gpu.h"
#include "llama-kv-slice-operations-gpu.h"
#include "llama-hybrid-kv-elimination.h"
#include "llama-decode-sync-elimination.h"
#include "llama-stream-ordering-enforce.h"
#include "llama-tensor-allocation-gpu.h"
#include "llama-tensor-metadata-gpu.h"
#include "llama-rnorm-matmul-fusion.h"
#include "llama-bias-activation-fusion.h"
#include "llama-mmq-enforcement.h"
#include "llama-kernel-fusion-enforce.h"
#include "llama-threading-discipline.h"
#include "llama-topology-freeze.h"
#include "llama-oversubscription-control.h"
#include "llama-decode-path-isolation.h"
#include "llama-server-decode-isolation.h"
#include "llama-json-isolation.h"
#include "llama-streaming-async.h"
#include "llama-decode-mutex-elimination.h"
#include "llama-decode-logging-disable.h"
#include "llama-config-freeze.h"
#include "llama-feature-freeze.h"
#include "llama-decode-probing-removal.h"
#include "llama-backend-disable.h"
#include "llama-debug-stripping.h"
#include "llama-mmq-backend-enforcement.h"
#include "llama-cublas-fallback-prevention.h"
#include "llama-quantization-format-freeze.h"
#include "llama-cpu-dequantization-elimination.h"
#include "llama-fused-kernel-coverage.h"
#include "llama-decode-allocation-freeze.h"
#include "llama-decode-buffer-freeze.h"
#include "llama-gpu-allocation-alignment.h"
#include "llama-host-access-prevention.h"
#include "llama-gpu-memory-fragmentation-monitor.h"
#include "llama-decode-cpu-execution-detector.h"
#include "llama-backend-audit-log.h"
#include "llama-gpu-utilization-probe.h"
#include "llama-pcie-traffic-watchdog.h"
#include "llama-decode-stability-harness.h"
#include "llama-decode-acceptance-criteria.h"
#include "llama-graph.h"
#include "llama.h"

#include <map>
#include <mutex>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0;  // memory allocated for the model
    size_t context = 0;  // memory allocated for the context
    size_t compute = 0;  // memory allocated for temporary compute buffers

    size_t total() const { return model + context + compute; }
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(const llama_model & model, llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    // [STRICT] Graph Persistence: Track active graph during decode
    // Valid only when is_decode_active is true.
    struct ggml_cgraph * active_decode_graph = nullptr;

    // [STRICT] Decode Boundary Enforcement: Prevents CPU↔GPU op boundary splitting
    // Enforces all decode ops execute on GPU with no intermediate tensor transfers
    llama_decode_boundary_state decode_boundary = {};

    // [STRICT] GPU-Exclusive Decode Invariant (Section 1)
    // Enforces GPU-exclusive execution: any operation gating next-token emission MUST be GPU-only
    // CPU is permanently forbidden from token-generation dependency chain
    struct llama_decode_invariant decode_invariant = {};

    // [STRICT] Task Taxonomy State (Section 2)
    // Implements exhaustive task classification system: DECODE_CRITICAL (GPU-only) vs NON_CRITICAL (CPU-only)
    // All work must be classified statically, explicitly, and irreversibly before execution
    // DECODE_CRITICAL tasks gate next-token emission; NON_CRITICAL tasks can scale independently
    struct {
        bool taxonomy_enabled;
        int total_tasks;
        int decode_critical_tasks;
        int non_critical_tasks;
    } task_taxonomy_state = {false, 0, 0, 0};

    // [STRICT] Decode Admission Control (Section 3)
    // Strict decode admission gate that allows execution to begin ONLY if GPU eligibility is fully satisfied
    // Decode never starts in hybrid or degraded mode. Failure is immediate and final.
    // Five exhaustive criteria: GPU backend, decode-critical ops GPU-bound, CUDA features, KV cache GPU-resident, backend frozen
    llama_decode_admission_control decode_admission = {};

    // [STRICT] Hard Failure on Decode-Critical CPU Execution (Section 4)
    // CPU execution on the decode-critical path is a fatal error, not a fallback option.
    // Any attempt to execute decode-critical work on CPU causes immediate hard failure.
    // Enforcement at: backend dispatch, kernel dispatch, graph execution, sampling, node execution
    struct {
        bool strict_enforcement_enabled;     // True = abort on violation, False = log but continue (testing)
        int cpu_violation_count;             // Count of detected CPU execution violations
    } decode_cpu_enforcement_state = {true, 0};

    // [STRICT] Token Dependency Chain Runtime Assertion (Section 5)
    // Verifies at runtime that CPU is never part of the dependency chain that gates token emission.
    // Token chain: Entry → Forward → Attention/MLP → KV Cache → Logits → Sampling → Commit
    // All stages must execute on GPU. CPU presence is a fatal invariant violation.
    struct {
        bool assertions_enabled;             // True = enable runtime checks
        bool in_decode_phase;               // True = currently in token-by-token decode phase
        uint64_t current_token_id;          // ID of token being processed
        int assertion_count;                // Count of tokens checked
    } token_chain_assert_state = {true, false, 0, 0};

    // [STRICT] Backend Immutability Enforcement (Section 6)
    // Eliminates all runtime backend switching during decode phase.
    // Backend selection is frozen before first decode token and remains immutable.
    // No per-token, per-layer, or per-operation backend re-evaluation allowed.
    // Backend changes trigger immediate failure.
    llama_backend_immutability_state backend_immutability = {};

    // [STRICT] Graph Backend Binding Enforcement (Section 7)
    // Enforces single backend binding at decode graph construction.
    // Backend ownership is fixed at graph build time before first token.
    // No per-node overrides permitted. CPU fallback impossible by design.
    llama_graph_backend_binding_record graph_binding = {};

    // [STRICT] Fallback Elimination (Section 8)
    // Eliminates all silent CPU backend fallbacks on decode path.
    // Any fallback from GPU to CPU is treated as fatal error, not recovery.
    // Decode-critical execution is GPU-only by construction.
    llama_fallback_elimination_state fallback_elimination = {};

    // [STRICT] CUDA Support Enforcement (Section 9)
    // Converts unsupported CUDA ops into hard decode errors.
    // Any decode-critical operation lacking CUDA support results in immediate hard error.
    // CPU fallback for unsupported ops is forbidden. All violations detected upfront.
    llama_cuda_support_validation_state cuda_support_validation = {};

    // [STRICT] Decode-Time Backend Lock (Section 10)
    // Guarantees backend ownership cannot change for entire duration of decode.
    // Once decode begins, backend selection is immutable until decode terminates.
    // Backend switching due to memory, capability, or state changes is structurally impossible.
    llama_backend_lock_validation_state backend_lock_validation = {};

    // [STRICT] Graph Freeze Enforcement (Section 11)
    // Ensures decode graph is constructed once, validated, and frozen before decode begins.
    // No structural graph changes permitted during decode (node add/remove, reorder, backend reassign).
    // Graph construction cost eliminated from hot path. Backend and execution structure stable.
    llama_graph_freeze_validation_state graph_freeze_validation = {};

    // [STRICT] Decode Rebuild Prohibition (Section 12)
    // Completely forbids graph rebuilds once decode has started.
    // Any attempt to rebuild, invalidate, or regenerate graph during decode is fatal error.
    // Graph immutability guaranteed for entire decode session. CPU cannot re-enter via rebuild.
    llama_rebuild_prohibition_validation_state rebuild_prohibition_validation = {};

    // [STRICT] Backend Cache Enforcement (Section 13)
    // Ensures backend selection is resolved once at graph build time and cached permanently.
    // Zero runtime backend re-evaluation or dynamic dispatch during decode.
    // Backend decisions are immutable, deterministic, and remove decision overhead from hot path.
    llama_graph_backend_cache_validation_state backend_cache_validation = {};

    // [STRICT] Graph Schedule Elimination (Section 14)
    // Eliminates all per-token graph scheduling and traversal logic from CPU during decode.
    // Decode graph executes as predefined, fixed execution plan computed once at build time.
    // Dynamic traversal, readiness checks, and topological sorts forbidden during decode.
    llama_graph_schedule_elimination_validation_state schedule_elimination_validation = {};

    // [STRICT] Token-Persistent Execution Model (Section 15)
    // Enforces single decode graph instance that persists across all tokens without CPU re-entry.
    // GPU owns long-lived execution context; graph lifetime exactly matches decode lifetime.
    // CPU does not rebuild, resubmit, or orchestrate graph per token.
    llama_token_persistent_execution_validation_state token_persistent_execution_validation = {};

    // [STRICT] Decode Loop Elimination (Section 16)
    // Eliminates CPU as owner of decode-loop progression. GPU drives decode loop autonomously.
    // CPU reduced to non-blocking initiator and signal observer.
    // No CPU per-token iteration, counter advancement, or loop ownership.
    llama_decode_loop_elimination_validation_state decode_loop_elimination_validation = {};

    // [STRICT] CPU Token-Step Gating Elimination (Section 17)
    // CPU cannot make conditional decisions about token progression or advancement.
    // All "can-proceed" checks, readiness barriers, and gating conditions eliminated from CPU.
    // GPU autonomously determines token steps via completion signals; CPU is signal observer only.
    llama_token_step_gating_validation_state decode_step_gating_elimination_validation = {};

    // [STRICT] CPU Sampling Elimination (Section 18)
    // CPU cannot invoke sampler, modify sampling parameters, or apply sampling logic during decode.
    // All sampling operations (temperature, top-k, top-p, penalties) must be GPU-resident.
    // Sampling becomes GPU-autonomous with CPU as signal observer only.
    llama_sampling_elimination_validation_state sampling_elimination_validation = {};

    // [STRICT] CPU KV-Cache Mutation Elimination (Section 19)
    // CPU cannot mutate KV-cache state, update offsets, or expand cache during decode.
    // All KV cache management becomes GPU-resident and GPU-autonomous.
    // KV cache becomes GPU-managed with CPU as read-only observer only.
    llama_kvcache_elimination_validation_state kvcache_elimination_validation = {};

    // [STRICT] CPU Tensor Metadata Elimination (Section 20)
    // CPU cannot update tensor metadata, shapes, or descriptors per-token during decode.
    // All tensor shapes and layouts are frozen before decode; no per-token metadata mutations.
    // Tensor metadata becomes immutable with GPU handling all positional variability.
    llama_tensor_metadata_elimination_validation_state tensor_metadata_elimination_validation = {};

    // [STRICT] Greedy Sampling GPU Execution (Section 21)
    // Greedy sampling (temperature=0) is GPU-exclusive. CPU sampling entry points bypassed.
    // Logits remain GPU-resident. GPU argmax kernel produces selected token on device.
    // No CPU involvement in greedy token selection; device-resident until final commit.
    llama_greedy_sampling_gpu_validation_state greedy_sampling_gpu_validation = {};

    // [STRICT] GPU Penalty Application (Section 22)
    // Enforces GPU-exclusive penalty computation: all repeat, frequency, presence penalties GPU-only
    // Token history stays GPU-resident; CPU performs zero penalty computation
    llama_gpu_penalty_validation_state penalty_gpu_validation = {};

    // [STRICT] GPU Top-K Filtering (Section 23)
    // Enforces GPU-exclusive top-k filtering: partial sorting, masking, candidate selection all GPU-only
    // Candidates stay GPU-resident; CPU performs zero top-k computation
    llama_gpu_topk_validation_state topk_gpu_validation = {};

    // [STRICT] GPU Top-P (Nucleus) Filtering (Section 24)
    // Enforces GPU-exclusive top-p filtering: softmax, sorting, cumsum, masking all GPU-only
    // Probabilities stay GPU-resident; CPU performs zero nucleus filtering computation
    llama_gpu_topp_validation_state topp_gpu_validation = {};

    // [STRICT] GPU-Exclusive Logits Access (Section 25)
    // Enforces phase-aware logits access control: logits GPU-resident during decode, CPU reads forbidden
    // CPU operations (get_data, backend_tensor_get, CPU buffer views) blocked during decode phase
    // Only selected token ID crosses PCIe; no logits arrays or probability vectors transferred to CPU
    llama_gpu_logits_validation_state logits_gpu_validation = {};

    // [STRICT] GPU-Only Token Selection Authority (Section 26)
    // Enforces GPU-exclusive token selection: all sampling decisions (penalties, filtering, argmax) on GPU
    // CPU sampling entry points forbidden; CPU may only observe committed token from GPU state
    // Sampling authority locked to GPU; no CPU participation in token selection during decode
    llama_gpu_token_selection_validation_state token_selection_gpu_validation = {};

    // [STRICT] GPU-Exclusive KV-Cache Position Tracking (Section 27)
    // Enforces GPU-resident position updates: position stays on GPU during decode
    // CPU cannot increment, update, or validate position; only read-only access allowed
    // Position locked to GPU; no CPU position modifications permitted during decode
    llama_gpu_kvcache_position_validation_state kvcache_position_gpu_validation = {};

    // [STRICT] GPU-Exclusive Context Position Tracking (Section 28)
    // Enforces GPU-exclusive n_past tracking: context position stays on GPU during decode
    // CPU cannot update, compare, or use n_past for gating during decode
    // Context position locked to GPU; only read-only access to current n_past allowed
    llama_gpu_context_position_validation_state context_position_gpu_validation = {};

    // [STRICT] GPU-Exclusive KV Metadata Tracking (Section 29)
    // Enforces GPU-only KV metadata: positions, offsets, validity all GPU-owned during decode
    // CPU cannot track, maintain, or validate KV metadata during decode
    // All KV mutations occur inside GPU kernels; CPU observes final KV state only
    llama_gpu_kv_metadata_validation_state kv_metadata_gpu_validation = {};

    // ===== Section 30: Prohibit Per-Token Host↔Device Transfers =====
    // Enforce transfer prohibition: no decode-critical tensor may cross PCIe during decode
    // Only final selected token ID permitted to cross PCIe per token
    // All buffers (logits, KV cache, sampling, attention) GPU-resident and persistent
    // All operations single-stream; only implicit sync allowed after token selection
    llama_gpu_transfer_prohibition_validation_state transfer_prohibition_gpu_validation = {};

    // [STRICT] Immutable KV Cache Layout (Originally Section 30, renumbered to integrate comprehensive transfer prohibition)
    // Enforces KV layout freeze: layout determined before decode and immutable during decode
    // CPU cannot resize, repartition, or adjust KV cache during decode
    // GPU operates on fixed KV layout for entire decode session; no runtime reconfiguration
    llama_kv_layout_freeze_validation_state kv_layout_freeze_validation = {};

    // ===== Section 31: Eliminate Host-Side Token Buffering =====
    // CPU cannot queue, enqueue, dequeue, or inspect token buffer during decode
    // Token queues (input/output) are GPU-resident; CPU cannot maintain token buffers
    // All token buffering operations (enqueue/dequeue) occur inside GPU kernels
    // CPU observes final buffer state only; no intermediate buffer state visible
    llama_gpu_token_buffer_validation_state token_buffer_gpu_validation = {};

    // ===== Section 32: Enforce GPU-Only Attention State Management =====
    // CPU cannot maintain, track, or validate attention state during decode
    // Attention state (query/key/value heads, attention scores) is GPU-resident
    // All attention computation and state mutations occur inside GPU kernels
    // CPU observes final attention state only; no intermediate state visible
    llama_gpu_attention_state_validation_state attention_state_gpu_validation = {};

    // ===== Section 33: GPU-Exclusive KV-Cache Slice Operations =====
    // CPU cannot perform KV cache slicing (row selection, range extraction, view creation)
    // All KV slice operations (row/range/view selection) are GPU-resident
    // All slice operations occur inside GPU kernels; CPU observes final slice state only
    llama_gpu_kv_slice_validation_state kv_slice_gpu_validation = {};

    // ===== Section 31: Eliminate Hybrid KV Cache Modes =====
    // Enforce GPU-only KV cache for decode phase
    // Hybrid KV cache modes (CPU+GPU split) forbidden during decode
    // All layers must have GPU-resident KV before decode begins
    // Per-layer CPU/GPU branching eliminated; single GPU backend enforced
    llama_gpu_kv_hybrid_elimination_validation_state hybrid_kv_elimination_validation = {};

    // ===== Section 32: Remove Decode-Path cudaDeviceSynchronize Calls =====
    // Eliminate global device synchronization from decode critical path
    // Replace with stream-ordered, GPU-driven execution model
    // Single dedicated decode CUDA stream enforced; CUDA events for final token only
    // No implicit syncs from host access; CPU blocks removed from GPU execution
    llama_gpu_sync_elimination_validation_state sync_elimination_validation = {};

    // Strict single-stream decode execution; all decode-critical ops in one CUDA stream
    // No cross-stream dependencies; no default stream usage; explicit stream binding required
    llama_gpu_stream_ordering_validation_state stream_ordering_validation = {};

    // All decode-critical tensors pre-allocated on GPU before decode begins
    // No host-side allocation permitted during decode phase; tensors pre-sized and reserved
    llama_gpu_tensor_allocation_validation_state tensor_allocation_validation = {};

    // Tensor metadata (shape, strides, type, buffer) immutable and GPU-resident during decode
    // No CPU tensor introspection, shape queries, or metadata modifications permitted in decode phase
    llama_gpu_tensor_metadata_validation_state tensor_metadata_validation = {};

    // RMSNorm + MatMul must fuse into single GPU kernel during decode; no separate execution
    // Normalized vector stays in register/shared memory; no intermediate materialization
    llama_gpu_fusion_validation_state fusion_validation = {};

    // Bias addition + activation must fuse into single GPU kernel during decode
    // Biased tensor stays device-local; no intermediate materialization or host sequencing
    llama_gpu_bias_act_fusion_validation_state bias_act_fusion_validation = {};

    // [STRICT] Quantized MatMul Enforcement via MMQ (Section 39)
    // All quantized decode matmuls must execute via MMQ fused CUDA kernels
    // No backend ambiguity, no CPU fallback, no cuBLAS path during decode
    llama_mmq_enforcement_state_t * mmq_enforcement_state = nullptr;

    // [STRICT] Kernel Fusion Enforcement (Section 41)
    // Minimizes CUDA kernel launches per token through aggressive fusion
    // Objective: execution density, not kernel speed
    llama_kernel_fusion_state kernel_fusion_state = {};

    // [STRICT] Threading Discipline Enforcement (Section 42)
    // Eliminates per-token thread wake/sleep cycles and synchronization churn
    // Decode workers must remain persistent with no condition variable signaling
    llama_threading_discipline_state threading_discipline = {};

    // [STRICT] Decode Topology Freeze Enforcement (Section 43)
    // Locks complete thread topology before decode loop enters
    // No thread creation, destruction, resizing during decode
    llama_topology_freeze_state topology_freeze = {};

    // [STRICT] Oversubscription Control (Section 45)
    // Prevents excessive CPU thread activation during decode
    // Minimal threads only: dispatch + optional orchestration
    llama_oversubscription_control oversubscription_control = {};

    // [STRICT] Decode Path Isolation (Section 46)
    // Eliminates all per-token thread pool interactions
    // No submissions, wake-sleep, or work-stealing during decode
    llama_decode_path_isolation_state decode_path_isolation = {};

    // [STRICT] Server-Decode Thread Isolation (Section 47)
    // Separates server threads from decode threads on dedicated CPU cores
    // Prevents server load from affecting decode performance
    std::unique_ptr<decode_isolation_engine> server_decode_isolation = nullptr;

    // [STRICT] JSON Serialization Isolation (Section 48)
    // Eliminates per-token JSON construction from decode loop
    // Decode produces minimal token records, server handles serialization
    // std::unique_ptr<json_isolation_engine> json_isolation = nullptr;

    // [STRICT] Asynchronous Streaming Decoupling (Section 49)
    // Completely separates decode and streaming execution domains
    // Lock-free queue for tokens, zero blocking in decode
    std::unique_ptr<async_streaming_engine> async_streaming = nullptr;

    // [STRICT] Decode Hot Path Mutex Elimination (Section 50)
    // Removes all mutex acquisitions from decode-critical path
    // Enforces single-owner model with lock-free synchronization
    std::unique_ptr<mutex_elimination_engine> mutex_elimination = nullptr;

    // [STRICT] Decode Logging Suppression (Section 51)
    // Disables all logging during decode-critical window
    // Ensures no log emission, formatting, or mutex acquisition in decode
    std::unique_ptr<decode_logging_suppression_engine> decode_logging_suppression = nullptr;

    // [STRICT] Configuration Freeze (Section 52)
    // Resolves all flags at startup, immutable during decode
    // Eliminates runtime flag evaluation from decode loop
    std::unique_ptr<config_freeze_engine> config_freeze = nullptr;

    // [STRICT] Feature Freeze (Section 53)
    // Freezes all feature flags at build time
    // Eliminates runtime feature branching from decode path
    std::unique_ptr<feature_freeze_engine> feature_freeze = nullptr;

    // [STRICT] Decode Probing Removal (Section 54)
    // Removes all runtime capability probing from decode
    // Validates capabilities at startup, locks them immutable
    std::unique_ptr<probing_removal_engine> probing_removal = nullptr;

    // [STRICT] Backend Disable at Build (Section 55)
    // Disables all unused backends at compile-time
    // Enforces single GPU backend, eliminates runtime dispatch
    std::unique_ptr<backend_disable_engine> backend_disable = nullptr;

    // [STRICT] Debug Stripping (Section 56)
    // Removes all debug and tracing instrumentation from decode
    // Ensures zero diagnostic branching in critical path
    std::unique_ptr<debug_stripping_engine> debug_stripping = nullptr;

    // [STRICT] MMQ Backend Enforcement (Section 57)
    // Enforces MMQ backend for quantized decode, rejects cuBLAS fallback
    // Guarantees quantized kernels only, no dense CUDA or CPU fallback
    std::unique_ptr<mmq_enforcement_engine> mmq_enforcement = nullptr;

    // [STRICT] cuBLAS Fallback Prevention (Section 58)
    // Prevents backend re-selection or fallback during decode
    // Locks decode backend immutable, blocks all fallback paths
    std::unique_ptr<fallback_prevention_engine> fallback_prevention = nullptr;

    // [STRICT] Quantization Format Freeze (Section 59)
    // Freezes quantization format as immutable decode invariant
    // Prevents promotion, dequantization, and format drift during decode
    std::unique_ptr<quantization_format_freeze_engine> quantization_format_freeze = nullptr;

    // [STRICT] CPU Dequantization Elimination (Section 60)
    // Eliminates all CPU dequantization paths during decode
    // All quantized compute GPU-resident and GPU-executed only
    std::unique_ptr<cpu_dequantization_elimination_engine> cpu_dequant_elimination = nullptr;

    // [STRICT] Fused Kernel Coverage Validation (Section 61)
    // Formally proves all quantized ops use fused CUDA kernels
    // Validates 100% coverage with no fallback paths
    std::unique_ptr<fused_kernel_coverage_engine> fused_kernel_coverage = nullptr;

    // [STRICT] Decode-Time Allocation Freeze (Section 62)
    // Guarantees zero dynamic memory allocations on decode-critical path
    // All memory is preallocated and fixed-layout
    std::unique_ptr<decode_allocation_freeze_engine> decode_allocation_freeze = nullptr;

    // [STRICT] Decode Buffer Freeze (Section 63)
    // Every buffer used during decode is fully allocated, sized, bound, and immutable
    // No buffer resizing, relocation, rebinding, or structural mutation allowed
    std::unique_ptr<decode_buffer_freeze_engine> decode_buffer_freeze = nullptr;

    // [STRICT] GPU Allocation Alignment Enforcement (Section 64)
    // All GPU-resident buffers allocated with explicit alignment guarantees
    // Supports Tensor Core MMA, vectorized loads, and fused kernels
    std::unique_ptr<gpu_allocation_alignment_engine> gpu_allocation_alignment = nullptr;

    // [STRICT] Host Access Prevention (Section 65)
    // No CPU-side code reads, writes, maps, or touches any decode-critical buffer
    // During token generation, all decode-path data remains GPU-resident and GPU-owned
    std::unique_ptr<host_access_prevention_engine> host_access_prevention = nullptr;

    // [STRICT] GPU Memory Fragmentation Monitoring (Section 66)
    // GPU memory remains structurally stable and fragmentation-free
    // across long-running decode sessions
    // std::unique_ptr<gpu_memory_fragmentation_monitor> gpu_memory_fragmentation_monitor = nullptr;

    // [STRICT] Decode-Path CPU Execution Detector (Section 67)
    // Guarantees that no decode-critical operations execute on CPU
    // Hard abort on any CPU execution of critical ops during decode
    std::unique_ptr<decode_cpu_execution_detector> gpu_decode_cpu_execution_detector = nullptr;

    // [STRICT] Backend Usage Audit Logging (Section 68)
    // Single, authoritative backend audit report per decode session
    // Proves which backend owns every decode-critical operation
    std::unique_ptr<backend_usage_audit_logger> backend_audit_logger = nullptr;
    bool backend_audit_done = false;

    // [STRICT] Per-Token GPU Utilization Probe (Section 69)
    // Decode-phase GPU utilization probe measuring actual GPU activity
    // Detects idle gaps and verifies GPU dominance
    std::unique_ptr<gpu_utilization_probe> gpu_utilization_probe_ptr = nullptr;

    // [STRICT] PCIe Traffic Watchdog (Section 70)
    // Decode-phase PCIe transfer watchdog detecting host↔device traffic
    // Enforces zero per-token H2D/D2H transfers during decode
    std::unique_ptr<pcie_traffic_watchdog> pcie_watchdog = nullptr;

    // [STRICT] Long-Run Decode Stability Harness (Section 71)
    // Comprehensive long-run stability validator
    // Detects regressions under 5k-50k token decode runs
    std::unique_ptr<decode_stability_harness> stability_harness = nullptr;

    // [STRICT] Decode-Exclusive Success Criteria (Section 72)
    // Binary acceptance gates for GPU-exclusive architecture
    // 12 non-negotiable criteria - partial compliance is failure
    std::unique_ptr<decode_acceptance_validator> acceptance_validator = nullptr;

    // [STRICT] autonomous Decode State (GPU-resident)
    struct ggml_tensor * t_decode_pos    = nullptr;
    struct ggml_tensor * t_decode_n_past = nullptr;
    struct ggml_tensor * t_decode_token  = nullptr;
    struct ggml_tensor * t_decode_stop   = nullptr; // GPU sets this to 1 when EOS or limit reached
    struct ggml_tensor * t_decode_history = nullptr; // Ring buffer for penalties (GPU)

    int autonomous_decode(const llama_batch & batch, int n_predict);

    void sched_reserve();

    void synchronize();

    const llama_model &   get_model() const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx() const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch() const;
    uint32_t n_ubatch() const;
    uint32_t n_seq_max() const;

    uint32_t n_threads() const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t              get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(ggml_threadpool_t threadpool, ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings(bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapter_lora(llama_adapter_lora * adapter, float scale);

    bool rm_adapter_lora(llama_adapter_lora * adapter);

    void clear_adapter_lora();

    bool apply_adapter_cvec(const float * data, size_t len, int32_t n_embd, int32_t il_start, int32_t il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(const llama_ubatch &     ubatch,
                                      llm_graph_type           gtype,
                                      llama_memory_context_i * mctx,
                                      ggml_status &            ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(const char *  filepath,
                         llama_token * tokens_out,
                         size_t        n_token_capacity,
                         size_t *      n_token_count_out);

    bool state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count);

    size_t state_seq_load_file(llama_seq_id  seq_id,
                               const char *  filepath,
                               llama_token * tokens_out,
                               size_t        n_token_capacity,
                               size_t *      n_token_count_out);

    size_t state_seq_save_file(llama_seq_id        seq_id,
                               const char *        filepath,
                               const llama_token * tokens,
                               size_t              n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void                    perf_reset();

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(ggml_opt_dataset_t      dataset,
                   ggml_opt_result_t       result_train,
                   ggml_opt_result_t       result_eval,
                   int64_t                 idata_split,
                   ggml_opt_epoch_callback callback_train,
                   ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(ggml_opt_dataset_t               dataset,
                        ggml_opt_result_t                result,
                        const std::vector<llama_token> & tokens,
                        const std::vector<llama_token> & labels_sparse,
                        llama_batch &                    batch,
                        ggml_opt_epoch_callback          callback,
                        bool                             train,
                        int64_t                          idata_in_loop,
                        int64_t                          ndata_in_loop,
                        int64_t                          t_loop_start);

  private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

  public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(uint32_t                       n_tokens,
                                uint32_t                       n_seqs,
                                uint32_t                       n_outputs,
                                const llama_memory_context_i * mctx,
                                llm_graph_type                 gtype = LLM_GRAPH_TYPE_DEFAULT,
                                bool                           split_only = false,
                                size_t *                       sizes      = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

  private:
    llm_graph_params graph_params(llm_graph_result *             res,
                                  const llama_ubatch &           ubatch,
                                  const llama_memory_context_i * mctx,
                                  llm_graph_type                 gtype) const;

    llm_graph_cb graph_get_cb(llm_graph_type gtype) const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data(llama_io_read_i & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams       cparams;
    llama_adapter_cvec  cvec;
    llama_adapter_loras loras;

    llama_cross cross;  // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0;  // capacity (of floats) for logits
    float * logits      = nullptr;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0;  // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // TODO: simplify
    struct sampling_info {
        std::map<llama_seq_id, llama_sampler *> samplers;

        float * logits      = nullptr;
        size_t  logits_size = 0;

        llama_token * sampled      = nullptr;
        size_t        sampled_size = 0;

        float * probs      = nullptr;
        size_t  probs_size = 0;

        llama_token * candidates      = nullptr;
        size_t        candidates_size = 0;

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0;           // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids;  // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t                backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size;  // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    std::map<size_t, llm_graph_result_ptr> gf_res_cache;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0;  // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0;  // number of eval calls

    mutable int32_t n_reused = 0;  // number of times the previous graph was reused

    mutable std::recursive_mutex mutex;
};
