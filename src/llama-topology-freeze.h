#pragma once

/**
 * Decode Topology Freeze Enforcement
 *
 * Locks the complete decode-thread topology before entering the decode loop.
 * No thread creation, destruction, resizing, or reassignment during decode.
 *
 * Goal: Immutable thread configuration for deterministic, jitter-free execution.
 */

#include <cstdint>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Thread domain type
 */
typedef enum {
    LLAMA_THREAD_DOMAIN_DECODE,      ///< Decode worker threads (latency-critical)
    LLAMA_THREAD_DOMAIN_CUDA_CONTROL, ///< CUDA dispatch/control thread
    LLAMA_THREAD_DOMAIN_SERVER       ///< Server/HTTP threads
} llama_thread_domain;

/**
 * Thread role type
 */
typedef enum {
    LLAMA_THREAD_ROLE_DECODE_WORKER,
    LLAMA_THREAD_ROLE_CUDA_DISPATCH,
    LLAMA_THREAD_ROLE_SERVER_WORKER,
    LLAMA_THREAD_ROLE_UNKNOWN
} llama_thread_role;

/**
 * Thread configuration snapshot
 */
typedef struct {
    int          thread_id;          ///< Thread OS ID
    int          logical_id;         ///< Logical thread index
    int          cpu_core;           ///< Pinned CPU core (-1 = not pinned)
    int          priority;           ///< Thread priority level
    uint64_t     affinity_mask;      ///< CPU affinity bitmask
    bool         is_pinned;          ///< Whether CPU affinity is set
    bool         is_elevated_priority; ///< Whether priority is elevated
    llama_thread_role role;          ///< Thread's assigned role
    llama_thread_domain domain;      ///< Thread's domain
} llama_thread_config;

/**
 * Per-thread structure allocations
 */
typedef struct {
    void *  thread_local_buffer;     ///< TLS storage
    size_t  buffer_size;             ///< TLS buffer size
    void *  temp_compute_region;     ///< Temporary compute buffers
    size_t  compute_region_size;     ///< Compute region size
    void *  sync_primitives;         ///< Synchronization structures
} llama_thread_allocations;

/**
 * Topology freeze state
 */
typedef struct {
    bool             enforce_active;          ///< Enforcement enabled
    bool             topology_frozen;         ///< Topology locked
    bool             in_decode_session;       ///< Currently decoding

    // Topology snapshot
    int              n_decode_threads;        ///< Number of decode workers
    int              n_cuda_threads;          ///< Number of CUDA threads (usually 1)
    int              n_server_threads;        ///< Number of server threads
    int              total_threads;           ///< Total thread count

    // Frozen configuration
    llama_thread_config * decode_configs;     ///< Frozen decode thread configs
    llama_thread_config cuda_config;          ///< Frozen CUDA thread config
    llama_thread_config * server_configs;     ///< Frozen server thread configs

    // Allocations
    llama_thread_allocations * decode_allocs; ///< Pre-allocated decode buffers
    llama_thread_allocations cuda_alloc;      ///< Pre-allocated CUDA buffers
    llama_thread_allocations * server_allocs; ///< Pre-allocated server buffers

    // Scheduling policy
    int              scheduling_policy;       ///< OS scheduling policy (SCHED_OTHER, etc.)
    int              priority_level;          ///< Base priority level
    bool             use_elevated_priority;   ///< Whether to use elevated priority

    // Validation state
    uint64_t         freeze_timestamp;        ///< When topology was frozen
    uint32_t         freeze_sequence_number;  ///< Sequence number at freeze
    uint32_t         current_sequence_number; ///< Current sequence for mutation detection

    // Metrics
    uint64_t         thread_creation_attempts;    ///< Attempts to create threads
    uint64_t         thread_destruction_attempts; ///< Attempts to destroy threads
    uint64_t         thread_resize_attempts;      ///< Attempts to resize pool
    uint64_t         role_change_attempts;        ///< Attempts to change roles
    uint64_t         affinity_change_attempts;    ///< Attempts to change affinity
} llama_topology_freeze_state;

/**
 * Initialize topology freeze enforcement
 */
void llama_topology_freeze_init(llama_topology_freeze_state * state);

/**
 * Pre-initialize all decode threads before entering decode loop
 *
 * Creates and configures all worker threads.
 * Allocates per-thread structures.
 * Warms up execution paths.
 *
 * Must be called before decode begins.
 *
 * @param state Topology freeze state
 * @param n_decode_threads Number of decode worker threads
 * @param n_cuda_threads Number of CUDA control threads (usually 1)
 * @param n_server_threads Number of server threads
 * @return true if pre-initialization successful, false otherwise
 */
bool llama_topology_freeze_pre_initialize(
    llama_topology_freeze_state * state,
    int n_decode_threads,
    int n_cuda_threads,
    int n_server_threads);

/**
 * [CRITICAL] Freeze thread topology
 *
 * Locks the thread configuration immutably.
 * After this point, no topology mutations allowed.
 *
 * @param state Topology freeze state
 * @return true if freeze successful, false if errors detected
 */
bool llama_topology_freeze_lock(llama_topology_freeze_state * state);

/**
 * [CRITICAL] Validate topology unchanged during decode
 *
 * Called at each decode iteration to ensure topology hasn't mutated.
 * Aborts if any changes detected.
 *
 * @param state Topology freeze state
 * @return true if topology unchanged, false if mutations detected
 */
bool llama_topology_freeze_validate_unchanged(
    llama_topology_freeze_state * state);

/**
 * [CRITICAL] Enforce separate thread domains
 *
 * Validates that decode, CUDA, and server threads are in distinct domains.
 * Ensures no thread borrowing or reassignment between domains.
 *
 * @param state Topology freeze state
 * @return true if domains properly separated, false otherwise
 */
bool llama_topology_freeze_enforce_domains(
    llama_topology_freeze_state * state);

/**
 * [CRITICAL] Lock thread count
 *
 * Captures the effective thread count and freezes it.
 * Rejects any runtime changes to thread count.
 *
 * @param state Topology freeze state
 * @param current_thread_count Current thread count to lock
 * @return true if locked successfully, false if inconsistent
 */
bool llama_topology_freeze_lock_thread_count(
    llama_topology_freeze_state * state,
    int current_thread_count);

/**
 * Set CPU affinity for decode threads
 *
 * Pins each decode thread to a specific CPU core.
 * Optional but preferred for jitter elimination.
 *
 * @param state Topology freeze state
 * @param core_ids Array of CPU core IDs (one per decode thread)
 * @param n_cores Number of cores to assign
 * @return true if affinity set, false otherwise
 */
bool llama_topology_freeze_set_affinity(
    llama_topology_freeze_state * state,
    const int * core_ids,
    int n_cores);

/**
 * [CRITICAL] Pre-allocate per-thread structures
 *
 * Allocates all thread-local buffers, temp regions, and sync primitives
 * before decode starts. No allocations allowed during decode.
 *
 * @param state Topology freeze state
 * @param buffer_size_per_thread Size of TLS buffer per thread
 * @param compute_region_size Size of temp compute region per thread
 * @return true if allocation successful, false if failed
 */
bool llama_topology_freeze_pre_allocate_structures(
    llama_topology_freeze_state * state,
    size_t buffer_size_per_thread,
    size_t compute_region_size);

/**
 * [CRITICAL] Enforce no dynamic role switching
 *
 * Validates that threads maintain fixed roles throughout decode.
 * Detects any attempts to repurpose or reassign threads.
 *
 * @param state Topology freeze state
 * @return true if roles unchanged, false if switches detected
 */
bool llama_topology_freeze_validate_no_role_switches(
    llama_topology_freeze_state * state);

/**
 * [CRITICAL] Freeze scheduling policy
 *
 * Locks the OS scheduling policy and priority level.
 * Prevents runtime priority boosting or adaptive tweaks.
 *
 * @param state Topology freeze state
 * @param policy OS scheduling policy (SCHED_OTHER, SCHED_FIFO, etc.)
 * @param priority Priority level
 * @param use_elevated Whether to use elevated priority
 * @return true if policy frozen, false otherwise
 */
bool llama_topology_freeze_lock_scheduling(
    llama_topology_freeze_state * state,
    int policy,
    int priority,
    bool use_elevated);

/**
 * [CRITICAL] Eliminate per-token topology checks
 *
 * Validates that topology recomputation code has been removed.
 * Per-token layout checks are forbidden.
 *
 * @param state Topology freeze state
 * @return true if no per-token checks detected, false otherwise
 */
bool llama_topology_freeze_eliminate_per_token_checks(
    llama_topology_freeze_state * state);

/**
 * [CRITICAL] Record topology mutation attempt
 *
 * Called whenever topology mutation is attempted.
 * Tracks creation, destruction, resize, role change, affinity change.
 *
 * @param state Topology freeze state
 * @param mutation_type "create", "destroy", "resize", "role_change", "affinity_change"
 */
void llama_topology_freeze_record_mutation_attempt(
    llama_topology_freeze_state * state,
    const char * mutation_type);

/**
 * [CRITICAL] Validate topology immutability
 *
 * Asserts that topology has not mutated.
 * Called at each decode iteration.
 *
 * @param state Topology freeze state
 * @return true if immutable (no mutations), false if changes attempted
 */
bool llama_topology_freeze_assert_immutable(
    llama_topology_freeze_state * state);

/**
 * Get topology configuration
 *
 * @param state Topology freeze state
 * @param thread_id Thread logical ID
 * @return Thread configuration
 */
llama_thread_config llama_topology_freeze_get_config(
    const llama_topology_freeze_state * state,
    int thread_id);

/**
 * Get topology metrics
 *
 * @param state Topology freeze state
 * @return Mutation attempt counts
 */
struct {
    uint64_t create_attempts;
    uint64_t destroy_attempts;
    uint64_t resize_attempts;
    uint64_t role_changes;
    uint64_t affinity_changes;
} llama_topology_freeze_get_metrics(const llama_topology_freeze_state * state);

/**
 * Dump topology configuration and status
 *
 * Logs complete thread topology and frozen state.
 *
 * @param state Topology freeze state
 */
void llama_topology_freeze_dump_config(
    const llama_topology_freeze_state * state);

/**
 * Release topology freeze at decode end
 *
 * @param state Topology freeze state
 */
void llama_topology_freeze_release(llama_topology_freeze_state * state);

#ifdef __cplusplus
}
#endif
