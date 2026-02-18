#pragma once

/**
 * Core Isolation Enforcement for LLAMA Decode Optimization
 *
 * Strict CPU core partitioning to prevent server threads from sharing
 * decode cores. Implements invariant: dedicated, immutable CPU domains
 * for decode and server operations with deterministic scheduling.
 *
 * Key Property: Once frozen, decode cores NEVER execute server work
 * and server cores NEVER execute decode work. No core sharing. Period.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Core domain type - partitions CPU cores into exclusive domains
 */
typedef enum {
    LLAMA_CORE_DOMAIN_UNINITIALIZED = 0,
    LLAMA_CORE_DOMAIN_DECODE = 1,    // Cores dedicated to decode workers only
    LLAMA_CORE_DOMAIN_SERVER = 2,    // Cores dedicated to server/HTTP threads only
    LLAMA_CORE_DOMAIN_CUDA_CONTROL = 3, // Core(s) for CUDA dispatch thread
    LLAMA_CORE_DOMAIN_RESERVED = 4   // Reserved/OS kernel cores
} llama_core_domain_t;

/**
 * CPU affinity enforcement mode
 */
typedef enum {
    LLAMA_AFFINITY_SOFT = 0,         // Allow OS migration (baseline)
    LLAMA_AFFINITY_HARD = 1,         // Pin threads to cores via sched_setaffinity
    LLAMA_AFFINITY_NUMA_AWARE = 2    // NUMA-aware pinning with local memory
} llama_affinity_mode_t;

/**
 * Core isolation state machine
 */
typedef enum {
    LLAMA_ISOLATION_UNINITIALIZED = 0,
    LLAMA_ISOLATION_CONFIGURED = 1,    // Cores partitioned, not yet locked
    LLAMA_ISOLATION_FROZEN = 2,        // Partitioning immutable, enforcement active
    LLAMA_ISOLATION_RELEASED = 3       // Session ended, cores released
} llama_isolation_state_t;

/**
 * Core affinity mask - represents which cores a thread can execute on
 */
typedef struct {
    uint64_t * masks;          // Bitmasks for core sets (up to 64 cores per uint64_t)
    int n_masks;               // Number of uint64_t elements needed
    int total_cores;           // Total physical cores available
} llama_core_affinity_mask;

/**
 * Per-thread core assignment
 */
typedef struct {
    uint32_t thread_id;                // OS thread ID
    llama_core_domain_t assigned_domain; // Assigned domain
    int primary_core;                  // Primary pinned core (-1 if unpinned)
    int secondary_cores[8];            // Secondary allowed cores (for worksteal)
    int n_secondary;                   // Count of secondary cores
    uint64_t assigned_at_ns;           // Timestamp of assignment (ns)
    int context_switches;              // Context switch count (for violation detection)
} llama_thread_core_assignment;

/**
 * Core isolation enforcement state
 */
typedef struct {
    // Topology
    int total_cores;                   // Total physical cores
    int cores_per_socket;              // Cores per NUMA socket (0 if no NUMA)
    int num_sockets;                   // NUMA socket count
    bool has_numa;                     // NUMA topology available

    // Domain partitioning
    int * decode_cores;                // Array of core IDs for decode domain
    int n_decode_cores;                // Count of decode cores (immutable)
    int * server_cores;                // Array of core IDs for server domain
    int n_server_cores;                // Count of server cores (immutable)
    int cuda_control_core;             // Dedicated CUDA control core (-1 if none)

    // Thread tracking
    llama_thread_core_assignment * thread_assignments; // Per-thread assignments
    int n_threads_tracked;             // Count of tracked threads
    int max_threads_capacity;          // Capacity of assignment array

    // Enforcement
    llama_affinity_mode_t affinity_mode;        // Affinity enforcement mode
    llama_isolation_state_t state;              // Current isolation state
    bool enforcement_active;            // true when frozen and enforcing
    uint64_t frozen_at_ns;              // Timestamp when frozen

    // Violation tracking
    uint64_t decode_on_server_count;    // Decode thread running on server core
    uint64_t server_on_decode_count;    // Server thread running on decode core
    uint64_t migration_violations;      // Thread migrated between domains
    uint64_t core_sharing_violations;   // Core sharing between domains detected
    uint64_t preemption_events;         // OS preemptions detected

    // Metrics
    uint64_t total_context_switches;    // Sum of all context switches
    uint64_t assertion_checks_passed;   // Successful validation checks
    uint64_t assertion_checks_failed;   // Failed validation checks
} llama_core_isolation_state;

/**
 * Initialize core isolation state
 *
 * Detects available CPU cores and NUMA topology.
 * Does not lock/freeze - just initializes state.
 *
 * @param state Isolation state to initialize
 * @return true if initialization successful
 */
bool llama_core_isolation_init(llama_core_isolation_state * state);

/**
 * Release core isolation state and cleanup resources
 *
 * @param state Isolation state to release
 */
void llama_core_isolation_release(llama_core_isolation_state * state);

/**
 * [CRITICAL] Partition CPU cores into decode and server domains
 *
 * After this call:
 * - decode_cores[] contains N cores exclusively for decode
 * - server_cores[] contains M cores exclusively for server
 * - Remaining cores go to RESERVED domain (kernel, OS)
 *
 * Example: On 8-core system with decode=4, server=3:
 *   Cores 0-3: DECODE domain (decode threads only)
 *   Cores 4-6: SERVER domain (server threads only)
 *   Core 7:    RESERVED domain (OS)
 *
 * @param state Isolation state
 * @param n_decode_cores Number of cores to allocate to decode
 * @param n_server_cores Number of cores to allocate to server
 * @return true if partition successful
 */
bool llama_core_isolation_partition_domains(
    llama_core_isolation_state * state,
    int n_decode_cores,
    int n_server_cores);

/**
 * [CRITICAL] Partition with NUMA awareness
 *
 * Allocates decode cores from a single NUMA node when possible,
 * minimizing cross-node memory access patterns.
 *
 * @param state Isolation state
 * @param n_decode_cores Number of decode cores (all from same NUMA node if possible)
 * @param n_server_cores Number of server cores
 * @param prefer_socket NUMA socket to prefer for decode (0 or 1)
 * @return true if NUMA-aware partition successful
 */
bool llama_core_isolation_partition_numa_aware(
    llama_core_isolation_state * state,
    int n_decode_cores,
    int n_server_cores,
    int prefer_socket);

/**
 * [CRITICAL] Assign thread to a core domain
 *
 * Registers thread and assigns it to a core domain.
 * Before freeze: multiple threads can be assigned.
 * After freeze: no new assignments allowed.
 *
 * @param state Isolation state
 * @param thread_id OS thread ID to assign
 * @param domain Domain to assign to (DECODE, SERVER, or CUDA_CONTROL)
 * @param primary_core Specific core to pin (-1 for no primary core)
 * @return true if assignment successful
 */
bool llama_core_isolation_assign_thread(
    llama_core_isolation_state * state,
    uint32_t thread_id,
    llama_core_domain_t domain,
    int primary_core);

/**
 * [CRITICAL] Apply CPU affinity to thread
 *
 * Sets OS-level CPU affinity using sched_setaffinity (Linux),
 * SetThreadAffinityMask (Windows), or pthread_setaffinity_np (POSIX).
 *
 * @param state Isolation state
 * @param thread_id OS thread ID
 * @param cores Array of allowed cores
 * @param n_cores Size of cores array
 * @return true if affinity set successfully
 */
bool llama_core_isolation_apply_affinity(
    llama_core_isolation_state * state,
    uint32_t thread_id,
    const int * cores,
    int n_cores);

/**
 * [CRITICAL] Freeze core isolation
 *
 * Locks core partitioning and thread assignments immutable.
 * After this:
 * - No new core partitions allowed
 * - No new thread assignments allowed
 * - Enforcement begins immediately
 * - Violations trigger abort
 *
 * @param state Isolation state
 * @return true if freeze successful
 */
bool llama_core_isolation_freeze(llama_core_isolation_state * state);

/**
 * [CRITICAL] Validate decode thread only runs on decode cores
 *
 * Checks that the thread_id (representing a decode thread)
 * is executing only on cores in the decode domain.
 *
 * This should be called periodically during decode to catch
 * OS migrations or preemption violations.
 *
 * @param state Isolation state
 * @param thread_id Decode thread to validate
 * @return true if thread is on decode cores only
 */
bool llama_core_isolation_validate_decode_thread_isolation(
    llama_core_isolation_state * state,
    uint32_t thread_id);

/**
 * [CRITICAL] Validate server thread only runs on server cores
 *
 * Checks that the thread_id (representing a server thread)
 * is executing only on cores in the server domain.
 *
 * @param state Isolation state
 * @param thread_id Server thread to validate
 * @return true if thread is on server cores only
 */
bool llama_core_isolation_validate_server_thread_isolation(
    llama_core_isolation_state * state,
    uint32_t thread_id);

/**
 * [CRITICAL] Validate decode and server domains don't share cores
 *
 * Audits all running threads and verifies:
 * - No decode thread is on server cores
 * - No server thread is on decode cores
 * - Core usage is strictly partitioned
 *
 * This is an expensive check (reads /proc/stat or Windows counters),
 * so should be called infrequently (e.g., every 100 tokens).
 *
 * @param state Isolation state
 * @return true if core partitioning valid
 */
bool llama_core_isolation_validate_no_core_sharing(
    llama_core_isolation_state * state);

/**
 * [CRITICAL] Prevent OS auto-migration of decode threads
 *
 * Increments preemption event counter and validates thread hasn't
 * moved off its assigned cores. Called to detect and prevent
 * OS migration of decode threads.
 *
 * @param state Isolation state
 * @param thread_id Thread to validate
 * @param current_core Core where thread is currently executing
 * @return true if thread is on assigned core(s)
 */
bool llama_core_isolation_detect_migration(
    llama_core_isolation_state * state,
    uint32_t thread_id,
    int current_core);

/**
 * [CRITICAL] Assert isolation state unchanged
 *
 * Validates that core partitioning and thread assignments
 * have not changed since freeze. This is a lightweight check.
 *
 * Aborts if violations detected.
 *
 * @param state Isolation state
 * @return true if isolation intact
 */
bool llama_core_isolation_assert_immutable(llama_core_isolation_state * state);

/**
 * Get core domain for a thread
 *
 * @param state Isolation state
 * @param thread_id Thread ID
 * @return Domain assigned to thread, or UNINITIALIZED if not found
 */
llama_core_domain_t llama_core_isolation_get_thread_domain(
    const llama_core_isolation_state * state,
    uint32_t thread_id);

/**
 * Get array of decode cores
 *
 * @param state Isolation state
 * @param out_cores Buffer to fill (caller-owned)
 * @param max_cores Size of out_cores buffer
 * @return Number of decode cores, or -1 on error
 */
int llama_core_isolation_get_decode_cores(
    const llama_core_isolation_state * state,
    int * out_cores,
    int max_cores);

/**
 * Get array of server cores
 *
 * @param state Isolation state
 * @param out_cores Buffer to fill (caller-owned)
 * @param max_cores Size of out_cores buffer
 * @return Number of server cores, or -1 on error
 */
int llama_core_isolation_get_server_cores(
    const llama_core_isolation_state * state,
    int * out_cores,
    int max_cores);

/**
 * Get current isolation state
 *
 * @param state Isolation state
 * @return Current isolation state enum
 */
llama_isolation_state_t llama_core_isolation_get_state(
    const llama_core_isolation_state * state);

/**
 * [DEBUG] Dump core isolation configuration
 *
 * Writes human-readable core partitioning and thread assignments
 * to log (via LLAMA_LOG_INFO).
 *
 * Example output:
 *
 * CORE ISOLATION CONFIG:
 *   Total cores: 8, NUMA sockets: 2
 *   Decode domain: cores [0,1,2,3] (4 cores)
 *   Server domain: cores [4,5,6] (3 cores)
 *   CUDA control: core 7
 *   State: FROZEN
 *   Threads tracked: 7
 *     Thread 1234 (DECODE): primary=0, secondary=[1,2,3]
 *     Thread 1235 (DECODE): primary=1, secondary=[0,2,3]
 *     Thread 2000 (SERVER): primary=4, secondary=[5,6]
 *   Violations: 0
 *
 * @param state Isolation state
 */
void llama_core_isolation_dump_config(const llama_core_isolation_state * state);

/**
 * [DEBUG] Get violation statistics
 *
 * Returns counts of detected isolation violations.
 *
 * @param state Isolation state
 * @param out_decode_on_server Count of decode threads on server cores
 * @param out_server_on_decode Count of server threads on decode cores
 * @param out_migrations Count of cross-domain migrations
 * @param out_preemptions Count of preemption events
 */
void llama_core_isolation_get_violations(
    const llama_core_isolation_state * state,
    uint64_t * out_decode_on_server,
    uint64_t * out_server_on_decode,
    uint64_t * out_migrations,
    uint64_t * out_preemptions);

/**
 * [DEBUG] Get metrics
 *
 * Returns performance and validation metrics.
 *
 * @param state Isolation state
 * @param out_total_context_switches Total context switches counted
 * @param out_checks_passed Successful validation checks
 * @param out_checks_failed Failed validation checks
 */
void llama_core_isolation_get_metrics(
    const llama_core_isolation_state * state,
    uint64_t * out_total_context_switches,
    uint64_t * out_checks_passed,
    uint64_t * out_checks_failed);

#ifdef __cplusplus
}
#endif
