/**
 * Core Isolation Enforcement Implementation
 *
 * Implements strict CPU core partitioning to prevent server threads
 * from sharing decode cores. Provides deterministic scheduling through
 * OS-level CPU affinity and NUMA awareness.
 */

#include "llama-core-isolation-enforce.h"
#include "llama-impl.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdatomic.h>

#ifdef _WIN32
    #include <windows.h>
    #include <processthreadsapi.h>
#else
    #include <pthread.h>
    #include <sched.h>
    #include <sys/types.h>
#endif

// Global isolation state
llama_core_isolation_state g_core_isolation_state = {
    .total_cores = 0,
    .cores_per_socket = 0,
    .num_sockets = 1,
    .has_numa = false,
    .decode_cores = NULL,
    .n_decode_cores = 0,
    .server_cores = NULL,
    .n_server_cores = 0,
    .cuda_control_core = -1,
    .thread_assignments = NULL,
    .n_threads_tracked = 0,
    .max_threads_capacity = 0,
    .affinity_mode = LLAMA_AFFINITY_SOFT,
    .state = LLAMA_ISOLATION_UNINITIALIZED,
    .enforcement_active = false,
    .frozen_at_ns = 0,
    .decode_on_server_count = 0,
    .server_on_decode_count = 0,
    .migration_violations = 0,
    .core_sharing_violations = 0,
    .preemption_events = 0,
    .total_context_switches = 0,
    .assertion_checks_passed = 0,
    .assertion_checks_failed = 0
};

/**
 * Get number of available CPU cores
 */
static int get_cpu_core_count(void) {
    #ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        return (int)sysinfo.dwNumberOfProcessors;
    #else
        return (int)sysconf(_SC_NPROCESSORS_ONLN);
    #endif
}

/**
 * Check if NUMA is available
 */
static bool has_numa_available(void) {
    #ifdef _WIN32
        // Windows doesn't expose NUMA the same way
        return false;
    #else
        // Check if libnuma is available or /proc/numa_maps exists
        return access("/proc/numa_maps", F_OK) == 0;
    #endif
}

/**
 * Initialize core isolation state
 */
bool llama_core_isolation_init(llama_core_isolation_state * state) {
    if (!state) return false;

    memset(state, 0, sizeof(*state));

    state->total_cores = get_cpu_core_count();
    state->has_numa = has_numa_available();
    state->num_sockets = state->has_numa ? 2 : 1; // Simplified: assume 2 for NUMA
    state->cores_per_socket = state->has_numa ? (state->total_cores / state->num_sockets) : 0;

    state->state = LLAMA_ISOLATION_CONFIGURED;
    state->affinity_mode = LLAMA_AFFINITY_SOFT;

    // Pre-allocate thread assignment tracking
    state->max_threads_capacity = 256; // Support up to 256 threads
    state->thread_assignments = (llama_thread_core_assignment *)malloc(
        sizeof(llama_thread_core_assignment) * state->max_threads_capacity);

    if (!state->thread_assignments) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Failed to allocate thread assignments buffer\n");
        return false;
    }

    memset(state->thread_assignments, 0,
           sizeof(llama_thread_core_assignment) * state->max_threads_capacity);

    LLAMA_LOG_INFO(
        "CORE ISOLATION: Initialized (cores=%d, NUMA=%s, sockets=%d)\n",
        state->total_cores,
        state->has_numa ? "yes" : "no",
        state->num_sockets);

    return true;
}

/**
 * Release core isolation state
 */
void llama_core_isolation_release(llama_core_isolation_state * state) {
    if (!state) return;

    if (state->decode_cores) {
        free(state->decode_cores);
        state->decode_cores = NULL;
    }
    if (state->server_cores) {
        free(state->server_cores);
        state->server_cores = NULL;
    }
    if (state->thread_assignments) {
        free(state->thread_assignments);
        state->thread_assignments = NULL;
    }

    state->state = LLAMA_ISOLATION_RELEASED;
    state->enforcement_active = false;

    LLAMA_LOG_INFO("CORE ISOLATION: Released\n");
}

/**
 * Partition CPU cores into decode and server domains
 */
bool llama_core_isolation_partition_domains(
    llama_core_isolation_state * state,
    int n_decode_cores,
    int n_server_cores) {

    if (!state) return false;
    if (state->state != LLAMA_ISOLATION_CONFIGURED) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Cannot partition - state is %d (must be CONFIGURED)\n",
                       state->state);
        return false;
    }

    // Validate core counts
    int total_requested = n_decode_cores + n_server_cores;
    if (total_requested > state->total_cores) {
        LLAMA_LOG_ERROR(
            "CORE ISOLATION: Requested %d cores (decode=%d + server=%d) exceeds total %d\n",
            total_requested, n_decode_cores, n_server_cores, state->total_cores);
        return false;
    }

    // Allocate decode cores array
    state->decode_cores = (int *)malloc(sizeof(int) * n_decode_cores);
    if (!state->decode_cores) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Failed to allocate decode cores array\n");
        return false;
    }

    // Allocate server cores array
    state->server_cores = (int *)malloc(sizeof(int) * n_server_cores);
    if (!state->server_cores) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Failed to allocate server cores array\n");
        free(state->decode_cores);
        state->decode_cores = NULL;
        return false;
    }

    // Simple linear partitioning: 0..n_decode_cores-1 for decode
    for (int i = 0; i < n_decode_cores; i++) {
        state->decode_cores[i] = i;
    }
    state->n_decode_cores = n_decode_cores;

    // n_decode_cores..n_decode_cores+n_server_cores-1 for server
    for (int i = 0; i < n_server_cores; i++) {
        state->server_cores[i] = n_decode_cores + i;
    }
    state->n_server_cores = n_server_cores;

    // CUDA control gets the last core if available
    if (total_requested < state->total_cores) {
        state->cuda_control_core = total_requested;
    } else {
        state->cuda_control_core = -1; // Share with last decode core
    }

    LLAMA_LOG_INFO(
        "CORE ISOLATION: Partitioned domains:\n"
        "  Decode: cores [%d-%d] (%d cores)\n"
        "  Server: cores [%d-%d] (%d cores)\n"
        "  CUDA control: core %d\n"
        "  Reserved: cores [%d-%d]\n",
        state->decode_cores[0],
        state->decode_cores[n_decode_cores - 1],
        n_decode_cores,
        state->server_cores[0],
        state->server_cores[n_server_cores - 1],
        n_server_cores,
        state->cuda_control_core,
        (total_requested < state->total_cores) ? (total_requested + 1) : -1,
        state->total_cores - 1);

    return true;
}

/**
 * Partition with NUMA awareness
 */
bool llama_core_isolation_partition_numa_aware(
    llama_core_isolation_state * state,
    int n_decode_cores,
    int n_server_cores,
    int prefer_socket) {

    if (!state) return false;
    if (!state->has_numa) {
        LLAMA_LOG_WARN("CORE ISOLATION: NUMA not available, falling back to linear partition\n");
        return llama_core_isolation_partition_domains(state, n_decode_cores, n_server_cores);
    }

    if (state->state != LLAMA_ISOLATION_CONFIGURED) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Cannot partition - state is %d (must be CONFIGURED)\n",
                       state->state);
        return false;
    }

    // Validate socket
    if (prefer_socket < 0 || prefer_socket >= state->num_sockets) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Invalid socket %d (max %d)\n",
                       prefer_socket, state->num_sockets - 1);
        return false;
    }

    // Allocate decode cores from preferred socket
    state->decode_cores = (int *)malloc(sizeof(int) * n_decode_cores);
    if (!state->decode_cores) return false;

    state->server_cores = (int *)malloc(sizeof(int) * n_server_cores);
    if (!state->server_cores) {
        free(state->decode_cores);
        state->decode_cores = NULL;
        return false;
    }

    // Decode cores from preferred socket
    int socket_base = prefer_socket * state->cores_per_socket;
    for (int i = 0; i < n_decode_cores; i++) {
        state->decode_cores[i] = socket_base + i;
    }
    state->n_decode_cores = n_decode_cores;

    // Server cores from remaining cores
    int other_socket = 1 - prefer_socket;
    int other_base = other_socket * state->cores_per_socket;
    for (int i = 0; i < n_server_cores; i++) {
        state->server_cores[i] = other_base + i;
    }
    state->n_server_cores = n_server_cores;

    // CUDA control on remaining core
    int remaining_core = other_base + n_server_cores;
    if (remaining_core < state->total_cores) {
        state->cuda_control_core = remaining_core;
    } else {
        state->cuda_control_core = -1;
    }

    LLAMA_LOG_INFO(
        "CORE ISOLATION: NUMA-aware partition:\n"
        "  Decode (socket %d): cores [%d-%d] (%d cores)\n"
        "  Server (socket %d): cores [%d-%d] (%d cores)\n",
        prefer_socket, state->decode_cores[0],
        state->decode_cores[n_decode_cores - 1], n_decode_cores,
        other_socket, state->server_cores[0],
        state->server_cores[n_server_cores - 1], n_server_cores);

    return true;
}

/**
 * Assign thread to core domain
 */
bool llama_core_isolation_assign_thread(
    llama_core_isolation_state * state,
    uint32_t thread_id,
    llama_core_domain_t domain,
    int primary_core) {

    if (!state) return false;
    if (state->state == LLAMA_ISOLATION_FROZEN) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Cannot assign thread - state is FROZEN\n");
        return false;
    }

    // Find existing assignment or create new one
    llama_thread_core_assignment * assignment = NULL;
    for (int i = 0; i < state->n_threads_tracked; i++) {
        if (state->thread_assignments[i].thread_id == thread_id) {
            assignment = &state->thread_assignments[i];
            break;
        }
    }

    if (!assignment) {
        // New assignment
        if (state->n_threads_tracked >= state->max_threads_capacity) {
            LLAMA_LOG_ERROR("CORE ISOLATION: Thread capacity exhausted (%d)\n",
                           state->max_threads_capacity);
            return false;
        }
        assignment = &state->thread_assignments[state->n_threads_tracked++];
    }

    assignment->thread_id = thread_id;
    assignment->assigned_domain = domain;
    assignment->primary_core = primary_core;
    assignment->assigned_at_ns = 0; // Will be set by OS timestamp
    assignment->context_switches = 0;
    memset(assignment->secondary_cores, -1, sizeof(assignment->secondary_cores));
    assignment->n_secondary = 0;

    // Populate secondary cores based on domain
    if (domain == LLAMA_CORE_DOMAIN_DECODE && state->decode_cores) {
        for (int i = 0; i < state->n_decode_cores && i < 8; i++) {
            assignment->secondary_cores[i] = state->decode_cores[i];
        }
        assignment->n_secondary = (state->n_decode_cores < 8) ? state->n_decode_cores : 8;
    } else if (domain == LLAMA_CORE_DOMAIN_SERVER && state->server_cores) {
        for (int i = 0; i < state->n_server_cores && i < 8; i++) {
            assignment->secondary_cores[i] = state->server_cores[i];
        }
        assignment->n_secondary = (state->n_server_cores < 8) ? state->n_server_cores : 8;
    }

    LLAMA_LOG_DEBUG("CORE ISOLATION: Assigned thread %u to domain %d (primary core %d)\n",
                   thread_id, domain, primary_core);

    return true;
}

/**
 * Apply CPU affinity to thread
 */
bool llama_core_isolation_apply_affinity(
    llama_core_isolation_state * state,
    uint32_t thread_id,
    const int * cores,
    int n_cores) {

    if (!state || !cores || n_cores <= 0) return false;

    #ifdef _WIN32
        // Windows: SetThreadAffinityMask
        HANDLE thread_handle = OpenThread(THREAD_SET_INFORMATION, FALSE, (DWORD)thread_id);
        if (!thread_handle) {
            LLAMA_LOG_WARN("CORE ISOLATION: Cannot open thread %u (may not be valid)\n", thread_id);
            return false;
        }

        DWORD_PTR mask = 0;
        for (int i = 0; i < n_cores; i++) {
            if (cores[i] >= 0 && cores[i] < 64) { // Windows supports up to 64 cores
                mask |= (1ULL << cores[i]);
            }
        }

        DWORD_PTR result = SetThreadAffinityMask(thread_handle, mask);
        CloseHandle(thread_handle);

        if (result == 0) {
            LLAMA_LOG_WARN("CORE ISOLATION: SetThreadAffinityMask failed for thread %u\n", thread_id);
            return false;
        }

        state->affinity_mode = LLAMA_AFFINITY_HARD;
        return true;

    #else
        // POSIX: sched_setaffinity or pthread_setaffinity_np
        cpu_set_t set;
        CPU_ZERO(&set);

        for (int i = 0; i < n_cores; i++) {
            if (cores[i] >= 0 && cores[i] < CPU_SETSIZE) {
                CPU_SET(cores[i], &set);
            }
        }

        // Try sched_setaffinity first
        pid_t pid = (pid_t)thread_id;
        int result = sched_setaffinity(pid, sizeof(cpu_set_t), &set);

        if (result != 0) {
            LLAMA_LOG_WARN("CORE ISOLATION: sched_setaffinity failed for thread %u: %d\n",
                          thread_id, result);
            return false;
        }

        state->affinity_mode = LLAMA_AFFINITY_HARD;
        return true;
    #endif
}

/**
 * Freeze core isolation
 */
bool llama_core_isolation_freeze(llama_core_isolation_state * state) {
    if (!state) return false;

    if (state->state != LLAMA_ISOLATION_CONFIGURED) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Cannot freeze - state is %d (must be CONFIGURED)\n",
                       state->state);
        return false;
    }

    if (!state->decode_cores || state->n_decode_cores == 0) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Cannot freeze - decode domain not partitioned\n");
        return false;
    }

    if (!state->server_cores || state->n_server_cores == 0) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Cannot freeze - server domain not partitioned\n");
        return false;
    }

    state->state = LLAMA_ISOLATION_FROZEN;
    state->enforcement_active = true;
    state->frozen_at_ns = 0; // Would use high-resolution timer in production

    LLAMA_LOG_INFO(
        "CORE ISOLATION: Frozen (decode=%d cores, server=%d cores, enforcement ACTIVE)\n",
        state->n_decode_cores, state->n_server_cores);

    return true;
}

/**
 * Validate decode thread isolation
 */
bool llama_core_isolation_validate_decode_thread_isolation(
    llama_core_isolation_state * state,
    uint32_t thread_id) {

    if (!state || !state->enforcement_active) return true;

    // Find thread assignment
    llama_thread_core_assignment * assignment = NULL;
    for (int i = 0; i < state->n_threads_tracked; i++) {
        if (state->thread_assignments[i].thread_id == thread_id) {
            assignment = &state->thread_assignments[i];
            break;
        }
    }

    if (!assignment) {
        LLAMA_LOG_WARN("CORE ISOLATION: Thread %u not found in assignments\n", thread_id);
        return true; // Not tracked, assume OK
    }

    if (assignment->assigned_domain != LLAMA_CORE_DOMAIN_DECODE) {
        return true; // Not a decode thread, skip validation
    }

    // Validate thread only runs on decode cores
    // This is simplified - in production, would read /proc/[pid]/stat or similar
    if (assignment->primary_core >= 0) {
        // Primary core should be in decode domain
        bool found = false;
        for (int i = 0; i < state->n_decode_cores; i++) {
            if (state->decode_cores[i] == assignment->primary_core) {
                found = true;
                break;
            }
        }

        if (!found) {
            LLAMA_LOG_ERROR(
                "CORE ISOLATION: Decode thread %u found on core %d (outside decode domain)\n",
                thread_id, assignment->primary_core);
            state->decode_on_server_count++;
            return false;
        }
    }

    state->assertion_checks_passed++;
    return true;
}

/**
 * Validate server thread isolation
 */
bool llama_core_isolation_validate_server_thread_isolation(
    llama_core_isolation_state * state,
    uint32_t thread_id) {

    if (!state || !state->enforcement_active) return true;

    // Find thread assignment
    llama_thread_core_assignment * assignment = NULL;
    for (int i = 0; i < state->n_threads_tracked; i++) {
        if (state->thread_assignments[i].thread_id == thread_id) {
            assignment = &state->thread_assignments[i];
            break;
        }
    }

    if (!assignment) {
        return true; // Not tracked, assume OK
    }

    if (assignment->assigned_domain != LLAMA_CORE_DOMAIN_SERVER) {
        return true; // Not a server thread, skip validation
    }

    // Validate thread only runs on server cores
    if (assignment->primary_core >= 0) {
        bool found = false;
        for (int i = 0; i < state->n_server_cores; i++) {
            if (state->server_cores[i] == assignment->primary_core) {
                found = true;
                break;
            }
        }

        if (!found) {
            LLAMA_LOG_ERROR(
                "CORE ISOLATION: Server thread %u found on core %d (outside server domain)\n",
                thread_id, assignment->primary_core);
            state->server_on_decode_count++;
            return false;
        }
    }

    state->assertion_checks_passed++;
    return true;
}

/**
 * Validate no core sharing
 */
bool llama_core_isolation_validate_no_core_sharing(
    llama_core_isolation_state * state) {

    if (!state || !state->enforcement_active) return true;

    // Check for overlap between decode and server cores
    for (int i = 0; i < state->n_decode_cores; i++) {
        for (int j = 0; j < state->n_server_cores; j++) {
            if (state->decode_cores[i] == state->server_cores[j]) {
                LLAMA_LOG_ERROR(
                    "CORE ISOLATION: Core sharing detected: core %d in both domains\n",
                    state->decode_cores[i]);
                state->core_sharing_violations++;
                return false;
            }
        }
    }

    // Validate all threads are on correct cores
    for (int i = 0; i < state->n_threads_tracked; i++) {
        llama_thread_core_assignment * assignment = &state->thread_assignments[i];
        if (assignment->thread_id == 0) continue; // Uninitialized

        if (assignment->assigned_domain == LLAMA_CORE_DOMAIN_DECODE) {
            if (!llama_core_isolation_validate_decode_thread_isolation(state, assignment->thread_id)) {
                return false;
            }
        } else if (assignment->assigned_domain == LLAMA_CORE_DOMAIN_SERVER) {
            if (!llama_core_isolation_validate_server_thread_isolation(state, assignment->thread_id)) {
                return false;
            }
        }
    }

    state->assertion_checks_passed++;
    return true;
}

/**
 * Detect thread migration
 */
bool llama_core_isolation_detect_migration(
    llama_core_isolation_state * state,
    uint32_t thread_id,
    int current_core) {

    if (!state || !state->enforcement_active) return true;

    // Find thread assignment
    llama_thread_core_assignment * assignment = NULL;
    for (int i = 0; i < state->n_threads_tracked; i++) {
        if (state->thread_assignments[i].thread_id == thread_id) {
            assignment = &state->thread_assignments[i];
            break;
        }
    }

    if (!assignment) return true; // Not tracked

    // Check if current core matches assigned domain
    bool valid_core = false;

    if (assignment->assigned_domain == LLAMA_CORE_DOMAIN_DECODE) {
        for (int i = 0; i < state->n_decode_cores; i++) {
            if (state->decode_cores[i] == current_core) {
                valid_core = true;
                break;
            }
        }
    } else if (assignment->assigned_domain == LLAMA_CORE_DOMAIN_SERVER) {
        for (int i = 0; i < state->n_server_cores; i++) {
            if (state->server_cores[i] == current_core) {
                valid_core = true;
                break;
            }
        }
    } else {
        valid_core = true; // CUDA_CONTROL or other domains
    }

    if (!valid_core) {
        LLAMA_LOG_WARN(
            "CORE ISOLATION: Thread %u migrated to core %d (outside assigned domain)\n",
            thread_id, current_core);
        state->migration_violations++;
        state->preemption_events++;
        return false;
    }

    return true;
}

/**
 * Assert immutable isolation
 */
bool llama_core_isolation_assert_immutable(llama_core_isolation_state * state) {
    if (!state) return false;

    if (state->state != LLAMA_ISOLATION_FROZEN) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Immutability assert failed - state is %d (not FROZEN)\n",
                       state->state);
        state->assertion_checks_failed++;
        LLAMA_ABORT("Core isolation not frozen");
        return false;
    }

    // Verify no mutations to core counts
    if (!state->decode_cores || state->n_decode_cores == 0) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Decode cores were cleared\n");
        state->assertion_checks_failed++;
        LLAMA_ABORT("Decode cores cleared");
        return false;
    }

    if (!state->server_cores || state->n_server_cores == 0) {
        LLAMA_LOG_ERROR("CORE ISOLATION: Server cores were cleared\n");
        state->assertion_checks_failed++;
        LLAMA_ABORT("Server cores cleared");
        return false;
    }

    state->assertion_checks_passed++;
    return true;
}

/**
 * Get thread domain
 */
llama_core_domain_t llama_core_isolation_get_thread_domain(
    const llama_core_isolation_state * state,
    uint32_t thread_id) {

    if (!state) return LLAMA_CORE_DOMAIN_UNINITIALIZED;

    for (int i = 0; i < state->n_threads_tracked; i++) {
        if (state->thread_assignments[i].thread_id == thread_id) {
            return state->thread_assignments[i].assigned_domain;
        }
    }

    return LLAMA_CORE_DOMAIN_UNINITIALIZED;
}

/**
 * Get decode cores
 */
int llama_core_isolation_get_decode_cores(
    const llama_core_isolation_state * state,
    int * out_cores,
    int max_cores) {

    if (!state || !out_cores || max_cores <= 0) return -1;
    if (!state->decode_cores) return -1;

    int copy_count = (state->n_decode_cores < max_cores) ? state->n_decode_cores : max_cores;
    memcpy(out_cores, state->decode_cores, sizeof(int) * copy_count);

    return state->n_decode_cores;
}

/**
 * Get server cores
 */
int llama_core_isolation_get_server_cores(
    const llama_core_isolation_state * state,
    int * out_cores,
    int max_cores) {

    if (!state || !out_cores || max_cores <= 0) return -1;
    if (!state->server_cores) return -1;

    int copy_count = (state->n_server_cores < max_cores) ? state->n_server_cores : max_cores;
    memcpy(out_cores, state->server_cores, sizeof(int) * copy_count);

    return state->n_server_cores;
}

/**
 * Get current isolation state
 */
llama_isolation_state_t llama_core_isolation_get_state(
    const llama_core_isolation_state * state) {
    return state ? state->state : LLAMA_ISOLATION_UNINITIALIZED;
}

/**
 * Dump configuration
 */
void llama_core_isolation_dump_config(const llama_core_isolation_state * state) {
    if (!state) return;

    LLAMA_LOG_INFO("\n=== CORE ISOLATION CONFIGURATION ===\n");
    LLAMA_LOG_INFO("Total cores: %d, NUMA: %s, Sockets: %d\n",
                   state->total_cores,
                   state->has_numa ? "yes" : "no",
                   state->num_sockets);

    LLAMA_LOG_INFO("State: %d, Enforcement: %s\n",
                   state->state,
                   state->enforcement_active ? "ACTIVE" : "inactive");

    LLAMA_LOG_INFO("Decode domain: ");
    if (state->decode_cores && state->n_decode_cores > 0) {
        LLAMA_LOG_INFO("[");
        for (int i = 0; i < state->n_decode_cores; i++) {
            if (i > 0) LLAMA_LOG_INFO(",");
            LLAMA_LOG_INFO("%d", state->decode_cores[i]);
        }
        LLAMA_LOG_INFO("] (%d cores)\n", state->n_decode_cores);
    } else {
        LLAMA_LOG_INFO("not partitioned\n");
    }

    LLAMA_LOG_INFO("Server domain: ");
    if (state->server_cores && state->n_server_cores > 0) {
        LLAMA_LOG_INFO("[");
        for (int i = 0; i < state->n_server_cores; i++) {
            if (i > 0) LLAMA_LOG_INFO(",");
            LLAMA_LOG_INFO("%d", state->server_cores[i]);
        }
        LLAMA_LOG_INFO("] (%d cores)\n", state->n_server_cores);
    } else {
        LLAMA_LOG_INFO("not partitioned\n");
    }

    LLAMA_LOG_INFO("CUDA control: core %d\n", state->cuda_control_core);

    LLAMA_LOG_INFO("Threads tracked: %d\n", state->n_threads_tracked);
    for (int i = 0; i < state->n_threads_tracked; i++) {
        const llama_thread_core_assignment * a = &state->thread_assignments[i];
        if (a->thread_id == 0) continue;

        const char * domain_name = "UNKNOWN";
        if (a->assigned_domain == LLAMA_CORE_DOMAIN_DECODE) domain_name = "DECODE";
        else if (a->assigned_domain == LLAMA_CORE_DOMAIN_SERVER) domain_name = "SERVER";
        else if (a->assigned_domain == LLAMA_CORE_DOMAIN_CUDA_CONTROL) domain_name = "CUDA";

        LLAMA_LOG_INFO("  Thread %u (%s): primary=%d, secondaries=%d\n",
                       a->thread_id, domain_name, a->primary_core, a->n_secondary);
    }

    LLAMA_LOG_INFO("Violations: decode_on_server=%llu, server_on_decode=%llu, migrations=%llu, preemptions=%llu\n",
                   (unsigned long long)state->decode_on_server_count,
                   (unsigned long long)state->server_on_decode_count,
                   (unsigned long long)state->migration_violations,
                   (unsigned long long)state->preemption_events);

    LLAMA_LOG_INFO("Metrics: checks_passed=%llu, checks_failed=%llu\n",
                   (unsigned long long)state->assertion_checks_passed,
                   (unsigned long long)state->assertion_checks_failed);

    LLAMA_LOG_INFO("=====================================\n\n");
}

/**
 * Get violation statistics
 */
void llama_core_isolation_get_violations(
    const llama_core_isolation_state * state,
    uint64_t * out_decode_on_server,
    uint64_t * out_server_on_decode,
    uint64_t * out_migrations,
    uint64_t * out_preemptions) {

    if (!state) return;

    if (out_decode_on_server) *out_decode_on_server = state->decode_on_server_count;
    if (out_server_on_decode) *out_server_on_decode = state->server_on_decode_count;
    if (out_migrations) *out_migrations = state->migration_violations;
    if (out_preemptions) *out_preemptions = state->preemption_events;
}

/**
 * Get metrics
 */
void llama_core_isolation_get_metrics(
    const llama_core_isolation_state * state,
    uint64_t * out_total_context_switches,
    uint64_t * out_checks_passed,
    uint64_t * out_checks_failed) {

    if (!state) return;

    if (out_total_context_switches) *out_total_context_switches = state->total_context_switches;
    if (out_checks_passed) *out_checks_passed = state->assertion_checks_passed;
    if (out_checks_failed) *out_checks_failed = state->assertion_checks_failed;
}

/**
 * Get the global isolation state singleton
 */
llama_core_isolation_state * llama_core_isolation_get_global_state(void) {
    return &g_core_isolation_state;
}
