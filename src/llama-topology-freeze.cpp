/**
 * Decode Topology Freeze Enforcement Implementation
 *
 * Locks complete thread topology before decode loop.
 * Ensures no dynamic thread mutations during execution.
 */

#include "llama-topology-freeze.h"
#include "llama-impl.h"

#include <cstring>
#include <algorithm>
#include <ctime>

/**
 * Initialize topology freeze state
 */
void llama_topology_freeze_init(llama_topology_freeze_state * state) {
    if (!state) return;

    state->enforce_active = false;
    state->topology_frozen = false;
    state->in_decode_session = false;

    state->n_decode_threads = 0;
    state->n_cuda_threads = 0;
    state->n_server_threads = 0;
    state->total_threads = 0;

    state->decode_configs = nullptr;
    state->server_configs = nullptr;
    state->decode_allocs = nullptr;
    state->server_allocs = nullptr;

    state->scheduling_policy = 0;
    state->priority_level = 0;
    state->use_elevated_priority = false;

    state->freeze_timestamp = 0;
    state->freeze_sequence_number = 0;
    state->current_sequence_number = 0;

    state->thread_creation_attempts = 0;
    state->thread_destruction_attempts = 0;
    state->thread_resize_attempts = 0;
    state->role_change_attempts = 0;
    state->affinity_change_attempts = 0;
}

/**
 * Pre-initialize all decode threads
 */
bool llama_topology_freeze_pre_initialize(
    llama_topology_freeze_state * state,
    int n_decode_threads,
    int n_cuda_threads,
    int n_server_threads) {

    if (!state || n_decode_threads <= 0) {
        return false;
    }

    state->n_decode_threads = n_decode_threads;
    state->n_cuda_threads = std::max(1, n_cuda_threads);
    state->n_server_threads = n_server_threads;
    state->total_threads = n_decode_threads + state->n_cuda_threads + n_server_threads;

    // Allocate configuration structures
    state->decode_configs = new llama_thread_config[n_decode_threads];
    if (n_server_threads > 0) {
        state->server_configs = new llama_thread_config[n_server_threads];
    }

    // Allocate buffer structures
    state->decode_allocs = new llama_thread_allocations[n_decode_threads];
    if (n_server_threads > 0) {
        state->server_allocs = new llama_thread_allocations[n_server_threads];
    }

    // Initialize decode thread configs
    for (int i = 0; i < n_decode_threads; i++) {
        state->decode_configs[i].thread_id = -1;  // Will be set when threads created
        state->decode_configs[i].logical_id = i;
        state->decode_configs[i].cpu_core = -1;   // Not pinned yet
        state->decode_configs[i].priority = 0;
        state->decode_configs[i].affinity_mask = 0;
        state->decode_configs[i].is_pinned = false;
        state->decode_configs[i].is_elevated_priority = false;
        state->decode_configs[i].role = LLAMA_THREAD_ROLE_DECODE_WORKER;
        state->decode_configs[i].domain = LLAMA_THREAD_DOMAIN_DECODE;
    }

    // Initialize CUDA thread config
    state->cuda_config.thread_id = -1;
    state->cuda_config.logical_id = n_decode_threads;
    state->cuda_config.cpu_core = -1;
    state->cuda_config.role = LLAMA_THREAD_ROLE_CUDA_DISPATCH;
    state->cuda_config.domain = LLAMA_THREAD_DOMAIN_CUDA_CONTROL;

    // Initialize server thread configs if any
    for (int i = 0; i < n_server_threads; i++) {
        state->server_configs[i].thread_id = -1;
        state->server_configs[i].logical_id = n_decode_threads + 1 + i;
        state->server_configs[i].cpu_core = -1;
        state->server_configs[i].role = LLAMA_THREAD_ROLE_SERVER_WORKER;
        state->server_configs[i].domain = LLAMA_THREAD_DOMAIN_SERVER;
    }

    state->enforce_active = true;

    LLAMA_LOG_INFO(
        "TOPOLOGY FREEZE: Pre-initialized (%d decode, %d CUDA, %d server threads)\n",
        n_decode_threads, state->n_cuda_threads, n_server_threads);

    return true;
}

/**
 * [CRITICAL] Freeze topology
 */
bool llama_topology_freeze_lock(llama_topology_freeze_state * state) {
    if (!state || !state->enforce_active) {
        return false;
    }

    // Validate all configurations are set
    if (state->n_decode_threads == 0) {
        LLAMA_LOG_ERROR("TOPOLOGY FREEZE: No decode threads configured\n");
        return false;
    }

    // Capture freeze timestamp
    state->freeze_timestamp = (uint64_t)std::time(nullptr);
    state->freeze_sequence_number = state->current_sequence_number;
    state->topology_frozen = true;
    state->in_decode_session = true;

    LLAMA_LOG_INFO(
        "TOPOLOGY FREEZE: Locked (total %d threads, sequence %u)\n",
        state->total_threads, state->freeze_sequence_number);

    return true;
}

/**
 * [CRITICAL] Validate topology unchanged
 */
bool llama_topology_freeze_validate_unchanged(
    llama_topology_freeze_state * state) {

    if (!state || !state->topology_frozen || !state->in_decode_session) {
        return true;
    }

    // Check if sequence number changed (indicates mutation attempt)
    if (state->current_sequence_number != state->freeze_sequence_number) {
        LLAMA_LOG_ERROR(
            "TOPOLOGY FREEZE: Topology mutation detected!\n"
            "  Frozen sequence: %u\n"
            "  Current sequence: %u\n"
            "  Mutations attempted: create=%lu, destroy=%lu, resize=%lu\n",
            state->freeze_sequence_number,
            state->current_sequence_number,
            state->thread_creation_attempts,
            state->thread_destruction_attempts,
            state->thread_resize_attempts);
        LLAMA_ABORT("Topology freeze violation");
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Enforce separate domains
 */
bool llama_topology_freeze_enforce_domains(
    llama_topology_freeze_state * state) {

    if (!state) {
        return false;
    }

    // Validate decode threads are in DECODE domain
    for (int i = 0; i < state->n_decode_threads; i++) {
        if (state->decode_configs[i].domain != LLAMA_THREAD_DOMAIN_DECODE) {
            LLAMA_LOG_ERROR(
                "TOPOLOGY FREEZE: Decode thread %d not in DECODE domain!\n", i);
            return false;
        }
    }

    // Validate CUDA thread is in CUDA_CONTROL domain
    if (state->cuda_config.domain != LLAMA_THREAD_DOMAIN_CUDA_CONTROL) {
        LLAMA_LOG_ERROR(
            "TOPOLOGY FREEZE: CUDA thread not in CUDA_CONTROL domain!\n");
        return false;
    }

    // Validate server threads are in SERVER domain
    for (int i = 0; i < state->n_server_threads; i++) {
        if (state->server_configs[i].domain != LLAMA_THREAD_DOMAIN_SERVER) {
            LLAMA_LOG_ERROR(
                "TOPOLOGY FREEZE: Server thread %d not in SERVER domain!\n", i);
            return false;
        }
    }

    return true;
}

/**
 * [CRITICAL] Lock thread count
 */
bool llama_topology_freeze_lock_thread_count(
    llama_topology_freeze_state * state,
    int current_thread_count) {

    if (!state) {
        return false;
    }

    if (current_thread_count != state->total_threads) {
        LLAMA_LOG_ERROR(
            "TOPOLOGY FREEZE: Thread count mismatch at lock!\n"
            "  Expected: %d\n"
            "  Current: %d\n",
            state->total_threads, current_thread_count);
        LLAMA_ABORT("Thread count lock failed");
        return false;
    }

    return true;
}

/**
 * Set CPU affinity
 */
bool llama_topology_freeze_set_affinity(
    llama_topology_freeze_state * state,
    const int * core_ids,
    int n_cores) {

    if (!state || !core_ids || n_cores <= 0) {
        return false;
    }

    if (n_cores > state->n_decode_threads) {
        LLAMA_LOG_WARN(
            "TOPOLOGY FREEZE: More cores than decode threads (%d > %d)\n",
            n_cores, state->n_decode_threads);
    }

    // Assign cores to decode threads
    for (int i = 0; i < n_cores && i < state->n_decode_threads; i++) {
        state->decode_configs[i].cpu_core = core_ids[i];
        state->decode_configs[i].is_pinned = true;
        state->decode_configs[i].affinity_mask = (1ULL << core_ids[i]);
    }

    LLAMA_LOG_INFO("TOPOLOGY FREEZE: CPU affinity set for %d threads\n", n_cores);
    return true;
}

/**
 * [CRITICAL] Pre-allocate structures
 */
bool llama_topology_freeze_pre_allocate_structures(
    llama_topology_freeze_state * state,
    size_t buffer_size_per_thread,
    size_t compute_region_size) {

    if (!state || buffer_size_per_thread == 0) {
        return false;
    }

    // Pre-allocate decode thread buffers
    for (int i = 0; i < state->n_decode_threads; i++) {
        state->decode_allocs[i].buffer_size = buffer_size_per_thread;
        state->decode_allocs[i].thread_local_buffer = malloc(buffer_size_per_thread);

        if (!state->decode_allocs[i].thread_local_buffer) {
            LLAMA_LOG_ERROR("TOPOLOGY FREEZE: Failed to allocate TLS buffer for thread %d\n", i);
            return false;
        }

        state->decode_allocs[i].compute_region_size = compute_region_size;
        if (compute_region_size > 0) {
            state->decode_allocs[i].temp_compute_region = malloc(compute_region_size);
            if (!state->decode_allocs[i].temp_compute_region) {
                LLAMA_LOG_ERROR(
                    "TOPOLOGY FREEZE: Failed to allocate compute region for thread %d\n", i);
                return false;
            }
        }
    }

    // Pre-allocate server thread buffers
    for (int i = 0; i < state->n_server_threads; i++) {
        state->server_allocs[i].buffer_size = buffer_size_per_thread;
        state->server_allocs[i].thread_local_buffer = malloc(buffer_size_per_thread);

        if (!state->server_allocs[i].thread_local_buffer) {
            LLAMA_LOG_ERROR("TOPOLOGY FREEZE: Failed to allocate TLS buffer for server thread %d\n", i);
            return false;
        }
    }

    LLAMA_LOG_INFO(
        "TOPOLOGY FREEZE: Pre-allocated structures (%zu bytes per thread)\n",
        buffer_size_per_thread);

    return true;
}

/**
 * [CRITICAL] Validate no role switches
 */
bool llama_topology_freeze_validate_no_role_switches(
    llama_topology_freeze_state * state) {

    if (!state || !state->topology_frozen) {
        return true;
    }

    // Check that decode thread roles haven't changed
    for (int i = 0; i < state->n_decode_threads; i++) {
        if (state->decode_configs[i].role != LLAMA_THREAD_ROLE_DECODE_WORKER) {
            LLAMA_LOG_ERROR(
                "TOPOLOGY FREEZE: Decode thread %d role changed!\n"
                "  Original: DECODE_WORKER\n"
                "  Current: %d\n",
                i, state->decode_configs[i].role);
            LLAMA_ABORT("Thread role mutation detected");
            return false;
        }
    }

    // Check that CUDA thread role unchanged
    if (state->cuda_config.role != LLAMA_THREAD_ROLE_CUDA_DISPATCH) {
        LLAMA_LOG_ERROR("TOPOLOGY FREEZE: CUDA thread role changed!\n");
        LLAMA_ABORT("CUDA thread role mutation");
        return false;
    }

    return true;
}

/**
 * [CRITICAL] Freeze scheduling policy
 */
bool llama_topology_freeze_lock_scheduling(
    llama_topology_freeze_state * state,
    int policy,
    int priority,
    bool use_elevated) {

    if (!state) {
        return false;
    }

    state->scheduling_policy = policy;
    state->priority_level = priority;
    state->use_elevated_priority = use_elevated;

    LLAMA_LOG_INFO(
        "TOPOLOGY FREEZE: Scheduling policy locked (policy=%d, priority=%d, elevated=%d)\n",
        policy, priority, use_elevated ? 1 : 0);

    return true;
}

/**
 * [CRITICAL] Eliminate per-token checks
 */
bool llama_topology_freeze_eliminate_per_token_checks(
    llama_topology_freeze_state * state) {

    if (!state) {
        return true;
    }

    // In a real implementation, this would scan the decode loop code
    // and verify that topology recomputation code has been removed.

    // For now, return true if topology is frozen (which indicates
    // that per-token checks should be eliminated)

    return state->topology_frozen;
}

/**
 * Record mutation attempt
 */
void llama_topology_freeze_record_mutation_attempt(
    llama_topology_freeze_state * state,
    const char * mutation_type) {

    if (!state || !mutation_type) {
        return;
    }

    // If frozen, this is a violation
    if (state->topology_frozen && state->in_decode_session) {
        LLAMA_LOG_ERROR(
            "TOPOLOGY FREEZE: Mutation attempt during decode: %s\n",
            mutation_type);
    }

    // Increment attempt counter
    if (strcmp(mutation_type, "create") == 0) {
        state->thread_creation_attempts++;
    }
    else if (strcmp(mutation_type, "destroy") == 0) {
        state->thread_destruction_attempts++;
    }
    else if (strcmp(mutation_type, "resize") == 0) {
        state->thread_resize_attempts++;
    }
    else if (strcmp(mutation_type, "role_change") == 0) {
        state->role_change_attempts++;
    }
    else if (strcmp(mutation_type, "affinity_change") == 0) {
        state->affinity_change_attempts++;
    }

    // Increment sequence number to indicate mutation
    state->current_sequence_number++;
}

/**
 * [CRITICAL] Assert immutable
 */
bool llama_topology_freeze_assert_immutable(
    llama_topology_freeze_state * state) {

    if (!state || !state->topology_frozen || !state->in_decode_session) {
        return true;
    }

    // Assert no mutations attempted
    if (state->thread_creation_attempts > 0 ||
        state->thread_destruction_attempts > 0 ||
        state->thread_resize_attempts > 0 ||
        state->role_change_attempts > 0 ||
        state->affinity_change_attempts > 0) {

        LLAMA_LOG_ERROR(
            "TOPOLOGY FREEZE: Immutability violated!\n"
            "  Create attempts: %lu\n"
            "  Destroy attempts: %lu\n"
            "  Resize attempts: %lu\n"
            "  Role changes: %lu\n"
            "  Affinity changes: %lu\n",
            state->thread_creation_attempts,
            state->thread_destruction_attempts,
            state->thread_resize_attempts,
            state->role_change_attempts,
            state->affinity_change_attempts);

        LLAMA_ABORT("Topology immutability assertion failed");
        return false;
    }

    return true;
}

/**
 * Get configuration
 */
llama_thread_config llama_topology_freeze_get_config(
    const llama_topology_freeze_state * state,
    int thread_id) {

    if (!state) {
        return {};
    }

    // Check if it's a decode thread
    if (thread_id < state->n_decode_threads) {
        return state->decode_configs[thread_id];
    }

    // Check if it's the CUDA thread
    if (thread_id == state->n_decode_threads) {
        return state->cuda_config;
    }

    // Server thread
    int server_idx = thread_id - state->n_decode_threads - 1;
    if (server_idx >= 0 && server_idx < state->n_server_threads) {
        return state->server_configs[server_idx];
    }

    return {};
}

/**
 * Get metrics - inline implementation
 */
// Metrics returned directly from function call

/**
 * Dump configuration
 */
void llama_topology_freeze_dump_config(
    const llama_topology_freeze_state * state) {

    if (!state) {
        return;
    }

    LLAMA_LOG_INFO("TOPOLOGY FREEZE CONFIGURATION:\n");
    LLAMA_LOG_INFO("  Decode threads: %d\n", state->n_decode_threads);
    LLAMA_LOG_INFO("  CUDA threads: %d\n", state->n_cuda_threads);
    LLAMA_LOG_INFO("  Server threads: %d\n", state->n_server_threads);
    LLAMA_LOG_INFO("  Total: %d\n", state->total_threads);
    LLAMA_LOG_INFO("  Frozen: %s\n", state->topology_frozen ? "YES" : "NO");
    LLAMA_LOG_INFO("  In decode: %s\n", state->in_decode_session ? "YES" : "NO");

    if (state->topology_frozen) {
        LLAMA_LOG_INFO("  Freeze timestamp: %lu\n", state->freeze_timestamp);
        LLAMA_LOG_INFO("  Freeze sequence: %u\n", state->freeze_sequence_number);
        LLAMA_LOG_INFO("  Current sequence: %u\n", state->current_sequence_number);
    }

    LLAMA_LOG_INFO("DECODE THREAD CONFIGURATION:\n");
    for (int i = 0; i < state->n_decode_threads; i++) {
        const char * affinity_str = state->decode_configs[i].is_pinned ? "pinned" : "unpinned";
        LLAMA_LOG_INFO(
            "  Thread %d: cpu_core=%d (%s), priority=%d\n",
            i,
            state->decode_configs[i].cpu_core,
            affinity_str,
            state->decode_configs[i].priority);
    }

    LLAMA_LOG_INFO("MUTATION ATTEMPT COUNTERS:\n");
    LLAMA_LOG_INFO("  Create: %lu\n", state->thread_creation_attempts);
    LLAMA_LOG_INFO("  Destroy: %lu\n", state->thread_destruction_attempts);
    LLAMA_LOG_INFO("  Resize: %lu\n", state->thread_resize_attempts);
    LLAMA_LOG_INFO("  Role changes: %lu\n", state->role_change_attempts);
    LLAMA_LOG_INFO("  Affinity changes: %lu\n", state->affinity_change_attempts);
}

/**
 * Release topology freeze
 */
void llama_topology_freeze_release(llama_topology_freeze_state * state) {
    if (!state) {
        return;
    }

    // Dump final configuration
    llama_topology_freeze_dump_config(state);

    // Free allocated structures
    if (state->decode_configs) {
        delete[] state->decode_configs;
        state->decode_configs = nullptr;
    }

    if (state->server_configs) {
        delete[] state->server_configs;
        state->server_configs = nullptr;
    }

    if (state->decode_allocs) {
        for (int i = 0; i < state->n_decode_threads; i++) {
            if (state->decode_allocs[i].thread_local_buffer) {
                free(state->decode_allocs[i].thread_local_buffer);
            }
            if (state->decode_allocs[i].temp_compute_region) {
                free(state->decode_allocs[i].temp_compute_region);
            }
        }
        delete[] state->decode_allocs;
        state->decode_allocs = nullptr;
    }

    if (state->server_allocs) {
        for (int i = 0; i < state->n_server_threads; i++) {
            if (state->server_allocs[i].thread_local_buffer) {
                free(state->server_allocs[i].thread_local_buffer);
            }
        }
        delete[] state->server_allocs;
        state->server_allocs = nullptr;
    }

    state->topology_frozen = false;
    state->in_decode_session = false;

    LLAMA_LOG_INFO("TOPOLOGY FREEZE: Released\n");
}
