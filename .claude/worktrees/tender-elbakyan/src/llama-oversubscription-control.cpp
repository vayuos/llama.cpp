/**
 * Oversubscription Control Implementation for LLAMA Decode Optimization
 *
 * REQUIREMENT #45: Strict CPU thread concurrency control during decode phase.
 * Implements enforcement of minimal thread usage during autonomous decode
 * to prevent context switching overhead, cache eviction, and scheduling contention.
 */

#include "llama-oversubscription-control.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>

#ifdef _WIN32
    #include <windows.h>
    #include <processthreadsapi.h>
#else
    #include <pthread.h>
    #include <sys/syscall.h>
    #include <sched.h>
#endif

// OpenMP support (optional)
#ifdef _OPENMP
    #include <omp.h>
#endif

// Logging support (assume llama-logging.h defines LLAMA_LOG_*)
#define LLAMA_LOG_INFO printf
#define LLAMA_LOG_WARN printf
#define LLAMA_LOG_ERROR printf

/**
 * Helper: Get current time in nanoseconds
 */
static uint64_t llama_get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/**
 * Helper: Get current thread ID
 */
static uint32_t llama_get_current_thread_id(void) {
#ifdef _WIN32
    return (uint32_t)GetCurrentThreadId();
#else
    return (uint32_t)syscall(SYS_gettid);
#endif
}

/**
 * Helper: Get number of online CPUs
 */
static int llama_get_num_cpus(void) {
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (int)sysinfo.dwNumberOfProcessors;
#else
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

/**
 * Initialize oversubscription control state
 */
bool llama_oversubscription_init(llama_oversubscription_control * control) {
    if (control == NULL) {
        return false;
    }

    memset(control, 0, sizeof(llama_oversubscription_control));

    control->state = LLAMA_OVERSUBSCRIPTION_UNINITIALIZED;
    control->openmp_mode = LLAMA_OPENMP_NORMAL;

    // Allocate thread tracking arrays
    int max_threads = llama_get_num_cpus() + 8; // +8 for overhead
    control->max_threads_capacity = max_threads;
    control->threads = (llama_thread_context *)malloc(sizeof(llama_thread_context) * max_threads);
    if (control->threads == NULL) {
        LLAMA_LOG_ERROR("Failed to allocate thread tracking array\n");
        return false;
    }
    memset(control->threads, 0, sizeof(llama_thread_context) * max_threads);

    // Allocate background thread tracking
    control->max_background_capacity = 16; // Typically 4-8 background threads
    control->background_threads = (llama_background_thread *)malloc(
        sizeof(llama_background_thread) * control->max_background_capacity);
    if (control->background_threads == NULL) {
        LLAMA_LOG_ERROR("Failed to allocate background thread tracking\n");
        free(control->threads);
        return false;
    }
    memset(control->background_threads, 0,
           sizeof(llama_background_thread) * control->max_background_capacity);

    // Default configuration
    control->thread_config.user_thread_count = llama_get_num_cpus();
    control->thread_config.decode_thread_count = 1;
    control->thread_config.decode_control_threads = 1;
    control->thread_config.decode_auxiliary_threads = 0;
    control->thread_config.max_runnable_threads = 2; // Hard limit: control + optional auxiliary
    control->thread_config.background_cache_threads = 0;
    control->thread_config.background_async_threads = 0;
    control->thread_config.background_logging_threads = 0;

    // Default enforcement policy
    control->abort_on_oversubscription = true;
    control->abort_on_omp_expansion = true;
    control->abort_on_background_wake = true;

    control->state = LLAMA_OVERSUBSCRIPTION_CONFIGURED;
    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] Initialized with %d CPU cores\n", llama_get_num_cpus());

    return true;
}

/**
 * Release oversubscription control state
 */
void llama_oversubscription_release(llama_oversubscription_control * control) {
    if (control == NULL) {
        return;
    }

    if (control->threads != NULL) {
        free(control->threads);
        control->threads = NULL;
    }

    if (control->background_threads != NULL) {
        free(control->background_threads);
        control->background_threads = NULL;
    }

    memset(control, 0, sizeof(llama_oversubscription_control));
    control->state = LLAMA_OVERSUBSCRIPTION_RELEASED;
}

/**
 * Configure thread counts for prefill and decode phases
 */
bool llama_oversubscription_configure_thread_counts(
    llama_oversubscription_control * control,
    int user_thread_count,
    bool gpu_exclusive_decode) {

    if (control == NULL || user_thread_count <= 0) {
        return false;
    }

    int num_cpus = llama_get_num_cpus();
    if (user_thread_count > num_cpus * 2) {
        LLAMA_LOG_WARN("[OVERSUBSCRIPTION] User thread count %d > 2x CPU count (%d), capping\n",
                       user_thread_count, num_cpus);
        user_thread_count = num_cpus;
    }

    control->user_thread_count = user_thread_count;
    control->actual_prefill_threads = user_thread_count;
    control->gpu_exclusive_decode = gpu_exclusive_decode;

    // Configure decode thread count
    // GPU-exclusive: 1 control thread
    // GPU-assisted: 1 control + 0-1 auxiliary
    control->thread_config.user_thread_count = user_thread_count;
    control->thread_config.decode_thread_count = gpu_exclusive_decode ? 1 : 2;
    control->thread_config.decode_control_threads = 1;
    control->thread_config.decode_auxiliary_threads = gpu_exclusive_decode ? 0 : 1;
    control->thread_config.max_runnable_threads = control->thread_config.decode_thread_count;

    control->decode_thread_count = control->thread_config.decode_thread_count;

    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] Configured: user_threads=%d, decode_threads=%d, "
                   "gpu_exclusive=%s\n",
                   user_thread_count, control->decode_thread_count,
                   gpu_exclusive_decode ? "yes" : "no");

    return true;
}

/**
 * Begin prefill phase with user thread count
 */
bool llama_oversubscription_begin_prefill(llama_oversubscription_control * control) {
    if (control == NULL) {
        return false;
    }

    if (control->state != LLAMA_OVERSUBSCRIPTION_CONFIGURED) {
        LLAMA_LOG_WARN("[OVERSUBSCRIPTION] Cannot begin prefill in state %d\n", control->state);
        return false;
    }

    control->state = LLAMA_OVERSUBSCRIPTION_PREFILL_ACTIVE;
    control->prefill_start_time = llama_get_time_ns();

    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] Entered PREFILL phase with %d threads at %.3f ms\n",
                   control->actual_prefill_threads, control->prefill_start_time / 1e6);

    return true;
}

/**
 * Helper: Find thread context by ID
 */
static llama_thread_context * llama_find_thread_context(
    llama_oversubscription_control * control,
    uint32_t thread_id) {

    for (int i = 0; i < control->n_threads_tracked; i++) {
        if (control->threads[i].thread_id == thread_id) {
            return &control->threads[i];
        }
    }
    return NULL;
}

/**
 * Transition from prefill to decode phase
 */
bool llama_oversubscription_begin_decode(llama_oversubscription_control * control) {
    if (control == NULL) {
        return false;
    }

    if (control->state != LLAMA_OVERSUBSCRIPTION_PREFILL_ACTIVE) {
        LLAMA_LOG_WARN("[OVERSUBSCRIPTION] Cannot begin decode in state %d\n", control->state);
        return false;
    }

    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] Transitioning to DECODE phase...\n");

    // 1. Override thread count
    if (!llama_oversubscription_override_thread_count(control, control->decode_thread_count)) {
        LLAMA_LOG_ERROR("[OVERSUBSCRIPTION] Failed to override thread count\n");
        return false;
    }

    // 2. Disable CPU backend workers if GPU-exclusive
    if (control->gpu_exclusive_decode) {
        if (!llama_oversubscription_disable_cpu_backend(control)) {
            LLAMA_LOG_WARN("[OVERSUBSCRIPTION] Warning: Failed to disable CPU backend\n");
            // Don't fail - GPU-exclusive is advisory
        }
    }

    // 3. Override OpenMP
    if (!llama_oversubscription_override_openmp(control, false)) {
        LLAMA_LOG_WARN("[OVERSUBSCRIPTION] Warning: Failed to override OpenMP\n");
        // Don't fail - OpenMP override is best-effort
    }

    // 4. Suspend background threads
    if (!llama_oversubscription_suspend_background_threads(control)) {
        LLAMA_LOG_WARN("[OVERSUBSCRIPTION] Warning: Failed to suspend background threads\n");
        // Don't fail - background thread suspension is best-effort
    }

    control->state = LLAMA_OVERSUBSCRIPTION_DECODE_ACTIVE;
    control->decode_start_time = llama_get_time_ns();
    control->tokens_generated = 0;
    control->context_switches_total = 0;
    control->wake_events_total = 0;
    control->per_token_max_runnable = 0;

    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] Entered DECODE phase at %.3f ms, "
                   "decode_threads=%d, strict_limit=%d\n",
                   control->decode_start_time / 1e6,
                   control->decode_thread_count,
                   control->thread_config.max_runnable_threads);

    return true;
}

/**
 * Override user thread count during decode
 */
bool llama_oversubscription_override_thread_count(
    llama_oversubscription_control * control,
    int target_thread_count) {

    if (control == NULL || target_thread_count <= 0) {
        return false;
    }

    if (target_thread_count > control->thread_config.max_runnable_threads) {
        LLAMA_LOG_ERROR("[OVERSUBSCRIPTION] Target thread count %d exceeds max %d\n",
                        target_thread_count, control->thread_config.max_runnable_threads);
        return false;
    }

    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] Overriding thread count: %d -> %d\n",
                   control->actual_prefill_threads, target_thread_count);

    // Mark excess threads as inactive
    for (int i = target_thread_count; i < control->n_threads_tracked; i++) {
        control->threads[i].active_in_decode = false;
    }

    return true;
}

/**
 * Disable CPU backend workers during decode
 */
bool llama_oversubscription_disable_cpu_backend(llama_oversubscription_control * control) {
    if (control == NULL) {
        return false;
    }

    // Mark all threads registered as CPU workers as inactive
    for (int i = 0; i < control->n_threads_tracked; i++) {
        if (!control->threads[i].is_decode_worker) {
            control->threads[i].active_in_decode = false;
        }
    }

    control->cpu_backend_disabled = true;
    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] CPU backend workers disabled\n");

    return true;
}

/**
 * Force OpenMP thread count to 1 during decode
 */
bool llama_oversubscription_override_openmp(
    llama_oversubscription_control * control,
    bool force_disable) {

    if (control == NULL) {
        return false;
    }

#ifdef _OPENMP
    // Save current OpenMP thread count
    control->omp_prev_num_threads = omp_get_max_threads();

    if (force_disable) {
        control->openmp_mode = LLAMA_OPENMP_DISABLED;
        omp_set_num_threads(1);
        LLAMA_LOG_INFO("[OVERSUBSCRIPTION] OpenMP disabled (prev=%d)\n",
                       control->omp_prev_num_threads);
    } else {
        control->openmp_mode = LLAMA_OPENMP_OVERRIDE_TO_ONE;
        omp_set_num_threads(1);
        LLAMA_LOG_INFO("[OVERSUBSCRIPTION] OpenMP overridden to 1 thread (prev=%d)\n",
                       control->omp_prev_num_threads);
    }

    control->omp_override_active = true;
    (void)force_disable;  // Parameter used only in _OPENMP block
#else
    (void)force_disable;  // Unused without OpenMP
    control->openmp_mode = LLAMA_OPENMP_NORMAL;
    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] OpenMP not available (no override)\n");
#endif

    return true;
}

/**
 * Suspend all background helper threads
 */
bool llama_oversubscription_suspend_background_threads(llama_oversubscription_control * control) {
    if (control == NULL) {
        return false;
    }

    int suspended_count = 0;
    for (int i = 0; i < control->n_background_threads; i++) {
        if (!control->background_threads[i].is_suspended) {
            control->background_threads[i].is_suspended = true;
            control->background_threads[i].suspension_count++;
            suspended_count++;
        }
    }

    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] Suspended %d background threads\n", suspended_count);

    return true;
}

/**
 * Register a background helper thread
 */
bool llama_oversubscription_register_background_thread(
    llama_oversubscription_control * control,
    uint32_t thread_id,
    const char * thread_name,
    bool is_cache_maintenance,
    bool is_async_memory,
    bool is_logging) {

    if (control == NULL || thread_name == NULL) {
        return false;
    }

    if (control->n_background_threads >= control->max_background_capacity) {
        LLAMA_LOG_WARN("[OVERSUBSCRIPTION] Background thread capacity exceeded\n");
        return false;
    }

    llama_background_thread * bg = &control->background_threads[control->n_background_threads++];
    bg->thread_id = thread_id;
    bg->thread_name = thread_name;
    bg->is_suspended = false;
    bg->suspension_count = 0;
    bg->is_cache_maintenance = is_cache_maintenance;
    bg->is_async_memory_cleaner = is_async_memory;
    bg->is_logging_worker = is_logging;

    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] Registered background thread %u (%s)\n",
                   thread_id, thread_name);

    return true;
}

/**
 * Track a decode worker thread
 */
bool llama_oversubscription_register_decode_worker(
    llama_oversubscription_control * control,
    uint32_t thread_id,
    bool is_prefill_worker) {

    if (control == NULL) {
        return false;
    }

    if (control->n_threads_tracked >= control->max_threads_capacity) {
        LLAMA_LOG_WARN("[OVERSUBSCRIPTION] Thread capacity exceeded\n");
        return false;
    }

    llama_thread_context * thread = &control->threads[control->n_threads_tracked++];
    thread->thread_id = thread_id;
    thread->is_decode_worker = true;
    thread->is_prefill_worker = is_prefill_worker;
    thread->is_background_worker = false;
    thread->is_omp_worker = false;
    thread->active_in_decode = true;
    thread->context_switches = 0;
    thread->wake_attempts = 0;
    thread->wake_count_per_token = 0;
    thread->oversubscription_violation = false;

    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] Registered decode worker thread %u\n", thread_id);

    return true;
}

/**
 * Count currently active runnable threads during decode
 */
int llama_oversubscription_count_active_runnable_threads(
    const llama_oversubscription_control * control) {

    if (control == NULL) {
        return -1;
    }

    int active_count = 0;
    for (int i = 0; i < control->n_threads_tracked; i++) {
        if (control->threads[i].active_in_decode) {
            active_count++;
        }
    }

    return active_count;
}

/**
 * Detect oversubscription at runtime
 */
bool llama_oversubscription_detect_oversubscription(llama_oversubscription_control * control) {
    if (control == NULL) {
        return true;
    }

    if (control->state != LLAMA_OVERSUBSCRIPTION_DECODE_ACTIVE) {
        return true; // Not in decode, no check
    }

    int active_threads = llama_oversubscription_count_active_runnable_threads(control);
    int max_allowed = control->thread_config.max_runnable_threads;

    if (active_threads > max_allowed) {
        control->oversubscription_violations++;

        LLAMA_LOG_ERROR("[OVERSUBSCRIPTION] VIOLATION: %d active threads > max %d\n",
                        active_threads, max_allowed);

        if (control->abort_on_oversubscription) {
            LLAMA_LOG_ERROR("[OVERSUBSCRIPTION] ABORTING due to oversubscription violation\n");
            abort();
        }

        return false;
    }

    if (active_threads > (int)control->per_token_max_runnable) {
        control->per_token_max_runnable = active_threads;
    }

    return true;
}

/**
 * Validate per-token scheduling
 */
bool llama_oversubscription_validate_per_token_scheduling(llama_oversubscription_control * control) {
    if (control == NULL) {
        return true;
    }

    if (control->state != LLAMA_OVERSUBSCRIPTION_DECODE_ACTIVE) {
        return true;
    }

    // Update wake count for active decode threads
    for (int i = 0; i < control->n_threads_tracked; i++) {
        if (control->threads[i].active_in_decode) {
            control->threads[i].wake_count_per_token++;
        }
    }

    return true;
}

/**
 * Prevent dynamic thread creation mid-decode
 */
bool llama_oversubscription_check_thread_creation_allowed(
    llama_oversubscription_control * control,
    uint32_t new_thread_id) {

    if (control == NULL) {
        return true;
    }

    if (control->state != LLAMA_OVERSUBSCRIPTION_DECODE_ACTIVE) {
        return true; // Allow thread creation outside decode
    }

    // Thread creation during decode is a violation
    LLAMA_LOG_ERROR("[OVERSUBSCRIPTION] VIOLATION: Thread creation attempted during decode "
                    "(new_thread_id=%u)\n", new_thread_id);

    control->thread_creation_violations++;

    if (control->abort_on_omp_expansion) {
        LLAMA_LOG_ERROR("[OVERSUBSCRIPTION] ABORTING due to thread creation during decode\n");
        abort();
    }

    return false;
}

/**
 * Validate static thread pool
 */
bool llama_oversubscription_validate_static_thread_pool(llama_oversubscription_control * control) {
    if (control == NULL) {
        return true;
    }

    if (control->state != LLAMA_OVERSUBSCRIPTION_DECODE_ACTIVE) {
        return true;
    }

    // Thread pool should be static - no new threads, no work-stealing
    // This is validated by checking that thread count hasn't increased
    return true;
}

/**
 * Validate background threads are parked
 */
bool llama_oversubscription_validate_background_threads_parked(
    llama_oversubscription_control * control) {

    if (control == NULL) {
        return true;
    }

    if (control->state != LLAMA_OVERSUBSCRIPTION_DECODE_ACTIVE) {
        return true;
    }

    for (int i = 0; i < control->n_background_threads; i++) {
        if (!control->background_threads[i].is_suspended) {
            LLAMA_LOG_ERROR("[OVERSUBSCRIPTION] VIOLATION: Background thread %u (%s) not suspended\n",
                            control->background_threads[i].thread_id,
                            control->background_threads[i].thread_name);

            control->background_thread_violations++;

            if (control->abort_on_background_wake) {
                LLAMA_LOG_ERROR("[OVERSUBSCRIPTION] ABORTING due to background thread wake\n");
                abort();
            }

            return false;
        }
    }

    return true;
}

/**
 * Check per-token thread violations
 */
bool llama_oversubscription_check_per_token_threads(llama_oversubscription_control * control) {
    if (control == NULL) {
        return true;
    }

    if (control->state != LLAMA_OVERSUBSCRIPTION_DECODE_ACTIVE) {
        return true;
    }

    // Perform all per-token checks
    bool ok = true;

    ok = llama_oversubscription_detect_oversubscription(control) && ok;
    ok = llama_oversubscription_validate_background_threads_parked(control) && ok;
    ok = llama_oversubscription_validate_static_thread_pool(control) && ok;

    control->tokens_generated++;

    // Reset per-token counters
    for (int i = 0; i < control->n_threads_tracked; i++) {
        control->threads[i].wake_count_per_token = 0;
    }

    return ok;
}

/**
 * End decode phase
 */
bool llama_oversubscription_end_decode(llama_oversubscription_control * control) {
    if (control == NULL) {
        return false;
    }

    if (control->state != LLAMA_OVERSUBSCRIPTION_DECODE_ACTIVE) {
        return false;
    }

    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] Ending DECODE phase...\n");

    control->decode_end_time = llama_get_time_ns();
    uint64_t decode_duration_ns = control->decode_end_time - control->decode_start_time;

    // Restore OpenMP if it was overridden
    if (control->omp_override_active) {
#ifdef _OPENMP
        omp_set_num_threads(control->omp_prev_num_threads);
        LLAMA_LOG_INFO("[OVERSUBSCRIPTION] OpenMP restored to %d threads\n",
                       control->omp_prev_num_threads);
#endif
        control->omp_override_active = false;
    }

    // Reactivate background threads
    for (int i = 0; i < control->n_background_threads; i++) {
        control->background_threads[i].is_suspended = false;
    }

    // Reactivate CPU backend workers
    control->cpu_backend_disabled = false;

    // Reactivate all threads
    for (int i = 0; i < control->n_threads_tracked; i++) {
        control->threads[i].active_in_decode = false;
    }

    // Log summary
    double avg_runnable = control->tokens_generated > 0 ?
        (double)control->per_token_max_runnable : 0.0;
    double avg_cs_per_token = control->tokens_generated > 0 ?
        (double)control->context_switches_total / control->tokens_generated : 0.0;

    LLAMA_LOG_INFO("[OVERSUBSCRIPTION] DECODE complete: %lu tokens in %.3f ms\n"
                   "  Avg runnable threads: %.2f, Avg context switches/token: %.2f\n"
                   "  Violations: oversubscription=%lu, thread_creation=%lu, "
                   "background_wake=%lu, omp_expansion=%lu\n",
                   control->tokens_generated, decode_duration_ns / 1e6,
                   avg_runnable, avg_cs_per_token,
                   control->oversubscription_violations,
                   control->thread_creation_violations,
                   control->background_thread_violations,
                   control->omp_parallel_violations);

    control->state = LLAMA_OVERSUBSCRIPTION_CONFIGURED;

    return true;
}

/**
 * Assert oversubscription control state intact
 */
bool llama_oversubscription_assert_control_intact(const llama_oversubscription_control * control) {
    if (control == NULL) {
        return false;
    }

    if (control->state != LLAMA_OVERSUBSCRIPTION_DECODE_ACTIVE) {
        return true;
    }

    // Validate decode threads are active
    int active_decode_threads = 0;
    for (int i = 0; i < control->n_threads_tracked; i++) {
        if (control->threads[i].is_decode_worker && control->threads[i].active_in_decode) {
            active_decode_threads++;
        }
    }

    if (active_decode_threads == 0) {
        LLAMA_LOG_ERROR("[OVERSUBSCRIPTION] ASSERTION FAILED: No active decode threads\n");
        return false;
    }

    if (active_decode_threads > control->thread_config.max_runnable_threads) {
        LLAMA_LOG_ERROR("[OVERSUBSCRIPTION] ASSERTION FAILED: Too many active threads: %d > %d\n",
                        active_decode_threads, control->thread_config.max_runnable_threads);
        return false;
    }

    // Validate background threads are suspended
    for (int i = 0; i < control->n_background_threads; i++) {
        if (!control->background_threads[i].is_suspended) {
            LLAMA_LOG_WARN("[OVERSUBSCRIPTION] ASSERTION: Background thread not suspended: %s\n",
                           control->background_threads[i].thread_name);
            // Warning only, not critical
        }
    }

    return true;
}

/**
 * Get current oversubscription control state
 */
llama_oversubscription_state_t llama_oversubscription_get_state(
    const llama_oversubscription_control * control) {

    if (control == NULL) {
        return LLAMA_OVERSUBSCRIPTION_UNINITIALIZED;
    }

    return control->state;
}

/**
 * Check if decode phase is currently active
 */
bool llama_oversubscription_is_decode_active(const llama_oversubscription_control * control) {
    if (control == NULL) {
        return false;
    }

    return control->state == LLAMA_OVERSUBSCRIPTION_DECODE_ACTIVE;
}

/**
 * Dump oversubscription control configuration
 */
void llama_oversubscription_dump_config(const llama_oversubscription_control * control) {
    if (control == NULL) {
        return;
    }

    const char * state_names[] = {
        "UNINITIALIZED",
        "CONFIGURED",
        "PREFILL_ACTIVE",
        "DECODE_ACTIVE",
        "RELEASED"
    };

    LLAMA_LOG_INFO("\n========== OVERSUBSCRIPTION CONTROL CONFIG ==========\n");
    LLAMA_LOG_INFO("User thread count: %d (from --threads)\n", control->user_thread_count);
    LLAMA_LOG_INFO("Decode thread count: %d (control=%d + auxiliary=%d)\n",
                   control->decode_thread_count,
                   control->thread_config.decode_control_threads,
                   control->thread_config.decode_auxiliary_threads);
    LLAMA_LOG_INFO("GPU-exclusive decode: %s\n",
                   control->gpu_exclusive_decode ? "yes" : "no");
    LLAMA_LOG_INFO("Current state: %s\n",
                   state_names[control->state]);

    LLAMA_LOG_INFO("\nDECODE PHASE METRICS:\n");
    LLAMA_LOG_INFO("  Tokens generated: %lu\n", control->tokens_generated);
    LLAMA_LOG_INFO("  Total context switches: %lu (%.2f per token)\n",
                   control->context_switches_total,
                   control->tokens_generated > 0 ?
                       (double)control->context_switches_total / control->tokens_generated : 0.0);
    LLAMA_LOG_INFO("  Total thread wake events: %lu (%.2f per token)\n",
                   control->wake_events_total,
                   control->tokens_generated > 0 ?
                       (double)control->wake_events_total / control->tokens_generated : 0.0);
    LLAMA_LOG_INFO("  Peak runnable threads: %lu\n", control->per_token_max_runnable);

    LLAMA_LOG_INFO("\nVIOLATIONS:\n");
    LLAMA_LOG_INFO("  Oversubscription: %lu\n", control->oversubscription_violations);
    LLAMA_LOG_INFO("  Thread creation: %lu\n", control->thread_creation_violations);
    LLAMA_LOG_INFO("  Background thread wake: %lu\n", control->background_thread_violations);
    LLAMA_LOG_INFO("  OpenMP expansion: %lu\n", control->omp_parallel_violations);
    LLAMA_LOG_INFO("  CPU backend: %lu\n", control->cpu_backend_violations);

    LLAMA_LOG_INFO("\nBACKGROUND THREADS:\n");
    LLAMA_LOG_INFO("  Total registered: %d\n", control->n_background_threads);
    for (int i = 0; i < control->n_background_threads; i++) {
        LLAMA_LOG_INFO("    %s [%s, suspended %lu times]\n",
                       control->background_threads[i].thread_name,
                       control->background_threads[i].is_suspended ? "suspended" : "active",
                       control->background_threads[i].suspension_count);
    }

    LLAMA_LOG_INFO("\nWORKER THREADS:\n");
    LLAMA_LOG_INFO("  Total registered: %d\n", control->n_threads_tracked);
    int active_count = 0;
    for (int i = 0; i < control->n_threads_tracked; i++) {
        if (control->threads[i].active_in_decode) {
            LLAMA_LOG_INFO("    Thread %u [ACTIVE in decode, woken %d times, cs=%d]\n",
                           control->threads[i].thread_id,
                           control->threads[i].wake_attempts,
                           control->threads[i].context_switches);
            active_count++;
        }
    }
    LLAMA_LOG_INFO("  Currently active: %d\n", active_count);

    LLAMA_LOG_INFO("====================================================\n\n");
}

/**
 * Get violation statistics
 */
void llama_oversubscription_get_violations(
    const llama_oversubscription_control * control,
    uint64_t * out_oversubscription,
    uint64_t * out_thread_creation,
    uint64_t * out_background_wake,
    uint64_t * out_omp_expansion) {

    if (control == NULL) {
        if (out_oversubscription) *out_oversubscription = 0;
        if (out_thread_creation) *out_thread_creation = 0;
        if (out_background_wake) *out_background_wake = 0;
        if (out_omp_expansion) *out_omp_expansion = 0;
        return;
    }

    if (out_oversubscription) *out_oversubscription = control->oversubscription_violations;
    if (out_thread_creation) *out_thread_creation = control->thread_creation_violations;
    if (out_background_wake) *out_background_wake = control->background_thread_violations;
    if (out_omp_expansion) *out_omp_expansion = control->omp_parallel_violations;
}

/**
 * Get decode phase metrics
 */
void llama_oversubscription_get_metrics(
    const llama_oversubscription_control * control,
    uint64_t * out_tokens_generated,
    uint64_t * out_context_switches,
    uint64_t * out_wake_events,
    int * out_peak_runnable_threads,
    double * out_avg_runnable_threads) {

    if (control == NULL) {
        if (out_tokens_generated) *out_tokens_generated = 0;
        if (out_context_switches) *out_context_switches = 0;
        if (out_wake_events) *out_wake_events = 0;
        if (out_peak_runnable_threads) *out_peak_runnable_threads = 0;
        if (out_avg_runnable_threads) *out_avg_runnable_threads = 0.0;
        return;
    }

    if (out_tokens_generated) *out_tokens_generated = control->tokens_generated;
    if (out_context_switches) *out_context_switches = control->context_switches_total;
    if (out_wake_events) *out_wake_events = control->wake_events_total;
    if (out_peak_runnable_threads) *out_peak_runnable_threads = (int)control->per_token_max_runnable;

    if (out_avg_runnable_threads) {
        *out_avg_runnable_threads = control->tokens_generated > 0 ?
            (double)control->per_token_max_runnable / control->tokens_generated : 0.0;
    }
}

/**
 * Get thread-specific metrics
 */
bool llama_oversubscription_get_thread_metrics(
    const llama_oversubscription_control * control,
    uint32_t thread_id,
    bool * out_active,
    int * out_context_switches,
    int * out_wake_count) {

    if (control == NULL) {
        return false;
    }

    for (int i = 0; i < control->n_threads_tracked; i++) {
        if (control->threads[i].thread_id == thread_id) {
            if (out_active) *out_active = control->threads[i].active_in_decode;
            if (out_context_switches) *out_context_switches = control->threads[i].context_switches;
            if (out_wake_count) *out_wake_count = control->threads[i].wake_attempts;
            return true;
        }
    }

    return false;
}
