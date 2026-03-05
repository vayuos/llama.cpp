#include "llama-server-decode-isolation.h"
#include "../common/log.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <stdexcept>

// ============================================================================
// GLOBALS AND STATICS
// ============================================================================

static bool g_isolation_verbose = false;
static std::string g_last_violation_message;
static std::mutex g_violation_message_mutex;

#define LOG_ISOLATION(fmt, ...) \
    do { \
        if (g_isolation_verbose) { \
            LOG_INF("[ISOLATION] " fmt __VA_OPT__(,) __VA_ARGS__); \
        } \
    } while(0)

#define LOG_VIOLATION(fmt, ...) \
    do { \
        LOG_ERR("[ISOLATION_VIOLATION] " fmt __VA_OPT__(,) __VA_ARGS__); \
        std::unique_lock<std::mutex> lock(g_violation_message_mutex); \
        g_last_violation_message = std::string("") + fmt; \
    } while(0)

// ============================================================================
// CORE SET IMPLEMENTATION
// ============================================================================

decode_core_set::decode_core_set() : core_count(0) {
    mask.fill(0);
}

bool decode_core_set::add_core(int32_t core_id) {
    if (core_id < 0 || core_id >= DECODE_ISOLATION_MAX_CORES) {
        return false;
    }
    if (contains(core_id)) {
        return false;  // Already added
    }
    mask[core_id / 8] |= (1 << (core_id % 8));
    core_count++;
    return true;
}

bool decode_core_set::remove_core(int32_t core_id) {
    if (core_id < 0 || core_id >= DECODE_ISOLATION_MAX_CORES) {
        return false;
    }
    if (!contains(core_id)) {
        return false;  // Not in set
    }
    mask[core_id / 8] &= ~(1 << (core_id % 8));
    core_count--;
    return true;
}

bool decode_core_set::contains(int32_t core_id) const {
    if (core_id < 0 || core_id >= DECODE_ISOLATION_MAX_CORES) {
        return false;
    }
    return (mask[core_id / 8] & (1 << (core_id % 8))) != 0;
}

std::vector<int32_t> decode_core_set::get_cores() const {
    std::vector<int32_t> cores;
    for (int32_t i = 0; i < DECODE_ISOLATION_MAX_CORES; ++i) {
        if (contains(i)) {
            cores.push_back(i);
        }
    }
    return cores;
}

bool decode_core_set::overlaps_with(const decode_core_set & other) const {
    return !is_disjoint_from(other);
}

bool decode_core_set::is_disjoint_from(const decode_core_set & other) const {
    for (size_t i = 0; i < mask.size(); ++i) {
        if ((mask[i] & other.mask[i]) != 0) {
            return false;
        }
    }
    return true;
}

void * decode_core_set::get_native_mask() const {
#ifdef _WIN32
    // Windows GROUP_AFFINITY - simplified placeholder
    return nullptr;
#else
    // Linux cpu_set_t - simplified placeholder
    return nullptr;
#endif
}

// ============================================================================
// DOMAIN IMPLEMENTATION
// ============================================================================

decode_domain::decode_domain()
    : is_enabled(false),
      thread_count(0), scheduling_priority(2),
      violations_detected(0),
      thread_migrations(0), scheduling_preemptions(0), lock_waits_detected(0) {
}

server_domain::server_domain()
    : is_enabled(false),
      thread_count(0), scheduling_priority(0),
      violations_detected(0),
      threads_on_decode_cores(0), admission_rejections(0) {
}

// ============================================================================
// ISOLATION ENGINE IMPLEMENTATION
// ============================================================================

decode_isolation_engine::decode_isolation_engine() {
}

bool decode_isolation_engine::initialize(
    const std::vector<int32_t> & decode_core_list,
    const std::vector<int32_t> & server_core_list) {

    LOG_ISOLATION("Initializing decode isolation engine");

    // Build decode core set
    for (int32_t core_id : decode_core_list) {
        if (!decode_dom.core_set.add_core(core_id)) {
            LOG_VIOLATION("Failed to add decode core");
            return false;
        }
    }
    decode_dom.thread_count = 0;
    decode_dom.scheduling_priority = 0;
    decode_dom.is_enabled = true;

    // Build server core set
    for (int32_t core_id : server_core_list) {
        if (!server_dom.core_set.add_core(core_id)) {
            LOG_VIOLATION("Failed to add server core");
            return false;
        }
    }
    server_dom.thread_count = 0;
    server_dom.is_enabled = true;

    // Validate no overlap
    if (!decode_dom.core_set.is_disjoint_from(server_dom.core_set)) {
        LOG_VIOLATION("Decode and server core sets overlap - isolation cannot be enforced");
        return false;
    }

    LOG_ISOLATION("Decode isolation initialized successfully");

    return true;
}

bool decode_isolation_engine::pin_decode_thread(std::thread::id tid, int32_t decode_thread_index) {
    if (decode_thread_index < 0 || decode_thread_index >= decode_dom.thread_count) {
        LOG_VIOLATION("Invalid decode thread index: %d", decode_thread_index);
        return false;
    }

    bool success = platform_set_affinity(tid, decode_dom.core_set);
    if (success) {
        LOG_ISOLATION("Pinned decode thread %d to cores", decode_thread_index);
    } else {
        LOG_VIOLATION("Failed to pin decode thread %d to cores", decode_thread_index);
    }
    return success;
}

bool decode_isolation_engine::pin_server_thread(std::thread::id tid, int32_t server_worker_index) {
    if (server_worker_index < 0 || server_worker_index >= server_dom.thread_count) {
        LOG_VIOLATION("Invalid server worker index: %d", server_worker_index);
        return false;
    }

    bool success = platform_set_affinity(tid, server_dom.core_set);
    if (success) {
        LOG_ISOLATION("Pinned server worker %d to cores", server_worker_index);
    } else {
        LOG_VIOLATION("Failed to pin server worker %d to cores", server_worker_index);
    }
    return success;
}

bool decode_isolation_engine::set_decode_priority(std::thread::id tid, int32_t priority) {
    bool success = platform_set_priority(tid, priority);
    if (success) {
        LOG_ISOLATION("Set decode thread priority to %d", priority);
    } else {
        LOG_VIOLATION("Failed to set decode thread priority to %d", priority);
    }
    return success;
}

bool decode_isolation_engine::validate_configuration() const {
    if (!decode_dom.is_enabled || !server_dom.is_enabled) {
        LOG_VIOLATION("Domains not properly initialized");
        return false;
    }

    if (decode_dom.core_set.core_count <= 0) {
        LOG_VIOLATION("Decode core set is empty");
        return false;
    }

    if (server_dom.core_set.core_count <= 0) {
        LOG_VIOLATION("Server core set is empty");
        return false;
    }

    if (!decode_dom.core_set.is_disjoint_from(server_dom.core_set)) {
        LOG_VIOLATION("Decode and server core sets overlap");
        return false;
    }

    if (decode_dom.thread_count <= 0 || decode_dom.thread_count > DECODE_ISOLATION_MAX_DECODE_THREADS) {
        LOG_VIOLATION("Invalid decode thread count: %d", decode_dom.thread_count);
        return false;
    }

    if (server_dom.thread_count <= 0 || server_dom.thread_count > DECODE_ISOLATION_MAX_SERVER_THREADS) {
        LOG_VIOLATION("Invalid server thread count: %d", server_dom.thread_count);
        return false;
    }

    LOG_ISOLATION("Configuration validation passed");
    return true;
}

bool decode_isolation_engine::validate_runtime() const {
    // This is a simplified validation. In production, this would:
    // - Use /proc/[pid]/task/[tid]/status to read actual CPU affinity
    // - Check for migration events
    // - Detect cross-domain lock contention
    // - Monitor scheduling preemptions
    return true;
}

bool decode_isolation_engine::is_thread_on_decode_cores(std::thread::id tid) const {
    decode_core_set current_affinity;
    if (!platform_get_affinity(tid, current_affinity)) {
        return false;
    }
    return !current_affinity.is_disjoint_from(decode_dom.core_set);
}

decode_core_set decode_isolation_engine::get_thread_affinity(std::thread::id tid) const {
    decode_core_set affinity;
    platform_get_affinity(tid, affinity);
    return affinity;
}

void decode_isolation_engine::record_violation(const std::string & violation_type, const std::string & details) {
    std::unique_lock<std::mutex> lock(metrics_mutex);
    violation_log.push_back(violation_type + ": " + details);
    server_dom.violations_detected++;
}

isolation_metrics decode_isolation_engine::get_metrics() const {
    std::unique_lock<std::mutex> lock(metrics_mutex);
    isolation_metrics metrics = {
        decode_dom.thread_migrations,                // non-atomic uint64_t
        decode_dom.scheduling_preemptions,           // non-atomic uint64_t
        decode_dom.lock_waits_detected,              // non-atomic uint64_t
        decode_dom.violations_detected.load(),       // atomic<uint64_t>
        server_dom.threads_on_decode_cores,          // non-atomic uint64_t
        server_dom.admission_rejections,             // non-atomic uint64_t
        0.0f,  // server_load_percent
        0.0f,  // decode_throughput_tokens_per_sec
        0.0f   // decode_latency_variance_us
    };
    return metrics;
}

const decode_domain & decode_isolation_engine::get_decode_domain() const {
    return decode_dom;
}

const server_domain & decode_isolation_engine::get_server_domain() const {
    return server_dom;
}

void decode_isolation_engine::abort_if_violated(bool check_runtime) const {
    if (!validate_configuration()) {
        std::cerr << "FATAL: Isolation configuration invalid - aborting\n";
        std::abort();
    }

    if (check_runtime && !validate_runtime()) {
        std::cerr << "FATAL: Runtime isolation violation detected - aborting\n";
        std::abort();
    }

    if (server_dom.violations_detected.load() > 0) {
        std::cerr << "FATAL: " << server_dom.violations_detected.load()
                  << " isolation violations recorded - aborting\n";
        std::abort();
    }
}

decode_isolation_engine & decode_isolation_engine::instance() {
    static decode_isolation_engine engine;
    return engine;
}

// Platform-specific implementations (Linux)
#if defined(__linux__) || defined(__unix__)

bool decode_isolation_engine::platform_set_affinity(std::thread::id tid, const decode_core_set & cores) {
#ifdef HAVE_PTHREAD_SETAFFINITY_NP
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (int32_t core_id : cores.get_cores()) {
        CPU_SET(core_id, &cpuset);
    }

    // Convert std::thread::id to pthread_t
    // This is platform-dependent and simplified here
    pthread_t thread = pthread_self();

    int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return result == 0;
#else
    (void)tid;
    (void)cores;
    LOG_VIOLATION("pthread_setaffinity_np not available on this platform");
    return false;
#endif
}

bool decode_isolation_engine::platform_get_affinity(std::thread::id tid, decode_core_set & cores) const {
#ifdef HAVE_PTHREAD_SETAFFINITY_NP
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    pthread_t thread = pthread_self();
    int result = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

    if (result == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &cpuset)) {
                cores.add_core(i);
            }
        }
    }
    return result == 0;
#else
    (void)tid;
    (void)cores;
    return false;
#endif
}

bool decode_isolation_engine::platform_set_priority(std::thread::id tid, int32_t priority) {
    (void)tid;  // Reserved for future platform-specific priority setting
    int policy = SCHED_OTHER;
    int native_priority = 0;

    // Map from ggml priority to POSIX priority
    if (priority == 3) {  // GGML_SCHED_PRIO_REALTIME
        policy = SCHED_FIFO;
        native_priority = 90;  // High FIFO priority
    } else if (priority == 2) {  // GGML_SCHED_PRIO_HIGH
        policy = SCHED_RR;
        native_priority = 50;
    } else {
        policy = SCHED_OTHER;
        native_priority = 0;
    }

    struct sched_param param;
    param.sched_priority = native_priority;

    pthread_t thread = pthread_self();
    int result = pthread_setschedparam(thread, policy, &param);

    return result == 0;
}

#elif defined(_WIN32)

bool decode_isolation_engine::platform_set_affinity(std::thread::id tid, const decode_core_set & cores) {
    DWORD_PTR mask = 0;
    for (int32_t core_id : cores.get_cores()) {
        if (core_id < 64) {  // Windows supports up to 64 cores in basic mask
            mask |= (1ULL << core_id);
        }
    }

    // Convert std::thread::id to HANDLE
    // This is simplified - real implementation would be more complex
    HANDLE thread = GetCurrentThread();
    DWORD_PTR result = SetThreadAffinityMask(thread, mask);

    return result != 0;
}

bool decode_isolation_engine::platform_get_affinity(std::thread::id tid, decode_core_set & cores) const {
    HANDLE thread = GetCurrentThread();
    DWORD_PTR process_mask = 0, system_mask = 0;

    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        for (int i = 0; i < 64; ++i) {
            if (process_mask & (1ULL << i)) {
                cores.add_core(i);
            }
        }
        return true;
    }
    return false;
}

bool decode_isolation_engine::platform_set_priority(std::thread::id tid, int32_t priority) {
    int priority_class = NORMAL_PRIORITY_CLASS;
    if (priority == 3) {
        priority_class = HIGH_PRIORITY_CLASS;  // Windows doesn't have realtime
    } else if (priority == 2) {
        priority_class = ABOVE_NORMAL_PRIORITY_CLASS;
    }

    HANDLE thread = GetCurrentThread();
    return SetThreadPriority(thread, priority_class) != 0;
}

#else

// Fallback for unknown platform
bool decode_isolation_engine::platform_set_affinity(std::thread::id tid, const decode_core_set & cores) {
    (void)tid;
    (void)cores;
    return false;
}

bool decode_isolation_engine::platform_get_affinity(std::thread::id tid, decode_core_set & cores) const {
    (void)tid;
    (void)cores;
    return false;
}

bool decode_isolation_engine::platform_set_priority(std::thread::id tid, int32_t priority) {
    (void)tid;
    (void)priority;
    return false;
}

#endif

// ============================================================================
// LOCK-FREE STREAMING QUEUE IMPLEMENTATION
// ============================================================================

template<typename T>
decode_streaming_queue<T>::decode_streaming_queue(size_t capacity)
    : head(0), tail(0) {
    // Round up to power of 2
    size_t c = 1;
    while (c < capacity) {
        c *= 2;
    }
    buffer.resize(c);
    mask = c - 1;
}

template<typename T>
bool decode_streaming_queue<T>::try_push(const T & token) {
    uint64_t current_tail = tail.load(std::memory_order_relaxed);
    uint64_t next_tail = current_tail + 1;

    if ((next_tail & mask) == (head.load(std::memory_order_acquire) & mask)) {
        return false;  // Queue full
    }

    buffer[current_tail & mask] = token;
    tail.store(next_tail, std::memory_order_release);
    return true;
}

template<typename T>
bool decode_streaming_queue<T>::try_pop(T & token) {
    uint64_t current_head = head.load(std::memory_order_relaxed);

    if ((current_head & mask) == (tail.load(std::memory_order_acquire) & mask)) {
        return false;  // Queue empty
    }

    token = buffer[current_head & mask];
    head.store(current_head + 1, std::memory_order_release);
    return true;
}

template<typename T>
size_t decode_streaming_queue<T>::depth() const {
    uint64_t h = head.load(std::memory_order_acquire);
    uint64_t t = tail.load(std::memory_order_acquire);
    return (size_t)(t - h);
}

template<typename T>
size_t decode_streaming_queue<T>::capacity_const() const {
    return buffer.size();
}

template<typename T>
bool decode_streaming_queue<T>::is_full() const {
    return depth() >= buffer.size();
}

template<typename T>
bool decode_streaming_queue<T>::is_empty() const {
    return depth() == 0;
}

template<typename T>
void decode_streaming_queue<T>::clear() {
    head.store(0, std::memory_order_release);
    tail.store(0, std::memory_order_release);
}

// Explicit instantiation for decode_token_event
template class decode_streaming_queue<decode_token_event>;

// ============================================================================
// STREAMING MANAGER IMPLEMENTATION
// ============================================================================

bool streaming_manager::initialize(size_t queue_capacity) {
    LOG_ISOLATION("Initializing streaming manager with capacity %zu", queue_capacity);
    return true;
}

bool streaming_manager::decode_push_token(const decode_token_event & event) {
    if (!queue.try_push(event)) {
        backpressure_events++;
        return false;  // Queue full - decode should detect backpressure
    }
    tokens_produced++;
    return true;
}

bool streaming_manager::server_consume_token(decode_token_event & event) {
    if (!queue.try_pop(event)) {
        return false;
    }
    tokens_consumed++;
    return true;
}

streaming_metrics streaming_manager::get_metrics() const {
    streaming_metrics metrics = {
        tokens_produced.load(),
        tokens_consumed.load(),
        backpressure_events.load(),
        0.0f  // throughput_tokens_per_sec - would be computed from timing
    };
    return metrics;
}

void streaming_manager::clear() {
    queue.clear();
    tokens_produced.store(0);
    tokens_consumed.store(0);
    backpressure_events.store(0);
}

streaming_manager & streaming_manager::instance() {
    static streaming_manager manager(DECODE_STREAMING_QUEUE_SIZE);
    return manager;
}

// ============================================================================
// CROSS-DOMAIN LOCK DETECTOR IMPLEMENTATION
// ============================================================================

static std::atomic<uint64_t> g_contention_count(0);
static thread_local std::string g_current_lock_name;
static thread_local int32_t g_current_domain = -1;

void cross_domain_lock_detector::enter_critical_section(const std::string & lock_name, int32_t domain) {
    g_current_lock_name = lock_name;
    g_current_domain = domain;
}

void cross_domain_lock_detector::exit_critical_section(const std::string & lock_name) {
    (void)lock_name;  // Reserved for future lock contention tracking
    g_current_lock_name = "";
    g_current_domain = -1;
}

bool cross_domain_lock_detector::has_decode_server_contention() const {
    // Would check if decode thread is currently holding server locks
    return false;
}

uint64_t cross_domain_lock_detector::get_contention_count() {
    return g_contention_count.load();
}

cross_domain_lock_detector & cross_domain_lock_detector::instance() {
    static cross_domain_lock_detector detector;
    return detector;
}

// ============================================================================
// ADMISSION CONTROL IMPLEMENTATION
// ============================================================================

bool admission_control::initialize(int64_t decode_latency_threshold_us, int32_t max_queue_depth_param) {
    decode_latency_threshold_us = decode_latency_threshold_us;
    max_queue_depth = max_queue_depth_param;
    pending_queue_depth.store(0);
    last_decode_latency_us.store(0);
    admissions_rejected.store(0);

    LOG_ISOLATION("Admission control initialized: latency_threshold=%ldus, max_queue=%d",
                  decode_latency_threshold_us, max_queue_depth);
    return true;
}

bool admission_control::try_admit_request() {
    int64_t latency = last_decode_latency_us.load();
    int32_t queue_depth = pending_queue_depth.load();

    // If decode is slow and queue is full, reject
    if (latency > decode_latency_threshold_us && queue_depth >= max_queue_depth) {
        admissions_rejected++;
        return false;
    }

    pending_queue_depth++;
    return true;
}

void admission_control::record_decode_latency(int64_t latency_us) {
    last_decode_latency_us.store(latency_us);
}

admission_metrics admission_control::get_metrics() const {
    admission_metrics metrics = {
        pending_queue_depth.load(),
        last_decode_latency_us.load(),
        admissions_rejected.load(),
        0.0f  // admission_rejection_rate
    };
    return metrics;
}

admission_control & admission_control::instance() {
    static admission_control controller;
    return controller;
}

// ============================================================================
// GLOBAL INITIALIZATION HELPERS
// ============================================================================

bool initialize_decode_isolation(
    int32_t total_cores,
    const std::vector<int32_t> & decode_core_ids,
    int32_t decode_priority,
    int32_t decode_thread_count,
    int32_t server_thread_count) {

    (void)total_cores;
    (void)decode_priority;
    (void)decode_thread_count;
    (void)server_thread_count;

    auto & engine = decode_isolation_engine::instance();
    auto & streaming = streaming_manager::instance();
    auto & admission = admission_control::instance();

    // Calculate server cores (all remaining cores)
    std::vector<int32_t> server_core_ids;
    std::vector<bool> is_decode_core(total_cores, false);

    for (int32_t id : decode_core_ids) {
        if (id >= 0 && id < total_cores) {
            is_decode_core[id] = true;
        }
    }

    for (int32_t i = 0; i < total_cores; ++i) {
        if (!is_decode_core[i]) {
            server_core_ids.push_back(i);
        }
    }

    // Initialize components
    if (!engine.initialize(decode_core_ids, server_core_ids)) {
        LOG_VIOLATION("Failed to initialize isolation engine");
        return false;
    }

    if (!streaming.initialize(DECODE_STREAMING_QUEUE_SIZE)) {
        LOG_VIOLATION("Failed to initialize streaming manager");
        return false;
    }

    if (!admission.initialize(10000, 1000)) {
        LOG_VIOLATION("Failed to initialize admission control");
        return false;
    }

    LOG_ISOLATION("Full decode isolation initialized successfully");
    return true;
}

bool pin_current_thread_to_decode(int32_t thread_index) {
    auto & engine = decode_isolation_engine::instance();
    return engine.pin_decode_thread(std::this_thread::get_id(), thread_index);
}

bool pin_current_thread_to_server(int32_t worker_index) {
    auto & engine = decode_isolation_engine::instance();
    return engine.pin_server_thread(std::this_thread::get_id(), worker_index);
}

void validate_isolation_config() {
    auto & engine = decode_isolation_engine::instance();
    engine.abort_if_violated(true);
}

// ============================================================================
// DIAGNOSTICS
// ============================================================================

void dump_isolation_state() {
    auto & engine = decode_isolation_engine::instance();
    auto & streaming = streaming_manager::instance();
    auto & admission = admission_control::instance();

    const decode_domain & decode_dom = engine.get_decode_domain();
    const server_domain & server_dom = engine.get_server_domain();

    std::cout << "\n=== ISOLATION STATE DUMP ===\n\n";

    std::cout << "DECODE DOMAIN:\n";
    std::cout << "  Cores: ";
    for (int32_t core : decode_dom.core_set.get_cores()) {
        std::cout << core << " ";
    }
    std::cout << "\n  Thread Count: " << decode_dom.thread_count << "\n";
    std::cout << "  Priority: " << decode_dom.scheduling_priority << "\n";
    std::cout << "  Enabled: " << (decode_dom.is_enabled ? "yes" : "no") << "\n";
    std::cout << "  Metrics:\n";
    std::cout << "    Migrations: " << decode_dom.thread_migrations << "\n";
    std::cout << "    Preemptions: " << decode_dom.scheduling_preemptions << "\n";
    std::cout << "    Lock Waits: " << decode_dom.lock_waits_detected << "\n";
    std::cout << "    Violations: " << decode_dom.violations_detected.load() << "\n";

    std::cout << "\nSERVER DOMAIN:\n";
    std::cout << "  Cores: ";
    for (int32_t core : server_dom.core_set.get_cores()) {
        std::cout << core << " ";
    }
    std::cout << "\n  Thread Count: " << server_dom.thread_count << "\n";
    std::cout << "  Priority: " << server_dom.scheduling_priority << "\n";
    std::cout << "  Enabled: " << (server_dom.is_enabled ? "yes" : "no") << "\n";
    std::cout << "  Metrics:\n";
    std::cout << "    Violations: " << server_dom.violations_detected.load() << "\n";
    std::cout << "    Threads on Decode Cores: " << server_dom.threads_on_decode_cores << "\n";
    std::cout << "    Admission Rejections: " << server_dom.admission_rejections << "\n";

    auto stream_metrics = streaming.get_metrics();
    std::cout << "\nSTREAMING METRICS:\n";
    std::cout << "  Backpressure Events: " << stream_metrics.backpressure_events << "\n";
    std::cout << "  Tokens Produced: " << stream_metrics.tokens_produced << "\n";
    std::cout << "  Tokens Consumed: " << stream_metrics.tokens_consumed << "\n";

    auto admission_metrics = admission.get_metrics();
    std::cout << "\nADMISSION CONTROL:\n";
    std::cout << "  Queue Depth: " << admission_metrics.queue_depth << "\n";
    std::cout << "  Decode Latency: " << admission_metrics.recent_decode_latency_us << " us\n";
    std::cout << "  Rejections: " << admission_metrics.admissions_rejected << "\n";

    std::cout << "\n=== END ISOLATION STATE DUMP ===\n\n";
}

// ============================================================================
// streaming_manager IMPLEMENTATION
// ============================================================================

streaming_manager::streaming_manager(size_t queue_capacity)
    : queue(queue_capacity), streaming_active(false), tokens_produced(0), tokens_consumed(0), backpressure_events(0) {
    initialize(queue_capacity);
}

// ============================================================================
// admission_control IMPLEMENTATION
// ============================================================================

admission_control::admission_control()
    : decode_latency_threshold_us(0), max_queue_depth(0),
      current_queue_depth(0), pending_queue_depth(0),
      recent_decode_latency_us(0), last_decode_latency_us(0),
      admissions_rejected(0) {
}

const char * get_last_violation_message() {
    std::unique_lock<std::mutex> lock(g_violation_message_mutex);
    static std::string message;
    message = g_last_violation_message;
    return message.c_str();
}

void set_isolation_verbose_logging(bool enable) {
    g_isolation_verbose = enable;
    if (enable) {
        LOG_INF("Isolation verbose logging ENABLED\n");
    }
}
