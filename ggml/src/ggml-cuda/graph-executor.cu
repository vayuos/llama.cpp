/**
 * SECTION 5: Persistent CUDA Graph Execution
 * Implementation: CUDA Graph Capture and Replay
 *
 * Replaces per-token graph construction with single-time capture and replay.
 * Eliminates per-op launch overhead, reducing kernel call latency to ~100ns.
 */

#include "graph-executor.cuh"
#include "common.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <map>
#include <vector>

// ============================================================================
// CUDA GRAPH STATE MANAGEMENT
// ============================================================================

struct cudagraph_exec_state {
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool is_captured;
    bool is_instantiated;
    size_t graph_nodes;
    uint64_t capture_time_ns;
    uint64_t instantiate_time_ns;
    uint64_t launch_count;
    uint64_t total_launch_time_ns;
};

static std::map<uint64_t, cudagraph_exec_state> g_graph_cache;
static uint64_t g_graph_id_counter = 1;
static bool g_cuda_graph_enabled = true;

// ============================================================================
// GRAPH CAPTURE
// ============================================================================

uint64_t ggml_cuda_graph_capture_begin(cudaStream_t stream) {
    if (!g_cuda_graph_enabled) {
        return 0;
    }

    uint64_t graph_id = g_graph_id_counter++;
    cudagraph_exec_state & state = g_graph_cache[graph_id];

    // Create empty graph
    CUDA_CHECK(cudaGraphCreate(&state.graph, 0));

    // Begin capture in default/row-wide mode
    // Stream-capture mode: captures all work from stream
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    state.is_captured = false;
    state.is_instantiated = false;
    state.graph_nodes = 0;
    state.launch_count = 0;
    state.total_launch_time_ns = 0;

    return graph_id;
}

int ggml_cuda_graph_capture_end(uint64_t graph_id, cudaStream_t stream) {
    if (!g_cuda_graph_enabled || graph_id == 0) {
        return -1;
    }

    auto it = g_graph_cache.find(graph_id);
    if (it == g_graph_cache.end()) {
        return -1;
    }

    cudagraph_exec_state & state = it->second;

    // End capture - graph is now populated
    CUDA_CHECK(cudaStreamEndCapture(stream, &state.graph));
    state.is_captured = true;

    // Query graph properties
    CUDA_CHECK(cudaGraphGetNodes(state.graph, NULL, &state.graph_nodes));

    // Record capture time
    auto now = std::chrono::high_resolution_clock::now();
    state.capture_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();

    return 0;
}

// ============================================================================
// GRAPH INSTANTIATION
// ============================================================================

int ggml_cuda_graph_instantiate(uint64_t graph_id, cudaStream_t stream) {
    GGML_UNUSED(stream);
    if (!g_cuda_graph_enabled || graph_id == 0) {
        return -1;
    }

    auto it = g_graph_cache.find(graph_id);
    if (it == g_graph_cache.end()) {
        return -1;
    }

    cudagraph_exec_state & state = it->second;

    if (!state.is_captured) {
        return -1;
    }

    if (state.is_instantiated) {
        // Already instantiated
        return 0;
    }

    // Instantiate executable graph
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&state.graph_exec, state.graph, 0));
    state.is_instantiated = true;

    // Record instantiation time
    auto now = std::chrono::high_resolution_clock::now();
    state.instantiate_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();

    return 0;
}

// ============================================================================
// GRAPH LAUNCH (ZERO-OVERHEAD REPLAY)
// ============================================================================

int ggml_cuda_graph_launch(uint64_t graph_id, cudaStream_t stream) {
    if (!g_cuda_graph_enabled || graph_id == 0) {
        return -1;
    }

    auto it = g_graph_cache.find(graph_id);
    if (it == g_graph_cache.end()) {
        return -1;
    }

    cudagraph_exec_state & state = it->second;

    if (!state.is_instantiated) {
        return -1;
    }

    // Single cudaGraphLaunch call - equivalent to launching N kernels in sequence
    // but with massively reduced CPU overhead (100ns vs ~1-5µs per kernel)
    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaGraphLaunch(state.graph_exec, stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    state.launch_count++;
    state.total_launch_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        t1 - t0).count();

    return 0;
}

// ============================================================================
// GRAPH CLEANUP
// ============================================================================

int ggml_cuda_graph_destroy(uint64_t graph_id) {
    auto it = g_graph_cache.find(graph_id);
    if (it == g_graph_cache.end()) {
        return -1;
    }

    cudagraph_exec_state & state = it->second;

    if (state.is_instantiated) {
        CUDA_CHECK(cudaGraphExecDestroy(state.graph_exec));
    }

    if (state.is_captured) {
        CUDA_CHECK(cudaGraphDestroy(state.graph));
    }

    g_graph_cache.erase(it);
    return 0;
}

// ============================================================================
// GRAPH STATISTICS
// ============================================================================

struct ggml_cuda_graph_stats ggml_cuda_graph_get_stats(uint64_t graph_id) {
    struct ggml_cuda_graph_stats stats;
    memset(&stats, 0, sizeof(stats));

    auto it = g_graph_cache.find(graph_id);
    if (it == g_graph_cache.end()) {
        return stats;
    }

    const cudagraph_exec_state & state = it->second;
    stats.graph_id = graph_id;
    stats.is_captured = state.is_captured;
    stats.is_instantiated = state.is_instantiated;
    stats.graph_nodes = state.graph_nodes;
    stats.capture_time_ns = state.capture_time_ns;
    stats.instantiate_time_ns = state.instantiate_time_ns;
    stats.launch_count = state.launch_count;
    stats.avg_launch_time_ns = state.launch_count > 0 ?
        state.total_launch_time_ns / state.launch_count : 0;

    return stats;
}

// ============================================================================
// GLOBAL CONTROL
// ============================================================================

void ggml_cuda_graph_set_enabled(bool enabled) {
    g_cuda_graph_enabled = enabled;
}

bool ggml_cuda_graph_is_enabled() {
    return g_cuda_graph_enabled;
}

void ggml_cuda_graph_cleanup_all() {
    std::vector<uint64_t> to_destroy;
    for (auto & p : g_graph_cache) {
        to_destroy.push_back(p.first);
    }
    for (uint64_t graph_id : to_destroy) {
        ggml_cuda_graph_destroy(graph_id);
    }
    g_graph_cache.clear();
}
