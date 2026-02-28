/**
 * CUDA Graph Executor: Capture-Replay for GPU-Exclusive Decode
 *
 * Phase 2.4: Graph capture/replay reduces kernel overhead from ~1-10µs to ~100ns
 */

#include "graph-executor.cuh"
#include "common.cuh"
#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>
#include <cstdio>

struct cuda_graph_state {
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool is_captured;
    bool is_instantiated;
    size_t graph_nodes;
    uint64_t launch_count;
};

static std::unordered_map<uint64_t, cuda_graph_state> g_graphs;
static std::mutex g_graph_mutex;
static uint64_t g_next_graph_id = 1;
static bool g_graphs_enabled = true;

#ifdef __cplusplus
extern "C" {
#endif

uint64_t ggml_cuda_graph_capture_begin(void * stream) {
    if (!g_graphs_enabled) return 0;
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    
    cudaStream_t s = (cudaStream_t)stream;
    if (!s) return 0;
    
    cudaError_t err = cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "ggml_cuda_graph_capture_begin failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    
    uint64_t graph_id = g_next_graph_id++;
    cuda_graph_state state = {};
    state.graph = nullptr;
    state.graph_exec = nullptr;
    state.is_captured = false;
    state.is_instantiated = false;
    state.graph_nodes = 0;
    state.launch_count = 0;
    
    g_graphs[graph_id] = state;
    // Debug: ggml_cuda_graph_capture_begin: graph_id created

    return graph_id;
}

int ggml_cuda_graph_capture_end(uint64_t graph_id, void * stream) {
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    
    if (g_graphs.find(graph_id) == g_graphs.end()) {
        fprintf(stderr, "ggml_cuda_graph_capture_end: graph_id %llu not found\n", (unsigned long long)graph_id);
        return -1;
    }
    
    cudaStream_t s = (cudaStream_t)stream;
    cudaGraph_t graph;
    cudaError_t err = cudaStreamEndCapture(s, &graph);

    if (err != cudaSuccess) {
        fprintf(stderr, "ggml_cuda_graph_capture_end failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    g_graphs[graph_id].graph = graph;
    g_graphs[graph_id].is_captured = true;
    
    size_t num_nodes = 0;
    cudaGraphGetNodes(graph, nullptr, &num_nodes);
    g_graphs[graph_id].graph_nodes = num_nodes;
    // Debug: ggml_cuda_graph_capture_end: graph captured with node count

    return 0;
}

int ggml_cuda_graph_instantiate(uint64_t graph_id, void * stream) {
    (void)stream;  // stream parameter unused - kept for API consistency
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    
    if (g_graphs.find(graph_id) == g_graphs.end()) {
        fprintf(stderr, "ggml_cuda_graph_instantiate: graph_id %llu not found\n", (unsigned long long)graph_id);
        return -1;
    }

    cuda_graph_state & state = g_graphs[graph_id];

    if (!state.is_captured || !state.graph) {
        fprintf(stderr, "ggml_cuda_graph_instantiate: graph not captured\n");
        return -1;
    }
    
    cudaGraphExec_t exec;
    cudaError_t err = cudaGraphInstantiate(&exec, state.graph, nullptr, nullptr, 0);

    if (err != cudaSuccess) {
        fprintf(stderr, "ggml_cuda_graph_instantiate failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    state.graph_exec = exec;
    state.is_instantiated = true;

    // Debug: ggml_cuda_graph_instantiate: graph instantiated

    return 0;
}

int ggml_cuda_graph_launch(uint64_t graph_id, void * stream) {
    std::lock_guard<std::mutex> lock(g_graph_mutex);

    if (g_graphs.find(graph_id) == g_graphs.end()) {
        fprintf(stderr, "ggml_cuda_graph_launch: graph_id %llu not found\n", (unsigned long long)graph_id);
        return -1;
    }

    cuda_graph_state & state = g_graphs[graph_id];

    if (!state.is_instantiated || !state.graph_exec) {
        fprintf(stderr, "ggml_cuda_graph_launch: graph not instantiated\n");
        return -1;
    }
    
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t err = cudaGraphLaunch(state.graph_exec, s);

    if (err != cudaSuccess) {
        fprintf(stderr, "ggml_cuda_graph_launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    state.launch_count++;
    return 0;
}

bool ggml_cuda_graph_is_enabled() {
    return g_graphs_enabled;
}

int ggml_cuda_graph_destroy(uint64_t graph_id) {
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    
    if (g_graphs.find(graph_id) == g_graphs.end()) {
        return -1;
    }
    
    cuda_graph_state & state = g_graphs[graph_id];
    
    if (state.is_instantiated && state.graph_exec) {
        cudaGraphExecDestroy(state.graph_exec);
        state.graph_exec = nullptr;
    }
    
    if (state.is_captured && state.graph) {
        cudaGraphDestroy(state.graph);
        state.graph = nullptr;
    }
    
    g_graphs.erase(graph_id);
    // Debug: ggml_cuda_graph_destroy: graph_id destroyed

    return 0;
}

int ggml_cuda_graph_cleanup_all() {
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    
    for (auto & pair : g_graphs) {
        cuda_graph_state & state = pair.second;
        if (state.is_instantiated && state.graph_exec) {
            cudaGraphExecDestroy(state.graph_exec);
        }
        if (state.is_captured && state.graph) {
            cudaGraphDestroy(state.graph);
        }
    }
    
    g_graphs.clear();
    g_next_graph_id = 1;

    // Debug: ggml_cuda_graph_cleanup_all: all graphs destroyed

    return 0;
}

int ggml_cuda_graph_get_count() {
    std::lock_guard<std::mutex> lock(g_graph_mutex);
    return g_graphs.size();
}

#ifdef __cplusplus
}
#endif
