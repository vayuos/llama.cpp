/**
 * SECTION 5: Persistent CUDA Graph Execution
 * Header: Graph capture, instantiation, and replay API
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <chrono>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CUDA GRAPH STATISTICS
// ============================================================================

struct ggml_cuda_graph_stats {
    uint64_t graph_id;
    bool is_captured;
    bool is_instantiated;
    size_t graph_nodes;
    uint64_t capture_time_ns;
    uint64_t instantiate_time_ns;
    uint64_t launch_count;
    uint64_t avg_launch_time_ns;
};

// ============================================================================
// GRAPH LIFECYCLE API
// ============================================================================

/**
 * Begin capturing a CUDA graph on the given stream.
 * Returns a unique graph_id for later reference.
 */
uint64_t ggml_cuda_graph_capture_begin(cudaStream_t stream);

/**
 * End capturing and populate the graph.
 * Must be called after capture_begin with same graph_id.
 */
int ggml_cuda_graph_capture_end(uint64_t graph_id, cudaStream_t stream);

/**
 * Instantiate the captured graph into an executable form.
 * Must be called before ggml_cuda_graph_launch.
 */
int ggml_cuda_graph_instantiate(uint64_t graph_id, cudaStream_t stream);

/**
 * Launch the instantiated graph on the given stream.
 * Zero-overhead replay: ~100ns launch time vs ~1-5µs per kernel.
 */
int ggml_cuda_graph_launch(uint64_t graph_id, cudaStream_t stream);

/**
 * Destroy the captured graph and free resources.
 */
int ggml_cuda_graph_destroy(uint64_t graph_id);

// ============================================================================
// QUERY API
// ============================================================================

/**
 * Get statistics for a graph (capture time, launch count, etc.)
 */
struct ggml_cuda_graph_stats ggml_cuda_graph_get_stats(uint64_t graph_id);

// ============================================================================
// GLOBAL CONTROL
// ============================================================================

/**
 * Enable/disable CUDA graph execution globally.
 * Useful for debugging or fallback to traditional execution.
 */
void ggml_cuda_graph_set_enabled(bool enabled);

/**
 * Check if CUDA graph execution is enabled.
 */
bool ggml_cuda_graph_is_enabled();

/**
 * Cleanup all cached graphs (called at shutdown).
 */
void ggml_cuda_graph_cleanup_all();

#ifdef __cplusplus
}
#endif
