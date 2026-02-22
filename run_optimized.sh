#!/bin/bash
###############################################################################
# run_optimized.sh - High-performance llama-server startup script
#
# This script optimizes system configuration for maximum throughput and
# minimum latency when running llama-server with CUDA GPU acceleration.
#
# Requirements:
#   - nvidia-smi (for GPU management)
#   - numactl (optional, for NUMA binding)
#   - jq (optional, for JSON parsing)
###############################################################################

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables (override via environment)
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-./build_cuda_mmq_moe/bin/llama-server}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
LOG_LEVEL="${LOG_LEVEL:-info}"
VERBOSE_LOGGING="${VERBOSE_LOGGING:-false}"
LOCK_GPU_CLOCKS="${LOCK_GPU_CLOCKS:-true}"
BIND_NUMA="${BIND_NUMA:-true}"
MODEL_PATH="${MODEL_PATH:-}"
SERVER_PORT="${SERVER_PORT:-8000}"

###############################################################################
# Utility Functions
###############################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

###############################################################################
# System Detection and Configuration
###############################################################################

detect_physical_cores() {
    # Detect physical core count (excluding hyperthreads)
    if command -v nproc &> /dev/null; then
        local logical_cores=$(nproc)
    else
        local logical_cores=$(grep -c ^processor /proc/cpuinfo)
    fi
    
    # Try to get physical cores from lscpu
    if command -v lscpu &> /dev/null; then
        local physical_cores=$(lscpu | grep "^Core(s) per socket:" | awk '{print $NF}')
        local num_sockets=$(lscpu | grep "^Socket(s):" | awk '{print $NF}')
        if [[ -n "$physical_cores" && -n "$num_sockets" ]]; then
            echo $((physical_cores * num_sockets))
            return 0
        fi
    fi
    
    # Fallback: assume hyperthreading (physical = logical / 2)
    local physical=$((logical_cores / 2))
    if [[ $physical -lt 1 ]]; then
        physical=$logical_cores
    fi
    echo $physical
}

get_numa_nodes() {
    if command -v numactl &> /dev/null; then
        numactl --show 2>/dev/null | grep "localalloc" | wc -l || echo 1
    else
        echo 1
    fi
}

###############################################################################
# GPU Configuration
###############################################################################

configure_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found. GPU configuration skipped."
        return 1
    fi
    
    log_info "Configuring GPU device $CUDA_DEVICE..."
    
    # Check GPU availability
    if ! nvidia-smi -i "$CUDA_DEVICE" &> /dev/null; then
        log_error "GPU device $CUDA_DEVICE not found!"
        return 1
    fi
    
    # Get GPU name
    local gpu_name=$(nvidia-smi -i "$CUDA_DEVICE" --query-gpu=name --format=csv,noheader)
    log_success "GPU: $gpu_name"
    
    # Get GPU compute capability (SM version)
    local gpu_compute_cap=$(nvidia-smi -i "$CUDA_DEVICE" --query-gpu=compute_cap --format=csv,noheader)
    log_success "Compute Capability: $gpu_compute_cap"
    
    # Lock GPU clocks for deterministic performance (requires root)
    if [[ "$LOCK_GPU_CLOCKS" == "true" ]]; then
        if [[ $EUID -eq 0 ]]; then
            log_info "Locking GPU clocks to maximum frequency..."
            nvidia-smi -i "$CUDA_DEVICE" -lgc 0 2>/dev/null && log_success "GPU clocks locked" || log_warning "Failed to lock GPU clocks (requires higher privileges)"
        else
            log_warning "GPU clock locking requires root privileges. Skipping. Run with sudo for best performance."
        fi
    fi
    
    # Query GPU memory
    local gpu_memory=$(nvidia-smi -i "$CUDA_DEVICE" --query-gpu=memory.total --format=csv,noheader | awk '{print $1}')
    log_success "GPU Memory: ${gpu_memory} MB"
    
    return 0
}

###############################################################################
# CPU Configuration
###############################################################################

configure_cpu() {
    local phys_cores=$(detect_physical_cores)
    log_success "Detected $phys_cores physical CPU cores"
    
    # Set OMP threads to physical cores only
    export OMP_NUM_THREADS=$phys_cores
    export OMP_DYNAMIC=false
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores
    
    log_success "OMP_NUM_THREADS set to $phys_cores"
    
    # NUMA binding
    if [[ "$BIND_NUMA" == "true" && -x "$(command -v numactl)" ]]; then
        local numa_nodes=$(get_numa_nodes)
        log_info "Detected $numa_nodes NUMA node(s)"
        
        if [[ $numa_nodes -gt 1 ]]; then
            log_info "NUMA binding: local allocations only"
            export NUMA_CMD="numactl --localalloc"
        else
            export NUMA_CMD=""
        fi
    else
        export NUMA_CMD=""
    fi
}

###############################################################################
# CUDA Configuration
###############################################################################

configure_cuda() {
    # Set CUDA device visibility
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
    log_success "CUDA_VISIBLE_DEVICES set to $CUDA_DEVICE"
    
    # Performance tuning
    export CUDA_LAUNCH_BLOCKING=0          # Async kernel launches
    export CUDA_DEVICE_ORDER=PCI_BUS_ID    # Deterministic device ordering
    export TF_FORCE_GPU_ALLOW_GROWTH=true  # Allow dynamic memory growth
    
    # CUDA graph optimization (uses CUDA graphs for better throughput)
    export CUDA_GRAPHS_ENABLED=1
    
    log_success "CUDA environment configured"
}

###############################################################################
# Server Configuration
###############################################################################

get_server_args() {
    local args=(
        # Core performance settings
        "--host 0.0.0.0"
        "--port $SERVER_PORT"
        "--n-gpu-layers $N_GPU_LAYERS"
        "--batch-size $BATCH_SIZE"
        "--threads $OMP_NUM_THREADS"
        
        # Memory optimization
        "--cache-type-k $CACHE_TYPE_K"
        "--cache-type-v $CACHE_TYPE_V"
        
        # Disable verbose output for cleaner profiling
        "--log-disable"
    )
    
    # Add optional flags based on configuration
    if [[ "$VERBOSE_LOGGING" == "true" ]]; then
        args=("${args[@]//--log-disable/}")  # Remove log-disable
    fi
    
    # Add model path if provided
    if [[ -n "$MODEL_PATH" && -f "$MODEL_PATH" ]]; then
        args+=(--model "$MODEL_PATH")
    fi
    
    echo "${args[@]}"
}

###############################################################################
# Main Entry Point
###############################################################################

main() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║      llama.cpp Optimized Server Startup Script              ║"
    echo "║                                                            ║"
    echo "║   GPU Performance Optimization & Tuning Framework           ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Check if server binary exists
    if [[ ! -f "$LLAMA_SERVER_BIN" ]]; then
        log_error "Server binary not found: $LLAMA_SERVER_BIN"
        log_info "Please build with: mkdir build_cuda_mmq_moe && cd build_cuda_mmq_moe && cmake -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CUDA=ON .."
        exit 1
    fi
    log_success "Server binary found: $LLAMA_SERVER_BIN"
    
    # Configure systems
    log_info "Configuring CPU..."
    configure_cpu
    
    log_info "Configuring GPU..."
    configure_gpu || true  # Don't fail if GPU config is unavailable
    
    log_info "Configuring CUDA..."
    configure_cuda
    
    # Prepare server command
    local server_args=$(get_server_args)
    local launch_cmd="$LLAMA_SERVER_BIN $server_args"
    
    if [[ -n "$NUMA_CMD" ]]; then
        launch_cmd="$NUMA_CMD $launch_cmd"
    fi
    
    # Display final configuration
    echo ""
    echo -e "${BLUE}=== Final Server Configuration ===${NC}"
    echo "Binary: $LLAMA_SERVER_BIN"
    echo "GPU Device: $CUDA_DEVICE"
    echo "CPU Threads: $OMP_NUM_THREADS (physical cores)"
    echo "Batch Size: $BATCH_SIZE"
    echo "GPU Layers: $N_GPU_LAYERS"
    echo "KV Cache Type: K=$CACHE_TYPE_K, V=$CACHE_TYPE_V"
    echo "Server Port: $SERVER_PORT"
    [[ -n "$NUMA_CMD" ]] && echo "NUMA Binding: Enabled (local alloc)" || echo "NUMA Binding: Disabled"
    echo ""
    echo -e "${YELLOW}Server Command:${NC}"
    echo "$launch_cmd"
    echo ""
    
    # Launch server with optimized settings
    log_info "Starting llama-server..."
    echo -e "${GREEN}=====================================${NC}"
    eval "$launch_cmd"
}

# Run main function
main "$@"
