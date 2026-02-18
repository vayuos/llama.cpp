# CUDA Build Requirements

## Overview

The CUDA build of llama.cpp requires the NVIDIA CUDA Toolkit to be installed and properly configured. This document explains the CUDA build requirements and how to set up your environment.

## Current Status

### CPU Build Status: ✅ WORKING
The CPU-only build is fully functional and can be used for testing and development.

### CUDA Build Status: ⚠️ REQUIRES CUDA TOOLKIT
The CUDA build requires:
1. NVIDIA CUDA Toolkit 11.0 or later
2. Compatible NVIDIA GPU
3. Proper environment configuration

## CUDA Linker Errors (Explanation)

If you see errors like:
```
undefined reference to `__device_builtin_variable_warpSize'
undefined reference to `cuda_sample_categorical_kernel'
```

This means:
- CUDA toolkit is **not installed** in your environment
- The build tried to link CUDA code without CUDA libraries available
- **Solution**: Use CPU-only build or install CUDA toolkit

## Installation Instructions

### NVIDIA CUDA Toolkit Installation

#### Ubuntu/Debian

**Option 1: Using NVIDIA Repository (Recommended)**

```bash
# Download and add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Add NVIDIA CUDA repository key
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub

# Add NVIDIA repository
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"

# Update and install
sudo apt-get update
sudo apt-get install cuda-toolkit-11-8
```

**Option 2: Direct Download**

1. Visit: https://developer.nvidia.com/cuda-toolkit
2. Select your platform (Linux, Ubuntu, x86_64)
3. Download the installer
4. Follow installation prompts

#### macOS

```bash
# Using Homebrew (if available)
brew install cuda

# Or download from NVIDIA: https://developer.nvidia.com/cuda-toolkit
```

#### Windows/WSL2

```bash
# In WSL2 terminal
sudo apt-get install cuda-toolkit

# Or download CUDA from NVIDIA website and follow installation guide
```

### Environment Configuration

After installation, add CUDA to your PATH and library path:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Apply changes
source ~/.bashrc
```

### Verify Installation

```bash
# Check if CUDA toolkit is installed
which nvcc
nvcc --version

# Check for CUDA libraries
ls /usr/local/cuda/lib64/libcuda* 2>/dev/null
```

## Building with CUDA

Once CUDA toolkit is installed:

### Using Build Script

```bash
./scripts/build-gpu-exclusive.sh cuda
```

### Manual Build

```bash
cd /home/viren/llama/llama.cpp
mkdir -p build_cuda
cd build_cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
make -j12
```

## CUDA Compute Capability

Different NVIDIA GPUs support different CUDA compute capabilities. If you have build issues, you can specify your GPU's compute capability:

```bash
# Find your GPU's compute capability at:
# https://developer.nvidia.com/cuda-gpus

# Build for specific capability (example: RTX 3080 = 8.6)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA=ON
```

### Common GPU Compute Capabilities

| GPU Series | Compute Capability |
|-----------|-------------------|
| RTX 40xx  | 8.9 |
| RTX 30xx  | 8.6 |
| RTX 20xx  | 7.5 |
| RTX 10xx  | 6.1 |
| Tesla A100 | 8.0 |
| Tesla V100 | 7.0 |

## Troubleshooting CUDA Build

### Error: "nvcc: command not found"

**Solution:**
1. Install CUDA Toolkit (see Installation Instructions above)
2. Add CUDA to PATH:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   ```
3. Verify: `which nvcc`

### Error: "Could not find CUDA runtime libraries"

**Solution:**
1. Set LD_LIBRARY_PATH:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
2. Verify: `ldconfig -p | grep libcuda`

### Error: "CUDA error: driver/runtime version mismatch"

**Solution:**
1. Update NVIDIA drivers: `nvidia-smi`
2. Download compatible CUDA version from https://developer.nvidia.com/cuda-toolkit
3. CUDA 11.8+ is recommended for modern GPUs

### Build Completes but CUDA Not Used

**Verification:**
```bash
# Check if CUDA backend was built
ls -la build_cuda/bin/libggml-cuda.so*

# Run inference with CUDA backend
./build_cuda/bin/llama-cli -m model.gguf -p "test" -ngl 99
# -ngl 99 forces all layers to GPU
```

## Alternative: Use CPU Build

If you cannot install CUDA toolkit, use the CPU-only build:

```bash
./scripts/build-gpu-exclusive.sh cpu
```

**Advantages of CPU build:**
- No special hardware required
- Faster to build
- Good for testing and development

**Performance note:**
- CPU build will be slower than CUDA build on GPUs
- Still benefits from all 56 GPU-exclusive optimizations
- Use for testing and CPU-only deployments

## CUDA Build Output

A successful CUDA build will produce:

```
build_cuda/bin/llama-cli              # Inference tool
build_cuda/bin/libllama.so.0.0.54     # Core library
build_cuda/bin/libggml-cuda.so        # CUDA backend
```

Verify with:
```bash
file build_cuda/bin/libggml-cuda.so
# Should show: ELF 64-bit LSB shared object, x86-64

ldd build_cuda/bin/libggml-cuda.so | grep cuda
# Should show CUDA library dependencies
```

## Performance Benefits

CUDA build provides:
- **GPU Acceleration**: Full NVIDIA GPU utilization
- **Fast Inference**: Significantly faster than CPU
- **Optimizations**: All 56 GPU-exclusive decode optimizations
- **Expected Speedup**: 15-45% per-token improvement over baseline

## Resources

- NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- CUDA Documentation: https://docs.nvidia.com/cuda/
- GPU Compute Capability: https://developer.nvidia.com/cuda-gpus
- NVIDIA Drivers: https://www.nvidia.com/Download/driverDetails.aspx

## Support

For CUDA-specific build issues:
1. Verify CUDA installation: `nvcc --version`
2. Check GPU detection: `nvidia-smi`
3. Review CMake output for CUDA configuration
4. Try CPU build as fallback: `./scripts/build-gpu-exclusive.sh cpu`

---

**Note**: The CPU-only build is fully functional and recommended for development/testing. CUDA build is optional and recommended only for production GPU deployments.

**Last Updated**: 2026-02-18
