#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Compute Capability: %d.%d\n", p.major, p.minor);
}
