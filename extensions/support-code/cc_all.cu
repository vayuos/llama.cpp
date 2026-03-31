#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int n;
    cudaGetDeviceCount(&n);
    for (int i = 0; i < n; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("Device %d: %s | CC %d.%d\n",
               i, p.name, p.major, p.minor);
    }
    return 0;
}
