#include "configurator.h"
#include <iostream>

// Assume your configurator1D function is defined here

__global__ void simpleKernel(float *data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] *= 2.0f; // Just double the value for testing
    }
}

int main() {
    const int N = 10007; // Array length
    float *h_data = new float[N];
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    auto config = configurator1D((void *)simpleKernel, N);
    std::cout << "Block size: (" << config.second.x << ", " << config.second.y << ", "
              << config.second.z << ")" << std::endl;
    std::cout << "Grid size: (" << config.first.x << ", " << config.first.y << ", "
              << config.first.z << ")" << std::endl;

    simpleKernel<<<config.first, config.second>>>(d_data, N);

    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verification
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_data[i] != static_cast<float>(i) * 2.0f) {
            correct = false;
            std::cerr << "Mismatch at " << i << " got: " << h_data[i]
                      << " expected: " << static_cast<float>(i) * 2.0f << std::endl;
        }
    }

    if (correct) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    delete[] h_data;
    cudaFree(d_data);
    return 0;
}
