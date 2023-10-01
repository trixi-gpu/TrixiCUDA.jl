// Kernel test file for 2D array problem

#include <iostream>

// Kernel Definition
__global__ void MyKernel(float *devPtr, size_t pitch, int width, int height) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int r = blockIdx.y * blockDim.y + threadIdx.y; // Row

    if (c < width && r < height) {
        float *row = (float *)((char *)devPtr + r * pitch);
        row[c] += 1.0f;
    }
}

int main() {
    const int width = 4, height = 4;
    float *devPtr;
    size_t pitch;
    float *h_data = new float[width * height]; // Create host memory

    // Initialize the host memory with some values
    for (int i = 0; i < width * height; i++) {
        h_data[i] = 1.0f;
    }

    cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);

    // Copy initialized data from host to device
    cudaMemcpy2D(devPtr, pitch, h_data, width * sizeof(float), width * sizeof(float), height,
                 cudaMemcpyHostToDevice);

    dim3 gridSize(16, 16);
    dim3 blockSize(16, 16);

    MyKernel<<<gridSize, blockSize>>>(devPtr, pitch, width, height);

    // Copy data back from device to host for validation
    cudaMemcpy2D(h_data, width * sizeof(float), devPtr, pitch, width * sizeof(float), height,
                 cudaMemcpyDeviceToHost);

    // Test the results
    bool success = true;
    for (int i = 0; i < width * height; i++) {
        if (h_data[i] != 2.0f) { // Check if the result is correct
            std::cerr << "Error: Value at " << i << " is " << h_data[i] << " (expected 2.0)"
                      << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    }

    // Cleanup
    delete[] h_data;
    cudaFree(devPtr);

    return 0;
}
