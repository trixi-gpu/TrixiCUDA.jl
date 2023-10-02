#include "matrix.h"
#include <iostream>

__global__ void testKernel(Matrix2D M) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M.height && col < M.width) {
        float val = getElement2D(M, row, col);
        setElement2D(M, row, col, val + 1.0f);
    }
}

int main() {
    const int width = 4, height = 4;
    Matrix2D hostMatrix, deviceMatrix;
    hostMatrix.initOnHost(4, 4);
    deviceMatrix.initOnDevice(4, 4);

    // Initialize matrix with some values.
    for (int i = 0; i < width * height; i++) {
        hostMatrix.elements[i] = i * 1.0f;
    }

    // Print the results
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << hostMatrix.elements[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Copy matrix to device
    cudaMemcpy(deviceMatrix.elements, hostMatrix.elements, width * height * sizeof(float),
               cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(2, 2);
    dim3 blocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
    testKernel<<<blocks, threadsPerBlock>>>(deviceMatrix);

    // Copy results back to host
    cudaMemcpy(hostMatrix.elements, deviceMatrix.elements, width * height * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << hostMatrix.elements[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    hostMatrix.freeOnHost();
    deviceMatrix.freeOnDevice();

    return 0;
}
