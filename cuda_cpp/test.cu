#include "matrix.h"
#include <iostream>

// ... Your Matrix5D struct and functions here ...

__global__ void testKernel(Matrix5D mat) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    int layer1 = threadIdx.z;
    int layer2 = blockIdx.x;
    int layer3 = blockIdx.y;

    float value = static_cast<float>(row + col + layer1 + layer2 + layer3);

    setElement5D(mat, row, col, layer1, layer2, layer3, value);

    float retrievedValue = getElement5D(mat, row, col, layer1, layer2, layer3);

    if (value != retrievedValue) {
        printf("Mismatch at (%d,%d,%d,%d,%d). Expected: %f, Got: %f\n", row, col, layer1, layer2,
               layer3, value, retrievedValue);
    }
}

int main() {
    const int width = 4, height = 4, depth1 = 4, depth2 = 4, depth3 = 4;
    Matrix5D mat(width, height, depth1, depth2, depth3);

    testKernel<<<dim3(depth2, depth3), dim3(width, height, depth1)>>>(mat);

    /*cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    } */

    return 0;
}
