// This file is for internal test purposes only and not part of the Trixi GPU framework
// Implements launch configuration and GPU kernels via CUDA and C++
// Focus on PDE solver with DG method for 1D problems

// Include libraries and header files
#include <cuda_runtime.h>
#include <iostream>

// Using namespaces
using namespace std;

// Kernel configurators 
//----------------------------------------------

// CUDA kernel configurator for 1D array computing
pair<dim3, dim3> configurator_1d(void* kernelFun, int arrayLength) {
    cudaFuncAttributes attributes;
    cudaFuncGetAttributes(&attributes, kernelFun);
    
    int threads = min(arrayLength, attributes.maxThreadsPerBlock);
    int blocks = ceil(static_cast<float>(arrayLength) / threads);
    
    return {dim3(blocks), dim3(threads)};
}

// CUDA kernel configurator for 2D array computing
pair<dim3, dim3> configurator_2d(void* kernelFun, int arrayWidth, int arrayHeight) {
    cudaFuncAttributes attributes;
    cudaFuncGetAttributes(&attributes, kernelFun);

    int threadsPerDimension = static_cast<int>(floor(sqrt(min(arrayWidth * arrayHeight, attributes.maxThreadsPerBlock))));

    dim3 threads(threadsPerDimension, threadsPerDimension);
    dim3 blocks(ceil(static_cast<float>(arrayWidth) / threads.x), ceil(static_cast<float>(arrayHeight) / threads.y));

    return {threads, blocks};
}

// CUDA kernel configurator for 3D array computing
pair<dim3, dim3> configurator_3d(void* kernelFun, int arrayWidth, int arrayHeight, int arrayDepth) {
    cudaFuncAttributes attributes;
    cudaFuncGetAttributes(&attributes, kernelFun);

    int threadsPerDimension = static_cast<int>(floor(cbrt(min(arrayWidth * arrayHeight * arrayDepth, attributes.maxThreadsPerBlock))));

    dim3 threads(threadsPerDimension, threadsPerDimension, threadsPerDimension);
    dim3 blocks(ceil(static_cast<float>(arrayWidth) / threads.x), ceil(static_cast<float>(arrayHeight) / threads.y), ceil(static_cast<float>(arrayDepth) / threads.z));

    return {threads, blocks};
}

// CUDA kernels
//----------------------------------------------

// Copy data from host to device (from double to float)
void copy_to_gpu(float*** du_device, const double*** du_host, float*** u_device, const double*** u_host, int dimX, int dimY, int dimZ) {
    int totalElements = dimX * dimY * dimZ;

    // Allocate memory on the GPU
    cudaMalloc(&du_device, totalElements * sizeof(float));
    cudaMalloc(&u_device, totalElements * sizeof(float));

    // Set du_device to zeros
    cudaMemset(*du_device, 0, totalElements * sizeof(float));

    // Convert double data on the CPU to float and then transfer to the GPU
    float* temp_u_float = new float[totalElements];

    int idx = 0;
    for (int i = 0; i < dimX; i++) {
        for (int j = 0; j < dimY; j++) {
            for (int k = 0; k < dimZ; k++) {
                temp_u_float[idx++] = static_cast<float>(u_host[i][j][k]);
            }
        }
    }

    cudaMemcpy(*u_device, temp_u_float, totalElements * sizeof(float), cudaMemcpyHostToDevice);

    delete[] temp_u_float;
}



