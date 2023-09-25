// This file is for internal test purposes only and not part of the Trixi GPU framework
// Implements launch configuration and GPU kernels via CUDA and C++
// Focus on PDE solver with DG method for 1D problems

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <algorithm>

using namespace std;

// CUDA kernel configurator for 1D array computing
dim3 configurator_1d(void* kernelFun, int arrayLength) {
    cudaFuncAttributes attributes;
    cudaFuncGetAttributes(&attributes, kernelFun);
    
    int threads = min(arrayLength, attributes.maxThreadsPerBlock);
    int blocks = ceil(static_cast<float>(arrayLength) / threads);
    
    return dim3(blocks, threads);
}

// CUDA kernel configurator for 2D array computing
dim3 configurator_2d(void* kernelFun, int arrayWidth, int arrayHeight) {
    cudaFuncAttributes attributes;
    cudaFuncGetAttributes(&attributes, kernelFun);

    int threadsPerDimension = static_cast<int>(floor(sqrt(min(arrayWidth * arrayHeight, attributes.maxThreadsPerBlock))));
    dim3 threads(threadsPerDimension, threadsPerDimension);
    
    dim3 blocks(ceil(static_cast<float>(arrayWidth) / threads.x), ceil(static_cast<float>(arrayHeight) / threads.y));

    return blocks;
}

// CUDA kernel configurator for 3D array computing
dim3 configurator_3d(void* kernelFun, int arrayWidth, int arrayHeight, int arrayDepth) {
    cudaFuncAttributes attributes;
    cudaFuncGetAttributes(&attributes, kernelFun);

    int threadsPerDimension = static_cast<int>(floor(cbrt(min(arrayWidth * arrayHeight * arrayDepth, attributes.maxThreadsPerBlock))));
    dim3 threads(threadsPerDimension, threadsPerDimension, threadsPerDimension);
    
    dim3 blocks(ceil(static_cast<float>(arrayWidth) / threads.x), ceil(static_cast<float>(arrayHeight) / threads.y), ceil(static_cast<float>(arrayDepth) / threads.z));

    return blocks;
}



