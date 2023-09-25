// This file is for internal test purposes only and not part of the Trixi GPU framework
// Implements launch configuration and GPU kernels via CUDA and C++
// Focus on PDE solver with DG method for 3D problems

// Include libraries and header files
#include <cuda_runtime.h>
#include <iostream>

// Using namespaces
using namespace std;

// TODO: Define matrix structs to simplify kernel calls

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
void copy_to_gpu(float*** &du_device, double*** du_host, float*** &u_device, double*** u_host, int width, int height, int depth) {
    
    // 3D extent for allocation
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height^3, depth); // We treat it as a 3D array with height = height^3
    
    // Allocate memory for du on the GPU and set to zero
    cudaPitchedPtr devDuPitchedPtr;
    cudaMalloc3D(&devDuPitchedPtr, extent);
    cudaMemset3D(devDuPitchedPtr, 0, extent);
    
    // Allocate memory for u on the GPU
    cudaPitchedPtr devUPitchedPtr;
    cudaMalloc3D(&devUPitchedPtr, extent);
    
    // Convert u from double to float and copy to GPU
    cudaMemcpy3DParms copyParams = {0};
    float* temp_u_float = new float[width * height^3 * depth];
    
    int idx = 0;
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height^3; y++) {
            for (int x = 0; x < width; x++) {
                temp_u_float[idx++] = static_cast<float>(u_host[z][y][x]);
            }
        }
    }

    copyParams.srcPtr   = make_cudaPitchedPtr((void*)temp_u_float, width * sizeof(float), width, height^3);
    copyParams.dstPtr   = devUPitchedPtr;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    // Assign the pointers to the device memory
    du_device = (float***)devDuPitchedPtr.ptr;
    u_device = (float***)devUPitchedPtr.ptr;
    
    delete[] temp_u_float;
}

// Copy data from device to host (from float to double)
void copy_to_cpu(float*** du_device, double*** &du_host, float*** u_device, double*** &u_host, int width, int height, int depth) {

    // 3D extent for copy
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height^3, depth); // We treat it as a 3D array with height = height^3
    
    // Temporary buffer for float data from the device
    float* temp_u_float = new float[width * height^3 * depth];
    float* temp_du_float = new float[width * height^3 * depth];

    cudaMemcpy3DParms copyParamsU = {0};
    copyParamsU.dstPtr   = make_cudaPitchedPtr((void*)temp_u_float, width * sizeof(float), width, height^3);
    copyParamsU.srcPtr   = make_cudaPitchedPtr((void*)u_device, width * sizeof(float), width, height^3);
    copyParamsU.extent   = extent;
    copyParamsU.kind     = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&copyParamsU);

    cudaMemcpy3DParms copyParamsDu = {0};
    copyParamsDu.dstPtr   = make_cudaPitchedPtr((void*)temp_du_float, width * sizeof(float), width, height^3);
    copyParamsDu.srcPtr   = make_cudaPitchedPtr((void*)du_device, width * sizeof(float), width, height^3);
    copyParamsDu.extent   = extent;
    copyParamsDu.kind     = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&copyParamsDu);

    // Convert float data back to double and store in u_host
    int idx = 0;
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height^3; y++) {
            for (int x = 0; x < width; x++) {
                u_host[z][y][x] = static_cast<double>(temp_u_float[idx]);
                du_host[z][y][x] = static_cast<double>(temp_du_float[idx]);
                idx++;
            }
        }
    }

    delete[] temp_u_float;
    delete[] temp_du_float;

    // Free GPU memory
    cudaFree(du_device);
    cudaFree(u_device);
}


