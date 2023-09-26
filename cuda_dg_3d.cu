/*
This file is for internal test purposes only and is not part of the Trixi GPU framework. It
implements launch configurations and GPU kernels using CUDA and C++. The focus is on solving PDEs
with the DG method for 3D problems.
*/

// Include libraries and header files
#include "header.h"
#include <cuda_runtime.h>
#include <iostream>

// Using namespaces
using namespace std;

// TODO: Define matrix structs to simplify CUDA kenerls and kernel calls

// Kernel configurators
//----------------------------------------------

// CUDA kernel configurator for 1D array computing
pair<dim3, dim3> configurator_1d(void *kernelFun, int arrayLength) {
    int blockSize;
    int minGridSize;

    // Get the potential block size for maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       kernelFun); // Use CUDA occupancy calculator

    int threads = blockSize;
    int blocks = ceil(static_cast<float>(arrayLength) / threads);

    return {dim3(blocks), dim3(threads)};
}

// CUDA kernel configurator for 2D array computing
pair<dim3, dim3> configurator_2d(void *kernelFun, int arrayWidth, int arrayHeight) {
    int blockSize;
    int minGridSize;

    // Get the potential block size for maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       kernelFun); // Use CUDA occupancy calculator

    int threadsPerDimension = static_cast<int>(sqrt(blockSize));

    dim3 threads(threadsPerDimension, threadsPerDimension);
    dim3 blocks(ceil(static_cast<float>(arrayWidth) / threads.x),
                ceil(static_cast<float>(arrayHeight) / threads.y));

    return {blocks, threads};
}

// CUDA kernel configurator for 3D array computing
pair<dim3, dim3> configurator_3d(void *kernelFun, int arrayWidth, int arrayHeight, int arrayDepth) {
    int blockSize;
    int minGridSize;

    // Get the potential block size for maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                       kernelFun); // Use CUDA occupancy calculator

    int threadsPerDimension = static_cast<int>(cbrt(blockSize));

    dim3 threads(threadsPerDimension, threadsPerDimension, threadsPerDimension);
    dim3 blocks(ceil(static_cast<float>(arrayWidth) / threads.x),
                ceil(static_cast<float>(arrayHeight) / threads.y),
                ceil(static_cast<float>(arrayDepth) / threads.z));

    return {blocks, threads};
}

// CUDA kernels
//----------------------------------------------

/*
Data Storage Decision: 1D (Flattened) vs. 3D Format on GPU

1D (Flattened) Format:
- Pros:
    - Linear memory access can provide better cache coherency and memory throughput in some cases.
    - Simplifies the indexing logic in kernels, as you only deal with a single index.
    - Easier interoperability with libraries or functions that expect linear memory.

- Cons:
    - The logic to map between 3D spatial coordinates and 1D indices might be less intuitive.
    - Can lead to divergent access patterns if neighboring threads access non-contiguous memory
locations.

3D Format:
- Pros:
    - More intuitive indexing based on spatial coordinates.
    - Can lead to coalesced memory accesses if neighboring threads access neighboring spatial
coordinates.
    - Easier to visualize and debug, especially when analyzing spatial patterns.

- Cons:
    - Might be slightly more overhead in indexing calculations.
    - Some GPU functions or libraries might expect linear memory and would require conversion.


Test both formats in the context of the specific application. Measure performance, ease of
development, and other relevant metrics. After careful consideration and based on empirical data and
specific application needs, we have currently chosen to use the 1D format.
*/

// TODO: Implement a function to convert du and u into du_host and u_host as flattened 1D arrays

// Copy data from host to device (from double to float)
void copy_to_gpu(float *&du_device, double *du_host, float *&u_device, double *u_host, int width,
                 int height, int depth) {

    // Calculate total size for the 1D array
    size_t totalSize = width * height ^ 3 * depth * sizeof(float);

    // Allocate linear memory for du on the GPU and set to zero
    cudaMalloc((void **)&du_device, totalSize);
    cudaMemset(du_device, 0, totalSize);

    // Allocate linear memory for u on the GPU
    cudaMalloc((void **)&u_device, totalSize);

    // Convert u from double to float and copy to GPU in 1D format
    float *temp_u_float = new float[width * height ^ 3 * depth];
    for (int i = 0; i < width * height ^ 3 * depth; i++) {
        temp_u_float[i] = static_cast<float>(u_host[i]);
    }

    // Copy the linear memory to the GPU
    cudaMemcpy(u_device, temp_u_float, totalSize, cudaMemcpyHostToDevice);

    delete[] temp_u_float;
}

// Copy data from device to host (from float to double)
void copy_to_cpu(float *du_device, double *&du_host, float *u_device, double *&u_host, int width,
                 int height, int depth) {

    // Calculate total size for the 1D array
    size_t totalSize = width * height ^ 3 * depth * sizeof(float);

    // Temporary buffers for float data from the device
    float *temp_u_float = new float[width * height ^ 3 * depth];
    float *temp_du_float = new float[width * height ^ 3 * depth];

    // Copy data from device (GPU) to temporary float buffers on host (CPU)
    cudaMemcpy(temp_u_float, u_device, totalSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_du_float, du_device, totalSize, cudaMemcpyDeviceToHost);

    // Convert float data back to double and store in 1D host arrays
    for (int idx = 0; idx < width * height ^ 3 * depth; idx++) {
        u_host[idx] = static_cast<double>(temp_u_float[idx]);
        du_host[idx] = static_cast<double>(temp_du_float[idx]);
    }

    delete[] temp_u_float;
    delete[] temp_du_float;

    // Free GPU memory
    cudaFree(du_device);
    cudaFree(u_device);
}

/* // Copy data from host to device (from double to float)
void copy_to_gpu(float ***&du_device, double ***du_host, float ***&u_device, double ***u_host,
                 int width, int height, int depth) {

    // 3D extent for allocation
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height ^ 3,
                                        depth); // We treat it as a 3D array with height = height^3

    // Allocate memory for du on the GPU and set to zero
    cudaPitchedPtr devDuPitchedPtr;
    cudaMalloc3D(&devDuPitchedPtr, extent);
    cudaMemset3D(devDuPitchedPtr, 0, extent);

    // Allocate memory for u on the GPU
    cudaPitchedPtr devUPitchedPtr;
    cudaMalloc3D(&devUPitchedPtr, extent);

    // Convert u from double to float and copy to GPU
    cudaMemcpy3DParms copyParams = {0};
    float *temp_u_float = new float[width * height ^ 3 * depth];

    int idx = 0;
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height ^ 3; y++) {
            for (int x = 0; x < width; x++) {
                temp_u_float[idx++] = static_cast<float>(u_host[z][y][x]);
            }
        }
    }

    copyParams.srcPtr =
        make_cudaPitchedPtr((void *)temp_u_float, width * sizeof(float), width, height ^ 3);
    copyParams.dstPtr = devUPitchedPtr;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    // Assign the pointers to the device memory
    du_device = (float ***)devDuPitchedPtr.ptr;
    u_device = (float ***)devUPitchedPtr.ptr;

    delete[] temp_u_float;
}

// Copy data from device to host (from float to double)
void copy_to_cpu(float ***du_device, double ***&du_host, float ***u_device, double ***&u_host,
                 int width, int height, int depth) {

    // 3D extent for copy
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height ^ 3,
                                        depth); // We treat it as a 3D array with height = height^3

    // Temporary buffer for float data from the device
    float *temp_u_float = new float[width * height ^ 3 * depth];
    float *temp_du_float = new float[width * height ^ 3 * depth];

    cudaMemcpy3DParms copyParamsU = {0};
    copyParamsU.dstPtr =
        make_cudaPitchedPtr((void *)temp_u_float, width * sizeof(float), width, height ^ 3);
    copyParamsU.srcPtr =
        make_cudaPitchedPtr((void *)u_device, width * sizeof(float), width, height ^ 3);
    copyParamsU.extent = extent;
    copyParamsU.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&copyParamsU);

    cudaMemcpy3DParms copyParamsDu = {0};
    copyParamsDu.dstPtr =
        make_cudaPitchedPtr((void *)temp_du_float, width * sizeof(float), width, height ^ 3);
    copyParamsDu.srcPtr =
        make_cudaPitchedPtr((void *)du_device, width * sizeof(float), width, height ^ 3);
    copyParamsDu.extent = extent;
    copyParamsDu.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&copyParamsDu);

    // Convert float data back to double and store in u_host
    int idx = 0;
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height ^ 3; y++) {
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
} */
