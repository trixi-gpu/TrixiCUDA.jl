/*
This file is for internal test purposes only and is not part of the Trixi GPU framework. It
implements launch configurations and GPU kernels using CUDA and C++. The focus is on solving PDEs
with the DG method for 1D problems.
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
    size_t totalSize = width * height * depth * sizeof(float);

    // Allocate linear memory for `du` on the GPU and set to zero
    cudaMalloc((void **)&du_device, totalSize);
    cudaMemset(du_device, 0, totalSize);

    // Allocate linear memory for `u` on the GPU
    cudaMalloc((void **)&u_device, totalSize);

    // Convert `u` from double to float and copy to GPU in 1D format
    float *temp_u_float = new float[width * height * depth];
    for (int i = 0; i < width * height * depth; i++) {
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
    size_t totalSize = width * height * depth * sizeof(float);

    // Temporary buffers for float data from the device
    float *temp_u_float = new float[width * height * depth];
    float *temp_du_float = new float[width * height * depth];

    // Copy data from device (GPU) to temporary float buffers on host (CPU)
    cudaMemcpy(temp_u_float, u_device, totalSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_du_float, du_device, totalSize, cudaMemcpyDeviceToHost);

    // Convert float data back to double and store in 1D host arrays
    for (int idx = 0; idx < width * height * depth; idx++) {
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
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);

    // Allocate memory for `du` on the GPU and set to zero
    cudaPitchedPtr devDuPitchedPtr;
    cudaMalloc3D(&devDuPitchedPtr, extent);
    cudaMemset3D(devDuPitchedPtr, 0, extent);

    // Allocate memory for `u` on the GPU
    cudaPitchedPtr devUPitchedPtr;
    cudaMalloc3D(&devUPitchedPtr, extent);

    // Convert `u` from double to float and copy to GPU
    cudaMemcpy3DParms copyParams = {0};
    float *temp_u_float = new float[width * height * depth];

    int idx = 0;
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                temp_u_float[idx++] = static_cast<float>(u_host[z][y][x]);
            }
        }
    }

    copyParams.srcPtr =
        make_cudaPitchedPtr((void *)temp_u_float, width * sizeof(float), width, height);
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
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);

    // Temporary buffer for float data from the device
    float *temp_u_float = new float[width * height * depth];
    float *temp_du_float = new float[width * height * depth];

    cudaMemcpy3DParms copyParamsU = {0};
    copyParamsU.dstPtr =
        make_cudaPitchedPtr((void *)temp_u_float, width * sizeof(float), width, height);
    copyParamsU.srcPtr =
        make_cudaPitchedPtr((void *)u_device, width * sizeof(float), width, height);
    copyParamsU.extent = extent;
    copyParamsU.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&copyParamsU);

    cudaMemcpy3DParms copyParamsDu = {0};
    copyParamsDu.dstPtr =
        make_cudaPitchedPtr((void *)temp_du_float, width * sizeof(float), width, height);
    copyParamsDu.srcPtr =
        make_cudaPitchedPtr((void *)du_device, width * sizeof(float), width, height);
    copyParamsDu.extent = extent;
    copyParamsDu.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&copyParamsDu);

    // Convert float data back to double and store in `u_host` and `du_host`
    int idx = 0;
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
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

// CUDA kernel for calculating fluxes along normal direction 1
__global__ void flux_kernel(float *flux_arr, float *u, int u_dim1, int u_dim2, int u_dim3,
                            AbstractEquations equations) { // TODO: `AbstractEquations`
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < u_dim2 && k < u_dim3) {
        float *u_node = get_nodes_vars(u, equations, j, k); // TODO: `get_nodes_vars`

        float *flux_node = flux(u_node, 1, equations); // TODO: `flux`

        for (int ii = 0; ii < u_dim1; ii++) {
            flux_arr[ii * u_dim2 * u_dim3 + j * u_dim3 + k] = flux_node[ii];
        }

        // Make sure to deallocate any memory you dynamically allocated
        delete[] u_node;
        delete[] flux_node;
    }
}

// CUDA kernel for calculating weak form
__global__ void weak_form_kernel(float *du, float *derivative_dhat, float *flux_arr, int du_dim1,
                                 int du_dim2, int du_dim3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < du_dim1 && j < du_dim2 && k < du_dim3) {
        for (int ii = 0; ii < du_dim2; ii++) {
            int du_idx = i * du_dim2 * du_dim3 + j * du_dim3 + k;
            int derivative_idx = j * du_dim2 + ii;
            int flux_idx = i * du_dim2 * du_dim3 + ii * du_dim3 + k;

            du[du_idx] += derivative_dhat[derivative_idx] * flux_arr[flux_idx];
        }
    }
}

// CUDA kernel for calculating volume fluxes in direction x
__global__ void volume_flux_kernel(float *volume_flux_arr, float *u, int u_dim1, int u_dim2,
                                   int u_dim3, AbstractEquations equations) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < u_dim2 * u_dim2 && k < u_dim3) {
        int j1 = j / u_dim2;
        int j2 = j % u_dim2;

        float *u_node = get_nodes_vars(u, equations, j1, k);  // TODO: `get_nodes_vars`
        float *u_node1 = get_nodes_vars(u, equations, j2, k); // TODO: `get_nodes_vars`

        float *volume_flux_node = volume_flux(u_node, u_node1, 1, equations); // TODO: `volume_flux`

        for (int ii = 0; ii < u_dim1; ii++) {
            volume_flux_arr[ii * u_dim2 * u_dim2 * u_dim3 + j1 * u_dim2 * u_dim3 + j2 * u_dim3 +
                            k] = volume_flux_node[ii];
        }

        // Make sure to deallocate any memory you dynamically allocated
        delete[] u_node;
        delete[] u_node1;
        delete[] volume_flux_node;
    }
}

// CUDA kernel for calculating symmetric and nonsymmetric fluxes in direction x
__global__ void symmetric_noncons_flux_kernel(float *symmetric_flux_arr, float *noncons_flux_arr,
                                              float *u, float *derivative_split, int u_dim1,
                                              int u_dim2, int u_dim3, AbstractEquations equations) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < u_dim2 * u_dim2 && k < u_dim3) {
        int j1 = j / u_dim2;
        int j2 = j % u_dim2;

        float *u_node = get_nodes_vars(u, equations, j1, k);
        float *u_node1 = get_nodes_vars(u, equations, j2, k);

        float *symmetric_flux_node =
            symmetric_flux(u_node, u_node1, 1, equations); // TODO: `symmetric_flux`
        float *noncons_flux_node =
            nonconservative_flux(u_node, u_node1, 1, equations); // TODO: `nonconservative_flux`

        for (int ii = 0; ii < u_dim1; ii++) {
            symmetric_flux_arr[ii * u_dim2 * u_dim2 * u_dim3 + j1 * u_dim2 * u_dim3 + j2 * u_dim3 +
                               k] = symmetric_flux_node[ii];
            noncons_flux_arr[ii * u_dim2 * u_dim2 * u_dim3 + j1 * u_dim2 * u_dim3 + j2 * u_dim3 +
                             k] = noncons_flux_node[ii] * derivative_split[j1 * u_dim2 + j2];
        }

        // Deallocate dynamically allocated memory
        delete[] u_node;
        delete[] u_node1;
        delete[] symmetric_flux_node;
        delete[] noncons_flux_node;
    }
}

// CUDA kernel for calculating volume integrals
__global__ void volume_integral_kernel(float *du, float *derivative_split, float *volume_flux_arr,
                                       int du_dim1, int du_dim2, int du_dim3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < du_dim1 && j < du_dim2 && k < du_dim3) {

        // The size of the second axis of `du` is used in the loop iteration
        // This assumes that the second dimension of `du` and `derivative_split` are the same
        for (int ii = 0; ii < du_dim2; ++ii) {
            du[i * du_dim2 * du_dim3 + j * du_dim3 + k] +=
                derivative_split[j * du_dim2 + ii] *
                volume_flux_arr[i * du_dim2 * du_dim2 * du_dim3 + j * du_dim2 * du_dim3 +
                                ii * du_dim3 + k];
        }
    }
}

// CUDA kernel for calculating symmetric and nonsymmetric volume integrals
__global__ void volume_integral_kernel(float *du, float *derivative_split,
                                       float *symmetric_flux_arr, float *noncons_flux_arr,
                                       int du_dim1, int du_dim2, int du_dim3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < du_dim1 && j < du_dim2 && k < du_dim3) {
        float integral_contribution = 0.0f;

        // The size of the second axis of du is used in the loop iteration
        // This assumes that the second dimension of `du` and `derivative_split` are the same
        for (int ii = 0; ii < du_dim2; ++ii) {
            du[i * du_dim2 * du_dim3 + j * du_dim3 + k] +=
                derivative_split[j * du_dim2 + ii] *
                symmetric_flux_arr[i * du_dim2 * du_dim2 * du_dim3 + j * du_dim2 * du_dim3 +
                                   ii * du_dim3 + k];

            integral_contribution += noncons_flux_arr[i * du_dim2 * du_dim2 * du_dim3 +
                                                      j * du_dim2 * du_dim3 + ii * du_dim3 + k];
        }

        du[i * du_dim2 * du_dim3 + j * du_dim3 + k] += 0.5f * integral_contribution;
    }
}

// Launch CUDA kernels to calculate volume integrals

// CUDA kernel for prolonging two interfaces in direction x
__global__ void prolong_interfaces_kernel(float *interfaces_u, float *u, int *neighbor_ids,
                                          int interfaces_u_dim2, int interfaces_u_dim3, int u_dim2,
                                          int u_dim3) {
    // Compute the indices
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure that we don't go out of bounds
    if (j < interfaces_u_dim2 && k < interfaces_u_dim3) {
        int left_element = neighbor_ids[k];
        int right_element = neighbor_ids[interfaces_u_dim3 + k];

        // Memory access (considering flattened arrays for simplicity)
        interfaces_u[j * interfaces_u_dim3 + k] =
            u[j * u_dim2 * u_dim3 + (u_dim2 - 1) * u_dim3 + left_element];
        interfaces_u[interfaces_u_dim2 * interfaces_u_dim3 + j * interfaces_u_dim3 + k] =
            u[j * u_dim2 * u_dim3 + right_element];
    }
}

// CUDA kernel for calculating surface fluxes
__global__ void surface_flux_kernel(float *surface_flux_arr, float *interfaces_u,
                                    int surface_flux_arr_dim2, int surface_flux_arr_dim3,
                                    AbstractEquations equations) {
    // Compute the indices
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < surface_flux_arr_dim3) {
        float *u_ll, *u_rr;
        get_surface_node_vars(interfaces_u, equations, k, u_ll,
                              u_rr); // TODO: `get_surface_node_vars`

        float *surface_flux_node = surface_flux(u_ll, u_rr, 1, equations); // TODO: `surface_flux`

        for (int jj = 0; jj < surface_flux_arr_dim2; jj++) {
            surface_flux_arr[jj * surface_flux_arr_dim3 + k] =
                surface_flux_node[jj]; // Adjusted for flattened array
        }
    }
}

// CUDA kernel for calculating surface and both nonconservative fluxes
__global__ void surface_noncons_flux_kernel(float *surface_flux_arr, float *interfaces_u,
                                            float *noncons_left_arr, float *noncons_right_arr,
                                            int surface_flux_arr_dim3,
                                            AbstractEquations *equations) {

    // Compute the indices
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (k < surface_flux_arr_dim3) {
        float *u_ll, *u_rr;
        get_surface_node_vars(interfaces_u, *equations, k, u_ll,
                              u_rr); // TODO: `get_surface_node_vars`

        float *surface_flux_node = surface_flux(u_ll, u_rr, 1, *equations); // TODO: `surface_flux`
        float *noncons_left_node =
            nonconservative_flux(u_ll, u_rr, 1, *equations); // TODO: `nonconservative_flux`
        float *noncons_right_node =
            nonconservative_flux(u_rr, u_ll, 1, *equations); // TODO: `nonconservative_flux`

        for (int jj = 0; jj < surface_flux_arr_dim3; ++jj) {
            surface_flux_arr[jj * surface_flux_arr_dim3 + k] =
                surface_flux_node[jj]; // Adjusted based on the 1D memory layout
            noncons_left_arr[jj * surface_flux_arr_dim3 + k] =
                noncons_left_node[jj]; // Adjusted based on the 1D memory layout
            noncons_right_arr[jj * surface_flux_arr_dim3 + k] =
                noncons_right_node[jj]; // Adjusted based on the 1D memory layout
        }
    }
}
