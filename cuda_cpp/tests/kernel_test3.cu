// Kernel test file for 3D array problem

#include <iostream>

// Kernel Definition
__global__ void My3DKernel(cudaPitchedPtr pitchedPtr, int width, int height, int depth) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < width && y < height && z < depth) {
        char *devPtr = (char *)pitchedPtr.ptr;
        size_t pitch = pitchedPtr.pitch;
        size_t slicePitch = pitch * height;

        float *slice = (float *)((char *)devPtr + z * slicePitch);
        float *row = (float *)((char *)slice + y * pitch);
        row[x] += 1.0f;
    }
}

int main() {
    int width = 4, height = 4, depth = 4;
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);

    // Initialize host data
    float *h_data = new float[width * height * depth];
    for (int i = 0; i < width * height * depth; i++) {
        h_data[i] = 1.0f;
    }

    // Allocate 3D memory on GPU
    cudaPitchedPtr pitchedPtr;
    cudaMalloc3D(&pitchedPtr, extent);

    // Copy data to 3D memory on GPU
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void *)h_data, width * sizeof(float), width, height);
    copyParams.dstPtr = pitchedPtr;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    // Set grid and block sizes
    dim3 threadsPerBlock(4, 4, 4);
    dim3 numBlocks(4, 4, 4);

    My3DKernel<<<numBlocks, threadsPerBlock>>>(pitchedPtr, width, height, depth);

    // Copy 3D data back to host
    copyParams.srcPtr = pitchedPtr;
    copyParams.dstPtr = make_cudaPitchedPtr((void *)h_data, width * sizeof(float), width, height);
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&copyParams);

    // Test the results
    bool success = true;
    for (int i = 0; i < width * height * depth; i++) {
        if (h_data[i] != 2.0f) {
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
    cudaFree(pitchedPtr.ptr);

    return 0;
}
