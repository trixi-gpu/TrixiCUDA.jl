#include <iostream>
#include <cmath>

// Host code
void copy_to_gpu(float*** &du_device, double*** du_host, float*** &u_device, double*** u_host, int width, int height, int depth) {
    
    // 3D extent for allocation
    cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
    
    // Allocate memory for du on the GPU and set to zero
    cudaPitchedPtr devDuPitchedPtr;
    cudaMalloc3D(&devDuPitchedPtr, extent);
    cudaMemset3D(devDuPitchedPtr, 0, extent);
    
    // Allocate memory for u on the GPU
    cudaPitchedPtr devUPitchedPtr;
    cudaMalloc3D(&devUPitchedPtr, extent);
    
    // Convert u from double to float and copy to GPU
    cudaMemcpy3DParms copyParams = {0};
    float* temp_u_float = new float[width * height * depth];
    
    int idx = 0;
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                temp_u_float[idx++] = static_cast<float>(u_host[z][y][x]);
            }
        }
    }

    copyParams.srcPtr   = make_cudaPitchedPtr((void*)temp_u_float, width * sizeof(float), width, height);
    copyParams.dstPtr   = devUPitchedPtr;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    // Assign the pointers (You might need a way to handle these pointers outside this function)
    du_device = (float***)devDuPitchedPtr.ptr;
    u_device = (float***)devUPitchedPtr.ptr;
    
    delete[] temp_u_float;
}


bool areArraysEqual(float*** array1, double*** array2, int width, int height, int depth) {
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (std::abs(array1[z][y][x] - array2[z][y][x]) > 1e-6) {  // using a small threshold for floating point comparison
                    return false;
                }
            }
        }
    }
    return true;
}

void test_copy_to_gpu() {
    const int width = 64, height = 64, depth = 64;

    // Allocate and initialize host memory
    double*** du_host = new double**[depth];
    double*** u_host = new double**[depth];
    for (int z = 0; z < depth; ++z) {
        du_host[z] = new double*[height];
        u_host[z] = new double*[height];
        for (int y = 0; y < height; ++y) {
            du_host[z][y] = new double[width];
            u_host[z][y] = new double[width];
            for (int x = 0; x < width; ++x) {
                du_host[z][y][x] = 0.0; // This should be zeros on the GPU as well
                u_host[z][y][x] = static_cast<double>(x + y + z); // Some arbitrary initialization for testing
            }
        }
    }

    // Pointers for device memory
    float*** du_device = nullptr;
    float*** u_device = nullptr;

    // Call our function
    copy_to_gpu(du_device, du_host, u_device, u_host, width, height, depth);

    // Print host arrays
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::cout << u_host[z][y][x] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "------" << std::endl;  // Separate different depths
    }

    // For now, let's print a message to know the function was called.
    std::cout << "copy_to_gpu function was called. Verification functionality should be added." << std::endl;

    // Cleanup host memory
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            delete[] du_host[z][y];
            delete[] u_host[z][y];
        }
        delete[] du_host[z];
        delete[] u_host[z];
    }
    delete[] du_host;
    delete[] u_host;

    // Cleanup device memory
    // cudaFree(du_device);  // Caution: This would be more complex with 3D memory allocations
    // cudaFree(u_device);
}

int main() {
    test_copy_to_gpu();
    return 0;
}
