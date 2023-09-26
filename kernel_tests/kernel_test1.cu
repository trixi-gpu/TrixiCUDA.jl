#include <iostream>

// CUDA kernel to add two arrays element-wise
__global__ void addArrays(float *a, float *b, float *result, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        result[tid] = a[tid] + b[tid];
    }
}

int main() {
    int N = 1024; // Size of arrays
    size_t size = N * sizeof(float);

    float *a, *b, *result;       // Host arrays
    float *d_a, *d_b, *d_result; // Device arrays

    // Allocate memory on the host
    a = new float[N];
    b = new float[N];
    result = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_result, size);

    // Copy input data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N);

    // Check kernel attributes
    cudaFuncAttributes attributes;
    cudaFuncGetAttributes(&attributes, addArrays);
    std::cout << "Max threads per block: " << attributes.maxThreadsPerBlock << std::endl;

    // Copy result from device to host
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < N; ++i) {
        if (result[i] != 3.0f) {
            std::cerr << "Mismatch at element " << i << ": " << result[i] << std::endl;
            break;
        }
    }

    // Clean up
    delete[] a;
    delete[] b;
    delete[] result;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}