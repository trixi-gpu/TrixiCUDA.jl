#include "test.h"
#include <iostream>

__global__ void addPointsKernel(Point *a, Point *b, Point *c) {
    int idx = threadIdx.x;
    c[idx].x = a[idx].x + b[idx].x;
    c[idx].y = a[idx].y + b[idx].y;
}

int main() {
    const int numPoints = 1;
    Point h_a[numPoints], h_b[numPoints], h_c[numPoints];
    Point *d_a, *d_b, *d_c;

    // Initialize host data
    h_a[0].x = 1.0f;
    h_a[0].y = 2.0f;
    h_b[0].x = 3.0f;
    h_b[0].y = 4.0f;

    // Allocate device memory
    cudaMalloc(&d_a, numPoints * sizeof(Point));
    cudaMalloc(&d_b, numPoints * sizeof(Point));
    cudaMalloc(&d_c, numPoints * sizeof(Point));

    // Copy data to device
    cudaMemcpy(d_a, h_a, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, numPoints * sizeof(Point), cudaMemcpyHostToDevice);

    // Call the kernel
    addPointsKernel<<<1, numPoints>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);

    std::cout << "Result: (" << h_c[0].x << ", " << h_c[0].y << ")\n";

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
